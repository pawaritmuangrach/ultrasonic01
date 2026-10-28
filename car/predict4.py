"""ใช้โมเดลรอบสี่/ห้า (model4 · ไฟล์ .pt จาก train_r4.py) ทำนายจากคลื่นสด

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ตัวกลางระหว่างโมเดลกับหน้าจอ live_map2.py)
═══════════════════════════════════════════════════════════════════════

หน้าตาเหมือน model3.MapPredictor ทุกอย่าง (push / warmup / score / dc / scale / amps)
live_map2 จึงสลับใช้ได้โดยไม่ต้องรู้ว่าข้างในเป็นโมเดลรุ่นไหน

ทำนายแบบเดียวกับ train_r4.predict ทุกขั้น ไม่งั้นภาพสดจะไม่ตรงกับคะแนนที่วัดไว้
  - ปรับขนาดคลื่นด้วย dc/scale ที่เก็บมากับโมเดล (คิดจากเฟรมเทรนเท่านั้น)
  - โหมดตัดช่วง: รันทุกช่วง 5 ms แล้วต่อเป็นภาพเดียว ช่องไหนหลายช่วงทายซ้อนกัน
    เอาช่วงที่มั่นใจที่สุด และระยะของช่วงนั้น
  - จุดตัดใช้ค่าที่ตั้งจากชุดเลือกตอนเทรน
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

import mapdata2 as MD
import model4 as M4
from mapdata2 import GW, GH, NEAR_CM, FAR_CM
from rig2 import PINS

TRAIN_RATE = 66313.0     # ค่าเดียวกับที่ train_r4.main ใช้วางช่วงเวลา ต้องตรงกัน
NSAMP = 871              # ความยาวเฟรมตอนเทรน (200 ซม.) · เฟรมสดที่ขาด/เกินเล็กน้อยจะถูกเติม/ตัด
LSTM_OUT = 320           # ขนาดเอาต์พุต LSTM ใน model4 · ใช้หาจำนวนช่วงจากน้ำหนักที่บันทึกไว้


class Predictor4:
    def __init__(self, path, smooth=1):
        import torch
        torch.set_num_threads(1)   # เธรดพูล torch ชนกับเธรดกล้อง (ดู model3.MapPredictor)
        self.torch = torch
        p = Path(path)
        if not p.exists():
            sys.exit(f"ไม่พบโมเดล {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.args = a
        self.stack = int(a["stack"])
        self.norm = a["norm"]
        self.dc, self.scale = ck["norm"]
        self.thr = float(ck["thr"])
        self.target = a.get("target", "scene")
        n_cond = ck["model"]["fc.1.weight"].shape[1] - LSTM_OUT
        if a["slice_ms"] > 0:
            self.plan = M4.slice_plan(TRAIN_RATE, NSAMP, a["slice_ms"], a["slice_step_ms"])
            nsamp = self.plan[0][1] - self.plan[0][0]
        else:
            self.plan, nsamp = None, NSAMP
        if len(self.plan or []) != n_cond:
            sys.exit(f"จำนวนช่วงเวลาไม่ตรงกับโมเดล ({len(self.plan or [])} เทียบ {n_cond}) "
                     f"· TRAIN_RATE/NSAMP ต้องตรงกับตอนเทรน")
        self.K = n_cond
        self.cond = np.eye(self.K, dtype=np.float32) if self.K else None
        self.net = M4.make_model(torch, torch.nn, GH, GW, self.stack, nsamp, n_cond, a["act"])
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.params = sum(t.numel() for t in self.net.parameters())
        self.score = self._score(p)
        self.name = p.stem
        self.buf = deque(maxlen=self.stack)
        self.occ = deque(maxlen=max(int(smooth), 1))
        self.dep = deque(maxlen=max(int(smooth), 1))
        self.amps = [0.0] * 4
        self.test_section = -1
        self.warmup()

    def _score(self, p):
        """คะแนนจากไฟล์ผล .json คู่กัน · ใช้ชุดรายงานถ้ามี ไม่งั้นชุดเลือก (บอกไว้ใน 'on')"""
        j = p.with_suffix(".json")
        if not j.exists():
            return {"thr": self.thr}
        r = json.loads(j.read_text(encoding="utf-8"))
        on = "test" if r.get("test") else "val"
        s = dict(r[on])
        s["on"] = r["args"]["test"] if on == "test" else r["args"]["val"]
        s["thr"] = self.thr
        return s

    def warmup(self, nsamp=NSAMP):
        """ซ้อม forward ก่อนเปิดกล้อง (เหตุผลเดียวกับ model3.MapPredictor.warmup)"""
        r = np.random.default_rng(0)
        fake = (self.dc + r.normal(0, 30, (4, int(nsamp)))).astype(np.uint16)
        for _ in range(self.stack):
            self.push({"counts": fake, "pins": list(PINS)})
        self.buf.clear()
        self.occ.clear()
        self.dep.clear()
        self.amps = [0.0] * 4

    def push(self, ping):
        x = fit_len(MD.read_counts(ping))
        v = x.astype(np.float32)
        self.amps = [float((v[i].max() - v[i].min()) / 4095 * 3.3 * 1000) for i in range(4)]
        if not self.buf:                      # ครั้งแรก เติมให้ครบทันที ไม่ต้องรอ 5 การยิง
            for _ in range(self.stack):
                self.buf.append(x)
        else:
            self.buf.append(x)
        w = M4.prep(np.stack(self.buf)[None], self.dc, self.scale, self.norm)
        o, d = self._forward(w)
        self.occ.append(o)
        self.dep.append(d)
        return np.median(self.occ, 0), np.median(self.dep, 0)

    def _forward(self, w):
        torch = self.torch
        with torch.no_grad():
            if not self.plan:
                ol, dl = self.net(torch.from_numpy(w).float())
                return (torch.sigmoid(ol)[0].numpy(),
                        (NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(dl))[0].numpy())
            # ทุกช่วงเป็นชุดเดียว (K ภาพ) รันครั้งเดียวเร็วกว่าวนทีละช่วง
            ws = np.concatenate([w[:, :, :, i0:i1] for i0, i1, _, _ in self.plan])
            ol, dl = self.net(torch.from_numpy(ws).float(), torch.from_numpy(self.cond))
            p = torch.sigmoid(ol).numpy()
            z = torch.sigmoid(dl).numpy()
        lo = np.array([s[2] for s in self.plan], np.float32)[:, None, None]
        hi = np.array([s[3] for s in self.plan], np.float32)[:, None, None]
        d = lo + (hi - lo) * z
        k = p.argmax(0)                       # ช่วงที่มั่นใจที่สุดของแต่ละช่อง
        return np.take_along_axis(p, k[None], 0)[0], np.take_along_axis(d, k[None], 0)[0]


def fit_len(x, n=NSAMP):
    """เฟรมสดยาวไม่เท่าตอนเทรนได้ ±1-2 จุด (นาฬิกา ADC) · เติมด้วยค่าท้าย หรือตัดทิ้ง"""
    if x.shape[-1] == n:
        return x
    if x.shape[-1] > n:
        return x[:, :n]
    return np.concatenate([x, np.repeat(x[:, -1:], n - x.shape[-1], 1)], 1)
