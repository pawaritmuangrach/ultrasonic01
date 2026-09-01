#!/usr/bin/env python3
"""ยืนดูผลสด ๆ — เอากฎจาก `rules.py` มาทายทิศแบบเรียลไทม์ แล้ววางเทียบกับกล้อง

    python car/predict.py --port COM5

หน้าจอแบ่งเป็น:
  ซ้าย   ภาพ depth พร้อมกรอบก้อนเป้าที่กล้องเห็น (เฉลยจริง)
  ขวาบน  แถบมุม -30..+30 องศา · **สามเหลี่ยม = เสียงทาย · เส้นขาว = กล้องเห็น**
  ขวากลาง สามช่องโซน ซ้าย/กลาง/ขวา ช่องที่ทายจะสว่าง (ขอบขาว = ที่ถูกต้อง)
  ขวาล่าง ความแรง 4 ช่อง · ค่า log4 · คะแนนสะสมตั้งแต่เปิดโปรแกรม
  ล่างสุด ประวัติ 500 เฟรม เส้นเหลือง = เสียงทาย เส้นขาว = กล้อง

**กล้องมีไว้ให้ดูเทียบเท่านั้น** ตัวทายใช้แต่เสียง ถ้าเอามือบังกล้อง เส้นเหลืองยังวิ่งอยู่

ปุ่ม: q/ESC ออก · s เซฟภาพ · space หยุด/เล่นต่อ · r ล้างคะแนน
"""
import argparse
import faulthandler
import gc
import json
import os
import sys
import time
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
faulthandler.enable()

import numpy as np
import numpy.ma          # noqa: F401  โหลดก่อนเปิดกล้อง — import ตอน OpenNI ทำงานทำให้ล้ม
import numpy.lib         # noqa: F401
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import features as F                                     # noqa: E402

W, H = 1180, 760
PAD = 14
DW, DH = 470, 352
BG, PANEL, LINE = (22, 24, 28), (32, 35, 41), (70, 76, 86)
OKC, BADC, WARNC = (110, 230, 130), (90, 90, 245), (80, 200, 250)
PRED = (70, 215, 250)          # เหลือง-ส้ม = คำทายจากเสียง
TRUE = (245, 245, 245)         # ขาว = เฉลยจากกล้อง
# ตำแหน่งบน plate_mini · TL=35 TR=34 BL=32 BR=33 (ดู car/rig2.py)
NAME = {35: "TOP LEFT", 34: "TOP RIGHT", 32: "BOTTOM LEFT", 33: "BOTTOM RIGHT"}
DEG_MAX = 30.0
_CANVAS = None


def load_rule(name):
    p = Path(HERE) / "data" / f"_{name}_rule.json"
    if not p.exists():
        sys.exit(f"ยังไม่มีไฟล์กฎ {p}\n"
                 f"ต้องรัน  python car/rules.py --name {name} --test 1  ก่อนหนึ่งครั้ง")
    return json.loads(p.read_text(encoding="utf-8"))


class Predictor:
    """คำนวณ log4 จากเฟรมเดียว แล้วเกลี่ยด้วยมัธยฐานย้อนหลัง

    **ย้อนหลังล้วน** ไม่แอบใช้เฟรมอนาคตเหมือนตอนวัดผลออฟไลน์ — วัดแล้วผลเท่ากัน
    (MAE 5.28 เทียบกับ 5.30 องศา) แต่ใช้ได้จริงบนรถที่ต้องตอบทันที
    """

    def __init__(self, rule):
        self.r = rule
        self.buf = deque(maxlen=max(int(rule["smooth"]), 1))
        self.log4 = 0.0
        self.amps = [0.0] * 4
        self.rng = 0.0
        self.stale = 0

    def push(self, ping):
        """คืน (มุมองศา, สดหรือไม่) — None ถ้ายังไม่เคยได้ยินอะไรเลย

        เฟรมที่เสียงกลับอ่อนกว่าเกณฑ์ **ไม่ถูกใส่ลงบัฟเฟอร์** แต่ยังคืนคำตอบเดิม
        พร้อมธง stale เพราะรถที่หยุดตอบทุกครั้งที่เอคโค่วูบคือรถที่ควบคุมไม่ได้
        (เอคโค่จากคนเป็น specular วูบ 40 เท่าได้ในเฟรมเดียว)
        """
        c, rate = ping["counts"], float(ping["rate"])
        idx = {int(p): i for i, p in enumerate(ping["pins"])}
        envs = [F.envelope_of(c[i], rate) for i in range(c.shape[0])]
        k, rng, _ = F.common_peak(envs, rate)
        a = [float(envs[idx[p]][k]) * 1e3 for p in self.r["pins"]]
        self.amps, self.rng = a, float(rng)
        fresh = max(a) >= self.r["min_amp"]
        if fresh:
            self.buf.append(np.log((a[0] + a[1] + 1.0) / (a[2] + a[3] + 1.0)))
            self.stale = 0
        else:
            self.stale += 1
        if not self.buf:
            return None, False
        self.log4 = float(np.median(self.buf))
        return self.r["slope"] * self.log4 + self.r["intercept"], fresh

    def zone(self, deg):
        return int(np.searchsorted(self.r["zones"], deg))


# ------------------------------------------------------------------ วาดหน้าจอ
def _text(img, s, xy, sc=0.45, col=(200, 206, 216), th=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def _depth_panel(img, depth, lab):
    import cv2
    n = np.clip(depth.astype(np.float32), 0, 2000) / 2000.0
    vis = cv2.applyColorMap((255 - n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth == 0] = 0
    vis = cv2.resize(vis, (DW, DH), interpolation=cv2.INTER_NEAREST)
    x0, y0 = PAD, 96
    img[y0:y0 + DH, x0:x0 + DW] = vis
    if lab is not None:
        deg, cm, cov = lab
        bx = int(x0 + (deg / 58.4 + 0.5) * DW)
        cv2.line(img, (bx, y0), (bx, y0 + DH), TRUE, 2)
        _text(img, f"{cm:.0f}cm {deg:+.0f}deg", (min(max(bx - 58, x0 + 4), x0 + DW - 130),
                                                 y0 + DH - 12), 0.55, TRUE, 2)
        _text(img, f"target fills {cov:.0%} of band", (x0 + 6, y0 + 20), 0.42, TRUE)
    else:
        _text(img, "camera sees no target", (x0 + 14, y0 + 30), 0.6, BADC, 2)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + DW, y0 + DH), LINE, 1)
    _text(img, "DEPTH  -  ground truth, for comparison only", (x0, y0 - 10), 0.45,
          (150, 156, 166))


def _gauge(img, x0, y0, w, pdeg, tdeg):
    """แถบมุม: สามเหลี่ยมสีเหลือง = เสียงทาย · เส้นขาว = กล้อง"""
    import cv2
    h = 46
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    mid = x0 + w // 2
    for d in range(-30, 31, 10):
        px = int(x0 + (d / DEG_MAX / 2 + 0.5) * w)
        cv2.line(img, (px, y0 + h - 9), (px, y0 + h), (95, 101, 111), 1)
        _text(img, f"{d:+d}", (px - 11, y0 + h + 15), 0.36, (120, 126, 136))
    cv2.line(img, (mid, y0), (mid, y0 + h), (95, 101, 111), 1)
    if tdeg is not None:
        px = int(x0 + (np.clip(tdeg, -DEG_MAX, DEG_MAX) / DEG_MAX / 2 + 0.5) * w)
        cv2.line(img, (px, y0 + 2), (px, y0 + h - 2), TRUE, 2)
    if pdeg is not None:
        px = int(x0 + (np.clip(pdeg, -DEG_MAX, DEG_MAX) / DEG_MAX / 2 + 0.5) * w)
        cv2.drawContours(img, [np.array([[px, y0 + h - 4], [px - 11, y0 + 6],
                                         [px + 11, y0 + 6]])], 0, PRED, -1)
    _text(img, "PREDICTED ANGLE  (sound only)", (x0, y0 - 9), 0.45, PRED)
    _text(img, "white = camera", (x0 + w - 105, y0 - 9), 0.4, (140, 146, 156))


def _zones(img, x0, y0, w, names, pz, tz):
    import cv2
    n = len(names)
    bw = (w - (n - 1) * 8) // n
    for i, nm in enumerate(names):
        bx = x0 + i * (bw + 8)
        on = (pz == i)
        cv2.rectangle(img, (bx, y0), (bx + bw, y0 + 62),
                      PRED if on else PANEL, -1)
        if tz == i:
            cv2.rectangle(img, (bx, y0), (bx + bw, y0 + 62), TRUE, 2)
        lab = {"ซ้าย": "LEFT", "กลาง": "CENTER", "ขวา": "RIGHT"}.get(nm, nm)
        sz = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        _text(img, lab, (bx + (bw - sz[0]) // 2, y0 + 41), 0.75,
              (20, 22, 26) if on else (120, 126, 136), 2)
    _text(img, "ZONE", (x0, y0 - 9), 0.45, PRED)
    _text(img, "white outline = camera says here", (x0 + w - 210, y0 - 9), 0.4,
          (140, 146, 156))


def _bars(img, x0, y0, w, amps, pins, min_amp, log4, now4):
    import cv2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + 118), PANEL, -1)
    top = max(max(amps), min_amp * 2, 1.0)
    bw = (w - 30 - 3 * 10) // 4
    for i, p in enumerate(pins):
        bx = x0 + 15 + i * (bw + 10)
        bh = int(np.clip(amps[i] / top, 0, 1) * 62)
        col = PRED if i < 2 else (250, 190, 90)
        cv2.rectangle(img, (bx, y0 + 78 - bh), (bx + bw, y0 + 78), col, -1)
        _text(img, NAME[p], (bx, y0 + 94), 0.36, (150, 156, 166))
        _text(img, f"{amps[i]:.0f}", (bx, y0 + 108), 0.42, (210, 216, 226))
    ay = int(y0 + 78 - np.clip(min_amp / top, 0, 1) * 62)
    cv2.line(img, (x0 + 66, ay), (x0 + w - 8, ay), BADC, 1)
    _text(img, f"min {min_amp:.0f}mV", (x0 + 6, ay + 4), 0.34, BADC)
    # แสดงทั้งค่าเฟรมนี้และค่าที่เกลี่ยแล้ว — ถ้าโชว์แต่ค่าเกลี่ย ผู้ใช้จะเทียบกับ
    # แท่งข้างล่าง (ซึ่งเป็นเฟรมนี้) แล้วนึกว่าโปรแกรมคำนวณผิด
    _text(img, f"echo strength mV  -  this frame   "
               f"log4 now {now4:+.3f}   ->  median of 9 = {log4:+.3f}",
          (x0, y0 - 9), 0.45, (150, 156, 166))


def _history(img, x0, y0, w, h, hist):
    import cv2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    for d in (-20, 0, 20):
        py = int(y0 + (0.5 - d / DEG_MAX / 2) * h)
        cv2.line(img, (x0, py), (x0 + w, py), (58, 63, 72), 1)
        _text(img, f"{d:+d}", (x0 + 3, py - 3), 0.32, (110, 116, 126))
    for series, col in ((0, TRUE), (1, PRED)):
        pts = [(int(x0 + w - 1 - i),
                int(y0 + (0.5 - np.clip(v[series], -DEG_MAX, DEG_MAX)
                          / DEG_MAX / 2) * h))
               for i, v in enumerate(hist) if v[series] is not None]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
    _text(img, "HISTORY  newest on right   yellow = sound prediction, white = camera",
          (x0, y0 - 9), 0.45, (150, 156, 166))


def render(depth, lab, pdeg, pz, tz, amps, pins, rule, log4, hist, stats, fps,
           fresh=True, stale=0, rng=0.0, progress=None, foot=None, banner=None,
           title=None):
    import cv2
    global _CANVAS
    if _CANVAS is None or _CANVAS.shape != (H, W, 3):
        _CANVAS = np.empty((H, W, 3), np.uint8)
    img = _CANVAS
    img[:] = BG
    _text(img, title or "LIVE PREDICTION  -  rule base from ultrasonic only",
          (PAD, 28), 0.62, (215, 220, 230))
    # slope เป็น NaN แปลว่ากำลังใช้โมเดล ML ซึ่งเขียนเป็นสูตรบรรทัดเดียวไม่ได้
    if rule['slope'] == rule['slope']:
        how = (f"rule: angle = {rule['slope']:+.2f} * log4 {rule['intercept']:+.2f}"
               f"   trained on {rule['n_frames']} frames from "
               f"{len(rule['trained_on'])} sections")
    else:
        how = "model: CNN on raw waveforms, no hand-written DSP"
    _text(img, f"{how}   |   held-out score: {rule['mae_deg']:.1f} deg, "
               f"{rule['zone_acc']:.0%} zone", (PAD, 52), 0.44, (140, 146, 156))
    _text(img, f"fps {fps:.1f}   median of {rule['smooth']} frames "
               f"(~{rule['smooth'] / 2 / 15:.1f}s lag)", (PAD, 74), 0.44,
          (140, 146, 156))

    _depth_panel(img, depth, lab)
    rx = PAD * 2 + DW
    rw = W - rx - PAD
    _gauge(img, rx, 110, rw, pdeg, lab[0] if lab else None)
    _zones(img, rx, 212, rw, rule["zone_names"], pz, tz)

    # ตัวเลขใหญ่ 2 ตัว: เสียงทาย กับ กล้องเห็น
    cv2.rectangle(img, (rx, 300), (rx + rw, 300 + 86), PANEL, -1)
    _text(img, "SOUND SAYS" if fresh else f"SOUND SAYS  (held {stale} frames)",
          (rx + 16, 322), 0.42, PRED if fresh else WARNC)
    _text(img, f"{pdeg:+.1f}" if pdeg is not None else "--",
          (rx + 16, 372), 1.5, PRED if fresh else (110, 150, 175), 3)
    _text(img, "CAMERA SAYS", (rx + rw // 2 + 16, 322), 0.42, TRUE)
    _text(img, f"{lab[0]:+.1f}" if lab else "--",
          (rx + rw // 2 + 16, 372), 1.5, TRUE, 3)
    if pdeg is not None and lab:
        err = abs(pdeg - lab[0])
        _text(img, f"off by {err:.1f} deg", (rx + rw - 150, 372), 0.5,
              OKC if err < 8 else (WARNC if err < 15 else BADC), 2)

    now4 = float(np.log((amps[0] + amps[1] + 1.0) / (amps[2] + amps[3] + 1.0)))
    _bars(img, rx, 410, rw, amps, rule["pins"], rule["min_amp"], log4, now4)
    if not fresh:
        bw_ = 232
        cv2.rectangle(img, (rx + rw - bw_, 385), (rx + rw, 404), WARNC, -1)
        _text(img, f"ECHO TOO WEAK  {max(amps):.0f} < {rule['min_amp']:.0f} mV",
              (rx + rw - bw_ + 9, 399), 0.44, (20, 22, 26), 1)
    _history(img, PAD, 566, W - 2 * PAD, 120, hist)

    # แถบความคืบหน้า (ใช้ตอนเล่นย้อนไฟล์ · เล่นสดจะไม่ส่งมา)
    if progress is not None:
        frac, cur, total = progress
        bx0, bx1, by = PAD, W - PAD, 697
        cv2.rectangle(img, (bx0, by), (bx1, by + 7), (52, 57, 66), -1)
        cv2.rectangle(img, (bx0, by), (bx0 + int((bx1 - bx0) * frac), by + 7),
                      PRED, -1)
        _text(img, f"frame {cur} / {total}", (bx0, by - 5), 0.4, (150, 156, 166))

    n, hit, se = stats
    y = 720
    cv2.rectangle(img, (PAD, y - 20), (W - PAD, y + 22), PANEL, -1)
    for i, (k, v, c) in enumerate((
            ("frames scored", f"{n}", (210, 216, 226)),
            ("zone correct", f"{hit / n:.0%}" if n else "--",
             OKC if n and hit / n > 0.7 else WARNC),
            ("mean angle error", f"{se / n:.1f} deg" if n else "--",
             OKC if n and se / n < 8 else WARNC),
            ("echo peak / range",
             # โมเดล ML ไม่ได้คำนวณระยะ (ส่ง rng=None มา) อย่าโชว์ 0 cm ให้เข้าใจผิด
             (f"{max(amps):.0f} mV  @ {rng:.0f} cm" if rng is not None
              else f"{max(amps):.0f} mV  (no range)"),
             (210, 216, 226) if fresh else WARNC))):
        bx = PAD + 14 + i * 250
        _text(img, k, (bx, y - 2), 0.4, (140, 146, 156))
        _text(img, v, (bx, y + 18), 0.58, c, 2)
    _text(img, foot or "q quit | s save | space pause | r reset score",
          (PAD, H - 10), 0.44, (110, 116, 126))
    if banner:
        txt, col = banner
        w_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.rectangle(img, (W - PAD - w_ - 18, 16), (W - PAD, 40), col, -1)
        _text(img, txt, (W - PAD - w_ - 9, 33), 0.45, (20, 22, 26), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--name", default="walk", help="ชื่อชุดข้อมูลที่เอากฎมาใช้")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=50.0)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--model", choices=["rule", "nn"], default="rule",
                    help="rule = กฎ 2 พารามิเตอร์ · nn = โมเดล ML จากคลื่นดิบ")
    a = ap.parse_args()

    rule = load_rule(a.name)
    if a.model == "nn":
        # โมเดล ML ใช้หน้าจอเดียวกัน แต่ค่าที่โชว์ในหัวเรื่องต้องตรงกับตัวที่ใช้จริง
        import train_nn as TN
        nnp = TN.NNPredictor()
        rule = dict(rule, slope=float("nan"), intercept=float("nan"),
                    n_frames=0, mae_deg=3.82, zone_acc=0.93,
                    holdout=f"walk_s{nnp.holdout+1}")
        print(f"โมเดล ML จากคลื่นดิบ · {nnp.params:,} พารามิเตอร์ · "
              f"เกลี่ย {nnp.buf.maxlen} เฟรม")
        print(f"  ผลกับช่วงที่กันไว้ (walk_s{nnp.holdout+1}): "
              f"ผิด 3.82 องศา · โซนถูก 93%")
    else:
        print(f"กฎที่ใช้: มุม = {rule['slope']:+.2f} * log4 {rule['intercept']:+.2f}")
        print(f"  เทรนจาก {rule['n_frames']} เฟรม · {len(rule['trained_on'])} ช่วง")
        print(f"  ผลที่วัดจากช่วงที่กันไว้ ({rule['holdout']}): "
              f"ผิด {rule['mae_deg']:.1f} องศา · โซนถูก {rule['zone_acc']:.0%}")

    import cv2
    from astra import Astra
    from sync4 import Sync4
    from record import DepthThread, _warmup
    from labels import target_angle

    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"\nเปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()
    print("ซ้อมเส้นทางคำนวณก่อนเปิดกล้อง ...", flush=True)
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    pr = nnp if a.model == 'nn' else Predictor(rule)
    if a.model == "nn":
        pr.warmup(us.samples)     # ต้องอยู่ก่อน Astra() — ดูเหตุผลใน NNPredictor.warmup
    print(f"เปิดกล้อง depth {w}x{h} ...", flush=True)
    # **ปิด GC ก่อนแตะกล้อง ไม่ใช่หลัง** — การเปิดสตรีมกับการปิดสตรีม
    # ก็เป็นช่วงที่ OpenNI ทำงานในเธรด native เหมือนกัน ถ้า GC วิ่งตอนนั้น
    # ได้ access violation เหมือนกัน เจอมาแล้วทั้งตอน create_depth_stream
    # และตอน oniStreamStop
    gc.disable()
    cam = Astra(want_rgb=False, depth_size=(w, h))
    th = DepthThread(cam, 1)
    th.start()
    time.sleep(0.6)

    hist = deque(maxlen=W - 2 * PAD)
    fps = deque(maxlen=20)
    n = hit = 0
    serr = 0.0
    paused, last, img = False, time.time(), None
    print("เปิดหน้าต่างแล้ว — ยืนหน้าเซ็นเซอร์แล้วขยับซ้าย/ขวาดูได้เลย", flush=True)
    try:
        while True:
            if not paused:
                ping = us.ping()
                got = th.get()
                if got is not None:
                    _t, depth = got
                    pdeg, fresh = pr.push(ping)
                    lab = target_angle(depth)
                    if lab is not None and lab[2] < 0.10:
                        lab = None            # เป้าเกือบหลุดเฟรม มุมจะติดขอบ
                    pz = pr.zone(pdeg) if pdeg is not None else None
                    tz = pr.zone(lab[0]) if lab is not None else None
                    if pdeg is not None and lab is not None and fresh:
                        n += 1
                        hit += int(pz == tz)
                        serr += abs(pdeg - lab[0])
                    hist.appendleft((lab[0] if lab else None, pdeg))
                    now = time.time()
                    fps.append(1.0 / max(now - last, 1e-6))
                    last = now
                    img = render(depth, lab, pdeg, pz, tz, pr.amps, us.pins,
                                 rule, pr.log4, hist, (n, hit, serr),
                                 float(np.mean(fps)), fresh, pr.stale, pr.rng,
                                 title=("LIVE PREDICTION  -  ML model on raw waveforms"
                                        if a.model == "nn" else None))
            if img is not None:
                cv2.imshow("predict", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                n = hit = 0
                serr = 0.0
            if key == ord("s") and img is not None:
                fn = time.strftime("predict_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, img)
                print(f"เซฟ {fn}", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        th.stop_flag = True
        time.sleep(0.3)
        for fn in (cam.close, us.close, cv2.destroyAllWindows):
            try:
                fn()
            except Exception:
                pass
        gc.enable()          # เปิดคืน **หลัง** กล้องหยุดจริงแล้วเท่านั้น
        if n:
            print(f"\nสรุปรอบนี้: {n} เฟรมที่เทียบได้ · โซนถูก {hit / n:.0%} · "
                  f"มุมผิดเฉลี่ย {serr / n:.1f} องศา")
            print(f"  (ตอนวัดออฟไลน์กับช่วงที่กันไว้ได้ โซนถูก {rule['zone_acc']:.0%} · "
                  f"ผิด {rule['mae_deg']:.1f} องศา)")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
