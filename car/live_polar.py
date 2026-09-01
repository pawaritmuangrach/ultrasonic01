#!/usr/bin/env python3
"""แผนที่เรดาร์สด จากโมเดล PolarScan — 13 ช่องมุมจากเสียงล้วน

    python car/live_polar.py --port COM5          ต่อฮาร์ดแวร์จริง
    python car/live_polar.py --replay walk_s5     เล่นข้อมูลที่อัดไว้ (ไม่ต้องต่ออะไร)

ต่างจาก `predict.py` ที่ตอบ "เป้าอยู่ทิศไหน" ตัวเดียว — ตัวนี้ตอบ **ทั้งแผนที่**
คือระยะของพื้นผิวที่ใกล้ที่สุดในทุกช่องมุม 13 ช่อง ซึ่งเป็นเป้าหมายปลายทางของโปรเจกต์
(ทำ mapping จากเสียงสะท้อนให้ได้เหมือนกล้อง depth)

โมเดลมาจาก testmodel01/Ultrasonic_2D_Map-main — เทรนด้วยข้อมูล walk ของเราเอง
โดยมีกล้อง depth เป็นครู พอเทรนเสร็จ **ใช้เสียงอย่างเดียว ไม่ต้องมีกล้อง**
กล้องบนหน้าจอนี้มีไว้เทียบให้ดูเท่านั้น ปิดกล้องแล้วเรดาร์ยังทำงาน

งบเวลาต่อเฟรม: เซ็นเซอร์ 47.9 + DSP 2.5 + โมเดล ~1 + วาดภาพ ~6 = ~57 ms (17 fps)

ปุ่ม: q/ESC ออก · s เซฟภาพ · space หยุด · g ซ่อน/โชว์เฉลยจากกล้อง
      (โหมด replay: , . เดินทีละเฟรม · < > กระโดด 30)
"""
import argparse
import faulthandler
import gc
import glob
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
faulthandler.enable()

import numpy as np
import numpy.ma          # noqa: F401  โหลดก่อนเปิดกล้อง — import ตอน OpenNI ทำงานทำให้ล้ม
import numpy.lib         # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
# โมเดลอยู่ในโฟลเดอร์ของ repo ภายนอก — ใส่ path ให้ import usmap ได้
USMAP = os.path.join(ROOT, "testmodel01", "Ultrasonic_2D_Map-main",
                     "Ultrasonic_2D_Map-main")
if USMAP not in sys.path:
    sys.path.insert(0, USMAP)

W, H = 1300, 820
PAD = 14
DW, DH = 400, 300
BG, PANEL = (22, 24, 28), (32, 35, 41)
OKC, BADC, WARNC = (110, 230, 130), (90, 90, 245), (80, 200, 250)
PRED = (70, 215, 250)          # เหลือง = โมเดล (เสียงล้วน)
TRUE = (245, 245, 245)         # ขาว = กล้อง
# ตำแหน่งบน plate_mini · TL=35 TR=34 BL=32 BR=33 (ดู car/rig2.py)
NAME = {35: "TOP LEFT", 34: "TOP RIGHT", 32: "BOTTOM LEFT", 33: "BOTTOM RIGHT"}
RMAX_CM = 150.0
_CANVAS = None


def _text(img, s, xy, sc=0.45, col=(200, 206, 216), th=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


# --------------------------------------------------------------- เรดาร์ --
def _pt(cx, cy, rpx, deg, r_cm):
    a = np.radians(deg)
    r = rpx * min(r_cm, RMAX_CM) / RMAX_CM
    return int(cx + r * np.sin(a)), int(cy - r * np.cos(a))


def _radar(img, x0, y0, w, h, edges, pred, pvalid, gt, gvalid, show_gt):
    """วาดสแกน 13 ช่อง — ลิ่มทึบ = พื้นที่ว่างจนถึงผิวที่เจอ · เส้นสว่าง = ตัวผิว"""
    import cv2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    cx, cy = x0 + w // 2, y0 + h - 40
    # เซกเตอร์กว้างแค่ 58 องศา ความกว้างที่ใช้จริง = 2*r*sin(29.2°) = 0.98*r
    # ถ้าคิด rpx จาก w/2 จะเหลือที่ว่างครึ่งแผง — คิดจากความกว้างเต็มแทน
    half = np.radians((edges[-1] - edges[0]) / 2)
    rpx = int(min(h - 70, (w - 40) / (2 * np.sin(half))))

    for r_cm in (50, 100, 150):                     # วงระยะ
        pts = [_pt(cx, cy, rpx, d, r_cm) for d in np.linspace(edges[0], edges[-1], 40)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (58, 63, 72), 1)
        p = _pt(cx, cy, rpx, edges[-1], r_cm)
        _text(img, f"{r_cm}", (p[0] + 6, p[1]), 0.38, (110, 116, 126))
    for d in (edges[0], 0.0, edges[-1]):            # เส้นขอบมุม
        cv2.line(img, (cx, cy), _pt(cx, cy, rpx, d, RMAX_CM), (58, 63, 72), 1)
        p = _pt(cx, cy, rpx, d, RMAX_CM)
        _text(img, f"{d:+.0f}", (p[0] - 12, p[1] - 8), 0.38, (110, 116, 126))

    # เบลนด์ทีเดียวบน ROI ของแผงเรดาร์ — ก่อนหน้านี้ copy ทั้งผืน 13 รอบ
    # (1300x820x3 = 3.2 MB x 13 = 42 MB ต่อเฟรม) ซึ่งกินเวลามากกว่าทุกอย่างรวมกัน
    roi = img[y0:y0 + h, x0:x0 + w]
    ov = roi.copy()
    for b in range(len(pred)):
        d = float(np.clip(pred[b], 0, RMAX_CM))
        conf = float(np.clip(pvalid[b], 0, 1))
        arc = np.linspace(edges[b], edges[b + 1], 6)
        pts = [_pt(cx - x0, cy - y0, rpx, t, d) for t in arc]
        cv2.fillPoly(ov, [np.array([(cx - x0, cy - y0)] + pts, np.int32)],
                     tuple(int(v * (0.35 + 0.65 * conf)) for v in PRED))
    cv2.addWeighted(ov, 0.45, roi, 0.55, 0, roi)
    for b in range(len(pred)):
        d = float(np.clip(pred[b], 0, RMAX_CM))
        arc = np.linspace(edges[b], edges[b + 1], 6)
        cv2.polylines(img, [np.array([_pt(cx, cy, rpx, t, d) for t in arc], np.int32)],
                      False, PRED, 3)

    if show_gt and gt is not None:
        for b in range(len(gt)):
            if not gvalid[b]:
                continue
            mid = (edges[b] + edges[b + 1]) / 2
            cv2.circle(img, _pt(cx, cy, rpx, mid, float(gt[b])), 5, TRUE, -1)

    cv2.drawContours(img, [np.array([(cx, cy - 13), (cx - 11, cy + 8),
                                     (cx + 11, cy + 8)])], 0, (170, 176, 186), -1)
    _text(img, "POLAR SCAN  -  13 bins from sound only", (x0 + 10, y0 + 22), 0.5, PRED)
    if show_gt:
        _text(img, "white dots = depth camera", (x0 + w - 210, y0 + 22), 0.42,
              (150, 156, 166))


def _bars(img, x0, y0, w, amps, tdoa, pins):
    import cv2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + 128), PANEL, -1)
    _text(img, "echo strength mV", (x0 + 8, y0 + 18), 0.42, (150, 156, 166))
    top = max(max(amps), 1.0)
    bw = (w - 28 - 3 * 8) // 4
    for i, p in enumerate(pins):
        bx = x0 + 14 + i * (bw + 8)
        bh = int(np.clip(amps[i] / top, 0, 1) * 52)
        cv2.rectangle(img, (bx, y0 + 82 - bh), (bx + bw, y0 + 82),
                      PRED if i < 2 else (250, 190, 90), -1)
        _text(img, NAME[p][:9], (bx, y0 + 98), 0.33, (150, 156, 166))
        _text(img, f"{amps[i]:.0f}", (bx, y0 + 114), 0.4, (210, 216, 226))

    cv2.rectangle(img, (x0, y0 + 136), (x0 + w, y0 + 232), PANEL, -1)
    _text(img, "TDOA per pair  (normalised -1..+1)", (x0 + 8, y0 + 154), 0.42,
          (150, 156, 166))
    mid = y0 + 196
    cv2.line(img, (x0 + 14, mid), (x0 + w - 14, mid), (70, 76, 86), 1)
    bw2 = (w - 28 - 5 * 6) // 6
    for i, v in enumerate(tdoa):
        bx = x0 + 14 + i * (bw2 + 6)
        hgt = int(np.clip(v, -1, 1) * 28)
        cv2.rectangle(img, (bx, mid - hgt if hgt > 0 else mid),
                      (bx + bw2, mid if hgt > 0 else mid - hgt),
                      OKC if hgt > 0 else BADC, -1)
        _text(img, f"{v:+.1f}", (bx - 1, y0 + 226), 0.32, (150, 156, 166))


def _depth(img, x0, y0, depth):
    import cv2
    if depth is None:
        cv2.rectangle(img, (x0, y0), (x0 + DW, y0 + DH), PANEL, -1)
        _text(img, "no camera (sound-only mode)", (x0 + 60, y0 + DH // 2), 0.6,
              (120, 126, 136))
        return
    n = np.clip(depth.astype(np.float32), 0, 2000) / 2000.0
    vis = cv2.applyColorMap((255 - n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth == 0] = 0
    img[y0:y0 + DH, x0:x0 + DW] = cv2.resize(vis, (DW, DH),
                                             interpolation=cv2.INTER_NEAREST)


def _history(img, x0, y0, w, h, hist):
    """ประวัติวัตถุใกล้สุด — ระยะ (บน) กับ มุม (ล่าง) เทียบกล้อง"""
    import cv2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    _text(img, "NEAREST OBJECT over time", (x0 + 10, y0 + 22), 0.45, PRED)
    hh = (h - 60) // 2
    for pane, (lab, lo, hi, kp, kg) in enumerate(
            (("distance  cm", 40.0, 110.0, 0, 2), ("angle  deg", -30.0, 30.0, 1, 3))):
        py = y0 + 40 + pane * (hh + 16)
        cv2.rectangle(img, (x0 + 8, py), (x0 + w - 8, py + hh), (26, 29, 34), -1)
        _text(img, lab, (x0 + 12, py - 4), 0.38, (140, 146, 156))
        for frac, v in ((0.0, hi), (0.5, (lo + hi) / 2), (1.0, lo)):
            yy = int(py + frac * hh)
            cv2.line(img, (x0 + 8, yy), (x0 + w - 8, yy), (46, 50, 58), 1)
            _text(img, f"{v:.0f}", (x0 + 11, yy - 3), 0.32, (100, 106, 116))
        n = min(len(hist), w - 20)
        for series, col in ((kg, TRUE), (kp, PRED)):
            pts = [(x0 + w - 10 - i,
                    int(py + (1 - (np.clip(hist[i][series], lo, hi) - lo) / (hi - lo)) * hh))
                   for i in range(n) if hist[i][series] is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1,
                              cv2.LINE_AA)
    _text(img, "yellow = model (sound)   white = camera", (x0 + 10, y0 + h - 8),
          0.36, (140, 146, 156))


def render(depth, pred, pvalid, gt, gvalid, amps, tdoa, pins, edges,
           near, stats, fps, show_gt, foot, sub, hist=()):
    import cv2
    global _CANVAS
    if _CANVAS is None or _CANVAS.shape != (H, W, 3):
        _CANVAS = np.empty((H, W, 3), np.uint8)
    img = _CANVAS
    img[:] = BG
    _text(img, "POLARSCAN LIVE  -  ultrasonic mapping", (PAD, 28), 0.62,
          (215, 220, 230))
    _text(img, sub, (PAD, 52), 0.43, (140, 146, 156))
    _text(img, f"fps {fps:.1f}", (W - 110, 28), 0.5, (170, 176, 186))

    _text(img, "DEPTH CAMERA  -  comparison only", (PAD, 82), 0.44, (150, 156, 166))
    _depth(img, PAD, 92, depth)
    _bars(img, PAD, 92 + DH + 16, DW, amps, tdoa, pins)
    rx0 = PAD * 2 + DW
    rw = 540
    _radar(img, rx0, 92, rw, H - 92 - 96, edges, pred, pvalid, gt, gvalid, show_gt)
    _history(img, rx0 + rw + PAD, 92, W - (rx0 + rw + 2 * PAD), H - 92 - 96, hist)

    y = H - 58
    cv2.rectangle(img, (PAD, y - 22), (W - PAD, y + 24), PANEL, -1)
    nd, na = near
    cells = [("nearest object", f"{nd:.0f} cm  {na:+.0f} deg", PRED)]
    if stats[0]:
        n, sd, sa = stats
        cells += [("vs camera  dist", f"{sd / n:.1f} cm",
                   OKC if sd / n < 10 else WARNC),
                  ("vs camera  angle", f"{sa / n:.1f} deg",
                   OKC if sa / n < 12 else WARNC),
                  ("frames scored", f"{n}", (210, 216, 226))]
    for i, (k, v, c) in enumerate(cells):
        bx = PAD + 16 + i * 270
        _text(img, k, (bx, y - 3), 0.4, (140, 146, 156))
        _text(img, v, (bx, y + 18), 0.58, c, 2)
    _text(img, foot, (PAD, H - 12), 0.42, (110, 116, 126))
    return img


# ------------------------------------------------------------- โมเดล --
class Model:
    def __init__(self):
        import torch
        from usmap.config import MODELS
        from usmap.polar_gt import N_ANG, FOV_H_DEG
        from usmap.polar_model import PolarNet
        p = Path(MODELS) / "polarscan.pt"
        if not p.exists():
            sys.exit(f"ยังไม่มีโมเดลที่ {p}\n"
                     f"ต้องเทรนก่อน:  cd {USMAP} && python -m usmap.polar_train")
        torch.set_num_threads(1)   # เธรดพูล torch ชนกับเธรดกล้อง — ดู warmup()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.net = PolarNet()
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.torch = torch
        self.edges = np.linspace(-FOV_H_DEG / 2, FOV_H_DEG / 2, N_ANG + 1)
        self.trained = ck.get("metrics", {})

    def warmup(self, nsamp=677, rate=66300.0):
        """ซ้อมเส้นทางคำนวณทั้งเส้น **ก่อนเปิดกล้อง**

        torch เตรียม kernel ตอน conv ครั้งแรก และ usmap.physics ยัง import
        โมดูลย่อยตอนถูกเรียกครั้งแรกด้วย ถ้าไปเกิดตอนเธรด OpenNI ทำงานอยู่
        โปรเซสตายเงียบ ๆ บน Windows (เจอมาแล้วกับ numpy, GC และ torch)
        """
        rng = np.random.default_rng(0)
        counts = (2048 + rng.normal(0, 30, (4, int(nsamp)))).astype(np.uint16)
        try:
            self(counts, rate)
        except Exception as e:            # เสียงปลอมอาจไม่ผ่านด่านหายอดคลื่น
            print(f"  (ซ้อมด้วยข้อมูลปลอมไม่ผ่าน: {e} — ของจริงยังทำงานได้)")

    def __call__(self, counts, rate):
        from usmap.physics import (envelopes, common_peak, pair_tdoa, PAIRS,
                                   range_profile_from_envs, max_lag_us)
        envs = envelopes(counts.astype(float), rate)
        k, dist_cm, _ = common_peak(envs, rate)
        amps = np.array([e[max(0, k - 20):k + 20].max() for e in envs])
        lo = max(1, int(1220e-6 * rate))
        noise = np.array([np.median(e[lo:]) for e in envs])
        snr = amps / np.maximum(noise, 1e-12)
        td = [pair_tdoa(envs[a], envs[b], rate, k, (a, b)) for a, b in PAIRS]
        tus = np.array([v for v, _ in td])
        tn = np.clip(tus / np.array([max_lag_us(p) for p in PAIRS]), -1, 1)
        prof = range_profile_from_envs(envs, rate)
        t = lambda x: self.torch.from_numpy(np.asarray(x, np.float32))[None]
        with self.torch.no_grad():
            o = self.net(t(prof), t(tn), t(amps / max(amps.max(), 1e-9)),
                         t(snr), t([dist_cm]))
        return (o["bin_dist"][0].numpy(),
                self.torch.sigmoid(o["bin_valid"][0]).numpy(),
                float(o["near_d"]), float(o["near_a"]),
                amps * 3.3 / 4095 * 1000, tn, dist_cm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=60.0)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--replay", default=None,
                    help="ชื่อช่วงที่อัดไว้ เช่น walk_s5 (ไม่ต้องต่อฮาร์ดแวร์)")
    ap.add_argument("--fps", type=float, default=15.0, help="ความเร็วตอน replay")
    a = ap.parse_args()

    import cv2
    from usmap.polar_gt import polar_label
    model = Model()
    print(f"โหลดโมเดลแล้ว · ผลตอนเทรน: "
          f"{ {k: round(v, 2) for k, v in model.trained.items()} }")

    show_gt, paused = True, False
    hist = deque(maxlen=400)
    n = 0
    sd = sa = 0.0
    fpsq = deque(maxlen=20)

    # ------------------------------------------------------ โหมดเล่นย้อน
    if a.replay:
        d = Path(HERE) / "data" / a.replay
        fs = sorted(glob.glob(str(d / "us_*.npz")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
        frames = [(u, d / f"depth_{Path(u).stem.split('_')[1]}.png") for u in fs]
        frames = [(u, p) for u, p in frames if p.exists()]
        if not frames:
            sys.exit(f"ไม่พบเฟรมใน {d}")
        print(f"เล่นย้อน {a.replay}: {len(frames)} เฟรม — q ออก")
        i, img = 0, None
        while True:
            u, dp = frames[i]
            z = np.load(u)
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            pred, pv, nd, na, amps, tn, phys = model(z["counts"], float(z["rate"]))
            gt, gv = polar_label(str(dp))
            cam_d = cam_a = None
            if gv.any():
                j = int(np.argmin(np.where(gv, gt, 1e9)))
                mid = (model.edges[:-1] + model.edges[1:]) / 2
                cam_d, cam_a = float(gt[j]), float(mid[j])
                n += 1
                sd += abs(nd - cam_d)
                sa += abs(na - cam_a)
            hist.appendleft((nd, na, cam_d, cam_a))
            img = render(depth, pred, pv, gt, gv, amps, tn,
                         [int(v) for v in z["pins"]], model.edges, (nd, na),
                         (n, sd, sa), a.fps, show_gt,
                         f"replay {a.replay}  frame {i + 1}/{len(frames)}  |  "
                         f"q quit | s save | space pause | , . step | g toggle GT",
                         f"physics-only range {phys:.0f} cm   |   "
                         f"model trained on walk_s1-s3, this is {a.replay}",
                         hist=hist)
            cv2.imshow("polarscan", img)
            key = cv2.waitKey(max(int(1000 / a.fps), 1) if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("g"):
                show_gt = not show_gt
            elif key == ord("s"):
                fn = f"polarscan_{a.replay}_{i:06d}.png"
                cv2.imwrite(fn, img)
                print("เซฟ", fn, flush=True)
            elif key in (ord(","), ord("<")):
                i = max(0, i - (30 if key == ord("<") else 1))
                paused = True
            elif key in (ord("."), ord(">")):
                i = min(len(frames) - 1, i + (30 if key == ord(">") else 1))
                paused = True
            elif not paused:
                i = (i + 1) % len(frames)
        cv2.destroyAllWindows()
        if n:
            print(f"\nเทียบกับกล้อง {n} เฟรม: ระยะผิด {sd / n:.1f} cm · "
                  f"มุมผิด {sa / n:.1f} องศา")
        return 0

    # ------------------------------------------------------ โหมดสด
    from astra import Astra
    from sync4 import Sync4
    from record import DepthThread, _warmup
    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()
    print("ซ้อมเส้นทางคำนวณก่อนเปิดกล้อง ...", flush=True)
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    model.warmup(us.samples, us.rate)    # ต้องอยู่ก่อน Astra() — ดู Model.warmup
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

    img, last = None, time.time()
    print("เปิดหน้าต่างแล้ว — q ออก · g ซ่อนเฉลยจากกล้อง", flush=True)
    try:
        while True:
            if not paused:
                ping = us.ping()
                got = th.get()
                depth = got[1] if got is not None else None
                pred, pv, nd, na, amps, tn, phys = model(ping["counts"],
                                                         float(ping["rate"]))
                gt = gv = None
                cam_d = cam_a = None
                if depth is not None:
                    gt, gv = polar_label_from_array(depth)
                    if gv is not None and gv.any():
                        j = int(np.argmin(np.where(gv, gt, 1e9)))
                        mid = (model.edges[:-1] + model.edges[1:]) / 2
                        cam_d, cam_a = float(gt[j]), float(mid[j])
                        n += 1
                        sd += abs(nd - cam_d)
                        sa += abs(na - cam_a)
                hist.appendleft((nd, na, cam_d, cam_a))
                now = time.time()
                fpsq.append(1.0 / max(now - last, 1e-6))
                last = now
                img = render(depth, pred, pv, gt, gv, amps, tn, us.pins,
                             model.edges, (nd, na), (n, sd, sa),
                             float(np.mean(fpsq)), show_gt,
                             "q quit | s save | space pause | g toggle camera GT",
                             f"physics-only range {phys:.0f} cm   |   "
                             f"{us.rate:.0f} Hz/ch  {us.samples} samples  "
                             f"period {us.period * 1e3:.0f} ms", hist=hist)
            if img is not None:
                cv2.imshow("polarscan", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("g"):
                show_gt = not show_gt
            elif key == ord("s") and img is not None:
                fn = time.strftime("polarscan_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, img)
                print("เซฟ", fn, flush=True)
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
            print(f"\nเทียบกับกล้อง {n} เฟรม: ระยะผิด {sd / n:.1f} cm · "
                  f"มุมผิด {sa / n:.1f} องศา")
        sys.stdout.flush()
        os._exit(0)
    return 0


def polar_label_from_array(depth):
    """polar_label ของ usmap รับ path เท่านั้น — โหมดสดมีแต่ array ในมือ
    จึงทำเวอร์ชันที่รับ array ตรง ๆ โดยลอกตรรกะเดียวกันเป๊ะ"""
    from usmap.polar_gt import FOV_H_DEG, N_ANG, V_BAND, MIN_MM, MAX_MM, MIN_PIX
    d = depth.astype(np.float32)
    h, w = d.shape
    fx = w / (2 * np.tan(np.deg2rad(FOV_H_DEG) / 2))
    band = d[int(h * (0.5 - V_BAND / 2)):int(h * (0.5 + V_BAND / 2)), :]
    yy, xx = np.mgrid[0:band.shape[0], 0:w]
    Z = band
    X = (xx - w / 2) * Z / fx
    ok = (Z >= MIN_MM) & (Z <= MAX_MM)
    if not ok.any():
        return np.zeros(N_ANG, np.float32), np.zeros(N_ANG, bool)
    ang = np.degrees(np.arctan2(X[ok], Z[ok]))
    z = Z[ok]
    edges = np.linspace(-FOV_H_DEG / 2, FOV_H_DEG / 2, N_ANG + 1)
    bi = np.clip(np.searchsorted(edges, ang) - 1, 0, N_ANG - 1)
    dist = np.zeros(N_ANG, np.float32)
    valid = np.zeros(N_ANG, bool)
    order = np.argsort(bi)
    bi_s, z_s = bi[order], z[order]
    bounds = np.searchsorted(bi_s, np.arange(N_ANG + 1))
    for b in range(N_ANG):
        seg = z_s[bounds[b]:bounds[b + 1]]
        if seg.size >= MIN_PIX:
            k = max(0, int((seg.size - 1) * 0.05))
            dist[b] = float(np.partition(seg, k)[k]) / 10.0
            valid[b] = True
    return dist, valid


if __name__ == "__main__":
    sys.exit(main())
