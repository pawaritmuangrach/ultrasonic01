#!/usr/bin/env python3
"""แผนที่กลุ่มจุด (point cloud) 2D — แสดง **ทุกเอคโค่ที่ได้ยิน** ไม่ใช่แค่ตัวที่ดังสุด

ต่างจาก live.py:
  live.py  เลือกยอดเดียว (ยอดร่วมที่ดังสุด) -> ได้ "วัตถุที่เด่นที่สุด" 1 จุดต่อปิง
  live2.py หายอดทุกยอดที่โผล่พ้นพื้นเสียง -> ได้หลายจุดต่อปิง แล้วสะสมเป็นกลุ่มเมฆ
           จึงเห็นทั้งผนัง มุมห้อง ขาโต๊ะ ฯลฯ ไม่ใช่แค่เป้าที่ตั้งใจวัด

มุมของ**แต่ละยอด** คำนวณแยกกัน: โมเดลมุมรับ feature ที่คิดจากหน้าต่างรอบยอดนั้น ๆ
(`_pair_direction_features(..., k=ยอดนั้น)`) จึงใช้โมเดลตัวเดิมกับยอดไหนก็ได้

ข้อจำกัดที่ต้องรู้ก่อนตีความภาพ:
  * โมเดลมุมเทรนจากฉากที่มี "วัตถุเดียว" เป็นหลัก ยอดที่มาจากหลายวัตถุพร้อมกัน
    (เช่นผนังกว้าง) จะให้มุมที่เป็นค่าเฉลี่ย ไม่ใช่ขอบจริงของวัตถุ
  * ความละเอียดเชิงมุมของอาเรย์ 4 ช่องนี้อยู่ราว 10° กลุ่มเมฆจึงเป็นภาพ "คร่าว ๆ"
  * ยอดที่ไกลกว่า ~150 cm มักเป็นเสียงก้องหลายทาง (multipath) ไม่ใช่วัตถุจริงเสมอไป

    python car/live2.py --port COM5
    python car/live2.py --port COM5 --snr 3.5 --max-peaks 8

ปุ่ม:  q/ESC ออก · c ล้างแผนที่ · s เซฟภาพ · เว้นวรรค หยุด/เล่นต่อ · d สลับโหมดจางหาย
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from features import (envelope_of, _pair_direction_features,          # noqa: E402
                      T0_US, C, GATE_MIN_CM, GATE_MAX_CM)
from live import (draw_traces, train_angle_model, zone, zone_en,      # noqa: E402
                  TR_TOP, TR_H, TR_GAP, TR_X0, TR_X1,
                  W, H, MAP_X0, MAP_X1, MAP_Y0, MAP_Y1, FOV,
                  BG, PANEL, GRID, HOT)

CELL = 2.0                      # ขนาดช่องกริดสะสม (cm) — 2 cm พอดีกับความละเอียดจริง
XMAX = GATE_MAX_CM * np.sin(np.radians(FOV))


def cm_to_index(cm, rate):
    return int((T0_US + 2 * cm / 100 / C * 1e6) * 1e-6 * rate)


def index_to_cm(k, rate):
    return (k / rate * 1e6 - T0_US) * 1e-6 * C / 2 * 100


def find_echoes(envs, rate, snr, max_peaks):
    """หายอดทุกยอดที่พ้นพื้นเสียง คืน [(k, amp, snr_ratio), ...] เรียงตามความแรง

    ใช้ค่าเฉลี่ยของทุกช่องเป็นตัวหายอด: เฉลี่ยแล้ว SNR ดีขึ้น และยอดของแต่ละช่อง
    เหลื่อมกันแค่ไม่กี่ไมโครวินาที (เบสไลน์สั้น) จึงไม่ทำให้ยอดเบลอ
    """
    from scipy.signal import find_peaks
    n = min(len(e) for e in envs)
    m = np.mean([e[:n] for e in envs], axis=0)
    lo = max(1, cm_to_index(GATE_MIN_CM, rate))
    hi = min(n, cm_to_index(GATE_MAX_CM, rate))
    if hi <= lo + 20:
        return []
    seg = m[lo:hi]
    nf = max(float(np.median(seg)), 1e-12)
    # ยอดต้องห่างกันอย่างน้อย 5 cm ไม่งั้นยอดเดียวถูกนับซ้ำจากการกระเพื่อม
    dist = max(3, cm_to_index(GATE_MIN_CM + 5, rate) - cm_to_index(GATE_MIN_CM, rate))
    idx, props = find_peaks(seg, height=nf * snr, distance=dist)
    if idx.size == 0:
        return []
    hgt = props["peak_heights"]
    order = np.argsort(hgt)[::-1][:max_peaks]
    return [(int(idx[i] + lo), float(hgt[i]), float(hgt[i] / nf)) for i in order]


def angle_at(envs, rate, k, mdl):
    """มุมของยอดที่ index k — ใช้โมเดลเดิมแต่คิด feature จากหน้าต่างรอบยอดนั้น"""
    try:
        f = np.concatenate([_pair_direction_features(envs[p], envs[p + 1], rate, k)
                            for p in range(0, len(envs) - 1, 2)]).reshape(1, -1)
        return float(mdl[1].predict(mdl[0].transform(f))[0])
    except Exception:
        return None


class CloudMap:
    """กริดสะสมในพิกัดฉาก (X ซ้าย-ขวา, Y ไปข้างหน้า) หน่วย cm"""

    def __init__(self):
        self.nx = int(2 * XMAX / CELL) + 1
        self.ny = int(GATE_MAX_CM / CELL) + 1
        self.g = np.zeros((self.ny, self.nx), np.float32)
        self.cx = (MAP_X0 + MAP_X1) // 2
        self.cy = MAP_Y1 - 18
        half = (MAP_X1 - MAP_X0) / 2 - 62
        self.s = min(half / XMAX, (self.cy - MAP_Y0 - 16) / GATE_MAX_CM)
        self.hits = 0

    def xy(self, cm, deg):
        t = np.radians(deg)
        return (int(self.cx + cm * np.sin(t) * self.s),
                int(self.cy - cm * np.cos(t) * self.s))

    def add(self, cm, deg, weight):
        X, Y = cm * np.sin(np.radians(deg)), cm * np.cos(np.radians(deg))
        ix, iy = int((X + XMAX) / CELL), int(Y / CELL)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.g[iy, ix] += weight
            self.hits += 1

    def decay(self, f):
        if f < 1.0:
            self.g *= f

    def draw(self, img, shots):
        import cv2
        cv2.rectangle(img, (MAP_X0, MAP_Y0), (MAP_X1, MAP_Y1), PANEL, -1)

        # ---- กริดสะสม: วาดเป็นภาพเดียวแล้วยืดเข้าที่ (เร็วกว่าวาดทีละจุด) ----
        if self.g.max() > 0:
            v = np.log1p(self.g)
            v = (v / v.max() * 255).astype(np.uint8)
            heat = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
            wpx = int(2 * XMAX * self.s)
            hpx = int(GATE_MAX_CM * self.s)
            heat = cv2.resize(heat, (wpx, hpx), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize((v > 0).astype(np.uint8) * 255, (wpx, hpx),
                              interpolation=cv2.INTER_LINEAR)
            heat = cv2.flip(heat, 0)          # Y ในกริดชี้ไปข้างหน้า แต่ y ในภาพชี้ลง
            mask = cv2.flip(mask, 0)
            x0, y0 = self.cx - wpx // 2, self.cy - hpx
            x1, y1 = min(x0 + wpx, MAP_X1), min(y0 + hpx, MAP_Y1)
            if x0 >= MAP_X0 and y0 >= MAP_Y0 and x1 > x0 and y1 > y0:
                roi = img[y0:y1, x0:x1]
                hh, ww = roi.shape[:2]
                m3 = (mask[:hh, :ww, None] / 255.0)
                img[y0:y1, x0:x1] = (roi * (1 - m3) + heat[:hh, :ww] * m3).astype(np.uint8)

        # ---- เส้นกริดอ้างอิง วาดทับกลุ่มเมฆ ----
        for r in range(50, int(GATE_MAX_CM) + 1, 50):
            rad = int(r * self.s)
            cv2.ellipse(img, (self.cx, self.cy), (rad, rad), 0,
                        -90 - FOV, -90 + FOV, GRID, 1)
            cv2.putText(img, f"{r}cm", (MAP_X0 + 8, self.cy - rad + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 126, 136), 1, cv2.LINE_AA)
        for g in range(int(-FOV), int(FOV) + 1, 15):
            cv2.line(img, (self.cx, self.cy), self.xy(GATE_MAX_CM, g),
                     GRID if g else (86, 92, 104), 1)
            x, y = self.xy(GATE_MAX_CM + 20, g)
            cv2.putText(img, f"{g:+d}" if g else "0", (x - 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 126, 136), 1, cv2.LINE_AA)

        # ---- ยอดของปิงล่าสุด: วงกลมโต = ยอดแรง ----
        for cm, deg, ratio in shots:
            x, y = self.xy(cm, deg)
            r = int(np.clip(3 + np.log1p(ratio) * 2.4, 3, 13))
            cv2.circle(img, (x, y), r, HOT, 1, cv2.LINE_AA)

        cv2.rectangle(img, (self.cx - 22, self.cy - 4), (self.cx + 22, self.cy + 8),
                      (150, 155, 165), -1)
        cv2.putText(img, "SENSOR", (self.cx - 26, self.cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 135, 145), 1, cv2.LINE_AA)


def render(envs, rate, shots, ks, pins, cmap, note, paused, fps, fade):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "point cloud: every echo above the noise floor",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 195, 205), 1, cv2.LINE_AA)
    draw_traces(img, envs, rate, None, pins)

    # ทำเครื่องหมายทุกยอดที่ตรวจพบบนกราฟคลื่น (live.py ทำแค่ยอดเดียว)
    n = min(len(e) for e in envs)
    i_lo = max(1, cm_to_index(GATE_MIN_CM, rate))
    i_hi = min(n, cm_to_index(GATE_MAX_CM, rate))
    for k in ks:
        if i_lo <= k < i_hi:
            x = int(TR_X0 + (k - i_lo) / max(1, i_hi - i_lo) * (TR_X1 - TR_X0))
            for ci in range(len(envs)):
                y0 = TR_TOP + ci * (TR_H + TR_GAP)
                cv2.line(img, (x, y0), (x, y0 + TR_H), (70, 110, 255), 1)

    cmap.draw(img, shots)

    by = TR_TOP + 4 * (TR_H + TR_GAP) + 46
    cv2.rectangle(img, (TR_X0 - 66, by), (TR_X1, by + 100), PANEL, -1)
    cv2.putText(img, f"{len(shots)}", (TR_X0 - 50, by + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (250, 250, 250), 3, cv2.LINE_AA)
    cv2.putText(img, "echoes this ping", (TR_X0 + 10, by + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 175, 185), 1, cv2.LINE_AA)
    if shots:
        cm, deg, ratio = shots[0]
        cv2.putText(img, f"strongest: {cm:.0f} cm  {deg:+.0f} deg  {zone_en(deg)}",
                    (TR_X0 - 50, by + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    HOT, 1, cv2.LINE_AA)
    cv2.putText(img, f"{note} | cells hit: {cmap.hits}", (TR_X0 + 200, by + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 135, 145), 1, cv2.LINE_AA)

    foot = "q quit | c clear | s save | space pause | d fade:" + ("on" if fade else "off")
    if paused:
        foot = "** PAUSED **   " + foot
    cv2.putText(img, foot, (14, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 125, 135), 1, cv2.LINE_AA)
    cv2.putText(img, f"{fps:.1f} fps", (W - 100, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (110, 115, 125), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--snr", type=float, default=4.0,
                    help="ยอดต้องสูงกว่าพื้นเสียงกี่เท่าจึงนับ (ต่ำ=จุดเยอะแต่มั่วขึ้น)")
    ap.add_argument("--max-peaks", type=int, default=6, help="รับกี่ยอดต่อปิง")
    a = ap.parse_args()

    import cv2
    from ultrasonic import Ultrasonic

    print("เทรนโมเดลมุมจาก car/data/ ...", flush=True)
    got = train_angle_model()
    if got is None:
        sys.exit("!! ข้อมูลไม่พอสำหรับโมเดลมุม — live2 ต้องใช้มุมถึงจะวางจุดบนแผนที่ได้")
    sc_, m_, nsamp = got
    mdl, note = (sc_, m_), f"model {nsamp} samples · snr>{a.snr:g}x"
    print(f"เทรนเสร็จ ({nsamp} ตัวอย่าง)")

    print(f"เปิด {a.port} · ช่อง {a.pins} ...", flush=True)
    us = Ultrasonic(port=a.port, pins=a.pins)
    cmap = CloudMap()
    cv2.namedWindow("cloud map", cv2.WINDOW_AUTOSIZE)
    paused, fade, last, fps = False, False, time.time(), 0.0
    envs = rate = None
    shots, ks = [], []
    try:
        while True:
            if not paused:
                ping = us.ping()
                if ping is not None:
                    rate, counts = ping["rate"], ping["counts"]
                    envs = [envelope_of(counts[c], rate)
                            for c in range(counts.shape[0])]
                    peaks = find_echoes(envs, rate, a.snr, a.max_peaks)
                    cmap.decay(0.97 if fade else 1.0)
                    shots, ks = [], []
                    for k, amp, ratio in peaks:
                        deg = angle_at(envs, rate, k, mdl)
                        if deg is None or abs(deg) > FOV:
                            continue
                        cm = index_to_cm(k, rate)
                        if not (GATE_MIN_CM <= cm <= GATE_MAX_CM):
                            continue
                        shots.append((cm, deg, ratio))
                        ks.append(k)
                        cmap.add(cm, deg, float(np.log1p(ratio)))
                    if shots:
                        s0 = shots[0]
                        print(f"\r  {len(shots)} echoes · แรงสุด {s0[0]:6.1f} cm "
                              f"{s0[1]:+6.1f}° {zone(s0[1]):<8}", end="", flush=True)
                    now = time.time()
                    fps = 0.8 * fps + 0.2 / max(now - last, 1e-6)
                    last = now
            if envs is not None:
                cv2.imshow("cloud map", render(envs, rate, shots, ks, us.pins,
                                               cmap, note, paused, fps, fade))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                cmap.g[:] = 0
                cmap.hits = 0
            if key == ord(" "):
                paused = not paused
            if key == ord("d"):
                fade = not fade
            if key == ord("s") and envs is not None:
                fn = time.strftime("cloud_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, render(envs, rate, shots, ks, us.pins,
                                       cmap, note, paused, fps, fade))
                print(f"\n  เซฟ {fn}")
    finally:
        us.close()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
