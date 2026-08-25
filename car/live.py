#!/usr/bin/env python3
"""ดูคลื่นสด + แผนที่ 2D มองจากด้านบน ว่าวัตถุอยู่ห่างกี่ cm และเบนซ้าย/ขวากี่องศา

ใช้สองวิธีคนละแบบตามที่พิสูจน์มาแล้วใน train.py:
  ระยะ = **ฟิสิกส์ล้วน** (เวลาเอคโค่จากยอดร่วม) — แม่นกว่า ML ราวเท่าตัว (7.3 vs 13.6 cm)
  มุม  = **โมเดลที่เทรน** (TDOA ข้ามช่อง) — คำนวณตรง ๆ ไม่ได้เพราะเบสไลน์สั้นและมี
         ความไม่สมมาตรของหัวรับปนอยู่ ต้องให้โมเดลเรียนเอง (11.8° vs เดา 14.8°)

โมเดลมุมเทรนสด ๆ ตอนเปิดโปรแกรมจากข้อมูลใน car/data/ จึงตรงกับข้อมูลล่าสุดเสมอ

    python car/live.py --port COM5
    python car/live.py --port COM5 --no-model     ดูคลื่น + ระยะอย่างเดียว (ไม่เทรน)

ปุ่ม:  q/ESC ออก · c ล้างร่องรอย · s เซฟภาพ · เว้นวรรค หยุด/เล่นต่อ
"""
import argparse
import os
import sys
import time
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from features import (envelope_of, common_peak, angle_features,      # noqa: E402
                      T0_US, C, GATE_MIN_CM, GATE_MAX_CM)

W, H = 1400, 730
TR_X0, TR_X1 = 74, 596              # พื้นที่กราฟคลื่น
TR_TOP, TR_H, TR_GAP = 72, 108, 10
MAP_X0, MAP_X1 = 624, W - 14        # พื้นที่แผนที่ 2D
MAP_Y0, MAP_Y1 = 52, H - 56
FOV = 45.0                          # องศาที่วาดในแผนที่ (ซ้าย-ขวาข้างละเท่านี้)

BG, PANEL, GRID = (24, 26, 31), (34, 38, 46), (58, 63, 72)
CH_COL = [(90, 220, 90), (250, 200, 90), (90, 190, 250), (200, 130, 250)]
HOT = (120, 230, 255)


def train_angle_model():
    """เทรนโมเดลมุมจากฉากทั้งหมดใน car/data/ คืน (scaler, model, n) หรือ None"""
    from train import load_all, DATA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scenes = sorted(p for p in DATA.glob("*")
                    if p.is_dir() and not p.name.startswith("test"))
    if not scenes:
        return None
    _, Xa, y, _ = load_all(scenes)
    if len(y) < 20:
        return None
    sc = StandardScaler().fit(Xa)
    m = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(sc.transform(Xa), y[:, 1])
    return sc, m, len(y)


def zone(deg):
    """คำอธิบายทิศสำหรับ terminal (ไทย) — ลบ = ซ้าย · บวก = ขวา"""
    a = abs(deg)
    if a < 7:
        return "กลาง"
    return ("ซ้าย" if deg < 0 else "ขวา") + ("สุด" if a >= 18 else "")


def zone_en(deg):
    """คำเดียวกันแต่เป็นอังกฤษ — ใช้วาดบนภาพ เพราะ OpenCV ไม่มีฟอนต์ไทย"""
    a = abs(deg)
    if a < 7:
        return "CENTER"
    return ("FAR " if a >= 18 else "") + ("LEFT" if deg < 0 else "RIGHT")


class MapView:
    """แผนที่มองจากด้านบน: sensor อยู่ล่างกลาง วัตถุกระจายออกเป็นรูปพัด"""

    def __init__(self):
        self.cx = (MAP_X0 + MAP_X1) // 2
        self.cy = MAP_Y1 - 18
        half = (MAP_X1 - MAP_X0) / 2 - 62
        # สเกลต้องพอดีทั้งด้านกว้าง (ที่มุมสุด) และด้านสูง (ที่ระยะไกลสุด)
        self.s = min(half / (GATE_MAX_CM * np.sin(np.radians(FOV))),
                     (self.cy - MAP_Y0 - 16) / GATE_MAX_CM)
        self.trail = deque(maxlen=500)

    def xy(self, cm, deg):
        t = np.radians(deg)
        return (int(self.cx + cm * np.sin(t) * self.s),
                int(self.cy - cm * np.cos(t) * self.s))

    def add(self, cm, deg):
        self.trail.append((cm, deg, time.time()))

    def draw(self, img, cm, deg):
        import cv2
        cv2.rectangle(img, (MAP_X0, MAP_Y0), (MAP_X1, MAP_Y1), PANEL, -1)

        # วงระยะ + ป้ายกำกับ
        for r in range(50, int(GATE_MAX_CM) + 1, 50):
            rad = int(r * self.s)
            cv2.ellipse(img, (self.cx, self.cy), (rad, rad), 0,
                        -90 - FOV, -90 + FOV, GRID, 1)
            cv2.putText(img, f"{r}cm", (MAP_X0 + 8, self.cy - rad + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 126, 136), 1, cv2.LINE_AA)
        # เส้นมุม
        for g in range(int(-FOV), int(FOV) + 1, 15):
            cv2.line(img, (self.cx, self.cy), self.xy(GATE_MAX_CM, g),
                     GRID if g else (86, 92, 104), 1)
            x, y = self.xy(GATE_MAX_CM + 20, g)
            cv2.putText(img, f"{g:+d}" if g else "0", (x - 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 126, 136), 1, cv2.LINE_AA)
        # เขตบอด (ใกล้กว่าเกตต่ำสุด) วาดจาง ๆ ให้รู้ว่าอ่านไม่ได้
        rad = int(GATE_MIN_CM * self.s)
        cv2.ellipse(img, (self.cx, self.cy), (rad, rad), 0,
                    -90 - FOV, -90 + FOV, (70, 55, 55), 1)

        # ร่องรอยเก่า — ยิ่งเก่ายิ่งจาง จึงเห็นเป็น "แผนที่" ตอนเลื่อนวัตถุ
        now = time.time()
        for c_, d_, t_ in self.trail:
            age = min((now - t_) / 20.0, 1.0)
            k = 1.0 - age
            col = (int(60 + 60 * k), int(90 + 110 * k), int(110 + 120 * k))
            cv2.circle(img, self.xy(c_, d_), 2, col, -1, cv2.LINE_AA)

        # ตำแหน่งปัจจุบัน
        if cm is not None and deg is not None:
            x, y = self.xy(cm, deg)
            cv2.line(img, (self.cx, self.cy), (x, y), (60, 110, 140), 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 11, HOT, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, HOT, -1, cv2.LINE_AA)
            cv2.putText(img, f"{cm:.0f}cm {deg:+.0f}", (x + 16, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, HOT, 1, cv2.LINE_AA)
        # ตัว sensor
        cv2.rectangle(img, (self.cx - 22, self.cy - 4), (self.cx + 22, self.cy + 8),
                      (150, 155, 165), -1)
        cv2.putText(img, "SENSOR", (self.cx - 26, self.cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 135, 145), 1, cv2.LINE_AA)


def draw_traces(img, envs, rate, k, pins):
    import cv2
    n = min(len(e) for e in envs)
    i_lo = max(1, int((T0_US + 2 * GATE_MIN_CM / 100 / C * 1e6) * 1e-6 * rate))
    i_hi = min(n, int((T0_US + 2 * GATE_MAX_CM / 100 / C * 1e6) * 1e-6 * rate))
    span = TR_X1 - TR_X0
    for ci, e in enumerate(envs):
        seg = e[i_lo:i_hi]
        y0 = TR_TOP + ci * (TR_H + TR_GAP)
        cv2.rectangle(img, (TR_X0, y0), (TR_X1, y0 + TR_H), PANEL, -1)
        if seg.size < 2:
            continue
        mx = max(float(seg.max()), 1e-9)
        xs = np.linspace(TR_X0, TR_X1, seg.size).astype(np.int32)
        ys = (y0 + TR_H - seg / mx * (TR_H - 8)).astype(np.int32)
        cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False,
                      CH_COL[ci % 4], 1, cv2.LINE_AA)
        cv2.putText(img, f"GPIO{pins[ci]}", (8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CH_COL[ci % 4], 1, cv2.LINE_AA)
        cv2.putText(img, f"{mx*1000:.0f}mV", (8, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 145, 155), 1, cv2.LINE_AA)
        if k is not None and i_lo <= k < i_hi:
            xk = int(TR_X0 + (k - i_lo) / max(1, i_hi - i_lo) * span)
            cv2.line(img, (xk, y0), (xk, y0 + TR_H), (70, 110, 255), 1)
    yb = TR_TOP + len(envs) * (TR_H + TR_GAP)
    for cm in range(int(GATE_MIN_CM), int(GATE_MAX_CM) + 1, 40):
        x = int(TR_X0 + (cm - GATE_MIN_CM) / (GATE_MAX_CM - GATE_MIN_CM) * span)
        cv2.line(img, (x, TR_TOP - 6), (x, yb), (46, 50, 58), 1)
        cv2.putText(img, str(cm), (x - 10, yb + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 126, 136), 1, cv2.LINE_AA)
    cv2.putText(img, "envelope per channel (x = distance, cm)", (TR_X0, yb + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 126, 136), 1, cv2.LINE_AA)


def render(envs, rate, cm, deg, k, pins, mv, note, paused, fps):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "distance = physics (echo time)   |   angle = trained model",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 195, 205), 1, cv2.LINE_AA)
    draw_traces(img, envs, rate, k, pins)
    mv.draw(img, cm, deg)

    # แถบตัวเลขใหญ่ใต้กราฟคลื่น
    by = TR_TOP + 4 * (TR_H + TR_GAP) + 46
    cv2.rectangle(img, (TR_X0 - 66, by), (TR_X1, by + 100), PANEL, -1)
    if cm is None:
        cv2.putText(img, "no echo in gate", (TR_X0 - 46, by + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (95, 100, 110), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, f"{cm:.1f}", (TR_X0 - 50, by + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, (250, 250, 250), 3, cv2.LINE_AA)
        cv2.putText(img, "cm", (TR_X0 + 108, by + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 165, 175), 2, cv2.LINE_AA)
        if deg is not None:
            cv2.putText(img, f"{deg:+.1f} deg", (TR_X0 + 190, by + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, HOT, 2, cv2.LINE_AA)
            cv2.putText(img, zone_en(deg), (TR_X0 + 190, by + 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(img, note, (TR_X0 - 60, by + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 135, 145), 1, cv2.LINE_AA)

    foot = "q quit | c clear trail | s save | space pause"
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
    ap.add_argument("--no-model", action="store_true", help="ไม่เทรนโมเดลมุม ดูแค่ระยะ")
    a = ap.parse_args()

    import cv2
    from ultrasonic import Ultrasonic

    mdl, note = None, "angle model: off"
    if not a.no_model:
        print("เทรนโมเดลมุมจาก car/data/ ...", flush=True)
        try:
            got = train_angle_model()
            if got is None:
                note = "angle model: ข้อมูลไม่พอ"
                print("!! ข้อมูลไม่พอ จะแสดงแต่ระยะ")
            else:
                sc_, m_, nsamp = got
                mdl, note = (sc_, m_), f"angle model: {nsamp} samples"
                print(f"เทรนเสร็จ ({nsamp} ตัวอย่าง)")
        except Exception as e:          # ข้อมูลเสีย/ไลบรารีหาย ไม่ควรล้มทั้งโปรแกรม
            note = "angle model: error"
            print("!! เทรนไม่สำเร็จ:", e)

    print(f"เปิด {a.port} · ช่อง {a.pins} ...", flush=True)
    us = Ultrasonic(port=a.port, pins=a.pins)
    mv = MapView()
    cv2.namedWindow("live map", cv2.WINDOW_AUTOSIZE)
    paused, last, fps = False, time.time(), 0.0
    envs = rate = k = cm = deg = None
    try:
        while True:
            if not paused:
                ping = us.ping()
                if ping is not None:
                    rate, counts = ping["rate"], ping["counts"]
                    envs = [envelope_of(counts[c], rate)
                            for c in range(counts.shape[0])]
                    k, cm, _ = common_peak(envs, rate)
                    if not (GATE_MIN_CM <= cm <= GATE_MAX_CM):
                        cm = None
                    deg = None
                    if mdl is not None and counts.shape[0] >= 2:
                        try:
                            f = angle_features(counts, rate).reshape(1, -1)
                            deg = float(mdl[1].predict(mdl[0].transform(f))[0])
                        except Exception:
                            deg = None
                    if cm is not None and deg is not None:
                        mv.add(cm, deg)
                        print(f"\r  {cm:6.1f} cm · {deg:+6.1f}°  {zone(deg):<8}",
                              end="", flush=True)
                    now = time.time()
                    fps = 0.8 * fps + 0.2 / max(now - last, 1e-6)
                    last = now
            if envs is not None:
                cv2.imshow("live map",
                           render(envs, rate, cm, deg, k, us.pins, mv, note, paused, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                mv.trail.clear()
            if key == ord(" "):
                paused = not paused
            if key == ord("s") and envs is not None:
                fn = time.strftime("live_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, render(envs, rate, cm, deg, k, us.pins,
                                       mv, note, paused, fps))
                print(f"\n  เซฟ {fn}")
    finally:
        us.close()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
