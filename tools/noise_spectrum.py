#!/usr/bin/env python3
"""แยกว่าเสียงรบกวนอยู่ที่ย่านความถี่ไหน — บอกได้ว่าต้องแก้ด้วยอะไร

อ่านค่าพื้นเสียงอย่างเดียวบอกได้แค่ "ดัง" แต่ไม่บอกว่าทำไม ย่านความถี่บอกได้:
  0-1 kHz      ไฟบ้าน 50 Hz และฮาร์มอนิก (มือคนแตะ · สายอินพุตยาว · กราวด์ลูป)
  1-20 kHz     เสียงในห้อง / การสั่นสะเทือน
  20-60 kHz    **ย่านที่เราใช้จริง** (40 kHz) — ตรงนี้ตัดทิ้งไม่ได้
  60 kHz ขึ้น  เกินย่านใช้งาน = ควรถูกกรองทิ้ง ถ้าเหลือเยอะแปลว่า
               แบนด์วิดท์ของภาคขยายกว้างเกิน (เช่นใช้ออปแอมป์เร็วโดยไม่มีตัวกรอง)

    python tools/noise_spectrum.py --port COM6 --pins 34,36
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
sys.path.insert(0, HERE)

BANDS = [(0, 1e3, "0-1 kHz   ไฟบ้าน/ฮัม"),
         (1e3, 20e3, "1-20 kHz  เสียง/สั่นสะเทือน"),
         (20e3, 60e3, "20-60 kHz **ย่านใช้งาน**"),
         (60e3, 150e3, "60-150 kHz เกินย่าน = ควรถูกกรอง"),
         (150e3, 1e9, "150 kHz+  เกินย่าน")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--pins", default="34,36")
    ap.add_argument("--pings", type=int, default=8)
    ap.add_argument("--samples", type=int, default=12000)
    a = ap.parse_args()

    from ultrasonic import Ultrasonic
    us = Ultrasonic(port=a.port, pins=a.pins, samples=a.samples)
    pins = us.pins
    acc = {i: [] for i in range(len(pins))}
    rate = None
    try:
        for _ in range(a.pings):
            fr = us.ping(fire=False)          # ไม่ยิง TX = เสียงรบกวนล้วน
            if fr is None:
                continue
            rate = fr["rate"]
            for i in range(len(pins)):
                v = fr["counts"][i].astype(np.float64) / 4095.0 * 3.3
                v -= v.mean()
                sp = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
                acc[i].append(sp)
    finally:
        us.close()

    if rate is None or not acc[0]:
        sys.exit("อ่านไม่ได้")
    f = np.fft.rfftfreq(a.samples, 1.0 / rate)
    print(f"\n  อัตราสุ่ม {rate/1000:.0f} kHz · ดูได้ถึง {rate/2000:.0f} kHz · "
          f"{len(acc[0])} ปิง (ไม่ยิง TX)\n")
    print(f"  {'ย่านความถี่':<28}" + "".join(f"{'g'+str(p):>14}" for p in pins))
    tot = {i: np.mean(acc[i], axis=0).sum() for i in acc}
    for lo, hi, name in BANDS:
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        row = []
        for i in range(len(pins)):
            e = np.mean(acc[i], axis=0)[m].sum()
            row.append(f"{e/tot[i]*100:9.1f} %  ")
        print(f"  {name:<28}" + "".join(row))
    print("\n  แปลผล: ถ้าพลังงานส่วนใหญ่อยู่เหนือ 60 kHz = แบนด์วิดท์ภาคขยายกว้างเกิน")
    print("         แก้ด้วยการใส่ C คร่อม Rf เพื่อจำกัดย่าน (ไม่ต้องรื้ออะไร)")
    print("         ถ้าส่วนใหญ่อยู่ 0-1 kHz = ฮัมจากไฟบ้าน แก้ที่สาย/กราวด์/ชีลด์")


if __name__ == "__main__":
    main()
