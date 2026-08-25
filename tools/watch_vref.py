#!/usr/bin/env python3
"""เฝ้าดูแรงดันจุดหนึ่งต่อเนื่อง แล้วบันทึก "เวลาที่มันเปลี่ยนสถานะ"

ใช้จับอาการติด ๆ ดับ ๆ (intermittent) ซึ่งการวัดทีละครั้งจับไม่ได้ —
วัดตอนมันดีก็บอกว่าดี วัดตอนมันเสียก็บอกว่าเสีย ต้องดูว่ามันเปลี่ยน "ตอนไหน"
แล้วเทียบกับสิ่งที่คนกำลังทำอยู่ ณ วินาทีนั้น (กดสายเส้นไหน งอบอร์ดตรงไหน)

    python tools/watch_vref.py --port COM6 --pin 36 --seconds 180

พิมพ์บรรทัดเฉพาะตอน "เปลี่ยนสถานะ" + heartbeat ทุก 15 วินาที จึงอ่านง่าย
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--pin", type=int, default=36, help="ขาที่เฝ้าดู DC (36=SP)")
    ap.add_argument("--pin2", type=int, default=39,
                    help="ขาที่เฝ้าดู AC ควบคู่ (39=SN) — ใช้ดูว่ามีสัญญาณเข้ามาไหม")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--lo", type=float, default=1.30, help="ต่ำกว่านี้ = ผิดปกติ")
    ap.add_argument("--hi", type=float, default=2.10, help="สูงกว่านี้ = ผิดปกติ")
    a = ap.parse_args()

    from ultrasonic import Ultrasonic
    us = Ultrasonic(port=a.port, pins=f"{a.pin},{a.pin2}")

    t0 = time.time()
    state = None          # True = ปกติ
    events, samples = [], []
    last_beat = 0.0
    print(f"\n  เฝ้าดู GPIO{a.pin} เป็นเวลา {a.seconds:.0f} วินาที "
          f"(ปกติ = {a.lo:.2f}-{a.hi:.2f} V)\n", flush=True)
    try:
        while time.time() - t0 < a.seconds:
            fr = us.ping(fire=False)
            if fr is None:
                continue
            v = float((fr["counts"][0].astype(np.float64) / 4095.0 * 3.3).mean())
            w = fr["counts"][1].astype(np.float64) / 4095.0 * 3.3
            pp = float(np.percentile(w, 99.5) - np.percentile(w, 0.5)) * 1000
            el = time.time() - t0
            samples.append(v)
            ok = a.lo <= v <= a.hi
            if state is None:
                state = ok
                print(f"  {el:6.1f}s  เริ่มต้น   {v:5.3f} V  "
                      f"{'ปกติ' if ok else '<<< ผิดปกติตั้งแต่แรก'}"
                      f"   · ขาออก {pp:6.1f} mVpp", flush=True)
            elif ok != state:
                state = ok
                events.append((el, v, ok))
                mark = "กลับมาปกติ" if ok else "<<<<<< ตกทันที"
                print(f"  {el:6.1f}s  เปลี่ยน!  {v:5.3f} V  {mark}"
                      f"   · ขาออก {pp:6.1f} mVpp", flush=True)
                last_beat = el
            elif el - last_beat >= 15:
                last_beat = el
                print(f"  {el:6.1f}s  ....      {v:5.3f} V  "
                      f"{'ปกติ' if ok else 'ยังผิดปกติ'}"
                      f"   · ขาออก {pp:6.1f} mVpp", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        us.close()

    if not samples:
        sys.exit("อ่านไม่ได้เลย")
    s = np.array(samples)
    good = float(((s >= a.lo) & (s <= a.hi)).mean() * 100)
    print(f"\n  สรุป: {len(s)} ตัวอย่าง · ปกติ {good:.0f}% · "
          f"ต่ำสุด {s.min():.3f} V · สูงสุด {s.max():.3f} V · เปลี่ยนสถานะ {len(events)} ครั้ง")
    if not events and good > 99:
        print("  => นิ่งตลอดช่วงที่ทดสอบ ไม่พบอาการติด ๆ ดับ ๆ")
    elif events:
        print("  => จดเวลาที่ 'เปลี่ยน!' ไว้ แล้วเทียบกับสิ่งที่กำลังแตะอยู่ ณ วินาทีนั้น")


if __name__ == "__main__":
    main()
