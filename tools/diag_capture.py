"""เก็บข้อมูลหนึ่งสถานะของ TX (เสียบ หรือ ถอด) แล้วบันทึก — รันแยกทีละสถานะ

    python tools/diag_capture.py --tx on       TX เสียบอยู่ตามปกติ
    python tools/diag_capture.py --tx off      ถอดสาย TX ออกแล้ว
    python tools/diag_capture.py --tx on --tag charger   เงื่อนไขอื่นที่ต่างไป (ตัวอย่าง: ต่อสายชาร์จ)

วิธีใช้
  1. ตั้งสถานะ TX ให้เรียบร้อย แล้วรันคำสั่ง
  2. มีนับถอยหลัง 15 วินาที (เปลี่ยนด้วย --wait) ใช้ถอยมือและตัวออกจากแผ่น
  3. จากนั้นวัด 60 วินาที (--seconds) **อย่าขยับอะไรทั้งนั้น** รวมถึงตัวคุณและของในห้อง
  4. ผลบันทึกที่ car/data/diag_tx-<on|off>_<เวลา>.npz แล้วพิมพ์ตารางสรุปให้เลย

ทุกปิงสลับ "ยิง" กับ "ไม่ยิง" (คำสั่ง q) เพื่อวัดสัญญาณจากเสียงยิงและนอยส์ของวงจรรับ
ในช่วงเวลาเดียวกัน · แนะนำลำดับ เสียบ -> ถอด -> เสียบ (รัน --tx on อีกรอบ)
จะรู้ว่าสถานะกลับมาเหมือนเดิมจริง ก่อนเชื่อผลของการถอด

เทียบหลายไฟล์ด้วย  python tools/diag_compare.py
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "car"))
sys.path.insert(0, HERE)


def collect(us, seconds, log=print):
    """สลับยิง/ไม่ยิง จนครบเวลา แล้วจบที่คู่ครบ (ยิงกับไม่ยิงจำนวนเท่ากัน)"""
    frames, times, fired = [], [], []
    fire, t0, last = True, time.time(), time.time()
    while True:
        if time.time() - t0 >= seconds and fire:      # ครบเวลา และเพิ่งจบคู่
            break
        p = us.ping(fire=fire)
        if p is None:
            continue                                  # เฟรมเสีย: ลองชนิดเดิมซ้ำ
        frames.append(p["counts"])
        times.append(time.time())
        fired.append(fire)
        fire = not fire
        if time.time() - last >= 10:
            last = time.time()
            log(f"  ... {time.time() - t0:3.0f}/{seconds:.0f} s · {len(frames)} ปิง")
    return np.stack(frames), np.array(times), np.array(fired, bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx", required=True, choices=("on", "off"),
                    help="สถานะ TX ตอนนี้ (บันทึกลงชื่อไฟล์ ไม่ได้ควบคุมอะไร)")
    ap.add_argument("--tag", default="",
                    help="บันทึกเงื่อนไขอื่นที่ต่างไปในรอบนี้ เช่น charger (ต่อสายชาร์จโน้ตบุ๊ก) "
                         "ต่อท้ายชื่อไฟล์และโชว์ในตารางเทียบ")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--wait", type=float, default=15.0)
    ap.add_argument("--max-cm", type=float, default=500.0)
    a = ap.parse_args()

    from sync4 import Sync4
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins="35,34,32,33", max_cm=a.max_cm, period_ms=None, verbose=False)
    print(f"สถานะ TX = {a.tx.upper()} · {us.samples} ตัวอย่าง/ช่อง · คาบ {us.period * 1e3:.0f} ms")
    for s in range(int(a.wait), 0, -1):
        print(f"  เริ่มใน {s:2d} s — ถอยออกจากแผ่น อย่าขยับอะไร", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 60)
    print(f"วัด {a.seconds:.0f} วินาที ...", flush=True)
    counts, t, fired = collect(us, a.seconds)
    lost = us.bad_frames
    us.close()

    tag = a.tag.replace("_", "-").strip()           # "_" ใช้แบ่งช่องเวลาในชื่อไฟล์
    out = os.path.join(os.path.dirname(HERE), "car", "data",
                       time.strftime(f"diag_tx-{a.tx}{'-' + tag if tag else ''}_%Y%m%d_%H%M%S.npz"))
    np.savez_compressed(out, counts=counts, t=t, fired=fired, rate=us.rate, tx=a.tx, tag=tag)
    print(f"\nบันทึก {len(counts)} ปิง ({int(fired.sum())} ยิง / {int((~fired).sum())} ไม่ยิง"
          f" · เฟรมเสียที่ยิงซ้ำ {lost}) -> {out}\n")
    from diag_compare import analyze, show
    show([analyze(out)])


if __name__ == "__main__":
    main()
