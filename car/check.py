#!/usr/bin/env python3
"""ตรวจสุขภาพระบบ — เปิดกล้อง depth + เซ็นเซอร์ แล้วดูว่าทั้งคู่ยังเห็นของเดียวกันไหม

รันได้ทุกเมื่อ ไม่บันทึกอะไรลงดิสก์ ใช้ก่อนอัด dataset ทุกครั้งเพื่อกันเสียเวลา 20 นาทีเปล่า

    python car/check.py --port COM5

**แสดงเฉพาะข้อมูลดิบ** ไม่กรอง ไม่หาเปลือกคลื่น ไม่คำนวณระยะจากเสียง —
สิ่งที่เห็นคือสิ่งที่บันทึกลงไฟล์เป๊ะ ๆ

ดูสามอย่าง:
  1. **valid** ของภาพ depth — ต่ำกว่า 25% แปลว่ากล้องมองไม่เห็นอะไร
  2. **raw peak per ch** — ความแรงดิบรายช่อง ถ้าช่องไหนต่ำผิดปกติแปลว่าวงจรช่องนั้นมีปัญหา
  3. **label dist / angle** — เฉลยที่จะถูกบันทึก ต้องตรงกับตำแหน่งเป้าจริง

ปุ่ม: q/ESC ออก · s เซฟภาพ · เว้นวรรค หยุด/เล่นต่อ
"""
import argparse
import faulthandler
import gc
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

import view              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=50.0)
    ap.add_argument("--size", default="320x240", help="ความละเอียด depth เช่น 320x240")
    a = ap.parse_args()

    import cv2
    from astra import Astra
    from sync4 import Sync4
    from record import DepthThread

    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()
    # ซ้อมเส้นทางคำนวณ+วาดภาพ **ก่อนเปิดกล้อง** — FFT ก้อนแรกจองหน่วยความจำใหญ่
    # ถ้าไปเกิดตอน OpenNI ทำงานอยู่คนละเธรด จะได้ heap corruption 0xc0000374
    print("ซ้อมเส้นทางแสดงผลก่อนเปิดกล้อง ...", flush=True)
    from record import _warmup
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    print(f"เปิดกล้อง depth {w}x{h} ...", flush=True)
    cam = Astra(want_rgb=False, depth_size=(w, h))
    gc.disable()      # ดูเหตุผลใน record.py — GC ชนกับ OpenNI ทำให้ heap พัง
    th = DepthThread(cam, 1)
    th.start()
    time.sleep(0.6)

    hist = {"amp": deque(maxlen=40), "fps": deque(maxlen=20)}
    strip = view.History()          # ประวัติค่าดิบสำหรับกราฟเลื่อน
    paused, last = False, time.time()
    img = None
    print("เปิดหน้าต่างแล้ว — กด q เพื่อออก", flush=True)
    try:
        while True:
            if not paused:
                ping = us.ping()
                got = th.get()
                if got is not None:
                    t_cam, depth = got
                    m = view.measure(depth, ping, gate=(40.0, a.max_cm))
                    strip.push(m)
                    now = time.time()
                    hist["fps"].append(1.0 / max(now - last, 1e-6))
                    last = now
                    if m["amps"]:
                        hist["amp"].append(float(np.max(m["amps"])))
                    sync = abs((ping["t"] - t_cam) * 1000.0) if ping else float("nan")

                    def stat(name, val, good, warn, unit, lower_better=True):
                        ok = (val <= good) if lower_better else (val >= good)
                        mid = (val <= warn) if lower_better else (val >= warn)
                        col = view.OKC if ok else (view.WARNC if mid else view.BADC)
                        return (f"{name} {val:.0f}{unit}", col)

                    lines = [
                        (f"fps {np.mean(hist['fps']):.1f}", (170, 175, 185)),
                        (f"sync {sync:.0f}ms", view.OKC if sync < 60 else view.WARNC),
                        stat("depth valid", m["valid"] * 100, 40, 25, "%", False),
                    ]
                    if hist["amp"]:
                        lines.append((f"raw peak(avg) {np.mean(hist['amp']):.0f}mV",
                                      (170, 175, 185)))
                    if us.bad_frames:
                        lines.append((f"bad frames {us.bad_frames}", view.BADC))
                    img = view.render(
                        depth, ping, m, us.pins,
                        "SYSTEM CHECK  -  camera + ultrasonic",
                        lines,
                        "q quit | s save | space pause     "
                        "raw only: no filtering, no envelope, every frame recorded",
                        hist=strip,
                        sub=(f"4ch simultaneous  {us.rate:.0f} Hz/ch  "
                             f"{us.samples} samples  period {us.period*1e3:.0f} ms  "
                             f"range {a.max_cm:.0f} cm   |   saved per frame: "
                             f"counts[4x{us.samples}] uint16 + depth PNG + label"))
            if img is not None:
                cv2.imshow("check", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s") and img is not None:
                fn = time.strftime("check_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, img)
                print(f"เซฟ {fn}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        th.stop_flag = True
        time.sleep(0.3)
        gc.enable()                     # เปิดคืนหลังกล้องหยุดแล้วเท่านั้น
        try:
            cam.close()
        except Exception:
            pass
        us.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if hist["diff"]:
            d, s = float(np.mean(hist["diff"])), float(np.mean(hist["snr"] or [0]))
            print(f"\nสรุป: diff เฉลี่ย {d:.0f} cm · SNR เฉลี่ย {s:.1f} เท่า · "
                  f"เฟรมเสีย {us.bad_frames}")
            print("  พร้อมอัด" if d < 20 and s >= 8 else
                  "  ยังไม่ดีนัก — diff ควร < 20 cm และ SNR ควร > 8 เท่า")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
