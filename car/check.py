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
import threading
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


class Grabber(threading.Thread):
    """อ่านเซ็นเซอร์ในเธรดแยก เก็บเฉพาะผลล่าสุด

    เดิมเรียก us.ping() ในเธรดเดียวกับที่วาดจอ ทุกครั้งที่เฟรมหายระหว่างทาง
    (วัดแล้วราว 2% ของเฟรมที่ช่วง 500 ซม.) การอ่านจะรอจน timeout ซึ่งตอนนั้น
    10 วินาที เธรดหลักไม่ได้เรียก waitKey เลย Windows จึงตีว่าหน้าต่างไม่ตอบสนอง
    """

    def __init__(self, us):
        super().__init__(daemon=True)
        self.us = us
        self.lock = threading.Lock()
        self.latest, self.seq = None, 0
        self.stop_flag, self.paused = False, False

    def run(self):
        while not self.stop_flag:
            if self.paused:
                time.sleep(0.02)
                continue
            try:
                p = self.us.ping()
            except Exception:           # พอร์ตหลุดกลางทาง: ตัวเลข seq ไม่ขยับ จอจะขึ้นป้ายเงียบ
                time.sleep(0.2)
                continue
            with self.lock:
                self.latest, self.seq = p, self.seq + 1

    def get(self):
        with self.lock:
            return self.latest, self.seq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    # ต้องตรงกับ rig2.PINS (TL TR BL BR) เหมือน record.py และ live_map2.py
    ap.add_argument("--pins", default="35,34,32,33")
    ap.add_argument("--no-cam", action="store_true",
                    help="ดูคลื่นดิบอย่างเดียว ไม่เปิดกล้อง")
    # 500 ซม. ไม่ใช่ 200: หน้านี้ไว้ดูคลื่นดิบ ไม่ได้อัดข้อมูล ดูไกลได้โดยไม่กระทบชุดข้อมูลเทรน
    # (record.py กับ live_map2.py ยังใช้ 200 เพราะโมเดลรอบ 3 เทรนมาที่ช่วงนั้น)
    ap.add_argument("--max-cm", type=float, default=500.0)
    ap.add_argument("--period-ms", type=float, default=None,
                    help="ไม่ใส่ = 50 ms ถ้าช่วง <= 200 ซม. นอกนั้นคำนวณให้เอง "
                         "(ช่วงไกลต้องใช้คาบยาว ไม่งั้นเอคโค่รอบก่อนทับรอบใหม่)")
    ap.add_argument("--size", default="320x240", help="ความละเอียด depth เช่น 320x240")
    a = ap.parse_args()

    import cv2
    from sync4 import Sync4

    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    period = a.period_ms if a.period_ms is not None else (
        50.0 if a.max_cm <= 200.0 else None)       # None = Sync4 คำนวณคาบต่ำสุดเอง
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=period, verbose=True)
    us.ping()
    # ซ้อมเส้นทางคำนวณ+วาดภาพ **ก่อนเปิดกล้อง** — FFT ก้อนแรกจองหน่วยความจำใหญ่
    # ถ้าไปเกิดตอน OpenNI ทำงานอยู่คนละเธรด จะได้ heap corruption 0xc0000374
    print("ซ้อมเส้นทางแสดงผลก่อนเปิดกล้อง ...", flush=True)
    from record import _warmup
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    cam = th = None
    blank = np.zeros((h, w), np.uint16)   # ใช้แทนภาพกล้องตอนไม่เปิดกล้อง
    if a.no_cam:
        print("ไม่เปิดกล้อง — ดูคลื่นดิบอย่างเดียว", flush=True)
    else:
        from astra import Astra
        from record import DepthThread
        print(f"เปิดกล้อง depth {w}x{h} ...", flush=True)
        gc.disable()  # ดูเหตุผลใน record.py — GC ชนกับ OpenNI ทำให้ heap พัง
        try:
            cam = Astra(want_rgb=False, depth_size=(w, h))
            th = DepthThread(cam, 1)
            th.start()
            time.sleep(0.6)
        except Exception as e:
            # กล้องเปิดไม่ได้ไม่ใช่เหตุให้เลิก คลื่นดิบยังดูได้
            # ต้องเปิด GC คืนตรงนี้ เพราะ finally ข้างล่างเช็ค th ว่ามีไหม
            gc.enable()
            cam = th = None
            print(f"  !! เปิดกล้องไม่ได้ ({e}) — ไปต่อแบบไม่มีกล้อง", flush=True)

    hist = {"amp": deque(maxlen=40), "fps": deque(maxlen=20)}
    strip = view.History()          # ประวัติค่าดิบสำหรับกราฟเลื่อน
    paused, last = False, time.time()
    img = None
    grab = Grabber(us)
    grab.start()
    last_seq, t_new = 0, time.time()
    state = {"depth": blank, "ping": None, "m": None, "stalled": False}

    def stat(name, val, good, warn, unit, lower_better=True):
        ok = (val <= good) if lower_better else (val >= good)
        mid = (val <= warn) if lower_better else (val >= warn)
        col = view.OKC if ok else (view.WARNC if mid else view.BADC)
        return (f"{name} {val:.0f}{unit}", col)

    def draw(stalled_s=None):
        depth, ping, m = state["depth"], state["ping"], state["m"]
        t_cam = state.get("t_cam", time.time())
        sync = abs((ping["t"] - t_cam) * 1000.0) if ping else float("nan")
        lines = [
            (f"fps {np.mean(hist['fps']) if hist['fps'] else 0:.1f}", (170, 175, 185)),
            (f"sync {sync:.0f}ms", view.OKC if sync < 60 else view.WARNC),
            stat("depth valid", m["valid"] * 100, 40, 25, "%", False),
        ]
        if hist["amp"]:
            lines.append((f"raw peak(avg) {np.mean(hist['amp']):.0f}mV", (170, 175, 185)))
        if us.bad_frames:
            lines.append((f"lost frames {us.bad_frames}", view.BADC))
        if stalled_s is not None:
            lines.append((f"SENSOR NOT ANSWERING {stalled_s:.0f}s", view.BADC))
        return view.render(
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

    print("เปิดหน้าต่างแล้ว — กด q เพื่อออก", flush=True)
    try:
        while True:
            # เธรดหลักไม่รอซีเรียลเลย: อ่านเฉพาะผลล่าสุดที่เธรดอ่านเก็บไว้
            # จึงยังส่งคำสั่งให้หน้าต่างได้ทุก ~10 ms แม้เฟรมหายแล้วเธรดอ่านติดรอ timeout
            ping, seq = grab.get()
            if seq != last_seq:
                got = th.get() if th else None
                if got is not None or th is None:
                    last_seq, t_new = seq, time.time()
                    t_cam, depth = got if got else (time.time(), blank)
                    m = view.measure(depth, ping, gate=(40.0, a.max_cm))
                    strip.push(m)
                    now = time.time()
                    hist["fps"].append(1.0 / max(now - last, 1e-6))
                    last = now
                    if m["amps"]:
                        hist["amp"].append(float(np.max(m["amps"])))
                    state.update(depth=depth, ping=ping, m=m, t_cam=t_cam, stalled=False)
                    img = draw()
            elif (state["m"] is not None and not paused and not state["stalled"]
                  and time.time() - t_new > 2.0):
                # เงียบเกิน 2 วินาที: วาดซ้ำพร้อมป้ายบอก ไม่ใช่ปล่อยจอนิ่งให้เข้าใจว่าปกติ
                state["stalled"] = True
                img = draw(time.time() - t_new)
            if img is not None:
                cv2.imshow("check", img)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                grab.paused = paused
            if key == ord("s") and img is not None:
                fn = time.strftime("check_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fn, img)
                print(f"เซฟ {fn}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        grab.stop_flag = True
        if th:
            th.stop_flag = True
            time.sleep(0.3)
            gc.enable()                 # เปิดคืนหลังกล้องหยุดแล้วเท่านั้น
            try:
                cam.close()
            except Exception:
                pass
        us.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # hist มีแค่ "amp" กับ "fps" — เดิมบล็อกนี้อ่านคีย์ "diff"/"snr" ที่ไม่มีแล้ว
        # จึงเกิด KeyError ทุกครั้งที่กด q ออก
        total = max(us.pings, 1)
        print(f"\nสรุป: ปิงทั้งหมด {us.pings} · เฟรมที่ส่งไม่ครบ/เสียหายต้องยิงซ้ำ "
              f"{us.bad_frames} ({100 * us.bad_frames / total:.1f}%)")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
