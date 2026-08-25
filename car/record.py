#!/usr/bin/env python3
"""เก็บ dataset แบบ **อัดต่อเนื่อง** — เปิดเซ็นเซอร์กับกล้อง depth พร้อมกัน อัดเป็นช่วงละ N นาที

ต่างจาก collect.py: ตัวนั้นเก็บทีละช็อตตอนกดปุ่ม (ฉากต้องนิ่ง) ตัวนี้อัดไม่หยุด
คนเลื่อนวัตถุไปมาได้ตลอดเวลา กล้อง depth เป็นเฉลย (ground truth) ทุกเฟรม

    python car/record.py --port COM5                  4 ช่วง ช่วงละ 5 นาที
    python car/record.py --port COM5 --sections 2 --minutes 1    ลองสั้น ๆ ก่อน

ความเร็วจริงที่วัดได้ (ส.ค. 2026 · 4 ช่อง · ระยะ 250 cm):
    เซ็นเซอร์ **~2.6 ครั้ง/วินาที** — คอขวดคือการส่งคลื่นดิบ 16 KB/ปิง ผ่านสาย USB
    ที่ 921600 baud (348 ms) ไม่ใช่ตัวเซ็นเซอร์หรือ ADC
    กล้อง depth 320x240 วิ่ง 30 fps (ฮาร์ดแวร์ไม่มีโหมด 15 fps) จึงเร็วกว่าเซ็นเซอร์ ~11 เท่า
    => ทุกปิงมีเฟรมกล้องสด ๆ รออยู่เสมอ ความเหลื่อมเวลาต่ำกว่า 35 ms

โครงสร้างที่ได้: car/data/<ชื่อ>_s1 .. _s4 โฟลเดอร์ละ 1 ช่วง (ใช้แบ่ง train/test ได้ตรง ๆ)
"""
import argparse
import faulthandler
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
faulthandler.enable()                 # เปลี่ยน crash เงียบ ๆ ของ native ให้เห็น stack

import numpy as np
import numpy.ma          # noqa: F401  โหลดไว้ก่อน — import ขณะ OpenNI ทำงานทำให้ล้ม
import numpy.lib         # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

DATA_ROOT = Path(HERE) / "data"
DEPTH_W, DEPTH_H = 320, 240


class DepthThread(threading.Thread):
    """อ่านกล้องไม่หยุดในเธรดแยก เก็บเฟรมล่าสุดไว้ให้หยิบ

    ทำไมต้องมีเธรด: OpenNI คิวเฟรมไว้ ถ้าเรียก read_frame เฉพาะตอนยิงเซ็นเซอร์
    (ทุก ~385 ms) จะได้เฟรมเก่าค้างคิวแทนเฟรมปัจจุบัน = เฉลยไม่ตรงกับคลื่น
    """

    def __init__(self, cam, keep_every):
        super().__init__(daemon=True)
        self.cam, self.keep_every = cam, keep_every
        self.lock = threading.Lock()
        self.latest = None            # (t, depth)
        self.frames = 0
        self.stop_flag = False
        self.err = None

    def run(self):
        try:
            i = 0
            while not self.stop_flag:
                d = self.cam.read_depth()
                i += 1
                self.frames += 1
                if i % self.keep_every:      # ทิ้งเฟรมส่วนเกิน = ลด fps ด้วยซอฟต์แวร์
                    continue
                with self.lock:
                    self.latest = (time.time(), d)
        except Exception as e:               # กล้องหลุดไม่ควรทำให้ทั้งโปรแกรมค้าง
            self.err = e

    def get(self):
        with self.lock:
            return self.latest


def _warmup(tmp_root, nsamp=677, rate=66300.0):
    """ซ้อมเส้นทางบันทึกจริงหนึ่งรอบด้วยข้อมูลปลอม **ก่อนเปิดกล้อง**

    การ import โมดูลใหม่ (numpy.ma ผ่าน percentile, zipfile ผ่าน savez_compressed)
    ขณะ OpenNI native ทำงานอยู่ ทำให้โปรเซสตายเงียบ ๆ บน Windows — เคยเสียเวลา
    ไล่หาสาเหตุมาแล้ว จึงบังคับให้ import ทุกอย่างตั้งแต่ยังไม่เปิดกล้อง
    """
    import shutil
    import tempfile
    import cv2
    import view
    from labels import depth_to_profile, near_object_count
    tmp = Path(tempfile.mkdtemp(prefix="rec_warm_", dir=tmp_root))
    try:
        depth = np.full((DEPTH_H, DEPTH_W), 900, np.uint16)
        depth[90:150, 140:190] = 700
        rng = np.random.default_rng(0)
        ping = {"counts": (2048 + rng.normal(0, 30, (4, nsamp))).astype(np.uint16),
                "rate": rate, "skew_us": 1.25, "pins": [34, 35, 32, 33], "t": 0.0}
        save_pair(tmp, 0, ping, depth, 0.0)
        depth_to_profile(depth)
        near_object_count(depth)
        # ซ้อมเส้นทางแสดงผลทั้งเส้น (FFT ของ scipy/numpy จองหน่วยความจำก้อนใหญ่รอบแรก)
        # ถ้าไปเกิดตอน OpenNI ทำงานอยู่ในอีกเธรด จะได้ heap corruption 0xc0000374
        m = view.measure(depth, ping)
        view.render(depth, ping, m, ping["pins"], "warmup", [("", (0, 0, 0))])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()
    gc.freeze()


def save_pair(out_dir, idx, ping, depth, dt_ms, near_cm=float("nan"), valid=float("nan")):
    """บันทึกคลื่นดิบ + ภาพ depth + ค่าบอกคุณภาพ (ไว้คัดกรองตอนเทรน ไม่ใช่ตอนเก็บ)"""
    import cv2
    np.savez_compressed(
        out_dir / f"us_{idx:06d}.npz",
        counts=ping["counts"], rate=ping["rate"], skew_us=ping["skew_us"],
        pins=np.array(ping["pins"]), t=ping["t"], sync_ms=dt_ms,
        cam_near_cm=near_cm, depth_valid=valid)
    cv2.imwrite(str(out_dir / f"depth_{idx:06d}.png"), depth)     # uint16 มิลลิเมตร


def wait_to_start(us, cam_thread, sec_no, total, gate, show=True):
    """แสดงภาพสดพร้อมข้อความ 'กดเพื่อเริ่ม' — คืน True ถ้าจะอัด · False ถ้าจะเลิก

    ทำไมต้องเห็นภาพก่อนเริ่ม: ต้องรู้ว่ากล้องจับเราถูกตัวและอยู่ในระยะแล้วจริง ๆ
    ก่อนกดอัด ไม่งั้นอัดครบนาทีแล้วเพิ่งมารู้ว่า label ผิดทั้งช่วง
    """
    import cv2
    import view
    if not show:
        input(f"  ช่วงที่ {sec_no} — กด Enter เพื่อเริ่มอัด ...")
        return True
    cv2.namedWindow("recording", cv2.WINDOW_AUTOSIZE)
    strip = view.History()
    print(f"  กดปุ่ม SPACE (หรือ Enter) ในหน้าต่างภาพเพื่อเริ่มอัด · q = เลิก",
          flush=True)
    while True:
        ping = us.ping()
        got = cam_thread.get()
        if ping is not None and got is not None:
            _t, depth = got
            m = view.measure(depth, ping, gate=gate)
            strip.push(m)
            lines = [
                (f"READY  s{sec_no}/{total}", (80, 200, 250)),
                ("PRESS SPACE TO START", (250, 250, 250)),
                (f"depth valid {m['valid']*100:.0f}%",
                 view.OKC if m["valid"] > 0.4 else view.WARNC),
            ]
            if m["amps"]:
                lines.append((f"raw peak {max(m['amps']):.0f}mV", (170, 175, 185)))
            cv2.imshow("recording", view.render(
                depth, ping, m, us.pins, f"READY  -  section {sec_no} of {total}",
                lines, "SPACE/ENTER = start recording     q = quit",
                hist=strip,
                sub=(f"4ch simultaneous  {us.rate:.0f} Hz/ch  {us.samples} samples  "
                     f"period {us.period*1e3:.0f} ms  range {gate[1]:.0f} cm")))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord(" "), 13, 10):
            return True
        if k in (ord("q"), 27):
            return False


def record_section(us, cam_thread, out_dir, seconds, min_valid, sec_no, show=True,
                   gate=(40.0, 150.0)):
    """อัดหนึ่งช่วง คืน (จำนวนที่บันทึก, จำนวนที่ข้าม, ค่าเฉลี่ยความเหลื่อมเวลา ms)

    มีหน้าต่างแสดงผลระหว่างอัด (กด q = หยุดช่วงนี้) เพราะดูจาก terminal อย่างเดียว
    ไม่รู้เลยว่ากล้องจับตัวเป้าถูกไหม — เคยเสียเวลาอัดยาวแล้วพบทีหลังว่า label ผิด
    """
    import cv2
    import view
    from labels import depth_to_profile
    out_dir.mkdir(parents=True, exist_ok=True)
    if show:
        cv2.namedWindow("recording", cv2.WINDOW_AUTOSIZE)
    t_end = time.time() + seconds
    idx, skipped, syncs, last_print = 0, 0, [], 0.0
    snrs, diffs, out_of_range = [], [], 0
    strip = view.History()          # ประวัติค่าดิบสำหรับกราฟเลื่อน
    while time.time() < t_end:
        ping = us.ping()
        if ping is None:
            skipped += 1
            continue
        got = cam_thread.get()
        if got is None:
            skipped += 1
            continue
        t_cam, depth = got
        dt_ms = (ping["t"] - t_cam) * 1000.0
        valid = float(np.count_nonzero(depth)) / depth.size
        dist, ok = depth_to_profile(depth)
        # **ไม่กรองอะไรทั้งนั้น** — บันทึกทุกเฟรมที่อ่านได้ครบทั้งคลื่นและภาพ depth
        # การคัดทิ้งย้ายไปทำตอนเทรนแทน (ที่นั่นย้อนกลับได้ ที่นี่ทิ้งแล้วทิ้งเลย)
        near_cm = (float(np.min(np.where(ok, dist, 1e9))) / 10.0
                   if ok.any() else float("nan"))
        if not (gate[0] <= near_cm <= gate[1]):
            out_of_range += 1            # นับไว้ดูเฉย ๆ ไม่ได้ข้าม
        save_pair(out_dir, idx, ping, depth, dt_ms, near_cm, valid)
        syncs.append(abs(dt_ms))
        idx += 1
        left = t_end - time.time()
        if show:
            m = view.measure(depth, ping, gate=gate)
            strip.push(m)
            if m["amps"]:
                snrs.append(float(np.max(m["amps"])))
            # แถวสถานะชุดเดียวกับ check.py (period · sync · depth valid · raw peak)
            # แล้วต่อท้ายด้วยของที่มีเฉพาะตอนอัด — จะได้อ่านหน้าจอแบบเดียวกันทั้งสองโหมด
            lines = [
                (f"REC s{sec_no}", (90, 90, 245)),
                (f"period {us.period*1e3:.0f}ms", (170, 175, 185)),
                (f"sync {np.mean(syncs):.0f}ms",
                 view.OKC if np.mean(syncs) < 60 else view.WARNC),
                (f"depth valid {m['valid']*100:.0f}%",
                 view.OKC if m["valid"] > 0.4 else view.WARNC),
            ]
            if snrs:
                lines.append((f"raw peak(avg) {np.mean(snrs[-40:]):.0f}mV",
                              (170, 175, 185)))
            per_ms, late_frac = us.cadence()
            lines.append((f"real period {per_ms:.0f}ms",
                          view.OKC if late_frac < 0.1 else view.WARNC))
            lines += [
                (f"time left {left:5.1f}s", (220, 225, 235)),
                (f"saved {idx}", (170, 175, 185)),
                (f"out-of-range {out_of_range}",
                 view.WARNC if out_of_range > idx else (170, 175, 185)),
            ]
            cv2.imshow("recording", view.render(
                depth, ping, m, us.pins, f"RECORDING  section {sec_no}", lines,
                "q = stop this section early     raw data only, every frame recorded",
                hist=strip,
                sub=(f"4ch simultaneous  {us.rate:.0f} Hz/ch  {us.samples} samples  "
                     f"period {us.period*1e3:.0f} ms  range {gate[1]:.0f} cm")))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("\n  หยุดช่วงนี้ก่อนกำหนดตามที่กด q", flush=True)
                break
        if time.time() - last_print > 0.5:
            last_print = time.time()
            print(f"\r  เหลือ {left:5.1f}s · เก็บ {idx:4d} · เกินระยะ {out_of_range:4d} "
                  f"· ใกล้สุด {near_cm:6.1f} cm · valid {valid*100:3.0f}% · "
                  f"เหลื่อม {np.mean(syncs):4.0f} ms", end="", flush=True)
    print()
    per_ms, late_frac = us.cadence()
    if late_frac > 0.1:
        print(f"  !! จังหวะไม่คงที่: ยิงช้ากว่าคาบที่ตั้ง ({us.period*1e3:.0f} ms) "
              f"{late_frac:.0%} ของเฟรม · คาบจริง {per_ms:.1f} ms")
        print(f"     งานในลูปยาวกว่าคาบ — ตั้ง --period-ms {np.ceil(per_ms/5)*5:.0f} "
              f"หรือใช้ --no-view เพื่อตัดเวลาวาดภาพออก")
    if out_of_range:
        print(f"  หมายเหตุ: {out_of_range}/{idx} เฟรม เฉลยอยู่นอกช่วง "
              f"{gate[0]:.0f}-{gate[1]:.0f} cm — **เก็บไว้ครบ** คัดทีหลังได้จากค่า "
              f"cam_near_cm ที่บันทึกไว้ในไฟล์")
    return idx, skipped, (float(np.mean(syncs)) if syncs else 0.0)


def _run(a):
    from astra import Astra
    from sync4 import Sync4

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"เปิดเซ็นเซอร์ {a.port} · ช่อง {a.pins} · 4 ช่องพร้อมกัน ...", flush=True)
    # โหมดนี้ **ไม่ยอมถอย** ไปใช้การยิงหลายรอบ ถ้าตั้ง 4 ช่องไม่ได้จะโยน error ทันที
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()                                  # เฟรมแรกทิ้ง (ตั้งค่า ADC)

    # ซ้อม **หลังรู้รูปร่างข้อมูลจริง แต่ก่อนเปิดกล้อง** — ลำดับนี้สำคัญ ถ้า FFT ก้อนแรก
    # ไปเกิดตอน OpenNI ทำงานอยู่คนละเธรด จะได้ heap corruption 0xc0000374
    print("ซ้อมเส้นทางบันทึก+แสดงผลก่อนเปิดกล้อง ...", flush=True)
    _warmup(DATA_ROOT, nsamp=us.samples, rate=us.rate)

    print(f"เปิดกล้อง depth {DEPTH_W}x{DEPTH_H} ...", flush=True)
    cam = Astra(want_rgb=False, depth_size=(DEPTH_W, DEPTH_H))
    keep = max(1, round(cam.depth_fps / a.cam_fps))
    print(f"  กล้องวิ่ง {cam.depth_fps} fps · เก็บทุก {keep} เฟรม "
          f"= {cam.depth_fps/keep:.0f} fps")
    # **ปิด GC ตลอดช่วงที่กล้องทำงาน** — stack trace ตอนล้มชี้ตรงว่าเป็น
    # "Garbage-collecting" ในเธรดหลักขณะ OpenNI อ่านเฟรมอยู่อีกเธรด
    # CPython คืนหน่วยความจำด้วยการนับอ้างอิงอยู่แล้ว ตัว GC มีไว้เก็บวงอ้างอิง
    # ซึ่งข้อมูลของเรา (numpy array + dict ธรรมดา) แทบไม่มี จึงปิดได้อย่างปลอดภัย
    gc.disable()
    th = DepthThread(cam, keep)
    th.start()
    time.sleep(0.6)                            # ให้เธรดมีเฟรมแรกก่อนเริ่มจับเวลา

    secs = a.minutes * 60.0
    report = []
    try:
        for s in range(a.start, a.start + a.sections):
            out = DATA_ROOT / f"{a.name}_s{s}"
            if out.exists() and any(out.glob("us_*.npz")) and not a.overwrite:
                print(f"!! {out.name} มีข้อมูลอยู่แล้ว — ข้าม "
                      f"(ใช้ --overwrite ถ้าต้องการอัดทับ)")
                continue
            print(f"\n=== ช่วงที่ {s}/{a.sections} → {out.name} "
                  f"({a.minutes:g} นาที) ===")
            if a.auto or not sys.stdin.isatty():
                gap = a.gap if s > a.start else min(a.gap, 10)  # ช่วงแรกไม่ต้องพักนาน
                for k in range(int(gap), 0, -1):
                    if k <= 5 or k % 5 == 0:
                        print(f"  เริ่มอัดใน {k} วินาที ...", flush=True)
                    time.sleep(1)
            elif not wait_to_start(us, th, s, a.start + a.sections - 1,
                                   (40.0, a.max_cm), show=not a.no_view):
                print("  เลิกตามที่กด q")
                break
            n, sk, sync = record_section(us, th, out, secs, a.min_valid,
                                         s, show=not a.no_view,
                                         gate=(40.0, a.max_cm))
            (out / "meta.json").write_text(json.dumps({
                "section": s, "samples": n, "skipped": sk,
                "seconds": secs, "pins": a.pins, "simultaneous": True,
                "us_rate_hz": us.rate, "us_samples": us.samples,
                "period_ms": us.period * 1e3, "max_cm": a.max_cm,
                "depth_size": [DEPTH_W, DEPTH_H],
                "depth_fps_saved": cam.depth_fps / keep,
                "us_fps": n / secs, "mean_sync_ms": sync,
                "depth_units": "uint16 PNG มิลลิเมตร 0=อ่านไม่ได้",
                "us_format": "npz: counts[ch][sample] 0..4095, rate, skew_us, pins, t, sync_ms",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            report.append((out.name, n, sk, n / secs, sync))
            print(f"  เสร็จ: {n} ตัวอย่าง ({n/secs:.1f}/วิ) · ข้าม {sk} · "
                  f"เหลื่อมเฉลี่ย {sync:.0f} ms")
    except KeyboardInterrupt:
        print("\nหยุดกลางคัน — ข้อมูลที่เก็บไปแล้วยังอยู่ครบ")
    finally:
        th.stop_flag = True
        time.sleep(0.3)
        gc.enable()                     # เปิดคืนหลังกล้องหยุดแล้วเท่านั้น
        try:
            cam.close()
        except Exception:
            pass
        us.close()

    if report:
        print("\nสรุป")
        print(f"  {'ช่วง':<16}{'ตัวอย่าง':>9}{'ข้าม':>7}{'fps':>7}{'เหลื่อม ms':>12}")
        for nm, n, sk, f, sy in report:
            print(f"  {nm:<16}{n:>9}{sk:>7}{f:>7.1f}{sy:>12.0f}")
        tot = sum(r[1] for r in report)
        print(f"  รวม {tot} ตัวอย่าง · เทรนด้วย 3 ช่วง ทดสอบ 1 ช่วง:")
        print(f"     python car/train_sections.py --name {report[0][0].rsplit('_s',1)[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--name", default="rec", help="ชื่อชุด → โฟลเดอร์ <ชื่อ>_s1.._sN")
    ap.add_argument("--sections", type=int, default=4, help="อัดกี่ช่วงในการรันครั้งนี้")
    ap.add_argument("--start", type=int, default=1,
                    help="เริ่มนับช่วงที่เลขนี้ — ใช้ตอนแยกรันทีละช่วง (s1, s2, ...)")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--max-cm", type=float, default=200.0,
                    help="ระยะสูงสุดที่สนใจ — กำหนดความยาวบันทึกและคาบยิง")
    ap.add_argument("--period-ms", type=float, default=50.0,
                    help="คาบยิง (50 ms = 20 ครั้ง/วินาที)")
    ap.add_argument("--cam-fps", type=float, default=15.0,
                    help="fps ที่ 'เก็บ' จากกล้อง (ฮาร์ดแวร์วิ่ง 30 เสมอ ตัวนี้คือหารลง)")
    ap.add_argument("--gap", type=float, default=30.0,
                    help="วินาทีพักระหว่าง section ตอนใช้ --auto")
    ap.add_argument("--no-view", action="store_true",
                    help="ไม่ต้องเปิดหน้าต่างแสดงผล (เร็วขึ้นเล็กน้อย)")
    ap.add_argument("--overwrite", action="store_true",
                    help="อัดทับช่วงที่มีข้อมูลอยู่แล้ว")
    ap.add_argument("--auto", action="store_true",
                    help="ไม่ต้องกด Enter ระหว่างช่วง (นับถอยหลัง 3 วิแทน)")
    ap.add_argument("--min-valid", type=float, default=0.0,
                    help="เก็บทุกเฟรมโดยไม่กรอง (ค่าเดิม 0) — ตั้งค่ามากกว่า 0 ถ้าอยากกรอง")
    a = ap.parse_args()
    code = 0
    try:
        _run(a)
    except Exception:
        # os._exit() ข้ามกลไกพิมพ์ traceback ปกติ ถ้าไม่พิมพ์เอง error จะหายเงียบ
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)     # ตัดจบก่อน GC แตะ native ที่ปิดไปแล้ว (เคยล้มตอนจบมาแล้ว)


if __name__ == "__main__":
    main()
