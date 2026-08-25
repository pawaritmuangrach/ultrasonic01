"""เก็บ dataset คู่ (คลื่นอัลตราซาวด์ 2 ช่อง + ความลึกจากกล้อง) — Stage A

หนึ่งตัวอย่าง = ยิงอัลตราซาวด์หนึ่งปิง + เก็บภาพความลึกของฉากเดียวกัน ณ ขณะนั้น
กล้องเป็น ground truth ว่า "มีอะไรอยู่ตรงไหนข้างหน้า" อัลตราซาวด์คือสิ่งที่ NN จะเรียน
ให้ทำนายออกมาให้ได้ โดยเฉพาะในจุดที่กล้องมองไม่เห็น (<0.6 m, กระจก, ผิวมัน)

โหมด:
    python car/collect.py --check
        เช็คว่าทั้งสองเซ็นเซอร์อ่านได้ ไม่เขียนไฟล์ ไม่ต้องมีกล้องก็ได้ (--no-cam)

    python car/collect.py --scene wall_01 --auto
        เก็บลง car/data/wall_01/ อัตโนมัติ · SPACE เก็บเอง · ESC/q ออก

เก็บโฟลเดอร์ละหนึ่งฉาก/หนึ่งมุม แล้วตอนแบ่ง train/val ให้แบ่งตามโฟลเดอร์ ไม่ใช่ตามเฟรม
(เฟรมติดกันเกือบซ้ำ ถ้าสุ่มแบ่งจะรั่วข้าม split ทำให้คะแนน val ไม่มีความหมาย —
บทเรียนจากโปรเจกต์กล้อง)

หน้าต่างพรีวิว: ครึ่งซ้าย = depth heatmap · ครึ่งขวา = โปรไฟล์เอคโค่สองช่อง
"""
import os
# cv2, numpy(MKL) และ OpenNI native ต่างมี OpenMP runtime ของตัวเอง พอโหลดซ้อนกัน
# บน Windows จะล้มระดับ native เงียบๆ (ไม่มี traceback, exit 127) — กับดักข้อ 4 ใน
# webcam/README ต้องตั้งก่อน import อะไรที่ดึง OpenMP เข้ามา
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ถ้าล้มระดับ native (OpenNI/OpenMP) ปกติจะเงียบสนิทไม่มี traceback
# faulthandler พิมพ์ C-level stack ให้ ทำให้ดีบักได้แทนที่จะเดา
import faulthandler
faulthandler.enable()

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
# โหลดโมดูลย่อยของ numpy ที่ถูก lazy-import ให้ครบตั้งแต่ตอนนี้ — ถ้าปล่อยให้โหลด
# กลางคันตอน OpenNI ทำงานอยู่ โปรเซสจะล้มระดับ native เงียบๆ (เจอมาแล้วกับ numpy.ma)
import numpy.ma          # noqa: F401
import numpy.lib         # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from ultrasonic import Ultrasonic                                   # noqa: E402
from scope_view import bandpass, envelope, counts_to_volts          # noqa: E402
from labels import near_object_count                                # noqa: E402

DATA_ROOT = Path(HERE) / "data"
C = 343.0


# โซนมุมสำหรับบอกตำแหน่งวัตถุตอนเก็บ — ให้คนเก็บรู้ว่ากวาดครบทุกทิศหรือยัง
# กล้องเห็น ±29° (FOV 58.4° ตาม labels.py) เกินกว่านี้คือหลุดเฟรม
ZONES = [(-90.0, -18.0, "ซ้ายสุด"), (-18.0, -7.0, "ซ้าย"), (-7.0, 7.0, "กลาง"),
         (7.0, 18.0, "ขวา"), (18.0, 90.0, "ขวาสุด")]


def scene_position(depth):
    """คืน (ระยะ cm, มุม deg, ชื่อโซน) ของสิ่งที่ใกล้ที่สุดที่กล้องเห็น · None ถ้าอ่านไม่ได้"""
    from labels import depth_to_profile, bin_angles
    dist, valid = depth_to_profile(depth)
    if not valid.any():
        return None
    kb = int(np.argmin(np.where(valid, dist, 1e9)))
    # กลับเครื่องหมายซ้าย-ขวาให้ตรงมุมมองคนวางของ (ยืนหน้ากล้อง = เห็นมิเรอร์ของกล้อง)
    # กระทบแค่ข้อความบนจอ · label ที่เทรน (labels.py) เป็นมุมกล้องจริง ไม่แตะ
    a = -float(bin_angles()[kb])
    name = next((n for lo, hi, n in ZONES if lo <= a < hi), "?")
    return float(dist[kb]) / 10.0, a, name


def coverage_line(counter):
    """สรุปว่าเก็บแต่ละโซนไปกี่รูป + ชี้โซนที่ยังน้อยสุด (ควรไปเก็บเพิ่มตรงนั้น)"""
    counts = [(n, counter.get(n, 0)) for _, _, n in ZONES]
    need = min(counts, key=lambda t: t[1])[0]
    return " ".join(f"{n}:{c}" for n, c in counts) + f"  ← ขาด {need}"


def echo_profile_img(counts, rate, pins, height=480, width=320, span_ms=8.0):
    """วาดโปรไฟล์เอคโค่สองช่องเป็นภาพ (ไม่ต้องมี matplotlib)"""
    import cv2
    img = np.zeros((height, width, 3), np.uint8)
    n = int(span_ms * 1e-3 * rate)
    colors = [(80, 200, 80), (80, 160, 255)]
    for i in range(counts.shape[0]):
        v = counts_to_volts(counts[i][:n].astype(float))
        e = envelope(bandpass(v - v.mean(), rate, 25e3, 60e3))
        if e.max() > 0:
            e = e / e.max()
        xs = (np.arange(len(e)) / len(e) * width).astype(int)
        ys = (height - 1 - e * (height * 0.45)).astype(int) - i * 4
        for x, y in zip(xs, ys):
            if 0 <= y < height:
                img[y, min(x, width - 1)] = colors[i % 2]
        cv2.putText(img, f"GPIO{pins[i]}", (8, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 2], 1)
    for cm in (20, 40, 60, 80):
        t_us = 2 * (cm / 100) / C * 1e6
        x = int(t_us * 1e-6 * rate / n * width)
        if 0 <= x < width:
            img[:, x] = (60, 60, 60)
            cv2.putText(img, f"{cm}", (x + 2, height - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    return img


def save_sample(out_dir, idx, ping, depth, rgb):
    import cv2
    np.savez_compressed(
        out_dir / f"us_{idx:06d}.npz",
        counts=ping["counts"], rate=ping["rate"],
        skew_us=ping["skew_us"], pins=np.array(ping["pins"]), t=ping["t"])
    cv2.imwrite(str(out_dir / f"depth_{idx:06d}.png"), depth)   # 16-bit mm
    if rgb is not None:
        cv2.imwrite(str(out_dir / f"rgb_{idx:06d}.png"), rgb)


def _write_meta(out_dir, a, total, rate):
    meta = out_dir / "meta.json"
    info = json.loads(meta.read_text()) if meta.exists() else {}
    info.update({
        "scene": a.scene, "samples": total, "pins": a.pins, "us_rate_hz": rate,
        "depth_units": "uint16 PNG มิลลิเมตร 0=อ่านไม่ได้",
        "us_format": "npz: counts[ch][sample] 0..4095, rate, skew_us, pins, t"})
    meta.write_text(json.dumps(info, indent=2, ensure_ascii=False))


def _warmup():
    """ซ้อมรันเส้นทางจริงหนึ่งรอบด้วยข้อมูลปลอม **ก่อนเปิดกล้อง**

    ทำไมต้องมี: การ import โมดูลใหม่ขณะ OpenNI (native) ทำงานอยู่ ทำให้โปรเซสล้ม
    (เจอจริงสองรอบ — `numpy.ma` ที่ `np.percentile` ลากมา และ `zipfile` ที่
    `np.savez_compressed` ลากมา ทั้งคู่ตายตอน `_compile_bytecode` heap พัง 0xc0000374)
    ไล่ปิดทีละโมดูลไม่มีวันจบ เพราะไม่รู้ว่ายังมีตัวไหนซ่อนอยู่อีก — วิธีที่ชัวร์คือ
    **รันโค้ดชุดเดียวกับตอนเก็บจริงหนึ่งรอบ** แล้ว lazy import ทุกตัวจะเกิดตรงนี้แทน
    """
    import shutil
    import tempfile
    import cv2
    tmp = Path(tempfile.mkdtemp(prefix="us_warmup_"))
    try:
        depth = np.full((480, 640), 900, np.uint16)
        depth[180:300, 280:380] = 700          # ก้อน "วัตถุ" ปลอมให้มีอะไรให้คำนวณ
        ping = {"counts": np.full((4, 4000), 2048, np.uint16), "rate": 266000.0,
                "skew_us": 1.25, "pins": [34, 35, 32, 33], "t": 0.0}
        save_sample(tmp, 0, ping, depth, None)   # -> savez_compressed (zipfile) + imwrite
        scene_position(depth)                    # -> depth_to_profile (percentile path)
        near_object_count(depth)
        cv2.imencode(".png", depth)
    except Exception as e:
        print(f"  (warmup เตือน: {e})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()
    gc.freeze()          # ย้ายของที่โหลดแล้วออกจากสายตา GC ลดงาน GC ตอนเก็บจริง


def _run(us_box):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--check", action="store_true", help="เช็คเซ็นเซอร์ ไม่เขียนไฟล์")
    ap.add_argument("--no-cam", action="store_true", help="ไม่ใช้กล้อง (ทดสอบอัลตราซาวด์อย่างเดียว)")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35")
    ap.add_argument("--min-valid", type=float, default=0.30,
                    help="สัดส่วนพิกเซล depth ที่ใช้ได้ขั้นต่ำจึงจะเก็บ")
    ap.add_argument("--auto-every", type=int, default=20)
    ap.add_argument("--auto", action="store_true",
                    help="เก็บอัตโนมัติทุก --auto-every เฟรม ตั้งแต่เปิด (ไม่ต้องกดปุ่ม)")
    ap.add_argument("--shots", type=int, default=0,
                    help="เก็บ N ตัวอย่างแบบไม่มีหน้าต่าง แล้วออก (กันปัญหา GUI ทั้งหมด)")
    ap.add_argument("--manual", action="store_true",
                    help="เก็บทีละรูปด้วยการกด ENTER ในเทอร์มินัล (ไม่มีหน้าต่าง) — "
                         "สำหรับกรณีต้องเดินไปขยับวัตถุเอง แล้วเดินออกก่อนค่อยกดเก็บ")
    ap.add_argument("--auto-still", action="store_true",
                    help="เก็บอัตโนมัติเมื่อฉากนิ่งหลังมีการขยับ (ไม่ต้องกดปุ่มเลย) — "
                         "ขยับวัตถุ เดินออก รอแป๊บ มันเก็บเอง วนไปเรื่อยๆ")
    ap.add_argument("--expect-cm", type=float, default=None,
                    help="ระยะที่ตั้งใจวางวัตถุ (cm) — เก็บเฉพาะเฟรมที่กล้องเห็นของใกล้สุด "
                         "อยู่ในช่วงนี้ ±--tol-cm (กันเฟรมที่ติดคน/ล็อกฉากหลัง)")
    ap.add_argument("--tol-cm", type=float, default=30.0,
                    help="ความคลาดที่ยอมได้ของ --expect-cm (ค่าเริ่มต้น 30)")
    ap.add_argument("--period", type=float, default=2.0,
                    help="วินาทีต่อหนึ่งภาพ ในโหมด --shots (ให้เวลากวาดวัตถุ) ค่าเริ่มต้น 2.0")
    ap.add_argument("--track", action="store_true",
                    help="เช็คสดว่าวัตถุเป็นเสียงเด่นไหม (us range ต้องตรงกับกล้อง) ไม่เขียนไฟล์")
    ap.add_argument("--rgb", action="store_true",
                    help="เปิดกล้องสีด้วย (ปิดโดยปริยาย เพราะ DSHOW มักค้าง และเราใช้ depth เป็น label)")
    a = ap.parse_args()
    if not a.check and not a.track and not a.scene:
        ap.error("ใส่ --check หรือ --track เพื่อทดสอบ หรือ --scene NAME เพื่อเก็บ")

    print("เปิดอัลตราซาวด์ ...")
    us = Ultrasonic(port=a.port, pins=a.pins)
    us_box['us'] = us
    cam = None
    depth_to_heatmap = None
    if not a.no_cam and not a.track:      # track ไม่เปิดกล้อง = เลี่ยง live cam.read ที่แครช native
        _warmup()                         # ให้ lazy import เกิดก่อนกล้องเข้ามา (ดู _warmup)
        print("เปิดกล้อง Astra ...")
        from astra import Astra, depth_to_heatmap
        try:
            cam = Astra(verbose=True, want_rgb=a.rgb)
            us_box['cam'] = cam
        except Exception as e:
            print(f"\nเปิดกล้องไม่ได้: {e}")
            print("เก็บ dataset ต้องมีกล้อง — แก้ตามด้านบนแล้วลองใหม่ "
                  "(หรือ --no-cam เพื่อทดสอบอัลตราซาวด์อย่างเดียว)")
            us.close()
            sys.exit(1)

    # --- โหมดเช็ค ---
    if a.check:
        ping = us.ping()
        if ping is None:
            print("อัลตราซาวด์: อ่านไม่ได้"); us.close()
            if cam: cam.close()
            sys.exit(1)
        c = ping["counts"]
        print(f"อัลตราซาวด์: {c.shape[0]} ช่อง x {c.shape[1]} ตัวอย่าง "
              f"@ {ping['rate']/1e3:.0f} kS/s  ดิบ {int(c.min())}..{int(c.max())}")
        if cam:
            rgb, depth = cam.read()
            print(f"กล้อง: depth {depth.shape} ใช้ได้ {cam.valid_fraction(depth):.0%}  "
                  f"ช่วง {int(depth[depth>0].min()) if (depth>0).any() else 0}.."
                  f"{int(depth.max())} mm  rgb {'มี' if rgb is not None else 'ไม่มี'}")
        print("เช็คผ่าน")
        us.close()
        if cam: cam.close()
        return

    # --- โหมดเช็คสด: เป้าเป็นเสียงเด่นไหม (US อย่างเดียว ไม่เปิดกล้อง = ไม่แครช) ---
    if a.track:
        # เปิดใหม่เป็นคู่เดียว: 4 ช่องต้อง reconfig ADC ทุกปิง = พ่นคอนฟิกท่วมจอ
        # track แค่ต้องดูว่า "ระยะ" ตามเป้า + amp สูงพอ ใช้สองช่องแรกพอ
        us.close()
        pair = a.pins.split(",")[:2]
        us = Ultrasonic(port=a.port, pins=",".join(pair))
        print("ขยับเป้า ดูว่า 'ระยะ' ของทั้งสองช่องขยับตามเป้าไหม + amp ยิ่งสูงยิ่งเด่น")
        print("เทียบ amp ตอนไม่มีเป้า (ยิงใส่ที่ว่าง) กับตอนมีเป้า — ต่างกันชัด = เป้าเด่นจริง")
        print("Ctrl-C หยุด (หยุดเองใน ~60 วิ)\n")
        try:
            for _ in range(200):                      # ~60 วิ กันดูเหมือนค้าง
                try:                                  # กันพลาดรายรอบ ลูปไม่ตายทั้งตัว
                    ping = us.ping()
                    if ping is None:
                        print("\r  (อ่านไม่ได้รอบนี้)              ", end="", flush=True)
                        time.sleep(0.3); continue
                    fs, counts, pn = ping["rate"], ping["counts"], ping["pins"]
                    parts = []
                    for ch in range(counts.shape[0]):
                        v = counts_to_volts(counts[ch].astype(float))
                        e = envelope(bandpass(v - v.mean(), fs, 25e3, 60e3))
                        sk = int(9e-4 * fs)
                        k = sk + int(np.argmax(e[sk:]))
                        rng = (k / fs * 1e6 - 435) * 1e-6 * C / 2 * 100
                        parts.append(f"g{pn[ch]} {rng:4.0f}cm amp{e[k]*1e3:4.0f}")
                    print("\r  " + "  |  ".join(parts) + "        ", end="", flush=True)
                    time.sleep(0.3)
                except Exception as ex:
                    print(f"\n  (ข้ามรอบ: {ex})", flush=True); time.sleep(0.3)
        except KeyboardInterrupt:
            print("\nหยุด")
        us.close()
        return

    out_dir = DATA_ROOT / a.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("us_*.npz")))

    # --- โหมดเก็บอัตโนมัติเมื่อฉากนิ่ง (ไม่ต้องกดปุ่ม) ---
    # ไหลงาน: ขยับวัตถุ (กล้องเห็นการเปลี่ยนแปลง) -> เดินออก -> ฉากนิ่ง -> เก็บอัตโนมัติ
    # ต้อง "เห็นการขยับก่อน" ถึงจะเก็บรอบถัดไป ไม่งั้นจะเก็บตำแหน่งเดิมซ้ำๆ เป็นร้อยรูป
    if a.auto_still:
        if cam is None:
            print("โหมด --auto-still ต้องมีกล้อง"); us.close(); sys.exit(1)
        from collections import Counter
        zones = Counter()
        CR = chr(13)

        target = a.shots if a.shots > 0 else 10 ** 9
        gate = (f" · รับเฉพาะที่กล้องเห็น {a.expect_cm:.0f}±{a.tol_cm:.0f} cm"
                if a.expect_cm else "")
        print(f"เก็บลง {out_dir} (มีอยู่แล้ว {existing}){gate}")
        print("ขยับวัตถุนิดเดียวก็ได้ → เดินออกจากเฟรม → พอวัตถุนิ่ง+อยู่ตำแหน่งใหม่ มันเก็บเอง")
        print("หยุดด้วย Ctrl-C" + (f" · ครบ {a.shots} รูปแล้วหยุด" if a.shots else ""))
        print("")
        saved, last_rate = 0, 0.0
        last_pos = None       # (ระยะ, มุม) ของรูปที่เก็บล่าสุด — กันเก็บซ้ำที่เดิม
        hist = []             # ตำแหน่งวัตถุไม่กี่เฟรมล่าสุด ดูว่านิ่งหรือยัง
        time.sleep(1.5)                # รอสตรีมกล้องนิ่งก่อนเริ่ม (กัน native crash)
        for _ in range(3):
            cam.read()
        try:
            while saved < target:
                time.sleep(0.15)          # หน่วง ไม่ให้อ่านกล้องถี่จน OpenNI ล้ม native
                rgb, depth = cam.read()
                if depth is None:
                    continue
                pos = scene_position(depth)               # (ระยะ, มุมแสดง, โซน) หรือ None
                valid = cam.valid_fraction(depth)
                nobj = near_object_count(depth) if pos else 0
                in_range = (not a.expect_cm or abs(pos[0] - a.expect_cm) <= a.tol_cm) if pos else False
                good = pos is not None and valid >= a.min_valid and in_range and nobj <= 1
                if not good:                              # คนอยู่ในเฟรม / นอกระยะ / ติดขา / depth น้อย
                    hist = []
                    if pos is None or valid < a.min_valid:
                        why = f"depth {valid:.0%} ต่ำ"
                    elif not in_range:
                        why = f"กล้องเห็น {pos[0]:.0f}cm นอกช่วง"
                    else:
                        why = "มีของใกล้ 2 จุด (ติดขา? เดินออกให้พ้น)"
                    print(f"{CR}  รอฉากพร้อม… ({why})                    ", end="", flush=True)
                    continue
                hist.append((pos[0], pos[1]))
                if len(hist) > 5:
                    hist.pop(0)
                arr = np.array(hist)
                if len(hist) < 5 or arr[:, 0].ptp() > 3 or arr[:, 1].ptp() > 4:
                    print(f"{CR}  …วัตถุยังขยับ รอให้นิ่ง                    ", end="", flush=True)
                    continue                              # ยังขยับ/เพิ่งขยับ รอให้นิ่งก่อน
                moved = (last_pos is None or abs(pos[0] - last_pos[0]) > 4
                         or abs(pos[1] - last_pos[1]) > 3)
                if not moved:                             # นิ่งแต่ตำแหน่งเดิม รอให้ขยับก่อน
                    print(f"{CR}  รอขยับวัตถุ (เดิม {pos[1]:+.0f}° {pos[0]:.0f}cm)          ",
                          end="", flush=True)
                    continue
                ping = us.ping()
                if ping is None:
                    continue
                last_rate = ping["rate"]
                try:
                    save_sample(out_dir, existing + saved, ping, depth, rgb)
                    saved += 1
                    zones[pos[2]] += 1
                    n_txt = f"{saved}/{a.shots}" if a.shots else str(saved)
                    print(f"{CR}  ✓ [{n_txt}] {pos[1]:+5.0f}° {pos[2]:<7s} {pos[0]:3.0f}cm"
                          f"   {coverage_line(zones)}      ")
                    last_pos, hist = (pos[0], pos[1]), []
                except Exception as e:
                    print(""); print(f"เขียนไฟล์พลาด: {e}")
        except KeyboardInterrupt:
            print(""); print("หยุด")
        if saved:
            _write_meta(out_dir, a, existing + saved, last_rate)
            print(""); print(f"เขียน {saved} ตัวอย่างใหม่ ({existing + saved} รวม) ที่ {out_dir}")
        us.close(); cam.close()
        return

    # --- โหมดกด ENTER ทีละรูป (ไม่มีหน้าต่าง) ---
    # ใช้ตอนต้องเดินไปขยับวัตถุเอง: ขยับเสร็จ เดินออกจากแนวเซนเซอร์ก่อน แล้วค่อยกด ENTER
    # ไม่งั้นเก็บอัตโนมัติจะจับ "ตัวคนเก็บ" ที่ใกล้กว่าวัตถุ = label เพี้ยน
    if a.manual:
        if cam is None:
            print("โหมด --manual ต้องมีกล้อง"); us.close(); sys.exit(1)
        from collections import Counter
        zones = Counter()      # นับว่าเก็บโซนไหนไปกี่รูป
        print(f"เก็บลง {out_dir} (มีอยู่แล้ว {existing})")
        print("ขยับท่อ → เดินออกจากแนวเซนเซอร์ → กด ENTER เก็บ 1 รูป · พิมพ์ q แล้ว ENTER = ออก")
        saved = 0
        last_rate = 0.0
        while True:
            try:
                cmd = input(f"[{existing + saved}] ENTER=เก็บ  q=ออก > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # ต้องจับ Ctrl-C ตรงนี้ ไม่งั้นเด้งออกโดยไม่ได้ปิดกล้อง -> อุปกรณ์ค้าง
                # รอบหน้าเปิดไม่ได้ ล้มเป็น access violation ตอน create_depth_stream
                print("")
                break
            if cmd == "q":
                break
            for _ in range(3):        # ทิ้งเฟรมกล้องที่ค้าง เอาเฟรมล่าสุด (ฉากปัจจุบัน)
                rgb, depth = cam.read()
            # ยิงอัลตราซาวด์ก่อนประมวล depth — เป็นลำดับเดิมที่เก็บ d70/d100/d130 ได้โดยไม่ล้ม
            # (การประมวล depth ทันทีหลัง cam.read() ทำให้ล้มระดับ native เงียบๆ)
            ping = us.ping()
            if ping is None:
                print("  อัลตราซาวด์อ่านไม่ได้ กดใหม่"); continue
            last_rate = ping["rate"]
            if depth is None:
                print("  ✗ กล้องไม่คืนภาพ กดใหม่"); continue
            pos = scene_position(depth)                 # (ระยะ, มุมแสดง, โซน) หรือ None
            valid = cam.valid_fraction(depth)
            # เช็คก่อนเก็บ แล้วบอกสถานะ: โซน (ถ้าดี) หรือเหตุที่ไม่เก็บ
            if pos is None or valid < a.min_valid:
                print(f"  ✗ depth ต่ำ ({valid:.0%}) — ปรับแล้วกดใหม่"); continue
            if near_object_count(depth) > 1:
                print(f"  ✗ ติดขา/ของใกล้ 2 จุด ({pos[2]}) — เดินออกให้พ้นแล้วกดใหม่"); continue
            if a.expect_cm and abs(pos[0] - a.expect_cm) > a.tol_cm:
                print(f"  ✗ นอกระยะ: {pos[2]} {pos[0]:.0f}cm — ต้องการ "
                      f"{a.expect_cm:.0f}±{a.tol_cm:.0f} cm · ไม่เก็บ"); continue
            try:
                save_sample(out_dir, existing + saved, ping, depth, rgb)
                saved += 1
                zones[pos[2]] += 1
                print(f"  ✓ [{saved}] {pos[2]:<7s} {pos[1]:+.0f}° {pos[0]:.0f}cm"
                      f"   {coverage_line(zones)}")
            except Exception as e:
                print(f"  เขียนไฟล์พลาด: {e}")
        if saved:
            _write_meta(out_dir, a, existing + saved, last_rate)
            print(f"เขียน {saved} ตัวอย่างใหม่ ({existing + saved} รวม) ที่ {out_dir}")
        us.close()
        if cam:
            cam.close()
        return

    # --- โหมดไม่มีหน้าต่าง: เก็บ N ตัวอย่างแล้วออก (เลี่ยงปัญหา GUI ทุกอย่าง) ---
    if a.shots > 0:
        if cam is None:
            print("โหมด --shots ต้องมีกล้อง")
            us.close(); sys.exit(1)
        total_s = a.shots * a.period
        print(f"เก็บ {a.shots} ตัวอย่างลง {out_dir} · ภาพละ {a.period:.1f} วิ รวม ~{total_s:.0f} วิ")
        print(f"กวาดวัตถุช้าๆ ซ้าย→กลาง→ขวา ให้พอดี {total_s:.0f} วินาที (ช้าลงถ้าเลื่อนไม่ทัน)")
        print("(เริ่มใน 3 วินาที ...)")
        time.sleep(3.0)
        saved = 0
        last_rate = 0.0
        while saved < a.shots:
            t0 = time.time()          # จับเวลาต่อภาพ เพื่อให้ครบ period พอดี
            ping = us.ping()
            rgb, depth = cam.read()
            if ping is None:
                continue
            last_rate = ping["rate"]
            valid = cam.valid_fraction(depth)
            if valid < a.min_valid:
                print(f"\r  ข้าม: depth ใช้ได้แค่ {valid:.0%} (< {a.min_valid:.0%}) "
                      f"— ลด --min-valid ถ้าอยู่ที่โล่ง        ", end="", flush=True)
            else:
                try:
                    save_sample(out_dir, existing + saved, ping, depth, rgb)
                    saved += 1
                    frac = saved / a.shots
                    sweep = "ซ้าย" if frac < 0.33 else ("กลาง" if frac < 0.66 else "ขวา")
                    print(f"\r  เก็บ {saved}/{a.shots}  depth {valid:.0%}  retries {us.bad_frames}"
                          f"   กวาดไป: {sweep}     ", end="", flush=True)
                except Exception as e:
                    print(f"\nเขียนไฟล์พลาด: {e}")
            # ทำให้แต่ละภาพใช้เวลาครบ period (หักเวลาที่ ping+save ใช้ไปแล้ว)
            rest = a.period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
        print()
        _write_meta(out_dir, a, existing + saved, last_rate)
        print(f"เขียน {saved} ตัวอย่างใหม่ ({existing + saved} รวม) ที่ {out_dir}")
        us.close(); cam.close()
        return

    # --- โหมดมีหน้าต่าง ---
    import cv2
    saved = 0
    auto = a.auto
    frame_no = 0
    print(f"เก็บลง {out_dir}  (มีอยู่แล้ว {existing})")
    print(f"SPACE เก็บทีละคู่ · ESC/q ออก · เก็บอัตโนมัติ = {'เปิด' if auto else 'ปิด (ใส่ --auto ถ้าต้องการ)'}")

    # สร้างหน้าต่างก่อนเข้าลูป แล้วดันขึ้นหน้าและวางตำแหน่งให้แน่นอน
    # ไม่งั้นบน Windows มันเปิดอยู่หลังหน้าต่างเทอร์มินัลจนดูเหมือนไม่ขึ้น
    WIN = "collect: depth (left) + echo (right)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)
    cv2.moveWindow(WIN, 60, 60)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass                      # บางบิลด์ของ OpenCV ไม่มีแฟล็กนี้
    print("(หน้าต่างจะเปิดมุมบนซ้าย — ต้องคลิกที่หน้าต่างก่อน ปุ่มถึงจะทำงาน)\n")

    last_right = np.zeros((480, 320, 3), np.uint8)
    last_rate = 0.0          # ปิงสุดท้ายอาจพลาด เก็บอัตราไว้เขียน meta ตอนจบ
    while True:
        # ปิงพลาดต้องไม่ข้าม GUI ไป ไม่งั้น waitKey ไม่ถูกเรียก = ปุ่มตายและดูเหมือนค้าง
        ping = us.ping()
        rgb, depth = (None, None)
        valid = 1.0
        if cam:
            rgb, depth = cam.read()
            valid = cam.valid_fraction(depth)

        frame_no += 1
        should = auto and ping is not None and frame_no % a.auto_every == 0

        left = (np.zeros((480, 640, 3), np.uint8) if depth is None
                else depth_to_heatmap(depth))
        if ping is not None:
            last_rate = ping["rate"]
            last_right = echo_profile_img(ping["counts"], ping["rate"], ping["pins"])
        view = np.hstack([cv2.resize(left, (480, 480)), last_right])
        usable = (depth is None) or (valid >= a.min_valid)
        # ข้อความบนภาพต้องเป็น ASCII — ฟอนต์ Hershey ของ OpenCV วาดไทยไม่ได้
        us_txt = "US FAILED" if ping is None else f"us ok (retries {us.bad_frames})"
        for i, txt in enumerate([
                f"scene {a.scene}  saved {existing+saved}  auto {'ON' if auto else 'off'}",
                f"depth valid {valid:.0%}  {'OK' if usable else 'LOW'}   {us_txt}"]):
            cv2.putText(view, txt, (10, 24 + i*24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3)
            cv2.putText(view, txt, (10, 24 + i*24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        cv2.imshow(WIN, view)

        # SPACE/ESC มี keycode คงที่ทุกภาษาคีย์บอร์ด (ตัวอักษร a/q เพี้ยนเมื่อเป็นไทย)
        # ไม่ใช้ TAB แล้ว เพราะมันสลับ focus หน้าต่างในระบบ Windows
        # ไม่เช็ค getWindowProperty แล้ว เพราะมันคืน 0 ตอนหน้าต่างแค่ไม่ใช่ตัวหน้าสุด
        # (ไม่ใช่ถูกปิด) ทำให้ดับเองโดยไม่ตั้งใจ — ออกด้วย ESC/q เท่านั้น
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):          # ESC หรือ q = ออก
            break
        if k == ord(" "):                 # SPACE = เก็บหนึ่งคู่
            should = True
        if should:
            if ping is None:
                print("\rข้าม: อัลตราซาวด์อ่านไม่ได้รอบนี้            ", end="", flush=True)
            elif not usable:
                print(f"\rข้าม: depth ใช้ได้แค่ {valid:.0%} (< {a.min_valid:.0%})       ",
                      end="", flush=True)
            else:
                # การเขียนไฟล์พลาดต้องไม่ทำให้เซสชันที่เก็บมาแล้วหายไปทั้งหมด
                try:
                    save_sample(out_dir, existing + saved, ping, depth, rgb)
                    saved += 1
                    print(f"\rเก็บ {existing+saved-1:06d}  depth {valid:.0%}       ", end="")
                except Exception as e:
                    print(f"\nเขียนไฟล์พลาด: {e}")

    if saved:
        _write_meta(out_dir, a, existing + saved, last_rate)
        print(f"\nเขียน {saved} ตัวอย่างใหม่ ({existing+saved} รวม) ที่ {out_dir}")

    cv2.destroyAllWindows()
    us.close()
    if cam:
        cam.close()


def main():
    """ห่อ _run เพื่อ **ปิดอุปกรณ์เสมอ** ไม่ว่าจะจบยังไง (Ctrl-C/ข้อผิดพลาด)

    ถ้ากล้องไม่ถูกปิด อุปกรณ์จะค้างที่ระดับไดรเวอร์ รอบหน้าเปิดใหม่จะล้มเป็น
    access violation ตอน create_depth_stream (ต้องถอด-เสียบสาย USB ถึงจะหาย)"""
    box = {}
    try:
        _run(box)
    except KeyboardInterrupt:
        print("")
        print("หยุด")
    finally:
        for key in ("us", "cam"):
            obj = box.get(key)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        # ปิดโปรเซสทันทีหลังปิดอุปกรณ์ — ไม่ให้ล่ามรัน GC/teardown ต่อ
        # เพราะ OpenNI คืนหน่วยความจำตอน teardown แล้ว heap พัง (0xc0000374)
        # ขึ้นข้อความน่าตกใจ *หลัง* เขียนไฟล์เสร็จแล้ว ทั้งที่ข้อมูลครบดี
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
