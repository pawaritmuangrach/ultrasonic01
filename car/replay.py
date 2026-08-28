#!/usr/bin/env python3
"""เล่นย้อนข้อมูลที่อัดไว้ ดูโมเดลทำงานทีละเฟรม ตั้งแต่เฟรมแรกจนเฟรมสุดท้าย

    python car/replay.py                     เล่นช่วงที่ 1 (ช่วงที่กันไว้ทดสอบ)
    python car/replay.py --section 3         เล่นช่วงอื่น
    python car/replay.py --save out.mp4      อัดเป็นวิดีโอแทนการเปิดหน้าต่าง

**ไม่ต้องต่อฮาร์ดแวร์** อ่านจากไฟล์ที่อัดไว้ล้วน ๆ จึงดูซ้ำได้เท่าที่อยาก

ใช้ `predict.Predictor` กับ `predict.render` **ตัวเดียวกับตอนรันสด** ไม่ได้เขียนใหม่
สิ่งที่เห็นบนจอนี้จึงเป็นสิ่งเดียวกับที่รถจะเห็นตอนวิ่งจริง ถ้าโค้ดสองทางแยกกัน
ผลที่เห็นตอนดูย้อนหลังจะไม่ผูกกับพฤติกรรมจริงอีกต่อไป ซึ่งเป็นกับดักคลาสสิก

**ตัวเลขที่โชว์เป็นตัวเลขที่ซื่อสัตย์**: ตอนเล่นช่วงที่กันไว้ทดสอบ จะใช้สัมประสิทธิ์
`holdout_slope/intercept` ซึ่งฟิตจากช่วงอื่นล้วน — โมเดลไม่เคยเห็นเฟรมพวกนี้มาก่อนเลย
ถ้าใช้สัมประสิทธิ์ตัวหลัก (ฟิตจากทุกช่วง) ตัวเลขจะสวยกว่าความจริงเพราะเคยเห็นมาแล้ว
เล่นช่วงที่ใช้เทรน โปรแกรมจะขึ้นป้ายเตือนว่า TRAINED ON THIS

ปุ่ม:
    space   หยุด / เล่นต่อ
    . ,     เดินหน้า / ถอยหลัง 1 เฟรม
    > <     กระโดด 30 เฟรม (2 วินาที)
    + -     เร็วขึ้น / ช้าลง
    r       กลับไปเฟรมแรก
    s       เซฟภาพเฟรมปัจจุบัน
    q/ESC   ออก
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import predict as P                                       # noqa: E402
from labels import target_angle                           # noqa: E402

DATA = Path(HERE) / "data"
MIN_COV = 0.10          # ต้องตรงกับ --min-cov ที่ใช้ตอนรัน rules.py


def load_section(name, sec_no):
    sc = DATA / f"{name}_s{sec_no}"
    if not sc.is_dir():
        have = sorted(p.name for p in DATA.glob(f"{name}_s*") if p.is_dir())
        sys.exit(f"ไม่มี {sc}\nที่มีอยู่: {', '.join(have) if have else '(ไม่มีเลย)'}")
    fs = sorted(glob.glob(str(sc / "us_*.npz")),
                key=lambda q: int(Path(q).stem.split("_")[1]))
    frames = [(f, sc / f"depth_{Path(f).stem.split('_')[1]}.png") for f in fs]
    frames = [(u, d) for u, d in frames if d.exists()]
    if not frames:
        sys.exit(f"{sc.name} ไม่มีเฟรมที่มีทั้ง npz และ png")
    return sc.name, frames


def prep(rule, name, sec_no):
    """เลือกสัมประสิทธิ์ให้ตรงกับว่าช่วงนี้เคยถูกใช้เทรนหรือไม่"""
    held = rule.get("holdout") == f"{name}_s{sec_no}"
    r = dict(rule)
    if held and "holdout_slope" in rule:
        r["slope"] = rule["holdout_slope"]
        r["intercept"] = rule["holdout_intercept"]
    return r, held


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="walk")
    ap.add_argument("--section", type=int, default=1, help="ช่วงที่จะเล่น (นับจาก 1)")
    ap.add_argument("--fps", type=float, default=15.0, help="ความเร็วเล่น (fps ที่อัดมา)")
    ap.add_argument("--save", default=None, help="อัดเป็นวิดีโอแทนการเปิดหน้าต่าง")
    ap.add_argument("--model", choices=["rule", "nn"], default="rule",
                    help="rule = กฎ 2 พารามิเตอร์ · nn = โมเดล ML จากคลื่นดิบ")
    a = ap.parse_args()

    import cv2
    rule_all = P.load_rule(a.name)
    rule, held = prep(rule_all, a.name, a.section)
    nnp = None
    if a.model == "nn":
        import train_nn as TN
        nnp = TN.NNPredictor()
        # โมเดล ML เทรนกันช่วงไหนไว้ ก็ต้องเช็คกับช่วงนั้น ไม่ใช่ของ rules.py
        held = (nnp.holdout == a.section - 1)
        rule = dict(rule, slope=float("nan"), intercept=float("nan"),
                    mae_deg=3.82, zone_acc=0.93, n_frames=0,
                    holdout=f"{a.name}_s{nnp.holdout+1}")
    sec_name, frames = load_section(a.name, a.section)

    print(f"เล่น {sec_name}: {len(frames)} เฟรม")
    if nnp is not None:
        print(f"  โมเดล ML จากคลื่นดิบ · {nnp.params:,} พารามิเตอร์ · "
              f"เกลี่ย {nnp.buf.maxlen} เฟรม")
    else:
        print(f"  กฎ: มุม = {rule['slope']:+.2f} * log4 {rule['intercept']:+.2f}")
    if held:
        print(f"  ** ช่วงนี้ถูกกันไว้ทดสอบ — ใช้สัมประสิทธิ์ที่ฟิตจากช่วงอื่นล้วน")
        print(f"     ตัวเลขที่เห็นจึงเป็นผลกับข้อมูลที่โมเดลไม่เคยเห็น (ซื่อสัตย์)")
    else:
        print(f"  ** ช่วงนี้อยู่ในชุดเทรน — ตัวเลขจะดีเกินจริง ดูไว้เทียบเฉย ๆ")

    pr = nnp if nnp is not None else P.Predictor(rule)
    n = hit = 0
    serr = 0.0
    i, paused, speed = 0, False, 1.0
    img = None
    vw = None
    if a.save:
        vw = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (P.W, P.H))
        if not vw.isOpened():
            sys.exit(f"เปิดไฟล์วิดีโอไม่ได้: {a.save}")
        print(f"  อัดลง {a.save} ...")

    cache = []      # ผลที่คำนวณแล้วของเฟรม 0..len(cache)-1

    def compute(k):
        """คำนวณเฟรมที่ k — **ต้องเรียกตามลำดับ** เพราะมัธยฐานย้อนหลังมีสถานะสะสม
        ผลถูกเก็บลง cache เพื่อให้ถอยหลังหรือกระโดดกลับไม่ต้องคำนวณซ้ำ
        (ถ้าไม่เก็บ การถอย 1 เฟรมตอนอยู่ท้ายช่วงต้องไล่ใหม่ตั้งแต่ต้น = รอ 25 วินาที)"""
        nonlocal n, hit, serr
        u, d = frames[k]
        z = np.load(u)
        ping = {"counts": z["counts"], "rate": float(z["rate"]),
                "pins": [int(v) for v in z["pins"]]}
        depth = cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
        pdeg, fresh = pr.push(ping)
        lab = target_angle(depth)
        if lab is not None and lab[2] < MIN_COV:
            lab = None
        pz = pr.zone(pdeg) if pdeg is not None else None
        tz = pr.zone(lab[0]) if lab is not None else None
        if pdeg is not None and lab is not None and fresh:
            n += 1
            hit += int(pz == tz)
            serr += abs(pdeg - lab[0])
        # **ไม่เก็บภาพ depth และไม่เก็บ hist ทั้งก้อน** — 865 เฟรมจะกิน 130 MB
        # ภาพอ่านใหม่จากดิสก์เร็วกว่า (~2 ms) ส่วน hist สร้างจาก cache ย้อนหลังได้
        cache.append(dict(lab=lab, pdeg=pdeg, pz=pz, tz=tz, amps=list(pr.amps),
                          pins=ping["pins"], log4=pr.log4, fresh=fresh,
                          stale=pr.stale, rng=pr.rng, stats=(n, hit, serr)))

    def draw(k):
        """วาดเฟรมที่ k จาก cache — คำนวณเพิ่มถ้ายังไม่ถึง"""
        while len(cache) <= k:
            compute(len(cache))
        c = cache[k]
        depth = cv2.imread(str(frames[k][1]), cv2.IMREAD_UNCHANGED)
        # ประวัติ = เฟรมใหม่อยู่ซ้าย จึงไล่ย้อนจาก k ลงมา
        lo = max(0, k - (P.W - 2 * P.PAD) + 1)
        h = [(cache[j]["lab"][0] if cache[j]["lab"] else None, cache[j]["pdeg"])
             for j in range(k, lo - 1, -1)]
        foot = (f"space pause | , . step | < > jump 30 | + - speed x{speed:.2f} "
                f"| r restart | s save | q quit")
        banner = (("TEST SECTION - model never saw these frames", P.OKC) if held
                  else ("TRAINED ON THIS - numbers look better than reality",
                        P.WARNC))
        return P.render(depth, c["lab"], c["pdeg"], c["pz"], c["tz"],
                        c["amps"], c["pins"], rule, c["log4"], h,
                        c["stats"], a.fps * speed, c["fresh"], c["stale"],
                        c["rng"],
                        progress=((k + 1) / len(frames), k + 1, len(frames)),
                        foot=foot, banner=banner,
                        title=("REPLAY  -  ML model on raw waveforms"
                               if nnp is not None else
                               "REPLAY  -  rule base from ultrasonic only"))

    def reset():
        """กลับไปเฟรมแรก — cache ยังใช้ได้ ไม่ต้องคำนวณใหม่"""
        nonlocal i
        i = 0

    # ---------------------------------------------------------------- อัดวิดีโอ
    if vw is not None:
        t0 = time.time()
        for k in range(len(frames)):
            vw.write(draw(k))
            if (k + 1) % 100 == 0:
                print(f"    {k + 1}/{len(frames)}", flush=True)
        vw.release()
        el = time.time() - t0
        print(f"เสร็จ {len(frames)} เฟรม ใน {el:.0f} วินาที -> {a.save}")
        if n:
            print(f"คะแนนทั้งช่วง: โซนถูก {hit / n:.0%} · มุมผิดเฉลี่ย {serr / n:.1f} องศา "
                  f"({n} เฟรมที่เทียบได้)")
        return 0

    # ---------------------------------------------------------------- เล่นบนจอ
    cv2.namedWindow("replay", cv2.WINDOW_AUTOSIZE)
    print("เปิดหน้าต่างแล้ว — space หยุด/เล่นต่อ · , . เดินทีละเฟรม · "
          "< > กระโดด 30 · q ออก")
    try:
        while True:
            if img is None:
                img = draw(i)
            elif not paused:
                if i + 1 >= len(frames):
                    paused = True                 # ถึงเฟรมสุดท้ายแล้วค้างไว้ให้ดู
                else:
                    i += 1
                    img = draw(i)
            cv2.imshow("replay", img)
            key = cv2.waitKey(max(int(1000.0 / (a.fps * speed)), 1)) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                reset()
                img = draw(i)
                paused = False
            elif key in (ord("+"), ord("=")):
                speed = min(speed * 1.5, 8.0)
            elif key in (ord("-"), ord("_")):
                speed = max(speed / 1.5, 0.125)
            elif key == ord("s") and img is not None:
                fn = f"replay_{sec_name}_{i:06d}.png"
                cv2.imwrite(fn, img)
                print(f"เซฟ {fn}", flush=True)
            # ใช้ , . < > แทนลูกศร เพราะ waitKey()&0xFF บนวินโดวส์อ่านลูกศรไม่ได้
            elif key in (ord(","), ord("."), ord("<"), ord(">")):
                step = 30 if key in (ord("<"), ord(">")) else 1
                if key in (ord(","), ord("<")):
                    step = -step
                i = max(0, min(len(frames) - 1, i + step))
                img = draw(i)
                paused = True

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if n:
            print(f"\nดูไปถึงเฟรม {i}/{len(frames)} · เทียบได้ {n} เฟรม")
            print(f"  โซนถูก {hit / n:.0%} · มุมผิดเฉลี่ย {serr / n:.1f} องศา")
            if held:
                print(f"  (ผลออฟไลน์ของช่วงนี้: โซนถูก {rule_all['zone_acc']:.0%} · "
                      f"ผิด {rule_all['mae_deg']:.1f} องศา)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
