"""ตรวจไฟล์ที่เขียนออกมาแล้ว โดยอ่านกลับจากไฟล์จริง ไม่ดูตัวแปรในหน่วยความจำ

    python pcb/check_out.py

จุดประสงค์ต่างจากตัวตรวจใน router.py:
  router  ตรวจ **ลายที่คิดได้** ว่าต่อถูกและไม่ช็อต
  ไฟล์นี้ ตรวจ **ไฟล์ที่จะส่งโรงงาน** ว่าเขียนออกมาครบและถูกไวยากรณ์

สองอย่างนี้พลาดคนละแบบ เคยพลาดมาแล้วจริง: แผ่นกราวด์ถูกต้องในหน่วยความจำ
และตัวตรวจของ router บอกว่าผ่าน แต่ตอนแรกไม่ได้เขียนลงไฟล์ Gerber เลย
บอร์ดที่ผลิตจะไม่มีกราวด์ ทั้งที่ทุกอย่าง 'ผ่าน'
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def check_gerber(path):
    txt = open(path, encoding="ascii").read()
    errs = []
    if not txt.rstrip().endswith("M02*"):
        errs.append("ไม่จบด้วย M02*")
    if "%FSLAX46Y46*%" not in txt or "%MOMM*%" not in txt:
        errs.append("ไม่ได้ประกาศรูปแบบพิกัดหรือหน่วย")
    defined = set(re.findall(r"%ADD(\d+)[CR],", txt))
    used = set(re.findall(r"^D(\d+)\*", txt, re.M)) - {"01", "02", "03"}
    miss = used - defined
    if miss:
        errs.append(f"ใช้รูรับแสงที่ไม่ได้ประกาศ: {sorted(miss)}")
    if txt.count("G36*") != txt.count("G37*"):
        errs.append(f"G36 กับ G37 ไม่เท่ากัน ({txt.count('G36*')}/{txt.count('G37*')})")
    if txt.count("%LPC*%") and not txt.rstrip().endswith("M02*"):
        errs.append("เปิดโหมดลบแล้วไม่ได้ปิด")
    # ต้องยึดต้นบรรทัด ไม่งั้นไปจับ %FSLAX46Y46*% เป็นพิกัด (0.000046 มม.)
    # แล้วขอบเขตเพี้ยน — ตอนแรกรายงานบอร์ด 61.8 มม. ว่ากว้าง 62.4
    co = re.findall(r"^X(-?\d+)Y(-?\d+)D", txt, re.M)
    if not co:
        return errs, None
    xs = [int(a) / 1e6 for a, _ in co]
    ys = [int(b) / 1e6 for _, b in co]
    return errs, (min(xs), max(xs), min(ys), max(ys))


def check_drill(path):
    txt = open(path, encoding="ascii").read()
    errs = []
    if not txt.startswith("M48"):
        errs.append("ไม่ได้ขึ้นต้นด้วย M48")
    if "M30" not in txt:
        errs.append("ไม่มี M30 ปิดไฟล์")
    if "METRIC" not in txt:
        errs.append("ไม่ได้ระบุหน่วยเมตริก")
    tools = set(re.findall(r"^T(\d+)C", txt, re.M))
    used = set(re.findall(r"^T(\d+)$", txt, re.M)) - {"0"}
    if used - tools:
        errs.append(f"ใช้ดอกสว่านที่ไม่ได้ประกาศ: {sorted(used - tools)}")
    n = len(re.findall(r"^X[-\d.]+Y[-\d.]+$", txt, re.M))
    return errs, n


def main():
    boards = sorted(d for d in os.listdir(OUT)
                    if os.path.isdir(os.path.join(OUT, d)))
    total = 0
    for b in boards:
        d = os.path.join(OUT, b)
        gerbers = sorted(f for f in os.listdir(d) if f.endswith(".gbr"))
        drill = [f for f in os.listdir(d) if f.endswith(".drl")]
        need = ["-F_Cu.gbr", "-B_Cu.gbr", "-F_Mask.gbr", "-B_Mask.gbr",
                "-F_Silkscreen.gbr", "-Edge_Cuts.gbr"]
        errs = [f"ขาดไฟล์ {b}{s}" for s in need
                if not any(f.endswith(s) for f in gerbers)]
        if not drill:
            errs.append("ขาดไฟล์เจาะ .drl")
        box = None
        for f in gerbers:
            e, bb = check_gerber(os.path.join(d, f))
            errs += [f"{f}: {x}" for x in e]
            if f.endswith("Edge_Cuts.gbr"):
                box = bb
        nh = 0
        if drill:
            e, nh = check_drill(os.path.join(d, drill[0]))
            errs += [f"{drill[0]}: {x}" for x in e]
            if nh == 0:
                errs.append("ไฟล์เจาะไม่มีรูเลย")
        # ทองแดงต้องอยู่ในขอบบอร์ด
        if box:
            for f in gerbers:
                if f.endswith("Edge_Cuts.gbr"):
                    continue
                _e, bb = check_gerber(os.path.join(d, f))
                if bb and (bb[0] < box[0] - 0.01 or bb[1] > box[1] + 0.01
                           or bb[2] < box[2] - 0.01 or bb[3] > box[3] + 0.01):
                    errs.append(f"{f}: มีของอยู่นอกขอบบอร์ด")
        cu = os.path.getsize(os.path.join(d, f"{b}-F_Cu.gbr"))
        print(f"{b:6s} ไฟล์ {len(gerbers)+len(drill)} · รูเจาะ {nh:3d} · "
              f"ขอบ {box[1]-box[0]:5.1f} x {box[3]-box[2]:5.1f} mm · "
              f"{'ผ่าน' if not errs else str(len(errs)) + ' ปัญหา'}")
        for e in errs:
            print("    -", e)
        total += len(errs)
    print(f"\n{'ไฟล์พร้อมส่งโรงงาน' if not total else f'ยังมี {total} ปัญหา'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
