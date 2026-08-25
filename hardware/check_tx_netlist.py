"""ตรวจวงจรภาคส่งบนแผ่นไข่ปลา **จากผังเอง** — ไล่โหนดไฟฟ้าว่าต่อถูกจริงไหม

Run:  python hardware/check_tx_netlist.py

ทำไมต้องมี: ผังที่ "ดูสวย" กับผังที่ "ต่อถูก" เป็นคนละเรื่อง เครื่องวาดตรวจได้แค่
เส้นทับกัน/สายลอดใต้ชิป/ขาอุปกรณ์ชนรู แต่ตรวจไม่ได้ว่าสายที่ลากไปนั้น *ต่อถึงกัน*
ตามที่ตั้งใจหรือเปล่า สคริปต์นี้อ่าน perf_tx_layout.py แล้วสร้างกราฟโหนดขึ้นมาใหม่

โมเดลที่ใช้ (สำคัญ — ผิดตรงนี้ผลตรวจจะหลอก):
  wire / solder   = ต่อถึงกันทางไฟตรง
  part(...,"R")   = ต่อถึงกันทางไฟตรง (ตัวต้านทานนำไฟตรง)
  part(...,"C")   = **ไม่ต่อ** — คาปาซิเตอร์กั้นไฟตรง
                    (ถ้านับ C เป็นสายด้วย คาปาบายพาสจะทำให้ +5V กับ GND
                     กลายเป็นโหนดเดียวกัน แล้วผลตรวจ "ไม่ช็อต" จะเชื่อไม่ได้)
  bus(col,lo,hi)  = ลวดแข็งเชื่อมทุกรูในคอลัมน์นั้นถึงกันหมด
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

PIN = {1: "L36", 2: "L35", 3: "L34", 4: "L33", 5: "L32", 6: "L31", 7: "L30",
       8: "O30", 9: "O31", 10: "O32", 11: "O33", 12: "O34", 13: "O35", 14: "O36"}
BUSES = (("A", 20, 44), ("C", 20, 44), ("U", 20, 44))
# หัวส่ง: (ชื่อ, ขาที่หัวต่อ, เกตแรก เข้า→ออก, เกตสอง เข้า→ออก, ปลาย R, ขั้วสัญญาณ, ขั้ว GND)
CHAINS = (("TX1", "E24", 1, 2, 3, 4, "S43", "T43", "T41"),
          ("TX2", "F24", 13, 12, 11, 10, "S32", "T32", "T30"),
          ("TX3", "G24", 5, 6, 9, 8, "S26", "T26", "T24"))


def build(src):
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def uni(a, b):
        par[find(a)] = find(b)

    for m in re.finditer(r'p\.(?:wire|solder)\("([A-Z]\d+)",\s*"([A-Z]\d+)"', src):
        uni(m.group(1), m.group(2))
    for m in re.finditer(
            r'p\.part\("([A-Z]\d+)",\s*"([A-Z]\d+)",\s*"[^"]*",\s*"([RC])"', src):
        if m.group(3) == "R":
            uni(m.group(1), m.group(2))
    for col, lo, hi in BUSES:
        for r in range(lo, hi):
            uni(f"{col}{r}", f"{col}{r + 1}")
    return find


def main():
    src = (HERE / "perf_tx_layout.py").read_text(encoding="utf-8")
    find = build(src)
    gnd, v5 = find("A30"), find("C30")
    ok = True

    def show(label, good):
        nonlocal ok
        ok &= good
        print(f"  {'✅' if good else '❌'}  {label}")

    print("ไฟเลี้ยง")
    show("ขา 14 (VCC) ถึงบัส +5V", find(PIN[14]) == v5)
    show("ขา 7 (GND) ถึงบัส GND", find(PIN[7]) == gnd)
    show("+5V ไม่ช็อตกับ GND", v5 != gnd)
    show("บัส GND ซ้าย (U) ต่อถึงบัส GND ขวา (A)", find("U30") == gnd)
    show("คาปาบายพาสคร่อม +5V กับ GND จริง",
         find("O38") == v5 and find("L38") == gnd)

    print("\nสายโซ่หัวส่งแต่ละตัว")
    for nm, hdr, gi, go, gi2, go2, rfar, sig, tgnd in CHAINS:
        show(f"{nm} หัวต่อ {hdr} → ขา {gi}", find(hdr) == find(PIN[gi]))
        show(f"{nm} ขา {go} → ขา {gi2} (เชื่อมสองเกต)",
             find(PIN[go]) == find(PIN[gi2]))
        show(f"{nm} ขา {go2} → ผ่าน R → ขั้ว {sig}", find(PIN[go2]) == find(sig))
        show(f"{nm} ไม่ช็อตข้ามตัวเกต",
             find(PIN[gi]) != find(PIN[go]) and find(PIN[gi2]) != find(PIN[go2]))
        show(f"{nm} ไม่มีขาไหนไปโดนไฟเลี้ยง",
             all(find(PIN[q]) not in (v5, gnd) for q in (gi, go, gi2, go2)))
        show(f"{nm} ขั้ว GND {tgnd} ถึงบัส GND", find(tgnd) == gnd)

    print("\nหัวส่งไม่ลัดถึงกัน")
    sigs = {c[0]: c[7] for c in CHAINS}
    for a in sigs:
        for b in sigs:
            if a < b:
                show(f"{a} กับ {b} แยกกัน", find(sigs[a]) != find(sigs[b]))

    # ขาที่ไม่ได้ใช้ต้องไม่มีเลย — 74HCT04 อินพุตลอยจะเหวี่ยงและกินกระแส
    print("\nขาชิปที่ยังไม่ได้ต่ออะไร (อินพุตลอยห้ามมี)")
    unconnected = [n for n in range(1, 15)
                   if all(find(PIN[n]) != find(o) for o in
                          [PIN[k] for k in PIN if k != n] + ["A30", "C30",
                                                             "E24", "F24", "G24",
                                                             "T43", "T32", "T26"])]
    show(f"ขาที่ลอย: {unconnected if unconnected else 'ไม่มี'}", not unconnected)

    print(f"\n{'ผ่านทั้งหมด — ผังนี้ต่อถูกทางไฟฟ้า' if ok else 'มีข้อผิดพลาด อย่าเพิ่งบัดกรี'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
