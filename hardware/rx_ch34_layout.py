"""ผังต่อช่องรับที่ 3 และ 4 บน MCP6004 ตัวเดียว บอร์ดทดลองอันเดียว (Stage B)

Run:  python hardware/rx_ch34_layout.py   ->  hardware/diagrams/rx34_*.svg

ทำไมชิปเดียวได้สองช่อง: MCP6004 มีออปแอมป์ 4 ตัว หนึ่งช่องใช้ 2 สเตจ จึงลงได้พอดี
สองช่อง และบอร์ดนี้ไม่ต้องมีบัฟเฟอร์ VREF (VREF มาจากบอร์ด RX) จึงใช้ครบทั้ง 4 ตัว
ข้อดีแถม: สองช่องอยู่ไดเดียวกัน อุณหภูมิเดียวกัน -> แมตช์กันดีกว่าคนละชิป ซึ่งสำคัญ
เพราะคู่ A80-C80 คือเบสไลน์ที่ใช้วัดมุม

การแบ่งชิป (chip_row0 = 12 ขาอยู่คอลัมน์ E กับ F):
    ช่อง 3  A80 -> GPIO32   ใช้ออปแอมป์ A+B   ขา 1-7   คอลัมน์ E   ทำงานฝั่งซ้าย A-E
    ช่อง 4  C80 -> GPIO33   ใช้ออปแอมป์ C+D   ขา 8-14  คอลัมน์ F   ทำงานฝั่งขวา F-J

ตำแหน่งขา (แถว):
    ซ้าย  ขา1 OUTA=12 · ขา2 INA-=13 · ขา3 INA+=14 · ขา4 VDD=15
          ขา5 INB+=16 · ขา6 INB-=17 · ขา7 OUTB=18
    ขวา   ขา14 OUTD=12 · ขา13 IND-=13 · ขา12 IND+=14 · ขา11 VSS=15
          ขา10 INC+=16 · ขา9 INC-=17 · ขา8 OUTC=18

สังเกตว่าฝั่งขวาเลขขาไล่กลับทาง (DIP นับทวนเข็ม) แถวของ OUT/IN จึงสลับกับฝั่งซ้าย
ผังนี้จัดวางให้ทั้งสองช่องใช้ "แถวเดียวกัน" สำหรับงานเดียวกัน จะได้จำง่ายและ
ปรสิตของสองช่องใกล้เคียงกัน (เลย์เอาต์เหมือนกัน = แมตช์ดี)

ค่าอุปกรณ์ทุกตัวเท่ากับช่องที่ 2 ที่ต่อสำเร็จแล้ว ดู hardware/rx_frontend.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_panel import Panel, TOP, COL, C_POS, C_NEG, C_SIG          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")

CHIP_L = dict(rail="left", chip_row0=12, chip_name="MCP6004 #3")
CHIP_R = dict(rail="right", chip_row0=12, chip_name="MCP6004 #3", width=800)
NXL, NXR = 516, 612


def arrive(p, row, col, text, sub, colour="#f08c00", x0=28):
    """สายที่มาจากบอร์ด RX เข้าที่รูหนึ่ง"""
    x, yy = p.p(f"{row}{col}")
    p._path(f"M{x0},{yy} L{x},{yy}", colour)
    p.b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{colour}"/>')
    p.b.append(f'<text x="{x0-2}" y="{yy-9}" font-size="11.5" font-weight="700" '
               f'text-anchor="start" fill="{colour}">{text}</text>')
    p.b.append(f'<text x="{x0-2}" y="{yy+16}" font-size="10.5" fill="#868e96">{sub}</text>')


def arrive_rail(p, row, which, text, sub):
    yy = p.y[row]
    rx = p.rp if which == "+" else p.rn
    colour = C_POS if which == "+" else C_NEG
    p._path(f"M28,{yy} L{rx},{yy}", colour)
    p.b.append(f'<circle cx="{rx}" cy="{yy}" r="5" fill="{colour}"/>')
    p.b.append(f'<text x="26" y="{yy-9}" font-size="11.5" font-weight="700" '
               f'fill="{colour}">{text}</text>')
    p.b.append(f'<text x="26" y="{yy+16}" font-size="10.5" fill="#868e96">{sub}</text>')


def out_arrow(p, ref, gpio, sub):
    x, yy = p.p(ref)
    p._path(f"M{x},{yy} L{COL['J']+52},{yy}", "#f08c00")
    p.b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="#f08c00"/>')
    p.b.append(f'<text x="{COL["J"]+56}" y="{yy-8}" font-size="11.5" font-weight="700" '
               f'fill="#f08c00">ไป {gpio}</text>')
    p.b.append(f'<text x="{COL["J"]+56}" y="{yy+10}" font-size="10.5" fill="#868e96">{sub}</text>')


# ------------------------------------------------------------- ผัง 1 ------

def panel_power():
    p = Panel("rx34_1_power_chip",
              "ช่อง 3+4 — ผัง 1 · ไฟเลี้ยง ชิป และ VREF ร่วม",
              "MCP6004 ตัวเดียวทำสองช่อง แถว 12-18 · ไม่มีตัวแบ่ง VREF บนบอร์ดนี้",
              [2, 9, 10, 12, 13, 14, 15, 16, 17, 18], **CHIP_L)
    p.chip()
    arrive_rail(p, 9, "+", "3V3 จากบอร์ด RX", "ต่อรางแดงถึงรางแดง")
    arrive_rail(p, 18, "-", "GND จากบอร์ด RX", "ต่อรางน้ำเงินถึงรางน้ำเงิน")
    p.rail_cap(9, "10µF", pol=True)
    p.rail_cap(10, "100nF")
    p.to_rail("15B", "+", below=False)               # ขา 4 VDD
    p.to_rail("15H", "-", lane=486, after=18)        # ขา 11 VSS (อ้อมขวาแล้วลงราง)
    p.tag("15B", "ขา 4 VDD", -14, 5, "end", C_POS)
    p.tag("15H", "ขา 11 VSS", 16, 5, "start", C_NEG)
    # VREF เข้าฝั่งซ้าย แล้วต่อข้ามร่องกลางไปฝั่งขวา ให้ทั้งสองช่องใช้ตัวแบ่งเดียวกัน
    arrive(p, 2, "A", "VREF จากบอร์ด RX", "แถว 2 ของบอร์ด RX")
    p.wire("2D", "2G", "#f08c00")
    p.tag("2G", "ข้ามร่องกลาง ไปเลี้ยงช่อง 4", 8, -14, "start", "#f08c00")
    p.note(NXL, TOP - 42, ["ชิปเดียวทำสองช่อง", "ขา 1-7 (ซ้าย) = ช่อง 3",
                           "ขา 8-14 (ขวา) = ช่อง 4", "อยู่ไดเดียวกัน",
                           "แมตช์ดีกว่าคนละชิป"], "#2f9e44", "#ebfbee")
    p.note(NXL, TOP + 160, ["ต้องต่อสะพานรางซ้าย-ขวา", "ทั้งราง + และราง −",
                            "เพราะช่อง 3 ลงรางซ้าย", "ช่อง 4 ลงรางขวา",
                            "ลืมข้อนี้ = ช่อง 4 ไม่มีกราวด์"], "#e03131", "#fff5f5")
    p.note(NXL, TOP + 330, ["VREF ต้องมาจาก", "ตัวแบ่งเดียวของบอร์ด RX",
                            "ห้ามสร้างตัวแบ่งใหม่", "ไม่งั้นสี่ช่องอยู่คนละเรฟ",
                            "แล้วมุมเพี้ยนทั้งหมด"], "#f08c00", "#fff4e6")
    return p


# --------------------------------------------------- ผัง 2-3 : ช่อง 3 -----

def panel_ch3_input():
    p = Panel("rx34_2_ch3_input",
              "ช่อง 3 (A80 → GPIO32) — ผัง 2 · อินพุตและสเตจขยายที่ 1",
              "ออปแอมป์ A · เข้าขา 3 (แถว 14) · Rf1 จากขา 1 (แถว 12)",
              [2, 5, 7, 12, 13, 14, 22, 25, 27], **CHIP_L)
    p.chip()
    p.transducer("5B", "TCT40-16R  A80")
    arrive(p, 2, "A", "VREF (จากผัง 1)", "แถว 2 ฝั่งซ้าย")
    p.part("5C", "7C", "100nF", "C")               # Cin
    p.part_lane("7D", "2D", 287, "10k", "R")       # R_bias -> VREF
    p.vlane("7A", "14A", 221)                      # โหนดอินพุต -> ขา 3 (INA+)
    p.part("12B", "22B", "100k", "R")              # Rf1  จากขา 1 (OUTA)
    p.vlane("13D", "22D", 287)                     # ขา 2 (INA-) -> ปลาย Rf1
    p.part("22A", "25A", "10k", "R")               # Rg1
    p.part("25B", "27B", "1nF", "C")               # Cg1
    p.to_rail("27C", "-")
    p.note(NXL, TOP - 42, ["เหมือนช่องที่ 2 ทุกรู", "ค่าทุกตัวเท่ากันเป๊ะ",
                           "ต่างแค่หัวรับเป็น A80", "และขาออกไป GPIO32"],
           "#2f9e44", "#ebfbee")
    p.note(NXL, TOP + 150, ["ขา 3 = INA+ อยู่แถว 14", "ขา 1 = OUTA อยู่แถว 12",
                            "ขา 2 = INA− อยู่แถว 13", "(ฝั่งซ้ายนับ 1→7 ลงล่าง)"],
           "#1971c2", "#e7f5ff")
    return p


def panel_ch3_stage2():
    p = Panel("rx34_3_ch3_stage2",
              "ช่อง 3 (A80 → GPIO32) — ผัง 3 · สเตจขยายที่ 2 และขาออก",
              "ออปแอมป์ B · ขา 1 → ขา 5 · Rf2 จากขา 7 (แถว 18) · ออกที่ 46D",
              [12, 13, 14, 15, 16, 17, 18, 30, 34, 37, 39, 42, 46, 48], **CHIP_L)
    p.chip()
    p.vlane("12C", "16C", 243)                     # ขา 1 (OUTA) -> ขา 5 (INB+)
    p.vlane("18B", "30A", 221)                     # ขา 7 (OUTB) -> เครือข่ายสเตจ 2
    p.part("30B", "34B", "100k", "R")              # Rf2
    p.vlane("17D", "34D", 265)                     # ขา 6 (INB-) -> ปลาย Rf2
    p.part("34A", "37A", "10k", "R")               # Rg2
    p.part("37B", "39B", "1nF", "C")               # Cg2
    p.to_rail("39C", "-")
    p.vlane("30C", "42C", 243)                     # โหนดขาออก -> ภาคขาออก
    p.part("42A", "46A", "2.2k", "R")              # R_out
    p.part("46B", "48B", "1nF", "C")               # C_out
    p.to_rail("48C", "-")
    out_arrow(p, "46D", "GPIO32", "แถว 48 หัวต่อฝั่ง ADC")
    p.note(NXL, TOP - 42, ["ขา 5 = INB+ แถว 16", "ขา 6 = INB− แถว 17",
                           "ขา 7 = OUTB แถว 18", "OUTB เป็นทั้งอินพุตของ Rf2",
                           "และต้นทางของขาออก"], "#1971c2", "#e7f5ff")
    p.note(NXL, TOP + 190, ["สายไป GPIO32 ควรยาว", "เท่ากับของช่อง 1,2,4",
                            "ความต่างของสาย", "= ความต่างของ Δt โดยตรง"],
           "#e03131", "#fff5f5")
    return p


# --------------------------------------------------- ผัง 4-5 : ช่อง 4 -----

def panel_ch4_input():
    p = Panel("rx34_4_ch4_input",
              "ช่อง 4 (C80 → GPIO33) — ผัง 4 · อินพุตและสเตจขยายที่ 1",
              "ออปแอมป์ C · เข้าขา 10 (แถว 16) · Rf1 จากขา 8 (แถว 18) · ฝั่งขวา F-J",
              [2, 5, 7, 12, 16, 17, 18, 22, 25, 27], **CHIP_R)
    p.chip()
    p.transducer("5I", "TCT40-16R  C80", cx=572, lead_side="right")
    # VREF ไม่ใช่สายใหม่ — มันมาอยู่ที่โหนด 2 ฝั่งขวา (2F-2J) แล้ว จากสายข้าม
    # ร่องกลาง 2D->2G ในผัง 1 ตรงนี้แค่มาร์กว่า "มีอยู่แล้ว" ไม่ต้องต่ออะไรเพิ่ม
    x2g, y2g = p.p("2G")
    p.b.append(f'<circle cx="{x2g}" cy="{y2g}" r="5" fill="#f08c00"/>')
    p.b.append(f'<text x="{x2g+12}" y="{y2g-8}" font-size="11.5" font-weight="700" '
               f'fill="#f08c00">VREF อยู่แล้ว</text>')
    p.b.append(f'<text x="{x2g+12}" y="{y2g+10}" font-size="10.5" fill="#868e96">'
               f'จากสายข้าม 2D→2G (ผัง 1) — ไม่ต้องต่อใหม่</text>')
    p.part("5H", "7H", "100nF", "C")               # Cin
    p.part_lane("7G", "2G", 353, "10k", "R")       # R_bias -> VREF (โหนด 2G)
    p.vlane("7J", "16J", 441)                      # โหนดอินพุต -> ขา 10 (INC+)
    p.part("18I", "22I", "100k", "R")              # Rf1  จากขา 8 (OUTC)
    p.vlane("17H", "22H", 397)                     # ขา 9 (INC-) -> ปลาย Rf1
    p.part("22J", "25J", "10k", "R")               # Rg1
    p.part("25I", "27I", "1nF", "C")               # Cg1
    p.to_rail("27H", "-")
    p.note(NXR, TOP - 42, ["ฝั่งขวาเลขขาไล่กลับทาง", "ขา 10 = INC+ แถว 16",
                           "ขา 9 = INC− แถว 17", "ขา 8 = OUTC แถว 18",
                           "(DIP นับทวนเข็มนาฬิกา)"], "#e03131", "#fff5f5", w=175)
    p.note(NXR, TOP + 200, ["ค่าทุกตัวเท่าช่อง 3", "วางเลย์เอาต์ให้เหมือนกัน",
                            "ปรสิตจะได้เท่ากัน", "= สองช่องแมตช์กันดี"],
           "#2f9e44", "#ebfbee", w=175)
    return p


def panel_ch4_stage2():
    p = Panel("rx34_5_ch4_stage2",
              "ช่อง 4 (C80 → GPIO33) — ผัง 5 · สเตจขยายที่ 2 และขาออก",
              "ออปแอมป์ D · ขา 8 → ขา 12 · Rf2 จากขา 14 (แถว 12) · ออกที่ 46J",
              [12, 13, 14, 15, 16, 17, 18, 30, 34, 37, 39, 42, 46, 48], **CHIP_R)
    p.chip()
    p.vlane("18G", "14G", 375)                     # ขา 8 (OUTC) -> ขา 12 (IND+)
    p.vlane("12H", "30H", 419)                     # ขา 14 (OUTD) -> เครือข่ายสเตจ 2
    p.part("30I", "34I", "100k", "R")              # Rf2
    p.vlane("13G", "34G", 353)                     # ขา 13 (IND-) -> ปลาย Rf2
    p.part("34J", "37J", "10k", "R")               # Rg2
    p.part("37I", "39I", "1nF", "C")               # Cg2
    p.to_rail("39H", "-")
    p.vlane("30J", "42J", 441)                     # โหนดขาออก -> ภาคขาออก
    p.part("42I", "46I", "2.2k", "R")              # R_out
    p.part("46H", "48H", "1nF", "C")               # C_out
    p.to_rail("48G", "-")
    x, yy = p.p("46J")
    p._path(f"M{x},{yy} L{x+90},{yy}", "#f08c00")
    p.b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="#f08c00"/>')
    p.b.append(f'<text x="{x+94}" y="{yy-8}" font-size="11.5" font-weight="700" '
               f'fill="#f08c00">ไป GPIO33</text>')
    p.b.append(f'<text x="{x+94}" y="{yy+10}" font-size="10.5" fill="#868e96">'
               f'แถว 49 หัวต่อฝั่ง ADC</text>')
    p.note(NXR, TOP - 42, ["ขา 14 = OUTD แถว 12", "ขา 13 = IND− แถว 13",
                           "ขา 12 = IND+ แถว 14", "OUTD อยู่บนสุด ต่างจาก",
                           "ช่อง 3 ที่ OUTB อยู่ล่างสุด"], "#1971c2", "#e7f5ff", w=175)
    p.note(NXR, TOP + 200, ["เช็คก่อนจ่ายไฟ:", "มัลติมิเตอร์โหมดปี๊บ",
                            "ขา 4 ↔ ราง + ต้องปี๊บ", "ขา 11 ↔ ราง − ต้องปี๊บ",
                            "ขา 4 ↔ ขา 11 ต้องเงียบ"], "#e03131", "#fff5f5", w=175)
    return p


PANELS = [panel_power, panel_ch3_input, panel_ch3_stage2,
          panel_ch4_input, panel_ch4_stage2]


def main():
    for fn in PANELS:
        print("wrote", fn().write(OUT))


if __name__ == "__main__":
    main()
