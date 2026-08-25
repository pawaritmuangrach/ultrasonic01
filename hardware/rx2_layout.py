"""Draw the SECOND receive channel, on its own breadboard.

Run:  python hardware/rx2_layout.py   ->  hardware/diagrams/rx2ch_*.svg

Why a third board rather than more room on the RX board: an MCP6004 has four
sections and channel 1 already uses three of them - A and B are its two gain
stages, C is the VREF buffer shared by every channel. Only D is left, which is
one stage, and a channel with one stage is not the same channel. Two receive
paths that differ turn straight into a Delta-t error, which is the one thing
Stage 2 measures. So channel 2 gets its own MCP6004, and the RX board has no
seven-row gap left to put one in.

Everything below is channel 1 copied hole for hole. That is the point: the two
paths must have identical group delay, so they get identical parts in identical
positions. What is NOT copied is the VREF divider - both channels must sit on
the SAME reference, so VREF arrives by wire from the RX board.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_panel import Panel, TOP, COL, C_POS, C_NEG, C_SIG          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")

CHIP = dict(rail="left", chip_row0=12, chip_name="MCP6004 #2")
NX = 516


def arrive(p, row, col, text, sub, colour):
    """A wire coming in from the RX board, entering at a hole."""
    x, yy = p.p(f"{row}{col}")
    p._path(f"M28,{yy} L{x},{yy}", colour)
    p.b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{colour}"/>')
    p.b.append(f'<text x="26" y="{yy-9}" font-size="11.5" font-weight="700" '
               f'text-anchor="start" fill="{colour}">{text}</text>')
    p.b.append(f'<text x="26" y="{yy+16}" font-size="10.5" fill="#868e96">{sub}</text>')


def arrive_rail(p, row, which, text, sub):
    yy = p.y[row]
    rx = p.rp if which == "+" else p.rn
    colour = C_POS if which == "+" else C_NEG
    p._path(f"M28,{yy} L{rx},{yy}", colour)
    p.b.append(f'<circle cx="{rx}" cy="{yy}" r="5" fill="{colour}"/>')
    p.b.append(f'<text x="26" y="{yy-9}" font-size="11.5" font-weight="700" '
               f'fill="{colour}">{text}</text>')
    p.b.append(f'<text x="26" y="{yy+16}" font-size="10.5" fill="#868e96">{sub}</text>')


def panel1():
    p = Panel("rx2ch_1_power_chip",
              "ช่องที่ 2 — ผัง 1 · ไฟเลี้ยงและตัวชิป",
              "MCP6004 ตัวที่สอง แถว 12-18 · ไม่มีตัวแบ่ง VREF · ออปแอมป์ที่ไม่ใช้สองตัว",
              [10, 12, 13, 14, 15, 16, 17, 18, 19, 21], **CHIP)
    p.chip()
    arrive_rail(p, 10, "+", "3V3 จากบอร์ด RX", "ต่อรางแดงถึงรางแดง")
    arrive_rail(p, 21, "-", "GND จากบอร์ด RX", "ต่อรางน้ำเงินถึงรางน้ำเงิน")
    p.rail_cap(12, "10µF", pol=True)
    p.rail_cap(13, "100nF")
    p.to_rail("15B", "+", below=False)            # pin 4  VDD
    p.to_rail("15H", "-", lane=486, after=18)     # pin 11 VSS
    p.vlane("15A", "19A", 221, C_POS)             # bypass: VDD down to row 19
    p.part("19D", "21D", "100nF", "C")
    p.to_rail("21C", "-")
    p.wire("12J", "15J", C_NEG)                   # pin 14 -> GND
    p.wire("13G", "14G")                          # pin 13 <-> pin 12
    p.wire("16G", "15G", C_NEG)                   # pin 10 -> GND
    p.wire("17G", "18G")                          # pin 9  <-> pin 8
    p.note(NX, TOP - 42, ["บอร์ดนี้ต่อรางร่วมกับ", "บอร์ด RX ได้ — ต่างจาก",
                          "บอร์ด TX เพราะไฟเท่ากัน", "และไม่มีกระแสสวิตช์"],
           "#2f9e44", "#ebfbee")
    p.note(NX, TOP + 130, ["ออปแอมป์สองตัวที่ไม่ใช้", "ขา 14 และขา 10 ลงกราวด์",
                           "ขา 13 ต่อขา 12", "ขา 9 ต่อขา 8",
                           "ห้ามปล่อยลอยแม้แต่ขาเดียว"], "#1971c2", "#e7f5ff")
    p.note(NX, TOP + 290, ["ไม่มีตัวแบ่ง VREF", "บนบอร์ดนี้",
                           "ทั้งสองช่องต้องใช้ VREF", "ตัวเดียวกันเป๊ะ",
                           "ไม่งั้นความต่างจะกลาย", "เป็น error ของ Δt"],
           "#f08c00", "#fff4e6")
    return p


def panel2():
    p = Panel("rx2ch_2_input_stage1",
              "ช่องที่ 2 — ผัง 2 · อินพุตและสเตจขยายที่ 1",
              "เหมือนช่องที่ 1 ทุกรู · VREF มาจากบอร์ด RX แถว 2",
              [2, 5, 7, 12, 13, 14, 22, 25, 27], **CHIP)
    p.chip()
    p.transducer("5B", "TCT40-16R")
    arrive(p, 2, "A", "VREF จากบอร์ด RX", "แถว 2 ของบอร์ด RX", "#f08c00")
    p.part("5C", "7C", "100nF", "C")              # Cin
    p.part_lane("7D", "2D", 287, "10k", "R")      # R_bias -> VREF
    p.vlane("7A", "14A", 221)                     # input node -> pin 3
    p.part("12B", "22B", "100k", "R")             # Rf1
    p.vlane("13D", "22D", 287)                    # pin 2 -> Rf1 bottom
    p.part("22A", "25A", "10k", "R")              # Rg1
    p.part("25B", "27B", "1nF", "C")              # Cg1
    p.to_rail("27C", "-")
    p.note(NX, TOP - 42, ["ค่าทุกตัวต้องเท่ากับ", "ช่องที่ 1 เป๊ะ",
                          "ถ้าเป็นไปได้ให้วัด R", "แล้วจับคู่ตัวที่ใกล้กันสุด"],
           "#e03131", "#fff5f5")
    p.note(NX, TOP + 175, ["หัวรับตัวนี้ใส่รู C140", "(หรือ B140 ถ้าสายถึงง่ายกว่า)",
                           "รัศมีเท่ากับ A140 เป๊ะ", "เป้าตรงหน้า → Δt = 0",
                           "ได้จุดคาลิเบรตฟรี", "โดยไม่ต้องเชื่อโมเดลเลย"],
           "#1971c2", "#e7f5ff")
    return p


def panel3a():
    p = Panel("rx2ch_3a_stage2",
              "ช่องที่ 2 — ผัง 3A · สเตจขยายที่ 2",
              "เหมือนช่องที่ 1 ทุกรู",
              [12, 13, 14, 15, 16, 17, 18, 30, 34, 37, 39], **CHIP)
    p.chip()
    p.vlane("12C", "16C", 243)                    # OUTA -> +INB
    p.vlane("18B", "30A", 221)                    # OUTB -> stage-2 network
    p.part("30B", "34B", "100k", "R")             # Rf2
    p.vlane("17D", "34D", 265)                    # -INB -> Rf2 bottom
    p.part("34A", "37A", "10k", "R")              # Rg2
    p.part("37B", "39B", "1nF", "C")              # Cg2
    p.to_rail("39C", "-")
    p.note(NX, TOP - 42, ["เทียบกับผัง rxv2_3a", "ของช่องที่ 1 ได้เลย",
                          "ต้องเหมือนกันทุกจุด"], "#2f9e44", "#ebfbee")
    return p


def panel3b():
    p = Panel("rx2ch_3b_output",
              "ช่องที่ 2 — ผัง 3B · ภาคขาออกไป GPIO35",
              "R 2.2k แถว 30-34 · C 1nF แถว 34-36 · ออกที่ 34I ไปแถว 47 ของบอร์ด RX",
              [18, 30, 34, 36], **CHIP)
    p.pin_marker(18, 7)
    p.route("18D", "30G", 265, 18)                # OUTB -> output side
    p.part("30H", "34H", "2.2k", "R")             # R_out
    p.part("34G", "36G", "1nF", "C")              # C_out
    p.to_rail("36F", "-", lane=320, after=36)
    x, yy = p.p("34I")
    p._path(f"M{x},{yy} L{COL['J']+52},{yy}", "#f08c00")
    p.b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="#f08c00"/>')
    p.b.append(f'<text x="{COL["J"]+56}" y="{yy-8}" font-size="11.5" font-weight="700" '
               f'fill="#f08c00">ไป GPIO35</text>')
    p.b.append(f'<text x="{COL["J"]+56}" y="{yy+10}" font-size="10.5" fill="#868e96">'
               f'แถว 47 รู H ของบอร์ด RX</text>')
    p.note(NX, TOP + 110, ["GPIO35 = ADC1_CH7", "อยู่ติดกับ GPIO34",
                           "ที่ช่องที่ 1 ใช้อยู่", "ADC สลับอ่านสองช่องนี้"],
           "#f08c00", "#fff4e6")
    p.note(NX, TOP + 230, ["สายเส้นนี้กับสายของ", "ช่องที่ 1 ควรยาวเท่ากัน",
                           "ความต่างของสายคือ", "ความต่างของ Δt โดยตรง"],
           "#e03131", "#fff5f5")
    return p


PANELS = [panel1, panel2, panel3a, panel3b]


def main():
    for fn in PANELS:
        print("wrote", fn().write(OUT))


if __name__ == "__main__":
    main()
