"""Draw the RX breadboard panels for the ESP32-straddling-two-boards layout.

Run:  python hardware/rx_layout_v2.py   ->  hardware/diagrams/rxv2_*.svg

Why this file exists rather than hand-drawn SVG: the netlist below is the only
copy of the wiring. Every panel renders from it, so a picture cannot show a
connection the netlist does not have, and moving a component is a one-line edit.
The drawing machinery lives in bb_panel.py, shared with the TX generator.

Row numbering: row 60 is the end of the board nearest the ESP32's USB socket,
row 1 the far end. The DevKit's 19 pins land on rows 42..60 in column J, which
leaves rows 1..41 for the circuit. Signal therefore flows from the far end
(transducer, row 5) down into the ESP32 (GPIO34, row 46). That direction is not
cosmetic: with the input at the near end the output would have to travel back
past it, and at a gain of 121 a few pF of stray coupling is positive feedback.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_panel import Panel, TOP, C_POS, C_NEG, C_SIG          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")

CHIP = dict(rail="left", chip_row0=12, chip_name="MCP6004")

# The DevKit's left-hand header, read from the USB end (row 60) upwards.
ESP = {42: "3V3", 43: "EN", 44: "SVP", 45: "SVN", 46: "IO34", 47: "IO35",
       48: "IO32", 49: "IO33", 50: "IO25", 51: "IO26", 52: "IO27", 53: "IO14",
       54: "IO12", 55: "GND", 56: "IO13", 57: "SD2", 58: "SD3", 59: "CMD",
       60: "5V"}


# ----------------------------------------------------------------- panels --

def panel1():
    p = Panel("rxv2_1_power_chip_vref",
              "ผังภาครับ 1 — ไฟเลี้ยง ชิป และ VREF",
              "MCP6004 แถว 12-18 · ตัวแบ่ง VREF แถว 9 · บัฟเฟอร์ส่งออกไปแถว 2 · บายพาสแถว 19-21",
              [2, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 42, 55], **CHIP)
    p.chip()
    p.esp_block(ESP)
    p.rail_cap(12, "10µF", note="", pol=True)
    p.rail_cap(13, "100nF", note="ฟิล์มหรือเซรามิกก็ได้")
    p.to_rail("42H", "+")
    p.to_rail("55H", "-")
    p.part_rail("9B", "+", "10k", "R", below=False)
    p.part_rail("9C", "-", "10k", "R", below=True)
    p.part_pol("9D", "11D", "10µF")
    p.to_rail("11C", "-")
    p.vlane("9E", "16G", 452)                     # VREF -> pin 10 (+INC)
    p.wire("17G", "18G")                          # pin 9 <-> pin 8 : follower
    p.route("18H", "2B", 470, 2, below=False)     # buffered VREF -> row 2
    p.to_rail("15B", "+", below=False)            # pin 4  VDD
    p.to_rail("15H", "-", lane=486, after=18)     # pin 11 VSS
    # Bypass: VDD is dragged down to row 19 and a local ground made at row 21
    # so the cap spans two rows (5.08 mm) - the lead pitch of a WIMA MKS2.
    # A cap straight across the chip would need 12.7 mm.
    p.vlane("15A", "19A", 221, C_POS)
    p.part("19D", "21D", "100nF", "C")
    p.to_rail("21C", "-")
    p.wire("12J", "15J", C_NEG)                   # pin 14 +IND -> GND
    p.wire("13G", "14G")                          # pin 13 -IND <-> pin 12 OUTD
    p.tag("2B", "โหนด VREF 1.65V", 96, 5, "start")
    p.note(516, TOP - 42, ["ราง + = 3.3V", "ราง – = GND", "อย่าลืมสะพานข้าม",
                           "ช่องว่างกลางบอร์ด", "ทั้งสองราง"])
    p.note(516, TOP + 130, ["คาปามีขั้ว (10µF)", "จุดแดง + = ขายาว",
                            "จุดดำ – = ขาสั้น ฝั่งที่มี", "แถบขาวคาดบนกระป๋อง",
                            "เสียบกลับ = ระเบิด"], "#e03131", "#fff5f5")
    p.note(516, TOP + 265, ["ออปแอมป์ที่ไม่ได้ใช้", "ขา 14 ลงกราวด์",
                            "ขา 13 ต่อขา 12", "ห้ามปล่อยลอย"], "#1971c2", "#e7f5ff")
    p.note(516, TOP + 385, ["บายพาสแถว 19–21", "ลาก VDD ลงมาที่ 19",
                            "ทำ GND ที่ 21", "คาปาจึงเว้น 2 แถว = 5.08 มม.",
                            "ฟิล์ม WIMA เสียบได้พอดี"], "#2f9e44", "#ebfbee")
    return p


def panel2():
    p = Panel("rxv2_2_input_stage1",
              "ผังภาครับ 2 — อินพุตและสเตจขยายที่ 1",
              "ทรานสดิวเซอร์แถว 5 · Cin แถว 5-7 · R ไบแอสแถว 7-2 · Rf1 / Rg1 / Cg1",
              [2, 5, 7, 12, 13, 14, 22, 25, 27], **CHIP)
    p.chip()
    p.transducer("5B", "TCT40-16R")
    p.part("5C", "7C", "100nF", "C")              # Cin
    p.part_lane("7D", "2D", 287, "10k", "R")      # R_bias -> VREF
    p.vlane("7A", "14A", 221)                     # input node -> pin 3
    p.part("12B", "22B", "100k", "R")             # Rf1
    p.vlane("13D", "22D", 287)                    # pin 2 -> Rf1 bottom
    p.part("22A", "25A", "10k", "R")              # Rg1
    p.part("25B", "27B", "1nF", "C")              # Cg1
    p.to_rail("27C", "-")
    p.tag("2D", "โหนด VREF", 60, 5, "start")
    p.note(516, TOP - 42, ["ขั้วทรานสดิวเซอร์", "สลับกันได้ ไม่มีบวกลบ",
                           "แต่ทั้ง 9 ช่องต้อง", "หันขาทางเดียวกัน"])
    p.note(516, TOP + 175, ["เกน = 1 + Rf/Rg", "100k / 10k = 11 เท่า",
                            "สองสเตจ = 121 เท่า"], "#2f9e44", "#ebfbee")
    p.note(516, TOP + 265, ["Cin ใช้ฟิล์ม WIMA", "ขา 5 มม. = 2 แถว พอดี",
                            "เซรามิก X7R เป็นพีโซฯ", "สั่นแล้วสร้างสัญญาณเอง",
                            "จุดนี้ขยาย 121 เท่า"], "#1971c2", "#e7f5ff")
    return p


def panel3a():
    p = Panel("rxv2_3a_stage2",
              "ผังภาครับ 3A — สเตจขยายที่ 2",
              "ขา 1 ป้อนเข้าขา 5 · Rf2 แถว 30-34 · Rg2 แถว 34-37 · Cg2 แถว 37-39",
              [12, 13, 14, 15, 16, 17, 18, 30, 34, 37, 39], **CHIP)
    p.chip()
    p.vlane("12C", "16C", 243)                    # OUTA -> +INB
    p.vlane("18B", "30A", 221)                    # OUTB -> stage-2 network
    p.part("30B", "34B", "100k", "R")             # Rf2
    p.vlane("17D", "34D", 265)                    # -INB -> Rf2 bottom
    p.part("34A", "37A", "10k", "R")              # Rg2
    p.part("37B", "39B", "1nF", "C")              # Cg2
    p.to_rail("39C", "-")
    p.note(516, TOP - 42, ["สเตจ 2 เหมือนสเตจ 1", "ทุกประการ",
                           "ต่างแค่ใช้ขา 5-6-7", "แทนขา 1-2-3"])
    p.note(516, TOP + 200, ["เกน DC ของทั้งสองสเตจ", "เป็น 1 เพราะคาปากั้นไว้",
                            "จึงต่อตรงถึงกันได้", "ไม่ต้องมีคาปาคั่นกลาง"],
           "#2f9e44", "#ebfbee")
    return p


def panel3b():
    p = Panel("rxv2_3b_output",
              "ผังภาครับ 3B — ภาคขาออกและสายไป ESP32",
              "R 2.2k แถว 30-34 ฝั่งขวา · C 1nF แถว 34-36 · ออกที่ 34I ไป GPIO34 แถว 46",
              [18, 30, 34, 36, 42, 43, 44, 45, 46], **CHIP)
    p.pin_marker(18, 7)
    p.esp_block(ESP)
    p.route("18D", "30G", 265, 18)                # OUTB -> output side
    p.part("30H", "34H", "2.2k", "R")             # R_out
    p.part("34G", "36G", "1nF", "C")              # C_out
    p.to_rail("36F", "-", lane=320, after=36)
    p.vlane("34I", "46H", 397, "#f08c00")         # -> GPIO34
    p.tag("44H", "ไป GPIO34", -16, 4, "end")
    p.note(516, TOP - 42, ["R 2.2k + C 1nF", "คือตัวกรองกันเอเลียส",
                           "และทำให้ ADC ที่สลับ", "ช่องอ่านค่าได้แม่น"])
    p.note(516, TOP + 150, ["สายเส้นนี้ให้สั้นที่สุด", "เท่าที่ทำได้",
                            "เป็นโหนดที่ไวที่สุด", "ของทั้งวงจร"], "#f08c00", "#fff4e6")
    return p


PANELS = [panel1, panel2, panel3a, panel3b]


def main():
    for fn in PANELS:
        print("wrote", fn().write(OUT))


if __name__ == "__main__":
    main()
