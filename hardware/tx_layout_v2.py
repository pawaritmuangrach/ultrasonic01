"""Draw the TX driver panels for the right-hand breadboard.

Run:  python hardware/tx_layout_v2.py   ->  hardware/diagrams/txv2_*.svg

Mirror image of the RX board in one respect that matters: the power rail is on
the OUTER (right) edge, and the DevKit's right-hand header lands in column A.
The renderer in bb_panel.py handles that with rail="right", so these panels are
drawn the way the board actually sits on the base plate - rail on the right,
ESP32 off the left edge - rather than flipped to match the RX drawings.

Row numbering matches the RX board: row 60 is the end nearest the USB socket,
so the DevKit's 19 pins occupy rows 42..60 and the circuit lives in 1..41.

The driver is Version A: gate 1 buffers the ESP32 pin, gates 2/3/4 run in
parallel to share the current, gates 5/6 are unused and their inputs are tied
to ground - a floating CMOS input is not merely untidy, it biases the gate into
its linear region where it draws milliamps and oscillates.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_panel import Panel, TOP, C_POS, C_NEG, C_SIG          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")

CHIP = dict(rail="right", chip_row0=28, chip_name="74HCT04", width=740)
NX = 560          # left edge of the side notes

# The DevKit's right-hand header, read from the USB end (row 60) upwards.
ESP_RIGHT = {42: "GND", 43: "IO23", 44: "IO22", 45: "TX0", 46: "RX0", 47: "IO21",
             48: "GND", 49: "IO19", 50: "IO18", 51: "IO5", 52: "IO17", 53: "IO16",
             54: "IO4", 55: "IO0", 56: "IO2", 57: "IO15", 58: "SD1", 59: "SD0",
             60: "CLK"}


def feed_5v(p, row):
    """The 5 V wire does not come from this board's ESP32 header - it comes
    from the 5V pin, which is on the RX side. Drawn entering from the left
    because that is the way it physically crosses."""
    y = p.y[row]
    yb = p.band(row, False)
    p._path(f"M40,{y} L120,{y} L120,{yb} L{p.rp},{yb}", C_POS)
    p.b.append(f'<circle cx="{p.rp}" cy="{yb}" r="4.5" fill="{C_POS}"/>')
    p.b.append(f'<text x="34" y="{y+18}" font-size="11.5" font-weight="700" '
               f'fill="{C_POS}">5V มาจากบอร์ดซ้าย</text>')
    p.b.append(f'<text x="34" y="{y+33}" font-size="11" fill="#868e96">แถว 60 รู H</text>')
    p.b.append(f'<text x="34" y="{y+47}" font-size="11" fill="#868e96">(ขา 5V อยู่หัวต่อฝั่ง ADC)</text>')


def panel1():
    p = Panel("txv2_1_power_chip",
              "ผังภาคส่ง 1 — ไฟเลี้ยง 5V และตัวชิป",
              "74HCT04 แถว 28-34 · คาปาที่ราง แถว 24 และ 26 · ขาอินพุตที่ไม่ใช้ลงกราวด์",
              [24, 26, 28, 29, 30, 31, 32, 33, 34, 48], **CHIP)
    p.chip()
    p.esp_marker(48, "GND")
    feed_5v(p, 24)
    p.rail_cap(24, "100nF")
    p.rail_cap(26, "100µF", pol=True)
    p.to_rail("48B", "-")                       # ESP32 GND -> rail
    p.to_rail("28G", "+")                       # pin 14 VCC
    p.to_rail("34A", "-", lane=186)             # pin 7 GND
    p.tag("34A", "① ขา 7 ของชิป", -16, 5, "end", "#343a40")
    p.tag("48B", "② GND ของ ESP32", 108, 5, "start", "#343a40")
    p.to_rail("29G", "-")                       # pin 13, unused input
    p.to_rail("31G", "-")                       # pin 11, unused input
    p.note(NX, TOP - 42, ["ไฟเลี้ยง 5V ไม่ใช่ 3.3V", "74HCT04 ที่ 5V เท่านั้น",
                          "ที่ 3.3V จะขับไม่แรงพอ", "และ VIH ไม่ผ่าน"], w=170)
    p.note(NX, TOP + 110, ["ต้องเป็น HCT ไม่ใช่ HC", "HC ที่ 5V ต้องการ",
                           "VIH 3.5V แต่ ESP32", "ให้แค่ 3.3V"], "#e03131", "#fff5f5", w=170)
    p.note(NX, TOP + 250, ["ขา 11 กับ 13 คืออินพุต", "ของเกตที่ไม่ได้ใช้",
                           "ปล่อยลอย = ชิปกินกระแส", "หลาย mA และแกว่งเอง",
                           "รบกวนภาครับที่เกน 121"], "#1971c2", "#e7f5ff", w=170)
    p.note(NX, TOP + 400, ["กราวด์ของบอร์ดนี้", "ใช้ขา GND ของ ESP32",
                           "คนละขากับบอร์ด RX", "จุดรวมอยู่ในตัว DevKit"],
           "#2f9e44", "#ebfbee", w=170)
    return p


def panel2():
    p = Panel("txv2_2_gates",
              "ผังภาคส่ง 2 — สัญญาณเข้า และเกต 3 ตัวต่อขนาน",
              "GPIO18 → ขา 1 · ขา 2 ป้อนขา 3/5/9 · ขาออก 4/6/8 มัดรวมที่ขา 8",
              [28, 29, 30, 31, 32, 33, 34, 50], **CHIP)
    p.chip()
    p.esp_marker(50, "IO18")
    p.vlane("50B", "28A", 186)                  # GPIO18 -> pin 1
    p.wire("29A", "30A")                        # pin 2 -> pin 3
    p.vlane("29B", "32B", 221)                  # pin 2 -> pin 5
    p.wire("29C", "33H")                        # pin 2 -> pin 9   (over the chip)
    p.wire("31C", "34I")                        # pin 4 -> pin 8   (over the chip)
    p.wire("33D", "34J")                        # pin 6 -> pin 8   (over the chip)
    p.note(NX, TOP - 42, ["เกต 1 (ขา 1-2)", "เป็นบัฟเฟอร์รับจาก ESP32",
                          "เกต 2,3,4 ต่อขนาน", "เพื่อแบ่งกระแสกัน"], w=170)
    p.note(NX, TOP + 110, ["ขาออก 4, 6, 8 มัดรวมกัน", "ที่ขา 8 แล้วออกไปหา R",
                           "ทำได้เพราะทุกเกตอยู่ในไดเดียว", "จึงสลับพร้อมกันเป๊ะ"],
           "#2f9e44", "#ebfbee", w=170)
    p.note(NX, TOP + 250, ["สาย 3 เส้นพาดข้ามชิป", "เป็นเรื่องปกติ",
                           "อย่าพยายามอ้อม", "ยิ่งอ้อมยิ่งยาว"], "#1971c2", "#e7f5ff", w=170)
    return p


def panel3():
    p = Panel("txv2_3_output",
              "ผังภาคส่ง 3 — ตัวต้านทานอนุกรม และหัวส่ง",
              "R 47Ω สองตัวอนุกรม แถว 34-37 และ 37-40 · TCT40-16T ที่ 40I",
              [34, 37, 40], **CHIP)
    p.pin_marker(34, 8)
    p.part("34G", "37G", "47Ω", "R")
    p.part("37H", "40H", "47Ω", "R")
    p.transducer("40I", "TCT40-16T")
    p.note(NX, TOP - 42, ["ต้องอนุกรม ไม่ใช่ขนาน", "ขนานได้ 23.5Ω",
                          "กระแสพีค 130 mA", "เกินพิกัด 50 mA ของชิป",
                          "ถึง 2.6 เท่า"], "#e03131", "#fff5f5", w=170)
    p.note(NX, TOP + 130, ["47+47 = 94Ω กระแสพีค 46 mA", "แถว 37 คือจุดต่อระหว่าง",
                           "R สองตัว — 37G กับ 37H", "อยู่แถวเดียวฝั่งเดียวกัน",
                           "จึงถึงกันใต้บอร์ดอยู่แล้ว"], "#2f9e44", "#ebfbee", w=180)
    p.note(NX, TOP + 280, ["หัวส่งต้องเป็นตัว T", "ไม่ใช่ R",
                           "อ่านตัวอักษรบนกระป๋อง", "ไม่มีขั้ว สลับขาได้",
                           "ถ้ามี R 100Ω ตัวเดียว ใช้ 34G→40G"],
           "#1971c2", "#e7f5ff", w=180)
    return p


PANELS = [panel1, panel2, panel3]


def main():
    for fn in PANELS:
        print("wrote", fn().write(OUT))


if __name__ == "__main__":
    main()
