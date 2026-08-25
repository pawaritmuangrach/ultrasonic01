"""Draw the ESP32 pin map across BOTH breadboards, rows 42-60.

Run:  python hardware/rig_2_pinmap.py  ->  diagrams/rig_2_esp32_pinmap.svg

The per-board panels each show one board, so a wire that starts on the other
board can only be described in words - which is exactly where "แถว 60 รู H"
becomes ambiguous, because row 60 exists on both boards and means 5V on one
and CLK (an internal flash line) on the other. This drawing puts both boards
and the module between them on one page so every hole that touches the ESP32
can be pointed at.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams", "rig_2_esp32_pinmap.svg")

ROWS = list(range(42, 61))
PITCH, TOP, W = 30, 132, 960

LCOL = {"F": 150, "G": 172, "H": 194, "I": 216, "J": 238}
RCOL = {"A": 386, "B": 408, "C": 430, "D": 452, "E": 474}
LRP, LRN = 70, 92                 # left board rails  (+ outer)
RRN, RRP = 522, 544               # right board rails (+ outer)
ESP_L, ESP_R = 238, 386

# name, category:  use / danger / plain
LEFT = {42: ("3V3", "use"), 43: ("EN", "plain"), 44: ("SVP", "plain"),
        45: ("SVN", "plain"), 46: ("IO34", "use"), 47: ("IO35", "plain"),
        48: ("IO32", "plain"), 49: ("IO33", "plain"), 50: ("IO25", "plain"),
        51: ("IO26", "plain"), 52: ("IO27", "plain"), 53: ("IO14", "plain"),
        54: ("IO12", "plain"), 55: ("GND", "use"), 56: ("IO13", "plain"),
        57: ("SD2", "danger"), 58: ("SD3", "danger"), 59: ("CMD", "danger"),
        60: ("5V", "use")}
RIGHT = {42: ("GND", "plain"), 43: ("IO23", "plain"), 44: ("IO22", "plain"),
         45: ("TX0", "danger"), 46: ("RX0", "danger"), 47: ("IO21", "plain"),
         48: ("GND", "use"), 49: ("IO19", "plain"), 50: ("IO18", "use"),
         51: ("IO5", "plain"), 52: ("IO17", "plain"), 53: ("IO16", "plain"),
         54: ("IO4", "plain"), 55: ("IO0", "danger"), 56: ("IO2", "danger"),
         57: ("IO15", "danger"), 58: ("SD1", "danger"), 59: ("SD0", "danger"),
         60: ("CLK", "danger")}

C_USE, C_DANGER, C_PLAIN = "#2f9e44", "#e03131", "#adb5bd"
C_POS, C_NEG, C_SIG = "#e03131", "#343a40", "#2f9e44"

y = {r: TOP + i * PITCH for i, r in enumerate(ROWS)}
BOT = y[60] + 34
o = []


def add(s):
    o.append(s)


def wire(d, colour, w=3):
    add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{w}" '
        f'stroke-linejoin="round" stroke-linecap="round"/>')


H = BOT + 96
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="Tahoma, Segoe UI, sans-serif" fill="#343a40">')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
add('<text x="20" y="30" font-size="18" font-weight="700" fill="#1971c2">'
    'ขา ESP32 ทั้ง 19 คู่ — รูไหนอยู่บอร์ดไหน</text>')
add('<text x="20" y="52" font-size="12.5" fill="#6c757d">'
    'แถวเดียวกันมีอยู่บนบอร์ดทั้งสองอัน และหมายถึงคนละขา — '
    '"แถว 60" ฝั่งซ้ายคือ 5V ฝั่งขวาคือ CLK</text>')

# ---------------------------------------------------------------- boards --
for x0, wdt, lab in ((60, 240, "บอร์ดซ้าย — RX"), (376, 240, "บอร์ดขวา — TX")):
    add(f'<rect x="{x0}" y="{TOP-38}" width="{wdt}" height="{BOT-TOP+38}" rx="6" '
        f'fill="#ffffff" stroke="#ced4da"/>')
    add(f'<text x="{x0+wdt/2}" y="{TOP-48}" font-size="13" font-weight="700" '
        f'text-anchor="middle" fill="#495057">{lab}</text>')
for x, c in ((LRP, "#ffc9c9"), (LRN, "#a5d8ff"), (RRN, "#a5d8ff"), (RRP, "#ffc9c9")):
    add(f'<rect x="{x-8}" y="{TOP-24}" width="16" height="{BOT-TOP+18}" rx="8" fill="{c}"/>')
for x, t, c in ((LRP, "+", C_POS), (LRN, "–", "#1971c2"),
                (RRN, "–", "#1971c2"), (RRP, "+", C_POS)):
    add(f'<text x="{x}" y="{TOP-30}" font-size="14" font-weight="700" '
        f'text-anchor="middle" fill="{c}">{t}</text>')
for name, x in list(LCOL.items()) + list(RCOL.items()):
    add(f'<text x="{x}" y="{TOP-30}" font-size="12" text-anchor="middle" '
        f'fill="#868e96">{name}</text>')

# ------------------------------------------------------------- ESP32 body --
add(f'<rect x="224" y="{TOP-26}" width="176" height="{BOT-TOP+22}" rx="7" fill="#343a40"/>')
add(f'<text x="312" y="{TOP-40}" font-size="13" font-weight="700" '
    f'text-anchor="middle" fill="#343a40">ESP32 DevKit</text>')
add(f'<text x="312" y="{BOT+18}" font-size="11.5" text-anchor="middle" '
    f'fill="#868e96">▼ ปลายนี้คือช่อง USB</text>')

# ------------------------------------------------------------------ rows --
CAT = {"use": C_USE, "danger": C_DANGER, "plain": C_PLAIN}
for r in ROWS:
    yy = y[r]
    add(f'<text x="44" y="{yy+4}" font-size="12.5" text-anchor="end" fill="#6c757d">{r}</text>')
    for x in list(LCOL.values())[:-1] + list(RCOL.values())[1:]:
        add(f'<circle cx="{x}" cy="{yy}" r="3.2" fill="#ced4da"/>')
    for x in (LRP, LRN, RRN, RRP):
        add(f'<circle cx="{x}" cy="{yy}" r="3.2" fill="#ffffff"/>')
    ln, lc = LEFT[r]
    rn, rc = RIGHT[r]
    add(f'<circle cx="{ESP_L}" cy="{yy}" r="5.5" fill="{CAT[lc]}"/>')
    add(f'<circle cx="{ESP_R}" cy="{yy}" r="5.5" fill="{CAT[rc]}"/>')
    add(f'<text x="250" y="{yy+4}" font-size="11" fill="{"#ffffff" if lc=="plain" else CAT[lc]}" '
        f'font-weight="{700 if lc!="plain" else 400}">{ln}</text>')
    add(f'<text x="374" y="{yy+4}" font-size="11" text-anchor="end" '
        f'fill="{"#ffffff" if rc=="plain" else CAT[rc]}" '
        f'font-weight="{700 if rc!="plain" else 400}">{rn}</text>')

# ------------------------------------------------------------- the wires --
wire(f"M{LCOL['H']},{y[42]} L{LCOL['H']},{y[42]-16} L{LRP},{y[42]-16}", C_POS)
add(f'<circle cx="{LRP}" cy="{y[42]-16}" r="5" fill="{C_POS}"/>')
add(f'<circle cx="{LCOL["H"]}" cy="{y[42]}" r="5" fill="{C_POS}"/>')

wire(f"M{LCOL['H']},{y[55]} L{LCOL['H']},{y[55]-15} L{LRN},{y[55]-15}", C_NEG)
add(f'<circle cx="{LRN}" cy="{y[55]-15}" r="5" fill="{C_NEG}"/>')
add(f'<circle cx="{LCOL["H"]}" cy="{y[55]}" r="5" fill="{C_NEG}"/>')

# the 5 V wire: the only one that crosses between the boards
wire(f"M{LCOL['H']},{y[60]} L{LCOL['H']},{BOT-8} L{RRP},{BOT-8} L{RRP},{y[60]}", C_POS, 3.6)
add(f'<circle cx="{LCOL["H"]}" cy="{y[60]}" r="6" fill="{C_POS}"/>')
add(f'<circle cx="{RRP}" cy="{y[60]}" r="6" fill="{C_POS}"/>')

wire(f"M{RCOL['B']},{y[48]} L{RCOL['B']},{y[48]-15} L{RRN},{y[48]-15}", C_NEG)
add(f'<circle cx="{RRN}" cy="{y[48]-15}" r="5" fill="{C_NEG}"/>')
add(f'<circle cx="{RCOL["B"]}" cy="{y[48]}" r="5" fill="{C_NEG}"/>')

wire(f"M{RCOL['B']},{y[50]} L{RCOL['E']},{y[50]}", C_SIG)
add(f'<circle cx="{RCOL["B"]}" cy="{y[50]}" r="5" fill="{C_SIG}"/>')
add(f'<text x="{RCOL["C"]+6}" y="{y[50]-13}" font-size="11" font-weight="700" '
    f'text-anchor="middle" fill="{C_SIG}">ไปแถว 28 รู A</text>')

wire(f"M{LCOL['H']},{y[46]} L{LCOL['F']},{y[46]}", "#f08c00")
add(f'<circle cx="{LCOL["H"]}" cy="{y[46]}" r="5" fill="#f08c00"/>')
add(f'<text x="{LCOL["H"]-6}" y="{y[46]-13}" font-size="11" font-weight="700" '
    f'text-anchor="middle" fill="#f08c00">มาจากแถว 34 รู I</text>')

# ------------------------------------------------------------ call-outs --
add(f'<rect x="{LCOL["F"]-26}" y="{y[60]-15}" width="{LCOL["J"]-LCOL["F"]+40}" '
    f'height="30" rx="6" fill="none" stroke="#f08c00" stroke-width="2.5"/>')
add(f'<rect x="{RCOL["A"]-24}" y="{y[60]-15}" width="{RCOL["E"]-RCOL["A"]+44}" '
    f'height="30" rx="6" fill="none" stroke="#e03131" stroke-width="2.5" '
    f'stroke-dasharray="6 4"/>')

NX = 640
def note(ytop, title, lines, colour, bg):
    add(f'<rect x="{NX}" y="{ytop}" width="300" height="{26+19*len(lines)}" rx="7" '
        f'fill="{bg}" stroke="{colour}"/>')
    add(f'<text x="{NX+14}" y="{ytop+23}" font-size="13" font-weight="700" '
        f'fill="#212529">{title}</text>')
    for i, t in enumerate(lines):
        add(f'<text x="{NX+14}" y="{ytop+44+i*19}" font-size="12" fill="#212529">{t}</text>')

note(TOP - 44, "รูที่ต้องต่อ ทั้งหมด 6 รู", [
    "บอร์ดซ้าย  42H → ราง +      (3V3)",
    "บอร์ดซ้าย  55H → ราง –      (GND)",
    "บอร์ดซ้าย  46H ← แถว 34 รู I (GPIO34)",
    "บอร์ดซ้าย  60H → ราง + ของบอร์ดขวา  (5V)",
    "บอร์ดขวา   48B → ราง –      (GND)",
    "บอร์ดขวา   50B → แถว 28 รู A (IO18)"], C_USE, "#ebfbee")

note(TOP + 190, "จุดสีแดง = ห้ามต่ออะไรทั้งนั้น", [
    "ฝั่งซ้าย  SD2 SD3 CMD  (แถว 57-59)",
    "ฝั่งขวา  TX0 RX0  (แถว 45-46)",
    "ฝั่งขวา  IO0 IO2 IO15 SD1 SD0 CLK",
    "            (แถว 55-60)",
    "ทั้งหมดนี้คือสายแฟลชในตัวชิป สายอัปโหลด",
    "และขาที่ชิปอ่านตอนบูต แตะเมื่อไหร่ชิปเงียบ"], C_DANGER, "#fff5f5")

note(TOP + 380, "จุดที่สับสนง่ายที่สุด", [
    "แถว 60 มีอยู่บนบอร์ดทั้งสองอัน",
    "",
    "ซ้าย  แถว 60 = 5V   ← เอาไฟจากตรงนี้",
    "ขวา   แถว 60 = CLK  ← สายแฟลช ห้ามแตะ",
    "",
    "สาย 5V เริ่มที่บอร์ดซ้าย ไปจบที่รางแดง",
    "ของบอร์ดขวา — เป็นสายเส้นเดียวที่ข้ามบอร์ด"], "#f08c00", "#fff4e6")

add('</svg>')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(o))
print("wrote", os.path.basename(OUT), f"({W} x {H})")
