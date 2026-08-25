"""Draw the top view showing which transducer goes in which hole.

Reads plate_coords.csv - the same file plate_model.py writes and the analysis
code reads - so the picture cannot drift away from the printed part.

Run:  python hardware/plate_sensor_map.py
"""

import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams", "plate_5_sensor_map.svg")

BOSS_OD, BORE = 23.0, 19.5
LEG_W, HUB_R, ARM_R1 = 28.0, 34.0, 170.0
ARM_ANGLES = (90.0, 210.0, 330.0)
JOINT_R0, JOINT_R1 = 53.0, 68.0
SLOT_R, SLOT_L, SLOT_W = 110.0, 20.0, 3.0
MOUNT_D, CENTER_D = 4.5, 6.5

S = 2.0                       # px per mm
X0, X1, Y0, Y1 = -152, 152, -96, 178
MARGIN = 14

W = (X1 - X0) * S + 2 * MARGIN
H = (Y1 - Y0) * S + 2 * MARGIN


def px(x, y):
    return (MARGIN + (x - X0) * S, MARGIN + (Y1 - y) * S)


def polar(r, deg):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


o = []
o.append('<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
         'viewBox="0 0 %.0f %.0f" font-family="Tahoma, Segoe UI, sans-serif">'
         % (W, H, W, H))
o.append('<rect width="%.0f" height="%.0f" fill="#ffffff"/>' % (W, H))

# ---- plate body -----------------------------------------------------------
for a in ARM_ANGLES:
    x0, y0 = px(*polar(0, a))
    x1, y1 = px(*polar(ARM_R1 - LEG_W / 2, a))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ced4da" '
             'stroke-width="%.1f" stroke-linecap="round"/>'
             % (x0, y0, x1, y1, LEG_W * S + 3))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#f1f3f5" '
             'stroke-width="%.1f" stroke-linecap="round"/>'
             % (x0, y0, x1, y1, LEG_W * S))
cx, cy = px(0, 0)
o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#f1f3f5" stroke="#ced4da" '
         'stroke-width="1.5"/>' % (cx, cy, HUB_R * S))

# ---- joint seam and acoustic slot ----------------------------------------
o.append('<g stroke="#868e96" stroke-width="1.2" stroke-dasharray="5 4" fill="none">')
for a in ARM_ANGLES:
    dx, dy = polar(1, a + 90)
    for r in (JOINT_R0, JOINT_R1):
        x, y = polar(r, a)
        p1 = px(x + dx * LEG_W / 2, y + dy * LEG_W / 2)
        p2 = px(x - dx * LEG_W / 2, y - dy * LEG_W / 2)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                 % (p1[0], p1[1], p2[0], p2[1]))
o.append('</g>')
for a in ARM_ANGLES:
    x, y = px(*polar(SLOT_R, a))
    o.append('<g transform="translate(%.1f,%.1f) rotate(%.1f)">'
             '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" '
             'fill="#ffffff" stroke="#868e96"/></g>'
             % (x, y, -(a - 90), -SLOT_L * S / 2, -SLOT_W * S / 2,
                SLOT_L * S, SLOT_W * S, SLOT_W * S / 2))

o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#ffffff" stroke="#868e96"/>'
         % (cx, cy, CENTER_D / 2 * S))

# ---- stage-1 pair: drawn before the bosses so it passes behind them --------
p1, p2 = px(17.32, 10.0), px(0.0, 140.0)
o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#f08c00" '
         'stroke-width="2.5" stroke-dasharray="7 5"/>' % (p1[0], p1[1], p2[0], p2[1]))

# ---- transducer positions -------------------------------------------------
STYLE = {"TX": ("#e03131", "#ffe3e3", "T"), "RX": ("#1971c2", "#d0ebff", "R")}
with open(os.path.join(HERE, "plate_coords.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    x, y = float(row["x_mm"]), float(row["y_mm"])
    sx, sy = px(x, y)
    if row["role"] == "MOUNT":
        o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#ffffff" '
                 'stroke="#2f9e44" stroke-width="1.5"/>' % (sx, sy, MOUNT_D / 2 * S))
        continue
    stroke, fill, kind = STYLE[row["role"]]
    o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" '
             'stroke-width="2"/>' % (sx, sy, BOSS_OD / 2 * S, fill, stroke))
    o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#ffffff" stroke="%s" '
             'stroke-width="1"/>' % (sx, sy, BORE / 2 * S, stroke))
    o.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="12" '
             'font-weight="bold" fill="%s">%s</text>'
             % (sx, sy - 1, stroke, row["id"]))
    o.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="11" '
             'font-weight="bold" fill="%s">%s</text>' % (sx, sy + 11, stroke, kind))

# ---- +Y marker: the free corridor at x=0 is only 11.6 mm wide, so the arrow
# goes there and the wording is parked in the empty upper-left with a leader --
ax, ay = px(0, -6)
bx, by = px(0, 26)
o.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" '
         'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#212529"/></marker></defs>')
o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#212529" '
         'stroke-width="2" marker-end="url(#ar)"/>' % (ax, ay, bx, by))
o.append('<polyline points="246,306 290,306 %.1f,%.1f" fill="none" stroke="#868e96" '
         'stroke-width="1.2"/>' % (bx - 6, by + 4))
o.append('<text x="240" y="300" text-anchor="end" font-size="12" font-weight="bold" '
         'fill="#212529">+Y = ทิศหน้าของอาเรย์</text>')
o.append('<text x="240" y="317" text-anchor="end" font-size="11" fill="#495057">'
         'บนดุมจริงมีอักษร +Y จมไว้</text>')

# ---- stage-1 callout ------------------------------------------------------
mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
o.append('<rect x="%.1f" y="%.1f" width="188" height="26" rx="5" fill="#fff9db" '
         'stroke="#f08c00"/>' % (mx + 34, my - 14))
o.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="bold" fill="#e67700">'
         'คู่ที่ต่อสายตอนนี้ ห่างกัน 131 มม.</text>' % (mx + 42, my + 4))

# ---- legend ---------------------------------------------------------------
lx, ly = px(-152, 178)
o.append('<rect x="%.1f" y="%.1f" width="265" height="102" rx="6" fill="#ffffff" '
         'stroke="#ced4da"/>' % (lx, ly))
o.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="bold" fill="#212529">'
         'ใส่กระป๋องแบบไหนลงรูไหน</text>' % (lx + 12, ly + 21))
for i, (st, fi, k, txt) in enumerate([
        ("#e03131", "#ffe3e3", "T", "กระป๋องพิมพ์ว่า T = ตัวส่ง  3 ตัว"),
        ("#1971c2", "#d0ebff", "R", "กระป๋องพิมพ์ว่า R = ตัวรับ  9 ตัว"),
        ("#2f9e44", "#ffffff", "", "รูยึด ⌀4.5 + จุดอ้างอิงกล้อง  3 รู")]):
    yy = ly + 43 + i * 22
    r = 9 if k else 4.5
    o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" stroke-width="2"/>'
             % (lx + 24, yy - 4, r, fi, st))
    if k:
        o.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="10" '
                 'font-weight="bold" fill="%s">%s</text>' % (lx + 24, yy, st, k))
    o.append('<text x="%.1f" y="%.1f" font-size="11" fill="#212529">%s</text>'
             % (lx + 42, yy, txt))

# ---- note box -------------------------------------------------------------
nx, ny = px(16, 178)
o.append('<rect x="%.1f" y="%.1f" width="262" height="86" rx="6" fill="#fff4e6" '
         'stroke="#f08c00"/>' % (nx, ny))
o.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="bold" fill="#e67700">'
         '3 ข้อที่ห้ามพลาดตอนใส่</text>' % (nx + 12, ny + 21))
for i, t in enumerate([
        "หน้ากระป๋องหันขึ้น สายออกทางใต้แผ่น",
        "หน้ากระป๋องเสมอปากบอสทุกตัว 1 มม. = 2.9 µs",
        "หมุนขาทั้ง 12 ตัวให้เรียงทิศเดียวกัน"]):
    o.append('<text x="%.1f" y="%.1f" font-size="11" fill="#212529">%d. %s</text>'
             % (nx + 12, ny + 42 + i * 18, i + 1, t))

o.append('</svg>')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(o))
print("wrote %s  (%.0f x %.0f px)" % (OUT, W, H))
