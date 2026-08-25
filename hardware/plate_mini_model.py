"""แผงทดลองแบบเล็ก 150x150 มม. — RX 4 ตัวที่มุมทั้งสี่ + TX 1 ตัวตรงกลาง

Run:  python hardware/plate_mini_model.py
      -> cad/plate_mini.step / .stl / plate_mini_preview.png / plate_mini_coords.csv

ทำไมมีแผงนี้ทั้งที่มี plate_model.py (3 TX + 9 RX) อยู่แล้ว:
  แผงเต็มต้องบัดกรี 12 ช่อง ซึ่งกินเวลาหลายวันก่อนจะได้ลองอะไรเลย แผงนี้ใช้ 5 ตัว
  (เท่าที่วงจรบนไข่ปลารองรับตอนนี้: RX 4 ช่อง + TX 1 ตัว) จึงลองของจริงได้ทันที
  พิมพ์ชิ้นเดียว ไม่ต้องแบ่งชิ้นเหมือนแผงเต็ม เพราะ 150 มม. ลงเตียงพิมพ์ทั่วไปได้สบาย

**ตำแหน่ง RX เลือกไว้ให้ใช้ rules.py ที่เทรนไว้แล้วได้ต่อทันที**
  กฎที่ได้จากชุด walk ใช้ log4 = ln((ซ้ายบน+ซ้ายล่าง+1)/(ขวาล่าง+ขวาบน+1))
  ซึ่งเป็นการเทียบ "ผลรวมฝั่งซ้าย กับ ผลรวมฝั่งขวา" ตราบใดที่ยังจับคู่แบบนี้
  สูตรเดิมใช้ได้เลย ไม่ต้องเทรนใหม่ (สเกลอาจต่างเพราะเบสไลน์เปลี่ยน — ฟิตใหม่ 2 ค่าพอ)

**ของแถมจากการวาง RX เป็นสี่เหลี่ยม**: ได้ทิศ *แนวตั้ง* มาด้วยฟรี ๆ
  ln((บนซ้าย+บนขวา+1)/(ล่างซ้าย+ล่างขวา+1)) บอกว่าเป้าอยู่สูงหรือต่ำกว่าแนวแผง
  ซึ่งอาเรย์เส้นตรงเดิมบอกไม่ได้เลย

ระยะห่าง RX (เบสไลน์) = 110 มม. ทั้งแนวนอนและแนวตั้ง
  ของเดิมบนบอร์ดทดลองสั้นกว่านี้มาก เบสไลน์ยาวขึ้น = ความละเอียดเชิงมุมดีขึ้น
  (ความคลาดเคลื่อนเชิงมุมแปรผกผันกับเบสไลน์)

รูเซ็นเซอร์โตขึ้น 1 มม. จากแบบเดิม (19.5 -> 20.5) ตามที่ผู้ใช้ขอ เพราะของเดิมฝืดตอนใส่
โอริง ยังใช้โอริงเบอร์เดิม (ID 15 หน้าตัด 2.0 มม.) ได้ แค่บีบน้อยลง — ซึ่งดีกว่า
เพราะโอริงที่ถูกอัดจนแบนจะนำเสียงได้มากขึ้น ผิดวัตถุประสงค์ของมัน

เขียนด้วย build123d โหมด algebra เหมือน plate_model.py (โหมด builder หาบริบทจาก
call stack ทำให้ย้ายโค้ดเข้าฟังก์ชันย่อยแล้วพัง)
"""

import csv
import math
import os
import struct

import numpy as np
from build123d import *

# ----------------------------------------------------------------- config --

SIZE = 150.0       # แผ่นสี่เหลี่ยมจัตุรัส ด้านละ 150 มม. (15 cm)
T = 4.0            # ความหนาแผ่น
CORNER_R = 12.0    # ลบมุมแผ่น กันบาดมือและกันมุมแหลมหลุดตอนพิมพ์

BORE = 20.5        # รูเซ็นเซอร์: ของเดิม 19.5 + 1.0 ตามที่ขอ
BOSS_OD = 24.0     # ผนังรอบรูหนา 1.75 มม. เท่าแบบเดิม (4 เส้น x 0.4 = 1.6 มม.)
BOSS_H = 10.0      # ความสูงบอส — โอริงสองวงซ้อนกันอยู่ข้างใน

RX_OFF = 55.0      # ระยะจากกึ่งกลางถึงศูนย์กลางรู RX ตามแกน x และ y
# ตรวจระยะขอบ: 55 + BOSS_OD/2 = 67 มม. เทียบกับครึ่งความกว้าง 75 -> เหลือเนื้อ 8 มม.

# ร่องตัดเสียงรอบ TX — บังคับให้เสียงที่วิ่งในเนื้อพลาสติกต้องอ้อมปลายร่อง
# วางร่องไว้ "ขวางทาง" TX->RX พอดี (ที่มุม 45/135/225/315) และเว้นสะพานไว้ที่
# 0/90/180/270 ซึ่งไม่มี RX อยู่ จึงไม่มีเส้นทางตรงเส้นไหนที่ไม่โดนร่องคั่น
SLOT_R, SLOT_L, SLOT_W = 33.0, 30.0, 3.0

MOUNT_R, MOUNT_D = 62.0, 4.5      # รูยึด 4 รูกลางด้าน — ใช้เป็นจุดอ้างอิงกล้องได้ด้วย
TIE_D, TIE_GAP, TIE_OFF = 3.5, 9.0, 8.0   # รูเคเบิลไทร์ คู่ละ 2 รู

LABEL_H, LABEL_DEPTH = 5.0, 0.6

# ตำแหน่ง RX: จับคู่ให้ (34,33) = ฝั่งซ้าย และ (32,35) = ฝั่งขวา
# เพื่อให้สูตร log4 ใน car/rules.py ใช้ได้โดยไม่ต้องแก้โค้ด
RX = (("RX_TL", -RX_OFF, +RX_OFF, 34, "ซ้ายบน"),
      ("RX_BL", -RX_OFF, -RX_OFF, 33, "ซ้ายล่าง"),
      ("RX_BR", +RX_OFF, -RX_OFF, 32, "ขวาล่าง"),
      ("RX_TR", +RX_OFF, +RX_OFF, 35, "ขวาบน"))
TX = ("TX1", 0.0, 0.0, 18, "กลาง")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cad")


# --------------------------------------------------------------- helpers --

def polar(r, deg):
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def boss(x, y):
    return Pos(x, y, T) * extrude(Circle(BOSS_OD / 2), BOSS_H)


def bore(x, y):
    return Pos(x, y, -0.5) * extrude(Circle(BORE / 2), T + BOSS_H + 1)


def hole(x, y, d):
    return Pos(x, y, -0.5) * extrude(Circle(d / 2), T + 1)


def slot(deg):
    """ร่องตัดเสียง วางขวางแนวรัศมีที่มุม deg"""
    x, y = polar(SLOT_R, deg)
    return (Pos(x, y, -0.5) * Rot(0, 0, deg)
            * extrude(RectangleRounded(SLOT_W, SLOT_L, SLOT_W / 2 - 0.01), T + 1))


def label(txt, x, y, deg=0.0):
    """ตัวอักษรจมบนผิวบน — ตัวนูนจะถูกหัวพิมพ์ปาดจนเสียรูป"""
    return (Pos(x, y, T) * Rot(0, 0, deg)
            * extrude(Text(txt, font_size=LABEL_H), -LABEL_DEPTH))


def tie_pair(x, y, deg):
    """รูเคเบิลไทร์สองรู อยู่ด้านในของบอส เพื่อรัดสายไม่ให้ดึงตัวเซ็นเซอร์"""
    r = BOSS_OD / 2 + TIE_OFF
    cx, cy = x + r * math.cos(math.radians(deg)), y + r * math.sin(math.radians(deg))
    tx, ty = -math.sin(math.radians(deg)), math.cos(math.radians(deg))
    return (hole(cx + tx * TIE_GAP / 2, cy + ty * TIE_GAP / 2, TIE_D)
            + hole(cx - tx * TIE_GAP / 2, cy - ty * TIE_GAP / 2, TIE_D))


# ----------------------------------------------------------------- build --

def build():
    part = extrude(RectangleRounded(SIZE, SIZE, CORNER_R), T)

    for _id, x, y, _gp, _th in RX:
        part += boss(x, y)
    part += boss(TX[1], TX[2])

    # เจาะรูเซ็นเซอร์ **หลังจาก** ก่อบอสครบแล้ว ถ้าเจาะก่อนบอสตัวถัดไปจะไปอุดรูเดิม
    for _id, x, y, _gp, _th in RX:
        part -= bore(x, y)
    part -= bore(TX[1], TX[2])

    for deg in (45.0, 135.0, 225.0, 315.0):
        part -= slot(deg)

    for deg in (0.0, 90.0, 180.0, 270.0):
        mx, my = polar(MOUNT_R, deg)
        part -= hole(mx, my, MOUNT_D)

    # รูเคเบิลไทร์: RX หันเข้าหากึ่งกลาง · TX ชี้ลงล่างซึ่งเป็นที่ว่าง
    for _id, x, y, _gp, _th in RX:
        part -= tie_pair(x, y, math.degrees(math.atan2(-y, -x)))
    part -= tie_pair(TX[1], TX[2], 270.0)

    for _id, x, y, gp, _th in RX:
        # ป้ายอยู่ด้านในของบอสเสมอ จึงไม่ตกขอบแผ่น
        lx = x - math.copysign(BOSS_OD / 2 + 12.0, x)
        part -= label(f"{_id[3:]}\nIO{gp}", lx, y - LABEL_H, 0.0)
    part -= label(f"TX\nIO{TX[3]}", TX[1] + BOSS_OD / 2 + 4, TX[2] - LABEL_H)
    part -= label("+Y", 0, SIZE / 2 - 6)      # เหนือรูยึด M90 ที่ y=62
    return part


def write_coords():
    path = os.path.join(HERE, "plate_mini_coords.csv")
    rows = [{"id": i, "x_mm": f"{x:.2f}", "y_mm": f"{y:.2f}", "z_mm": f"{T + BOSS_H:.2f}",
             "role": r, "gpio": g, "note": n}
            for i, x, y, g, n, r in
            [(*t, "RX") for t in RX] + [(*TX, "TX")]]
    for deg in (0.0, 90.0, 180.0, 270.0):
        mx, my = polar(MOUNT_R, deg)
        rows.append({"id": f"M{int(deg)}", "x_mm": f"{mx:.2f}", "y_mm": f"{my:.2f}",
                     "z_mm": "0.00", "role": "MOUNT", "gpio": "",
                     "note": f"รูยึด ⌀{MOUNT_D}"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "x_mm", "y_mm", "z_mm", "role",
                                          "gpio", "note"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def render_preview(part, path):
    """มองจากด้านบน วาดจากไฟล์ STL ที่จะพิมพ์จริง ภาพจึงขัดกับไฟล์ไม่ได้"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    with open(os.path.join(OUT, "plate_mini.stl"), "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tris = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    z = tris[:, :, 2].mean(axis=1)
    order = np.argsort(z)
    tris, z = tris[order], z[order]

    fig, ax = plt.subplots(figsize=(8.4, 8), dpi=110)
    shade = plt.cm.Blues(0.25 + 0.7 * (z - z.min()) / (np.ptp(z) + 1e-9))
    ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=shade,
                                     edgecolors="none"))
    colour = {"TX": "#e03131", "RX": "#1971c2", "MOUNT": "#2f9e44"}
    with open(os.path.join(HERE, "plate_mini_coords.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x, y = float(row["x_mm"]), float(row["y_mm"])
            c = colour[row["role"]]
            ax.plot(x, y, "+", color=c, ms=11, mew=1.8)
            tag = row["id"] + (f"  IO{row['gpio']}" if row["gpio"] else "")
            ax.annotate(tag, (x, y), textcoords="offset points", xytext=(11, 7),
                        fontsize=8.5, color=c, weight="bold")
    bb = part.bounding_box()
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"plate mini - top view   {bb.size.X:.0f} x {bb.size.Y:.0f} mm   "
                 f"bore ⌀{BORE}")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    part = build()
    export_step(part, os.path.join(OUT, "plate_mini.step"))
    export_stl(part, os.path.join(OUT, "plate_mini.stl"))
    bb = part.bounding_box()
    v = part.volume / 1000
    # ปริมาตรเนื้อตัน — ของจริงพิมพ์ด้วยผนัง 4 เส้น + อินฟิล 25% ใช้เส้นราว 45-55%
    print(f"plate_mini  {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} มม."
          f"   เนื้อตัน {v:.1f} cm3"
          f"   (พิมพ์จริง ~{v * 0.5 * 1.27:.0f} g PETG ที่อินฟิล 25%)")
    print(f"รูเซ็นเซอร์ ⌀{BORE} มม. (ของเดิม 19.5 + 1.0) · บอส ⌀{BOSS_OD} สูง {BOSS_H}")
    print(f"เบสไลน์ RX {2 * RX_OFF:.0f} มม. ทั้งแนวนอนและแนวตั้ง")
    print(f"เขียน plate_mini_coords.csv ({write_coords()} จุด)")
    render_preview(part, os.path.join(OUT, "plate_mini_preview.png"))
    print("วาดภาพตัวอย่าง cad/plate_mini_preview.png")


if __name__ == "__main__":
    main()
