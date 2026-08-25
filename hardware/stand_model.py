"""ขาตั้งสำหรับ plate_mini + ที่วางกล้อง Orbbec Astra Pro — พิมพ์ 2 ชิ้น

Run:  python hardware/stand_model.py
      -> cad/stand_base.stl / cad/cam_cradle.stl (+ .step + ภาพตัวอย่าง)

ชิ้นที่ 1  stand_base   ฐาน + เสาตั้ง ยึด plate_mini ด้วยรูยึดเดิม **ครบทั้ง 4 รู**
ชิ้นที่ 2  cam_cradle   รางวาง Astra Pro ขันติดหน้าฐาน ล็อกซ้าย-ขวาอย่างเดียว

**ตัวเลขเดียวที่ต้องวัดก่อนพิมพ์: CAM_W (ความยาวตัวกล้อง)**
  สเปกที่ Orbbec ประกาศคือ 165 x 30 x 40 มม. ซึ่งใช้เป็นค่าตั้งต้นในไฟล์นี้
  แต่ตัวจริงอาจต่างถ้ามีฐานเอียงติดมาด้วย **วัดด้วยไม้บรรทัดแล้วแก้ CAM_W บรรทัดเดียว**
  แล้วรันใหม่ ใช้เวลา 10 วินาที เทียบกับพิมพ์ผิด 3 ชั่วโมง

ทำไมรางล็อกแค่ซ้าย-ขวา (ตามที่ขอ):
  กล้องแค่ "วางลงไป" ไม่ต้องหนีบ — ผนังสองข้างกันเลื่อนซ้าย-ขวาซึ่งเป็นทิศที่ทำให้
  มุมที่กล้องเห็นเพี้ยน ส่วนหน้า-หลังไม่มีผนัง จึงไม่ต้องรู้ความลึกกล้องให้แม่น
  **มีมิติเดียวที่ต้องถูก** = ความเสี่ยงพิมพ์ผิดเหลือน้อยที่สุด
  ถ้าอยากได้กันไหลไปข้างหลังด้วย ตั้ง BACK_LIP = 6.0 แล้วรันใหม่

ทำไมวางกล้องชิดแผง (ขอบหลังรางที่ x=30 มม.) และยกแผงสูง 125 มม.:
  **ยิ่งวางกล้องใกล้แผง ยิ่งบังลำเสียงน้อยลง ไม่ใช่มากขึ้น** — ลำเสียงเป็นกรวยที่กางออก
  ตามระยะ ขอบล่างของกรวย 30 องศาจาก RX ล่าง (สูง 70 มม. หน้าเซ็นเซอร์ x=26) ตกลงมาเป็น
      z = 70 - tan(30°) x (x - 26)
  ที่ x=30 ขอบอยู่ 67.7 · ที่ x=60 อยู่ 50.4 · ที่ x=95 เหลือ 30.2 มม.
  หลังคากล้องอยู่ 46 มม. (พื้นราง 6 + ตัวกล้อง 40) จึง **พ้นตลอดทั้งตัวกล้อง** เมื่อกล้อง
  อยู่ในช่วง x=30 ถึง 60 แต่จะโดนบังถ้าเลื่อนออกไปไกลกว่านั้น
  (ฉบับแรกวางไว้ที่ x=95 ซึ่งกลับเป็นตำแหน่งที่แย่ที่สุด — คิดกลับด้าน)
  ยกแผงจาก 115 เป็น 125 มม. เพื่อให้ปลายหน้าของกล้อง (x=60) พ้นขอบกรวยด้วย
  ขอบหลังรางหยุดที่ x=30 เพราะบอส RX ล่างกินพื้นที่ถึง x=26

ทิศพิมพ์: ทั้งสองชิ้นวางหงายบนเตียงตามที่โมเดลวางไว้ **ไม่ต้องใช้ซัพพอร์ต**
  เสาตั้งกับผนังรางเป็นผนังตั้งฉาก ช่องน็อตใต้ฐานเปิดลงล่างซึ่งเป็นชั้นแรกพอดี
"""

import math
import os
import struct

import numpy as np
from build123d import *

# ------------------------------------------------------- กล้อง (วัดก่อนพิมพ์!) --

CAM_W = 165.0      # ความยาวตัวกล้องซ้าย-ขวา — **วัดจริงแล้วแก้ตรงนี้**
CAM_FIT = 2.0      # ช่องว่างรวมซ้าย+ขวา ให้วางลงง่ายโดยยังไม่โคลง
WALL_H = 18.0      # ความสูงผนังราง — ต่ำกว่าตัวกล้อง (40) มาก จึงไม่บังเลนส์
BACK_LIP = 0.0     # ตั้ง 6.0 ถ้าอยากได้ขอบกันกล้องไหลไปข้างหลังด้วย

# ------------------------------------------------------------------ ฐาน+เสา --

BASE_D, BASE_W, BASE_T = 145.0, 150.0, 6.0   # ลึก(X) x กว้าง(Y) x หนา(Z)
BASE_BACK = 42.0                             # ฐานยื่นหลังเสา — ต้องคลุมปลายครีบค้ำด้วย
UP_T, UP_H = 8.0, 200.0                      # เสา: หนา(X) x สูง(Z) — สูงพอรับรูยึดครบ 4 รู
GUS_T, GUS_D, GUS_H = 6.0, 30.0, 90.0        # ครีบค้ำหลังเสา (เสาสูงขึ้นจึงค้ำสูงขึ้นด้วย)
GUS_Y = 50.0

PLATE_CZ = 125.0        # ความสูงกึ่งกลาง plate_mini เหนือผิวบนฐาน
PLATE_MOUNT_R = 62.0    # ต้องตรงกับ MOUNT_R ใน plate_mini_model.py
PLATE_SIZE, PLATE_T = 150.0, 4.0
BOLT_D = 4.5            # รูยึด M4

# เดือยรองแผง — **จำเป็น ไม่ใช่ของตกแต่ง**
# ตัวเซ็นเซอร์ TCT40 นั่งจมอยู่ในรูลึก 14 มม. และ **สายออกทางด้านหลังแผง** ซึ่งเป็น
# ด้านที่แนบเสาพอดี ถ้าแผงแนบเสาสนิทสายจะถูกหนีบ — สายที่ถูกดึง/หนีบคือจุดสัมผัสแข็ง
# ที่ทำลายการแยกเสียงพอ ๆ กับการทากาว (plate_mini_spec.md ข้อ 3)
# เดือยยกแผงให้ลอย 12 มม. สายจึงมีที่เดินแล้วลอดช่องร้อยสายไปหลังเสาได้
SO_H, SO_D = 12.0, 14.0

CABLE_W, CABLE_H, CABLE_Z = 50.0, 22.0, 95.0   # ช่องร้อยสายผ่านเสา

WIN_CX, WIN_L, WIN_W = 53.0, 78.0, 76.0        # ช่องลดเนื้อกลางฐาน (เว้นรูขันราง)
UWIN_Y, UWIN_W, UWIN_Z, UWIN_H = 25.0, 32.0, 26.0, 36.0   # ช่องลดเนื้อบนเสา (คู่ล่าง)
UWIN2_W, UWIN2_Z, UWIN2_H = 70.0, 156.0, 34.0             # ช่องลดเนื้อบนเสา (ช่องบน)

# ------------------------------------------------------------- จุดขันรางกล้อง --

CRADLE_X0 = 30.0        # ขอบหลังราง — ชิดแผงที่สุดเท่าที่ไม่ชนบอส RX ล่าง
CRADLE_D = 45.0         # ความลึกพื้นราง
TAB_X = 52.0            # ตำแหน่งรูขันบนฐาน (X) — กึ่งกลางราง
TAB_Y = 55.0            # ตำแหน่งรูขันบนฐาน (+-Y)
SLOT_L = 15.0           # ร่องบนรางให้ปรับหน้า-หลังได้ +-7.5 มม.
NUT_AF, NUT_T = 7.2, 3.4    # ช่องน็อต M4 ใต้ฐาน (หกเหลี่ยม 7 มม. + เผื่อ)
CBORE_D, CBORE_T = 8.6, 3.0  # เจาะบ่าให้หัวสกรูจมใต้ตัวกล้อง

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cad")


# --------------------------------------------------------------- helpers --
# หมายเหตุ build123d: extrude(sketch, h) ไปทาง +normal จาก h=0 · ห้ามใส่ both=True
# ถ้าไม่ได้ตั้งใจ เพราะมันจะได้ความหนา 2h ไม่ใช่ h (พลาดมาแล้วรอบแรก ฐานหนาเป็นสองเท่า)

def xhole(x, y, z, d, length):
    """รูแนวนอนทะลุเสา — Cylinder อยู่กึ่งกลางแกนตัวเอง จึงคุมตำแหน่งง่ายกว่า extrude"""
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(d / 2, length)


def hexpocket(x, y, z0, af, h):
    """ช่องน็อตหกเหลี่ยม เปิดลงล่าง — af = ระยะระหว่างด้านตรงข้าม"""
    return Pos(x, y, z0) * extrude(RegularPolygon(af / math.sqrt(3), 6), h)


def slot_z(x, y, z0, d, length, h):
    """ร่องยาวตามแกน X เจาะตามแนว Z"""
    return (Pos(x, y, z0)
            * extrude(RectangleRounded(length + d, d, d / 2 - 0.01), h))


# ------------------------------------------------------------- ชิ้นที่ 1 ----

def build_base():
    cx = (BASE_D - 2 * BASE_BACK) / 2          # กึ่งกลางฐานตามแกน X
    # ฐาน: ผิวบนอยู่ที่ z = 0 พอดี ทุกความสูงในไฟล์นี้จึงวัดจากผิวบนฐาน
    part = (Pos(cx, 0, -BASE_T)
            * extrude(RectangleRounded(BASE_D, BASE_W, 10.0), BASE_T))
    # เสาตั้ง: ผิวหน้าอยู่ที่ x = 0 (แผงจะแนบหน้านี้)
    part += Pos(-UP_T / 2, 0, 0) * extrude(Rectangle(UP_T, BASE_W), UP_H)

    # ครีบค้ำอยู่ "ด้านหลัง" เสา จึงไม่ชนแผงที่แนบอยู่ด้านหน้า
    tri = Polygon((-UP_T, 0), (-UP_T - GUS_D, 0), (-UP_T, GUS_H), align=None)
    for sy in (+1, -1):
        part += (Pos(0, sy * GUS_Y - GUS_T / 2, 0)
                 * extrude(Plane.XZ * tri, GUS_T))

    # ---- ลดเนื้อ ----
    # ช่องกลางฐาน: ต้องเว้นให้พ้นรูขันรางกล้องที่ (112, ±55) ไม่งั้นช่องน็อตจะแหว่ง
    part -= (Pos(WIN_CX, 0, -BASE_T - 1)
             * extrude(RectangleRounded(WIN_L, WIN_W, 10.0), BASE_T + 2))
    # ช่องบนเสา: ต้องอยู่ระหว่างครีบค้ำสองอัน (y=±50) ไม่งั้นจะเจาะทะลุโคนครีบ
    # ซึ่งเป็นจุดที่ครีบมีไว้เสริมพอดี
    for sy in (+1, -1):
        part -= (Pos(-UP_T / 2, sy * UWIN_Y, UWIN_Z)
                 * Box(UP_T + 2, UWIN_W, UWIN_H))

    # ช่องลดเนื้อช่องบน อยู่ระหว่างรูยึดข้าง (z=125) กับรูยึดบน (z=187)
    part -= Pos(-UP_T / 2, 0, UWIN2_Z) * Box(UP_T + 2, UWIN2_W, UWIN2_H)

    # เดือยรองแผง **ครบทั้ง 4 รูของแผง** ยื่นออกหน้าเสา ให้แผงลอยพ้นเสาไว้เดินสาย
    mounts = ((+PLATE_MOUNT_R, PLATE_CZ), (-PLATE_MOUNT_R, PLATE_CZ),
              (0.0, PLATE_CZ - PLATE_MOUNT_R), (0.0, PLATE_CZ + PLATE_MOUNT_R))
    for y, z in mounts:
        part += Pos(0, y, z) * Rot(0, 90, 0) * extrude(Circle(SO_D / 2), SO_H)

    # รูยึดแผง 3 จุด เจาะทะลุทั้งเดือยและเสา
    for y, z in mounts:
        part -= xhole(0, y, z, BOLT_D, 2 * (SO_H + UP_T + 2))

    # ช่องร้อยสายผ่านเสา อยู่ระหว่างรูยึดข้าง (115) กับรูล่าง (53) ไม่ชนทั้งคู่
    part -= Pos(-UP_T / 2, 0, CABLE_Z) * Box(UP_T + 2, CABLE_W, CABLE_H)

    # รูขันรางกล้อง + ช่องน็อตใต้ฐาน (เปิดลงล่าง = ชั้นแรก ไม่ต้องซัพพอร์ต)
    for sy in (+1, -1):
        part -= Pos(TAB_X, sy * TAB_Y, -BASE_T - 1) * extrude(
            Circle(BOLT_D / 2), BASE_T + 2)
        part -= hexpocket(TAB_X, sy * TAB_Y, -BASE_T - 0.01, NUT_AF, NUT_T)
    return part


# ------------------------------------------------------------- ชิ้นที่ 2 ----

def cradle_iw():
    return CAM_W + CAM_FIT


WALL_T, FLOOR_T = 5.0, 6.0


def build_cradle():
    iw = cradle_iw()
    ow = iw + 2 * WALL_T
    part = (Pos(CRADLE_D / 2, 0, 0)
            * extrude(RectangleRounded(CRADLE_D, ow, 6.0), FLOOR_T))
    for sy in (+1, -1):
        part += (Pos(CRADLE_D / 2, sy * (iw + WALL_T) / 2, FLOOR_T)
                 * extrude(Rectangle(CRADLE_D, WALL_T), WALL_H))
    if BACK_LIP > 0:
        part += (Pos(WALL_T / 2, 0, FLOOR_T)
                 * extrude(Rectangle(WALL_T, iw), BACK_LIP))

    # ร่องขันยึด ปรับหน้า-หลังได้ + เจาะบ่าให้หัวสกรูจมพ้นใต้ตัวกล้อง
    cx = TAB_X - CRADLE_X0
    for sy in (+1, -1):
        part -= slot_z(cx, sy * TAB_Y, -1, BOLT_D, SLOT_L, FLOOR_T + 2)
        part -= slot_z(cx, sy * TAB_Y, FLOOR_T - CBORE_T, CBORE_D, SLOT_L,
                       CBORE_T + 1)
    return part


# ----------------------------------------------------------------- ตรวจ ----

def mesh_of(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def watertight(tris):
    from collections import Counter
    e = Counter()
    q = np.round(tris.reshape(-1, 3), 3)
    for i in range(0, len(q), 3):
        a, b, c = [tuple(v) for v in q[i:i + 3]]
        for u, v in ((a, b), (b, c), (c, a)):
            e[(u, v) if u < v else (v, u)] += 1
    return sum(1 for v in e.values() if v != 2)


def check_clearances():
    """ตรวจว่าฟีเจอร์ที่เจาะเข้าไปไม่กินเนื้อรอบฟีเจอร์อื่นจนบาง

    เขียนไว้เพราะรอบแรกช่องลดเนื้อกลางฐานกินขอบช่องน็อต M4 พอดี (ห่างแค่ 0 มม.)
    และช่องลดเนื้อบนเสาเจาะทะลุโคนครีบค้ำ — ทั้งสองอย่างมองจากภาพด้านบนไม่เห็นเลย
    """
    MIN = 4.0   # เนื้อบางกว่านี้ = ผนัง 4 เส้นเต็ม 1.6 มม. + อินฟิลไม่พอ
    out = []

    def gap(name, d):
        out.append((name, d, d >= MIN))

    # ช่องกลางฐาน vs รูขัน/ช่องน็อตราง (ใช้ครึ่งความกว้างช่องน็อตเป็นรัศมี)
    nut_r = NUT_AF / math.sqrt(3)
    # จุดอยู่นอกสี่เหลี่ยมถ้าพ้นทาง *แกนใดแกนหนึ่ง* ระยะเนื้อจึงเป็นค่ามากสุดของสองแกน
    gap("ช่องฐาน ↔ ช่องน็อตราง",
        max((TAB_X - nut_r) - (WIN_CX + WIN_L / 2),
            (TAB_Y - nut_r) - WIN_W / 2))
    gap("รูขันราง ↔ ขอบฐาน (แกน Y)", BASE_W / 2 - (TAB_Y + nut_r))
    gap("รูขันราง ↔ ขอบฐาน (แกน X)",
        (BASE_D - BASE_BACK) - (TAB_X + nut_r))
    # ช่องบนเสา vs ครีบค้ำ / รูยึดแผง / ช่องสาย
    gap("ช่องเสา ↔ โคนครีบค้ำ", (GUS_Y - GUS_T / 2) - (UWIN_Y + UWIN_W / 2))
    gap("ช่องเสา ↔ ขอบเสา", BASE_W / 2 - (UWIN_Y + UWIN_W / 2))
    gap("ช่องเสา ↔ รูยึดล่างกลาง",
        math.hypot(max(0, UWIN_Y - UWIN_W / 2),
                   max(0, (PLATE_CZ - PLATE_MOUNT_R) - (UWIN_Z + UWIN_H / 2)))
        - BOLT_D / 2)
    gap("ช่องเสา ↔ ช่องร้อยสาย", (CABLE_Z - CABLE_H / 2) - (UWIN_Z + UWIN_H / 2))
    gap("ช่องร้อยสาย ↔ รูยึดข้าง",
        (PLATE_CZ - BOLT_D / 2) - (CABLE_Z + CABLE_H / 2))
    gap("รูยึดข้าง ↔ ขอบเสา", BASE_W / 2 - (PLATE_MOUNT_R + BOLT_D / 2))
    gap("รูยึดบน ↔ ยอดเสา", UP_H - (PLATE_CZ + PLATE_MOUNT_R + SO_D / 2))
    gap("ช่องบนเสา ↔ รูยึดข้าง",
        (UWIN2_Z - UWIN2_H / 2) - (PLATE_CZ + BOLT_D / 2))
    gap("ช่องบนเสา ↔ รูยึดบน",
        (PLATE_CZ + PLATE_MOUNT_R - BOLT_D / 2) - (UWIN2_Z + UWIN2_H / 2))
    gap("ช่องบนเสา ↔ ขอบเสา", BASE_W / 2 - UWIN2_W / 2)
    # ราง: ร่องขัน vs ผนังราง
    # ปลายครีบค้ำต้องอยู่บนเนื้อฐาน ไม่งั้นจะพิมพ์ลอยกลางอากาศ
    gap("ปลายครีบค้ำ ↔ ขอบหลังฐาน", BASE_BACK - (UP_T + GUS_D))
    gap("ราง ↔ ขอบหน้าฐาน (ต้องไม่ยื่นเกิน)",
        (BASE_D - BASE_BACK) - (CRADLE_X0 + CRADLE_D))
    gap("เดือยรองแผง ↔ ขอบเดือย-รูสกรู", (SO_D - BOLT_D) / 2)
    gap("ร่องขันราง ↔ ผนังราง",
        (cradle_iw() / 2) - (TAB_Y + (SLOT_L + CBORE_D) / 2 - SLOT_L / 2))
    print(f"\nระยะเนื้อขั้นต่ำระหว่างฟีเจอร์ (ต้อง >= {MIN:.0f} มม.)")
    bad = 0
    for name, d, ok in out:
        bad += (not ok)
        print(f"  {'✅' if ok else '❌'}  {name:32} {d:6.1f} มม.")
    return bad


def render_preview(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle as R

    fig, ax = plt.subplots(figsize=(9, 6.4), dpi=115)
    iw = cradle_iw()
    # มองจากด้านข้าง (แกน X ไปหน้า, Z ขึ้น)
    ax.add_patch(R((-BASE_BACK, -BASE_T), BASE_D, BASE_T, fc="#adb5bd", ec="#495057"))
    ax.add_patch(R((-UP_T, 0), UP_T, UP_H, fc="#adb5bd", ec="#495057"))
    ax.add_patch(plt.Polygon([(-UP_T, 0), (-UP_T - GUS_D, 0), (-UP_T, GUS_H)],
                             fc="#ced4da", ec="#868e96"))
    ax.add_patch(R((SO_H, PLATE_CZ - PLATE_SIZE / 2), PLATE_T, PLATE_SIZE,
                   fc="#4dabf7", ec="#1864ab", lw=2))
    for _y, mz in ((0, PLATE_CZ), (0, PLATE_CZ - PLATE_MOUNT_R)):
        ax.add_patch(R((0, mz - SO_D / 2), SO_H, SO_D, fc="#adb5bd", ec="#495057"))
    ax.annotate(f"standoff {SO_H:.0f} mm\n(cable space)", (SO_H / 2, PLATE_CZ - 30),
                xytext=(28, -22), textcoords="offset points", fontsize=8.5,
                color="#495057", arrowprops=dict(arrowstyle="->", color="#868e96"))
    ax.add_patch(R((CRADLE_X0, 0), CRADLE_D, 6.0, fc="#ffd43b", ec="#e67700"))
    ax.add_patch(R((CRADLE_X0, 6.0), CRADLE_D, WALL_H, fc="none", ec="#e67700",
                   ls="--"))
    ax.add_patch(R((CRADLE_X0 + 5, 6.0), 30, 40, fc="#ffa8a8", ec="#c92a2a",
                   alpha=.75))
    for z, lab in ((PLATE_CZ + 55, "RX top"), (PLATE_CZ, "TX"),
                   (PLATE_CZ - 55, "RX bottom")):
        ax.plot([SO_H + PLATE_T], [z], "o", color="#1864ab", ms=7)
        ax.annotate(lab, (SO_H + PLATE_T, z), xytext=(10, -3),
                    textcoords="offset points", fontsize=9, color="#1864ab")
    ax.annotate("Astra Pro", (CRADLE_X0 + 20, 46), xytext=(0, 8),
                textcoords="offset points", fontsize=9, color="#c92a2a",
                ha="center", weight="bold")
    ax.plot([SO_H + PLATE_T, CRADLE_X0 + 40], [PLATE_CZ - 55, 46], "--",
            color="#868e96", lw=1)
    ax.annotate(f"gap {PLATE_CZ - 55 - 46:.0f} mm",
                ((SO_H + PLATE_T + CRADLE_X0 + 40) / 2, (PLATE_CZ - 55 + 46) / 2),
                fontsize=8.5, color="#495057", ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-BASE_BACK - 15, BASE_D - BASE_BACK + 20)
    ax.set_ylim(-20, PLATE_CZ + PLATE_SIZE / 2 + 25)
    ax.set_xlabel("X - front [mm]")
    ax.set_ylabel("Z - height [mm]")
    ax.set_title(f"stand + camera cradle - side view   "
                 f"cradle inner {iw:.0f} mm (camera {CAM_W:.0f})")
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, part in (("stand_base", build_base()), ("cam_cradle", build_cradle())):
        export_step(part, os.path.join(OUT, name + ".step"))
        export_stl(part, os.path.join(OUT, name + ".stl"))
        bb = part.bounding_box()
        tris = mesh_of(os.path.join(OUT, name + ".stl"))
        bad = watertight(tris)
        print(f"{name:12s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} มม."
              f"   เนื้อตัน {part.volume/1000:5.1f} cm3"
              f"   (~{part.volume/1000*0.5*1.27:4.0f} g)"
              f"   เมช {'ปิดสนิท ✅' if not bad else f'รั่ว {bad} ขอบ ❌'}")
    print(f"\nรางกล้อง: กว้างภายใน {cradle_iw():.1f} มม. "
          f"(CAM_W {CAM_W:.0f} + เผื่อ {CAM_FIT:.0f})")
    print(f"แผงอยู่สูง {PLATE_CZ:.0f} มม. · RX คู่ล่างสูง {PLATE_CZ-55:.0f} มม. · "
          f"หลังคากล้องสูง ~46 มม. -> ช่องว่าง {PLATE_CZ-55-46:.0f} มม.")
    bad = check_clearances()
    if bad:
        print(f"\n!! มี {bad} จุดที่เนื้อบางเกินไป แก้ค่าคงที่ก่อนพิมพ์")
    render_preview(os.path.join(OUT, "stand_preview.png"))
    print("วาดภาพตัวอย่าง cad/stand_preview.png")


if __name__ == "__main__":
    main()
