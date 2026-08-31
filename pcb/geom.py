"""แปลง footprint เป็นพิกัดรูเจาะจริง และเก็บรูปทรงพื้นฐานของบอร์ด

หน่วยเป็นมิลลิเมตรทั้งไฟล์ จุดกำเนิดอยู่มุมซ้ายล่างของบอร์ด แกน y ชี้ขึ้น
(เหมือน Gerber) ไม่ใช่ชี้ลงแบบภาพ — ตอนวาดภาพค่อยพลิกทีเดียวที่จุดเดียว
"""
from netlist import FP

# กติกาการผลิตที่เลือกไว้ — เผื่อไว้เยอะกว่ามาตรฐานโรงงาน เพื่อให้ทำเองก็ได้
TRACE = 0.4          # ความกว้างลายสัญญาณ
POWER = 0.8          # ความกว้างลายไฟ
CLEAR = 0.4          # ระยะห่างขั้นต่ำระหว่างทองแดงคนละเน็ต
ANNULAR = 0.4        # เนื้อทองแดงรอบรู (ด้านเดียว) ขั้นต่ำ
EDGE = 1.5           # ระยะจากขอบบอร์ดถึงทองแดง


def pads(fp_name, x, y, rot=0):
    """คืนพิกัดขาทุกขาของ footprint หนึ่งตัว เป็น list ของ (x, y, drill, pad)

    rot = 0 หมายถึงวางตามแนวนอน · 90 คือหมุนทวนเข็มให้เป็นแนวตั้ง
    ขา 1 อยู่ตำแหน่งแรกเสมอ (มุมซ้ายล่างของตัวถัง)
    """
    f = FP[fp_name]
    k = f["kind"]
    if k == "dip":
        half = f["n"] // 2
        pts = [(i * f["pitch"], 0.0) for i in range(half)]              # ขา 1..7
        pts += [((half - 1 - i) * f["pitch"], f["span"]) for i in range(half)]
    elif k == "axial":
        pts = [(0.0, 0.0), (f["pitch"], 0.0)]
    elif k == "header":
        pts = [(i * f["pitch"], 0.0) for i in range(f["n"])]
    else:
        raise ValueError(k)
    out = []
    for px, py in pts:
        if rot == 90:
            px, py = -py, px
        elif rot == 180:
            px, py = -px, -py
        elif rot == 270:
            px, py = py, -px
        out.append((round(x + px, 4), round(y + py, 4), f["drill"], f["pad"]))
    return out


def body(fp_name, x, y, rot=0):
    """กรอบตัวถังไว้พิมพ์บนซิลค์สกรีน คืน (x0, y0, x1, y1)"""
    f = FP[fp_name]
    k = f["kind"]
    if k == "dip":
        half = f["n"] // 2
        w, h = (half - 1) * f["pitch"], f["span"]
        box = (-1.4, -1.4, w + 1.4, h + 1.4)
    elif k == "axial":
        box = (-1.6, -1.4, f["pitch"] + 1.6, 1.4)
    else:
        box = (-1.4, -1.4, (f["n"] - 1) * f["pitch"] + 1.4, 1.4)
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    out = []
    for cx, cy in corners:
        if rot == 90:
            cx, cy = -cy, cx
        elif rot == 180:
            cx, cy = -cx, -cy
        elif rot == 270:
            cx, cy = cy, -cx
        out.append((x + cx, y + cy))
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    return (min(xs), min(ys), max(xs), max(ys))


def place(board, positions):
    """รวม netlist + ตำแหน่งที่วาง -> โครงสร้างที่ตัวเดินลายใช้ได้

    positions: ref -> (x, y, rot)
    คืน dict ที่มี pad['REF.PIN'] = (x, y, drill, pad_dia) และ net -> [pin, ...]
    """
    fps = {r: f for r, _, f in board["parts"]}
    vals = {r: v for r, v, _ in board["parts"]}
    pad = {}
    bodies = {}
    for ref, fp_name in fps.items():
        if ref not in positions:
            raise KeyError(f"ยังไม่ได้วาง {ref}")
        x, y, rot = positions[ref]
        for i, p in enumerate(pads(fp_name, x, y, rot), start=1):
            pad[f"{ref}.{i}"] = p
        bodies[ref] = body(fp_name, x, y, rot)
    return dict(name=board["name"], title=board["title"], pad=pad,
                nets=board["nets"], bodies=bodies, vals=vals, fps=fps)


def extent(pl, margin=5.5):
    """ขนาดบอร์ดที่พอดีกับของที่วาง + ขอบ

    ขอบ 5.5 มม. เพราะต้องมีที่ให้รูยึด M2.5 ที่มุมทั้งสี่ พร้อมพื้นที่ปลอดทองแดง
    รอบหัวสกรู ถ้าเผื่อน้อยกว่านี้ หัวสกรูจะทับลายหรือทับแผ่นกราวด์
    """
    xs, ys = [], []
    for x0, y0, x1, y1 in pl["bodies"].values():
        xs += [x0, x1]
        ys += [y0, y1]
    for x, y, _, d in pl["pad"].values():
        xs += [x - d / 2, x + d / 2]
        ys += [y - d / 2, y + d / 2]
    return (round(max(xs) - min(xs) + 2 * margin, 2),
            round(max(ys) - min(ys) + 2 * margin, 2),
            min(xs) - margin, min(ys) - margin)
