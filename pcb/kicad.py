"""ส่งออกเป็นไฟล์ `.kicad_pcb` — รูปแบบที่ EasyEDA Pro นำเข้าได้ (File > Import > KiCad)

ทำไมต้องมีทั้งที่มี Gerber แล้ว:
  Gerber มีแต่ **รูปร่างทองแดง** ไม่มีชื่อเน็ต ไม่มีขา ไม่มีชิ้นส่วน
  โปรแกรมอื่นเปิดดูได้ว่าลายหน้าตายังไง แต่ตรวจไม่ได้ว่า "ขานี้ควรต่อกับขานั้น"
  ไฟล์ KiCad พาข้อมูลเน็ตไปด้วย EasyEDA จึงตรวจการเชื่อมต่อและรัน DRC ให้ได้จริง
  = ได้ตัวตรวจที่สาม ที่ไม่ใช่โค้ดของเราเอง

**แกน y กลับด้าน** KiCad นับ y ลงล่าง ส่วนไฟล์อื่นในโฟลเดอร์นี้นับขึ้นบนตาม Gerber
แปลงที่ฟังก์ชัน `_y` จุดเดียว ไม่กระจายไปทั่ว

**แผ่นกราวด์ส่งเป็น zone ไม่ใช่รูปหลายเหลี่ยมที่เทแล้ว**
ปล่อยให้ EasyEDA คำนวณการเทเอง ได้ระยะเผื่อกับคูกันความร้อนตามกติกาของมัน
ซึ่งเชื่อถือได้กว่าการที่เรายัดรูปที่เทเองเข้าไป
"""
import os

from geom import CLEAR

VERSION = 20221018        # รูปแบบของ KiCad 6/7 ซึ่งเป็นช่วงที่ตัวนำเข้าส่วนใหญ่รองรับ
THICK = 1.6

LAYERS = """  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )"""


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def export(path, pl, r, w, h, x0, y0):
    """เขียนไฟล์ .kicad_pcb ของบอร์ดหนึ่ง"""
    top = y0 + h

    def _y(v):
        """แกน y ของ KiCad ชี้ลง ของเราชี้ขึ้น — กลับด้านที่นี่จุดเดียว"""
        return round(top - v, 4)

    def _x(v):
        return round(v - x0, 4)

    nets = list(pl["nets"])
    nid = {n: i for i, n in enumerate(nets, start=1)}
    pin_net = {p: n for n, ps in pl["nets"].items() for p in ps}

    out = [f"(kicad_pcb (version {VERSION}) (generator {_q('ultrasonic01-pcb')})",
           f"  (general (thickness {THICK}))",
           f"  (paper {_q('A4')})", LAYERS,
           "  (setup (pad_to_mask_clearance 0.05))",
           f"  (net 0 {_q('')})"]
    for n in nets:
        out.append(f"  (net {nid[n]} {_q(n)})")

    # ---- ชิ้นส่วน: หนึ่ง footprint ต่อหนึ่งตัว ขาเป็นรูทะลุทั้งหมด
    for ref, fp in pl["fps"].items():
        val = pl["vals"][ref]
        pins = sorted((k for k in pl["pad"] if k.startswith(ref + ".")),
                      key=lambda s: int(s.split(".")[1]))
        px, py, _d, _dia = pl["pad"][pins[0]]
        out.append(f"  (footprint {_q('ultrasonic:' + fp)} (layer {_q('F.Cu')}) "
                   f"(at {_x(px)} {_y(py)})")
        out.append(f"    (attr through_hole)")
        out.append(f"    (fp_text reference {_q(ref)} (at 0 -2.2) (layer {_q('F.SilkS')})"
                   f" (effects (font (size 1 1) (thickness 0.15))))")
        out.append(f"    (fp_text value {_q(val)} (at 0 2.2) (layer {_q('F.Fab')})"
                   f" (effects (font (size 1 1) (thickness 0.15))))")
        for p in pins:
            x, y, drill, dia = pl["pad"][p]
            num = p.split(".")[1]
            net = pin_net.get(p)
            # ขา 1 เป็นสี่เหลี่ยม ที่เหลือกลม — มาตรฐานเดียวกับที่ช่างดูออกทันที
            shape = "rect" if num == "1" else "circle"
            ntxt = f" (net {nid[net]} {_q(net)})" if net else ""
            out.append(f"    (pad {_q(num)} thru_hole {shape} "
                       f"(at {round(_x(x) - _x(px), 4)} {round(_y(y) - _y(py), 4)}) "
                       f"(size {dia} {dia}) (drill {drill}) "
                       f"(layers {_q('*.Cu')} {_q('*.Mask')}){ntxt})")
        out.append("  )")

    # ---- รูยึด: footprint ที่ไม่มีทองแดง มีแต่รู
    for i, (mx, my, md) in enumerate(r.mounts, start=1):
        out.append(f"  (footprint {_q('ultrasonic:MountingHole')} (layer {_q('F.Cu')}) "
                   f"(at {_x(mx)} {_y(my)})")
        out.append(f"    (attr through_hole exclude_from_pos_files exclude_from_bom)")
        out.append(f"    (fp_text reference {_q('H' + str(i))} (at 0 -3) "
                   f"(layer {_q('F.SilkS')}) (effects (font (size 1 1) (thickness 0.15))))")
        out.append(f"    (fp_text value {_q('M2.5')} (at 0 3) (layer {_q('F.Fab')})"
                   f" (effects (font (size 1 1) (thickness 0.15))))")
        out.append(f"    (pad {_q('')} np_thru_hole circle (at 0 0) (size {md} {md}) "
                   f"(drill {md}) (layers {_q('F&B.Cu')} {_q('*.Mask')}))")
        out.append("  )")

    # ---- ขอบบอร์ด
    c = [(0, 0), (w, 0), (w, h), (0, h)]
    for (ax, ay), (bx, by) in zip(c, c[1:] + c[:1]):
        out.append(f"  (gr_line (start {ax} {round(h - ay, 4)}) "
                   f"(end {bx} {round(h - by, 4)}) "
                   f"(stroke (width 0.15) (type solid)) (layer {_q('Edge.Cuts')}))")

    # ---- ลายทองแดง
    for L, pts, width, net in r.tracks:
        lay = "F.Cu" if L == 0 else "B.Cu"
        nn = nets[net - 1] if 1 <= net <= len(nets) else None
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            out.append(f"  (segment (start {_x(ax)} {_y(ay)}) (end {_x(bx)} {_y(by)}) "
                       f"(width {width}) (layer {_q(lay)}) "
                       f"(net {nid.get(nn, 0)}))")
    for vx, vy in r.vias:
        out.append(f"  (via (at {_x(vx)} {_y(vy)}) (size 1.2) (drill 0.6) "
                   f"(layers {_q('F.Cu')} {_q('B.Cu')}) (net {nid.get('GND', 0)}))")

    # ---- แผ่นกราวด์: ส่งเป็นกรอบ ให้ปลายทางเทเอง
    if "GND" in nid:
        pts = " ".join(f"(xy {ax} {round(h - ay, 4)})" for ax, ay in c)
        for lay in ("F.Cu", "B.Cu"):
            out.append(f"""  (zone (net {nid['GND']}) (net_name {_q('GND')}) (layer {_q(lay)})
    (hatch edge 0.5) (connect_pads (clearance {CLEAR}))
    (min_thickness 0.25) (filled_areas_thickness no)
    (fill yes (thermal_gap 0.4) (thermal_bridge_width 0.5))
    (polygon (pts {pts}))
  )""")

    out.append(")")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


def summary(pl, r):
    """ตัวเลขไว้เทียบกับที่ EasyEDA อ่านได้ หลังนำเข้า"""
    return {"ชิ้นส่วน": len(pl["fps"]), "ขา": len(r.pads), "เน็ต": len(pl["nets"]),
            "ลาย": len(r.tracks), "เวีย": len(r.vias), "รูยึด": len(r.mounts)}
