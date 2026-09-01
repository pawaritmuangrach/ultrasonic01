"""แปลงผังวงจรจาก schematic.py เป็นไฟล์ `.kicad_sch` ที่ EasyEDA Pro นำเข้าได้

ทำไมต้องมีนอกจาก SVG:
  SVG เป็นภาพ เปิดดูได้อย่างเดียว แก้ต่อในโปรแกรมออกแบบวงจรไม่ได้
  ไฟล์ผังจริงพาข้อมูลไฟฟ้าไปด้วย — ขาไหนต่อกับขาไหน — EasyEDA จึงลากเส้นต่อ
  เพิ่มชิ้นส่วน หรือตรวจกับบอร์ดได้

**ไฟล์ผังต้องมีครบทั้ง 8 ช่องรับ** ต่างจากภาพที่วาดช่องเดียวพอ
เพราะถ้าผังมีไม่ครบ ผังกับบอร์ดจะไม่ตรงกัน แล้วเครื่องมือจะเตือนหรือลบของทิ้ง
จึงใช้แผ่น 4-rx-all ที่มีครบ แทนแผ่น 2 ที่วาดช่องเดียว

**เรื่องแกน y ที่ต้องระวัง** KiCad ใช้สองระบบพิกัดในไฟล์เดียวกัน
  ในหน้ากระดาษ  y เพิ่มลงล่าง (เหมือนภาพ)
  ในนิยามสัญลักษณ์ y เพิ่มขึ้นบน (เหมือนคณิตศาสตร์)
แปลงกลับด้านเฉพาะตอนคำนวณตำแหน่งขาในนิยามสัญลักษณ์ ที่ฟังก์ชัน _lib_xy

**หน่วยและการปัดพิกัด** ภาพวาดเป็นพิกเซล ไฟล์ผังเป็นมิลลิเมตร
คูณ 0.254 แล้วปัดลงตาราง 1.27 มม. ซึ่งเป็นตารางมาตรฐานของผังวงจร
การปัดไม่ทำให้สายหลุดจากขา เพราะจุดที่ตรงกันอยู่แล้วปัดแล้วยังตรงกันเสมอ
"""
import hashlib
import os

LIB = "ultrasonic"     # ชื่อคลังสัญลักษณ์ · ต้องใช้ตัวเดียวกันทั้งนิยามและที่อ้างถึง
SCALE = 0.254          # มม. ต่อพิกเซล · 10 พิกเซล = 2.54 มม. พอดี
GRID = 1.27
VERSION = 20230121     # รูปแบบของ KiCad 7 ซึ่งตัวนำเข้าส่วนใหญ่รองรับ

# ขาไฟของชิป · แยกจาก section ปกติเพราะเป็นคนละ unit ในสัญลักษณ์เดียวกัน
GATE_UNIT = {("1", "2"): 1, ("3", "4"): 2, ("5", "6"): 3,
             ("9", "8"): 4, ("11", "10"): 5, ("13", "12"): 6}
SECT_UNIT = {"A": 1, "B": 2, "C": 3, "D": 4}


def snap(v):
    return round(round(v / GRID) * GRID, 4)


def _uid(*parts):
    """สร้าง uuid แบบเดิมทุกครั้งจากชื่อ — ไฟล์ที่สร้างซ้ำจะได้ผลเหมือนเดิม

    ถ้าสุ่ม uuid ใหม่ทุกครั้ง ไฟล์จะต่างกันทุกรอบทั้งที่วงจรไม่เปลี่ยน
    แล้ว git diff ก็อ่านไม่ได้ว่าอะไรเปลี่ยนจริง
    """
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _eff(size=1.27, hide=False, justify=""):
    j = f" (justify {justify})" if justify else ""
    return (f"(effects (font (size {size} {size})){j}"
            f"{' hide' if hide else ''})")


def _pin(num, name, etype, x, y, ang, length=2.54):
    return (f"        (pin {etype} line (at {x} {y} {ang}) (length {length})\n"
            f"          (name {_q(name)} {_eff()})\n"
            f"          (number {_q(num)} {_eff()})\n"
            f"        )")


def _angle(dx, dy):
    """ทิศจากปลายขาเข้าหาตัวสัญลักษณ์ · พิกัดแบบ y ขึ้นบน"""
    if abs(dx) >= abs(dy):
        return 180 if dx > 0 else 0
    return 270 if dy > 0 else 90


def _rect(x1, y1, x2, y2, fill="none"):
    return (f"        (rectangle (start {x1} {y1}) (end {x2} {y2})\n"
            f"          (stroke (width 0.254) (type default)) "
            f"(fill (type {fill})))")


def _poly(pts, fill="none"):
    d = " ".join(f"(xy {a} {b})" for a, b in pts)
    return (f"        (polyline (pts {d})\n"
            f"          (stroke (width 0.254) (type default)) "
            f"(fill (type {fill})))")


def _circle(x, y, r):
    return (f"        (circle (center {x} {y}) (radius {r})\n"
            f"          (stroke (width 0.254) (type default)) "
            f"(fill (type none)))")


# ---------------------------------------------------------------- สัญลักษณ์
def body_graphics(kind, offs):
    """รูปร่างของสัญลักษณ์ · offs คือตำแหน่งขาในพิกัดสัญลักษณ์ (y ขึ้นบน)"""
    pts = list(offs.values())
    if kind == "res":
        (ax, ay), (bx, by) = pts[0], pts[1]
        if abs(bx - ax) >= abs(by - ay):
            return [_rect(-2.54, -1.27, 2.54, 1.27)]
        return [_rect(-1.27, -2.54, 1.27, 2.54)]
    if kind == "cap":
        (ax, ay), (bx, by) = pts[0], pts[1]
        if abs(bx - ax) >= abs(by - ay):
            return [_poly([(-0.635, -2.2), (-0.635, 2.2)]),
                    _poly([(0.635, -2.2), (0.635, 2.2)])]
        return [_poly([(-2.2, 0.635), (2.2, 0.635)]),
                _poly([(-2.2, -0.635), (2.2, -0.635)])]
    if kind == "opamp":
        return [_poly([(-5.08, 5.84), (-5.08, -5.84), (5.08, 0),
                       (-5.08, 5.84)], "background")]
    if kind == "inv":
        return [_poly([(-4.45, 3.81), (-4.45, -3.81), (2.54, 0),
                       (-4.45, 3.81)], "background"),
                _circle(3.43, 0, 0.889)]
    if kind == "term":
        return [_rect(-3.94, -4.32, 3.94, 4.32, "background"),
                _circle(0, 2.29, 1.14), _circle(0, -2.29, 1.14)]
    if kind in ("opwr", "icpwr"):
        return [_rect(-6.35, -4.45, 6.35, 4.45, "background")]
    if kind == "header":
        ys = [q[1] for q in pts]
        return [_rect(-6.1, min(ys) - 2.29, 6.1, max(ys) + 2.29, "background")]
    return [_rect(-2.54, -2.54, 2.54, 2.54)]


ETYPE = {"res": "passive", "cap": "passive", "term": "passive",
         "header": "passive", "opwr": "power_in", "icpwr": "power_in"}


def pin_etype(kind, num, offs):
    if kind in ETYPE:
        return ETYPE[kind]
    # ออปแอมป์กับเกต: ขาที่อยู่ขวาสุดคือขาออก ที่เหลือเป็นขาเข้า
    return "output" if offs[num][0] == max(q[0] for q in offs.values()) \
        else "input"


BASE = {"res": "R", "cap": "C", "term": "TERM2", "header": "HDR20"}


def variant(kind, offs):
    """คำต่อท้ายชื่อสัญลักษณ์ตามทิศทางการวาง

    **บั๊กที่ทำให้ผังพัง**: เดิมตัวต้านทานทั้งแนวนอน แนวตั้ง และแบบสลับขา
    ใช้นิยามสัญลักษณ์ตัวเดียวกันหมด ตัวที่บันทึกทีหลังทับตัวก่อนหน้า
    ขาของอีก 3 ทิศจึงไปโผล่ผิดที่ สายกับขาไม่บรรจบกัน ผังเลยกลายเป็นเส้นลอย ๆ
    54 ตัวจาก 142 โดนแบบนี้

    แยกนิยามตามทิศไปเลย ปลอดภัยกว่าการหมุนสัญลักษณ์ เพราะไม่ต้องพึ่ง
    การแปลงพิกัดของโปรแกรมปลายทางว่าจะหมุนตรงกับที่เราคิดไหม
    """
    if kind in ("opamp", "opwr", "inv", "icpwr", "header"):
        return ""
    # ตั้งชื่อตาม **ทิศของขา 1 เทียบจุดอ้างอิง** ไม่ใช่ตามแกนที่ยาวกว่า
    # เพราะเทอร์มินอลมีขาซ้อนกันแนวตั้งทั้งคู่ ต่างกันแค่อยู่ซ้ายหรือขวา
    # ถ้าดูแกนที่ยาวกว่าจะได้ชื่อเดียวกันทั้งสองแบบ แล้วนิยามก็ทับกันอีก
    dx, dy = offs["1"]
    if abs(dx) >= abs(dy):
        return "_R" if dx > 0 else "_L"
    return "_U" if dy > 0 else "_D"


def lib_name(kind, ref, offs=None):
    if kind in ("opamp", "opwr"):
        return "MCP6024"
    if kind in ("inv", "icpwr"):
        return "SN74HCT04N"
    return BASE[kind] + (variant(kind, offs) if offs else "")


def unit_of(kind, val, offs):
    if kind == "opamp":
        return SECT_UNIT[val]
    if kind == "opwr":
        return 5
    if kind == "inv":
        return GATE_UNIT[tuple(offs)]
    if kind == "icpwr":
        return 7
    return 1


def build_symbols(insts):
    """รวมชิ้นส่วนทั้งหมดเป็นนิยามสัญลักษณ์ · ชิปหลาย section = หลาย unit"""
    lib = {}
    for kind, ref, val, pins, anchor in insts:
        offs = {n: (snap((x - anchor[0]) * SCALE),
                    snap(-(y - anchor[1]) * SCALE)) for n, (x, y) in pins.items()}
        name = lib_name(kind, ref, offs)
        unit = unit_of(kind, val, tuple(pins))
        prev = lib.setdefault(name, {}).get(unit)
        if prev and prev[1] != offs:
            raise SystemExit(f"สัญลักษณ์ {name} unit {unit} ถูกใช้กับสองรูปทรง "
                             f"— นิยามจะทับกันแล้วขาไปผิดที่ ({ref})")
        lib[name][unit] = (kind, offs)
    out = []
    for name, units in sorted(lib.items()):
        # **ชื่อในนิยามต้องมีชื่อคลังนำหน้าให้ตรงกับที่ชิ้นส่วนอ้างถึง**
        # เดิมนิยามชื่อ "C_L" แต่ชิ้นส่วนอ้าง "ultrasonic:C_L" ชื่อไม่ตรงกัน
        # ตัวนำเข้าหานิยามไม่เจอ เลยทิ้งสัญลักษณ์ทุกตัว เหลือแต่สายลอย ๆ
        # ส่วนชื่อของ unit ข้างในใช้ชื่อเปล่า ไม่ต้องมีคลังนำหน้า ตามรูปแบบ KiCad
        full = f"{LIB}:{name}"
        # ตัวอักษรนำหน้าเลขอ้างอิง · ตัดคำต่อท้ายที่บอกทิศออกก่อน
        pref = {"R": "R", "C": "C", "TERM2": "J",
                "HDR20": "J"}.get(name.split("_")[0], "U")
        out.append(f'    (symbol {_q(full)} (pin_names (offset 0.508)) '
                   f'(in_bom yes) (on_board yes)')
        out.append(f'      (property "Reference" {_q(pref)} (at 0 6.35 0) '
                   f'{_eff()})')
        out.append(f'      (property "Value" {_q(name)} (at 0 -6.35 0) '
                   f'{_eff()})')
        out.append(f'      (property "Footprint" "" (at 0 0 0) '
                   f'{_eff(hide=True)})')
        out.append(f'      (property "Datasheet" "" (at 0 0 0) '
                   f'{_eff(hide=True)})')
        for unit, (kind, offs) in sorted(units.items()):
            out.append(f'      (symbol {_q(f"{name}_{unit}_1")}')
            out += body_graphics(kind, offs)
            for num, (dx, dy) in sorted(offs.items(), key=lambda kv: int(kv[0])):
                out.append(_pin(num, "~", pin_etype(kind, num, offs),
                                dx, dy, _angle(dx, dy)))
            out.append("      )")
        out.append("    )")
    return out, lib


# ---------------------------------------------------------------- ไฟล์ผัง
def export(placed, path, project="main8"):
    """เขียนไฟล์ .kicad_sch หนึ่งไฟล์ · placed = [(แผ่น, เลื่อนแกน x, เลื่อนแกน y)]

    วางเป็นคอลัมน์ได้ ไม่ใช่ต่อกันลงล่างอย่างเดียว เพราะถ้าเรียงลงล่างหมด
    หน้ากระดาษจะกลายเป็นแถบสูง 2 เมตรกว้างครึ่งเมตร เปิดมาแล้วมองไม่เห็นอะไรเลย
    """
    insts, wires, juncs, labels = [], [], [], []
    nc, ncpt = set(), []
    for sh, dx, dy in placed:
        for pin, net in sh.tied.items():
            if net.startswith("NC_"):
                nc.add(pin)
        for kind, ref, val, pins, anchor in sh.inst:
            insts.append((kind, ref, val,
                          {n: (x + dx, y + dy) for n, (x, y) in pins.items()},
                          (anchor[0] + dx, anchor[1] + dy)))
        wires += [((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy))
                  for a, b in sh.wires]
        juncs += [(x + dx, y + dy) for x, y in sh.juncs]
        labels += [(x + dx, y + dy, n) for x, y, n in sh.labels]

    dup = [r for r, c in
           _count(f"{k}:{r}" if k in ("opamp", "inv", "opwr", "icpwr") else r
                  for k, r, _v, _p, _a in insts).items() if c > 1 and ":" not in r]
    if dup:
        raise SystemExit(f"อ้างอิงซ้ำในผัง: {', '.join(sorted(dup))} "
                         f"— แผ่นที่ให้มาซ้อนทับกัน")

    for _k, ref, _v, ps, _a in insts:
        for n, (x, y) in ps.items():
            if f"{ref}.{n}" in nc:
                ncpt.append((x, y))
    check_geometry(insts, wires, labels, nc)
    sym_defs, _lib = build_symbols(insts)
    w = snap(max(sh.w + dx for sh, dx, _ in placed) * SCALE) + 40
    h = snap(max(sh.h + dy for sh, _, dy in placed) * SCALE) + 40
    out = [f"(kicad_sch (version {VERSION}) (generator {_q('ultrasonic01')})",
           f"  (uuid {_uid(project)})",
           f"  (paper {_q('User')} {w} {h})",
           "  (lib_symbols"]
    out += sym_defs
    out.append("  )")

    for x1, y1, x2, y2 in ((a[0], a[1], b[0], b[1]) for a, b in wires):
        if (x1, y1) == (x2, y2):
            continue
        out.append(f"  (wire (pts (xy {snap(x1 * SCALE)} {snap(y1 * SCALE)}) "
                   f"(xy {snap(x2 * SCALE)} {snap(y2 * SCALE)}))")
        out.append(f"    (stroke (width 0) (type default)) "
                   f"(uuid {_uid('w', x1, y1, x2, y2)}))")
    for x, y in juncs:
        out.append(f"  (junction (at {snap(x * SCALE)} {snap(y * SCALE)}) "
                   f"(diameter 0) (color 0 0 0 0) (uuid {_uid('j', x, y)}))")
    # ขาที่ตั้งใจไม่ต่อ ต้องติดเครื่องหมายไว้ ไม่งั้นโปรแกรมจะเตือนว่าลืมต่อ
    for x, y in ncpt:
        out.append(f"  (no_connect (at {snap(x * SCALE)} {snap(y * SCALE)}) "
                   f"(uuid {_uid('nc', x, y)}))")
    for x, y, name in labels:
        out.append(f"  (label {_q(name)} (at {snap(x * SCALE)} "
                   f"{snap(y * SCALE)} 0) {_eff(1.27, justify='left bottom')} "
                   f"(uuid {_uid('l', x, y, name)}))")

    for kind, ref, val, pins, anchor in insts:
        offs = {n: (snap((x - anchor[0]) * SCALE),
                    snap(-(y - anchor[1]) * SCALE)) for n, (x, y) in pins.items()}
        name = lib_name(kind, ref, offs)
        unit = unit_of(kind, val, tuple(pins))
        ax, ay = snap(anchor[0] * SCALE), snap(anchor[1] * SCALE)
        uid = _uid("s", ref, unit, ax, ay)
        out.append(f"  (symbol (lib_id {_q(LIB + ':' + name)}) "
                   f"(at {ax} {ay} 0) (unit {unit})")
        out.append(f"    (in_bom yes) (on_board yes) (dnp no) (uuid {uid})")
        out.append(f"    (property \"Reference\" {_q(ref)} "
                   f"(at {ax} {ay - 7.62} 0) {_eff()})")
        out.append(f"    (property \"Value\" {_q(val or name)} "
                   f"(at {ax} {ay + 7.62} 0) {_eff()})")
        for num in sorted(pins, key=int):
            out.append(f"    (pin {_q(num)} (uuid {_uid('p', ref, unit, num)}))")
        out.append(f"    (instances (project {_q(project)} "
                   f"(path {_q('/' + _uid(project))} "
                   f"(reference {_q(ref)}) (unit {unit}))))")
        out.append("  )")
    out.append(")")

    txt = "\n".join(out) + "\n"
    check_names(txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return len(insts), len(wires), len(labels)


def check_names(txt):
    """ทุกชื่อที่ชิ้นส่วนอ้างถึง ต้องมีนิยามอยู่ในไฟล์เดียวกันจริง

    บั๊กที่ตัวตรวจนี้เกิดมาเพื่อจับ: นิยามตั้งชื่อว่า "C_L" แต่ชิ้นส่วนอ้างถึง
    "ultrasonic:C_L" ชื่อไม่ตรงกัน ตัวนำเข้าหานิยามไม่เจอเลยทิ้งสัญลักษณ์
    ทุกตัว เปิดมาเห็นแต่สายลอย ๆ ไม่มีตัวต้านทานหรือชิปสักตัว

    ตัวตรวจเดิมสองตัวผ่านหมด เพราะตัวหนึ่งดูว่าขาอยู่เน็ตถูกไหม อีกตัวดูว่า
    สายบรรจบกับขาไหม ทั้งคู่ไม่ได้ดูว่า **ปลายทางจะหานิยามสัญลักษณ์เจอไหม**
    """
    import re
    defined = set(re.findall(r'^    \(symbol "([^"]+)"', txt, re.M))
    used = set(re.findall(r'\(lib_id "([^"]+)"', txt))
    missing = sorted(used - defined)
    if missing:
        raise SystemExit(f"ชิ้นส่วนอ้างถึงนิยามที่ไม่มีอยู่: {', '.join(missing)}"
                         f" — ตัวนำเข้าจะทิ้งสัญลักษณ์ เหลือแต่สาย")
    print(f"ตรวจชื่อ: อ้างถึงนิยาม {len(used)} ชื่อ มีครบทุกชื่อ")


def check_geometry(insts, wires, labels, nc):
    """ทุกขาต้องมีอะไรมาบรรจบจริงในพิกัดสุดท้าย

    นี่คือตัวตรวจที่จะจับบั๊กเดิมได้ตั้งแต่แรก: schematic.py ตรวจว่า
    **ขาไหนควรอยู่เน็ตอะไร** ซึ่งผ่านหมด แต่ไม่ได้ตรวจว่า
    **สายไปบรรจบกับขาตรงพิกัดจริงหรือเปล่า** พอสัญลักษณ์ถูกนิยามผิดทิศ
    ขาก็ย้ายที่ สายลอยอยู่เฉย ๆ และไม่มีอะไรร้อง

    ขาหนึ่งนับว่าต่อแล้วถ้าเจอ ปลายสาย หรือ ป้ายชื่อเน็ต หรือ ขาอื่น ที่พิกัดเดียวกัน
    """
    def key(x, y):
        return (snap(x * SCALE), snap(y * SCALE))

    segs = [(key(*a), key(*b)) for a, b in wires]
    ends = {p for seg in segs for p in seg}
    lab = {key(x, y) for x, y, _ in labels}
    pins = {}
    for _k, ref, _v, ps, _a in insts:
        for n, (x, y) in ps.items():
            pins.setdefault(key(x, y), []).append(f"{ref}.{n}")

    def on_a_wire(pt):
        """ขาที่เกาะ **กลาง** สายก็ต่อถึงกัน ไม่ใช่ต้องอยู่ปลายสายเท่านั้น
        เช่นคาปาที่ห้อยลงมาจากรางไฟตรงกลางราง"""
        px, py = pt
        for (ax, ay), (bx, by) in segs:
            if ax == bx == px and min(ay, by) <= py <= max(ay, by):
                return True
            if ay == by == py and min(ax, bx) <= px <= max(ax, bx):
                return True
        return False

    loose = [(pt, who) for pt, who in pins.items()
             if pt not in ends and pt not in lab and len(who) < 2
             and not all(p in nc for p in who) and not on_a_wire(pt)]
    if loose:
        for pt, who in loose[:12]:
            print(f"  ขาลอย {who[0]} ที่ {pt} — ไม่มีสายหรือป้ายมาบรรจบ")
        raise SystemExit(f"ขาที่ไม่มีอะไรต่อ {len(loose)} จาก {len(pins)} จุด "
                         f"— ไฟล์ผังจะเปิดมาเป็นเส้นลอย ๆ ไม่สร้างไฟล์")
    print(f"ตรวจพิกัด: ทุกขาใน {len(pins)} จุดมีสายหรือป้ายมาบรรจบจริง "
          f"· ขาที่ตั้งใจไม่ต่อ {len(nc)} ขา ใส่เครื่องหมายไม่ต่อให้แล้ว")


def _count(it):
    d = {}
    for v in it:
        d[v] = d.get(v, 0) + 1
    return d


def main():
    import schematic as S
    grid, tx, vref = S.sheet_rx_grid(), S.sheet_power_tx(), S.sheet_vref_mcu()
    S.verify([grid, tx, vref])
    placed = [(grid, 0, 0), (tx, grid.w + 120, 0),
              (vref, grid.w + 120, tx.h + 160)]
    path = os.path.join(S.OUT, "main8.kicad_sch")
    n, w, l = export(placed, path)
    print(f"ไฟล์ผังวงจร: ชิ้นส่วน {n} · สาย {w} · ชื่อเน็ต {l}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
