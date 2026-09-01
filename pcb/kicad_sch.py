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


def lib_name(kind, ref):
    if kind in ("opamp", "opwr"):
        return "MCP6024"
    if kind in ("inv", "icpwr"):
        return "SN74HCT04N"
    return {"res": "R", "cap": "C", "term": "TERM2", "header": "HDR20"}[kind]


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
        name = lib_name(kind, ref)
        offs = {n: (snap((x - anchor[0]) * SCALE),
                    snap(-(y - anchor[1]) * SCALE)) for n, (x, y) in pins.items()}
        unit = unit_of(kind, val, tuple(pins))
        lib.setdefault(name, {})[unit] = (kind, offs)
    out = []
    for name, units in sorted(lib.items()):
        pref = {"R": "R", "C": "C", "TERM2": "J", "HDR20": "J"}.get(name, "U")
        out.append(f'    (symbol {_q(name)} (pin_names (offset 0.508)) '
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
def export(sheets, path, project="main8"):
    """เขียนไฟล์ .kicad_sch หนึ่งไฟล์ รวมทุกแผ่นที่ให้มาวางต่อกันลงล่าง"""
    insts, wires, juncs, labels = [], [], [], []
    dy = 0
    for sh in sheets:
        for kind, ref, val, pins, anchor in sh.inst:
            insts.append((kind, ref, val,
                          {n: (x, y + dy) for n, (x, y) in pins.items()},
                          (anchor[0], anchor[1] + dy)))
        wires += [((a[0], a[1] + dy), (b[0], b[1] + dy)) for a, b in sh.wires]
        juncs += [(x, y + dy) for x, y in sh.juncs]
        labels += [(x, y + dy, n) for x, y, n in sh.labels]
        dy += sh.h + 200

    dup = [r for r, c in
           _count(f"{k}:{r}" if k in ("opamp", "inv", "opwr", "icpwr") else r
                  for k, r, _v, _p, _a in insts).items() if c > 1 and ":" not in r]
    if dup:
        raise SystemExit(f"อ้างอิงซ้ำในผัง: {', '.join(sorted(dup))} "
                         f"— แผ่นที่ให้มาซ้อนทับกัน")

    sym_defs, _lib = build_symbols(insts)
    w = snap(max(sh.w for sh in sheets) * SCALE) + 40
    h = snap(dy * SCALE) + 40
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
    for x, y, name in labels:
        out.append(f"  (label {_q(name)} (at {snap(x * SCALE)} "
                   f"{snap(y * SCALE)} 0) {_eff(1.27, justify='left bottom')} "
                   f"(uuid {_uid('l', x, y, name)}))")

    for kind, ref, val, pins, anchor in insts:
        name = lib_name(kind, ref)
        unit = unit_of(kind, val, tuple(pins))
        ax, ay = snap(anchor[0] * SCALE), snap(anchor[1] * SCALE)
        uid = _uid("s", ref, unit, ax, ay)
        out.append(f"  (symbol (lib_id {_q('ultrasonic:' + name)}) "
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

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return len(insts), len(wires), len(labels)


def _count(it):
    d = {}
    for v in it:
        d[v] = d.get(v, 0) + 1
    return d


def main():
    import schematic as S
    sheets = [S.sheet_power_tx(), S.sheet_vref_mcu(), S.sheet_rx_all()]
    S.verify(sheets)
    path = os.path.join(S.OUT, "main8.kicad_sch")
    n, w, l = export(sheets, path)
    print(f"ไฟล์ผังวงจร: ชิ้นส่วน {n} · สาย {w} · ชื่อเน็ต {l}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
