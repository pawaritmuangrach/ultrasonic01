"""สร้างไฟล์ทั้งหมดของทุกบอร์ด — รันไฟล์เดียวจบ

    python pcb/build.py

ออกไฟล์ที่ pcb/out/<ชื่อบอร์ด>/ :
  *.gbr          ลายทองแดง หน้ากากประสาน ซิลค์สกรีน เส้นตัดขอบ  (ส่งโรงงาน)
  *.drl          ตำแหน่งและขนาดรูเจาะ                          (ส่งโรงงาน)
  preview.png    ภาพตรวจด้วยตา ก่อนส่งจริง
  assembly.png   ผังลงชิ้นส่วน ไว้ถือตอนบัดกรี
  *.kicad_pcb    เปิดใน KiCad หรือนำเข้า EasyEDA (File > Import > KiCad)
  BOM.md         รายการชิ้นส่วน
  nets.md        ตารางว่าขาไหนต่อกับขาไหน  (ไว้ไล่วัดด้วยมิเตอร์)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                        # noqa: E402

import geom                                               # noqa: E402
import layout                                             # noqa: E402
import netlist as N                                       # noqa: E402
from gerber import write, bom_rows                        # noqa: E402
from kicad import export as kicad_export, summary as kicad_summary  # noqa: E402
from router import route_board                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PX = 12          # จุดต่อมิลลิเมตร ตอนวาดภาพตรวจ

TOP = (90, 90, 235)          # ลายด้านบน (BGR)
BOT = (60, 175, 60)          # ลายด้านล่าง
PAD = (40, 200, 245)
BG = (22, 22, 26)


def _px(x, y, x0, y0, h):
    """มม. -> พิกเซล · พลิกแกน y ที่จุดเดียวตรงนี้ เพราะภาพนับ y ลง แต่บอร์ดนับขึ้น"""
    return int(round((x - x0) * PX)), int(round((h - (y - y0)) * PX))


def preview(path, pl, r, w, h, x0, y0, assembly=False):
    import cv2
    img = np.full((int(h * PX) + 1, int(w * PX) + 1, 3), BG, np.uint8)
    if not assembly:
        # แผ่นกราวด์ก่อน ลายทับทีหลัง — ลำดับเดียวกับในไฟล์ Gerber
        for L, net in r.pours:
            m = (r.cop[L] == net).astype(np.uint8)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = np.flipud(m).astype(bool)
            img[m] = (34, 74, 34) if L == 1 else (60, 46, 90)
        for L, pts, width, _n in r.tracks:
            col = TOP if L == 0 else BOT
            p = [_px(x, y, x0, y0, h) for x, y in pts]
            for a, b in zip(p, p[1:]):
                cv2.line(img, a, b, col, max(1, int(width * PX)), cv2.LINE_AA)
    for ref, (bx0, by0, bx1, by1) in pl["bodies"].items():
        a = _px(bx0, by0, x0, y0, h)
        b = _px(bx1, by1, x0, y0, h)
        cv2.rectangle(img, a, b, (150, 150, 160), 1, cv2.LINE_AA)
    for pin, (x, y, drill, dia, _n) in r.pads.items():
        c = _px(x, y, x0, y0, h)
        cv2.circle(img, c, max(2, int(dia / 2 * PX)), PAD, -1, cv2.LINE_AA)
        cv2.circle(img, c, max(1, int(drill / 2 * PX)), BG, -1, cv2.LINE_AA)
    for x, y in r.vias:
        cv2.circle(img, _px(x, y, x0, y0, h), max(2, int(0.6 * PX)), (200, 200, 90), -1)
    for ref, (bx0, by0, bx1, by1) in pl["bodies"].items():
        tx, ty = _px(bx0, by1, x0, y0, h)
        lab = f"{ref} {pl['vals'][ref]}" if assembly else ref
        cv2.putText(img, lab, (tx, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (235, 235, 240), 1, cv2.LINE_AA)
    # ขา 1 ทำเครื่องหมายสี่เหลี่ยม — ใส่ชิปกลับด้านคือพังทันที
    for ref in pl["fps"]:
        x, y, _d, dia, _n = r.pads[f"{ref}.1"]
        c = _px(x, y, x0, y0, h)
        k = max(3, int(dia / 2 * PX) + 2)
        cv2.rectangle(img, (c[0] - k, c[1] - k), (c[0] + k, c[1] + k),
                      (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (120, 120, 130), 1)
    head = f"{pl['title']}   {w:.1f} x {h:.1f} mm"
    sub = ("component side - place parts here" if assembly
           else "blue = top copper   green = bottom copper   yellow = pad")
    img = np.vstack([np.full((44, img.shape[1], 3), BG, np.uint8), img])
    cv2.putText(img, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(img, sub, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (150, 150, 160), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def nets_md(pl, name):
    out = [f"# {name} — ตารางการต่อ", "",
           "ไว้ไล่วัดด้วยมิเตอร์โหมดเสียงปี๊บ ขาในแถวเดียวกันต้องปี๊บถึงกันหมด",
           "ขาคนละแถวต้องไม่ปี๊บ ถ้าปี๊บ = ลายช็อต", "",
           "| เน็ต | ขาที่ต่อถึงกัน |", "|---|---|"]
    for net, pins in pl["nets"].items():
        out.append(f"| `{net}` | {' · '.join(pins)} |")
    return "\n".join(out) + "\n"


# หมายเหตุที่ต้องรู้ตอนซื้อและตอนประกอบ — ใส่ไว้ในไฟล์ BOM ของบอร์ดนั้นเลย
# จะได้ไม่ต้องถือเอกสารสองใบตอนไปร้าน
BOM_NOTES = {
    "main8": """**ซื้อซ็อกเก็ต DIP-14 มาด้วย 6 ตัว** อย่าบัดกรีชิปลงบอร์ดตรง ๆ
รูเหมือนกันเป๊ะ ใส่ซ็อกเก็ตแทนได้เลยไม่ต้องแก้บอร์ด เหตุผล:
MCP6024 ไวต่อความร้อนตอนบัดกรี และถ้าชิปตัวไหนเสียก็ถอดเปลี่ยนได้ทันที
ไม่ต้องดูดตะกั่ว 14 ขาบนบอร์ดที่มีแผ่นกราวด์ดูดความร้อน

**JA กับ JB ใช้เฮดเดอร์ตัวเมีย 1x20 ระยะขา 2.54 มม.** สองแถวห่างกัน 17.78 มม.
เป็นที่เสียบ STM32F103C8T6 (Blue Pill) ไม่ต้องบัดกรีบอร์ด STM32 ลงไปตรง ๆ
จะได้ถอดไปแฟลชด้วย ST-Link ได้ และเปลี่ยนตัวได้ถ้าเสีย

**เทอร์มินอลขันสกรู KF301 ระยะขา 5.08 มม. 12 ตัว** สำหรับ TX1-3 · RX1-8 · ไฟ 5V เข้า

**ตัวต้านทาน 1/4W ระยะขา 7.62 มม.** ถ้าซื้อมาแล้วตัวยาวกว่านี้ ให้งอขาลงตรง ๆ ได้""",
}


def bom_md(pl, name, extra=""):
    rows = bom_rows(pl)
    out = [f"# {name} — รายการชิ้นส่วน", "", extra, "",
           "| ค่า | ตัวถัง | จำนวน | อ้างอิง |", "|---|---|---|---|"]
    for val, fp, n, refs in rows:
        out.append(f"| **{val}** | {fp} | {n} | {refs} |")
    out.append("")
    out.append(f"รวม {sum(r[2] for r in rows)} ชิ้น")
    return "\n".join(out) + "\n"


def unrouted(name):
    """ส่งออกไฟล์ KiCad ที่ **วางชิ้นส่วนครบและมีเน็ตครบ แต่ยังไม่เดินลาย**

    ให้เครื่องมือที่มี autorouter จริง (เช่น EasyEDA) เดินลายต่อ
    ส่วนที่โค้ดนี้ทำได้ดีคือความถูกต้องของ netlist กับการวางชิ้นส่วน
    ส่วนการเดินลาย 90 เน็ตบนบอร์ด 340 ขา เกินกำลังตัวเดินลายง่าย ๆ ที่ไม่มีการรื้อ
    """
    from router import Router
    b = N.BOARDS[name]()
    if N.check(b):
        raise SystemExit(f"netlist ของ {name} ผิด")
    pl = geom.place(b, layout.PLACE[name])
    cl = geom.clashes(pl)
    if cl:
        raise SystemExit(f"{name}: ขาวางชนกัน {len(cl)} คู่")
    w, h, x0, y0 = geom.extent(pl)
    r = Router(w, h, x0, y0)
    for mx, my in ((x0 + 3.5, y0 + 3.5), (x0 + w - 3.5, y0 + 3.5),
                   (x0 + 3.5, y0 + h - 3.5), (x0 + w - 3.5, y0 + h - 3.5)):
        r.add_mount(mx, my)
    ids = {net: k for k, net in enumerate(pl["nets"], start=1)}
    for net, pins in pl["nets"].items():
        for pin in pins:
            x, y, d, dia = pl["pad"][pin]
            r.add_pad(pin, x, y, d, dia, ids[net])
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{name}-unrouted.kicad_pcb")
    kicad_export(out, pl, r, w, h, x0, y0)
    # บอร์ดที่ให้เครื่องมืออื่นเดินลายต่อจะไม่ได้ผ่าน build() จึงต้องเขียน
    # เอกสารประกอบตรงนี้เอง ไม่งั้น BOM จะค้างเป็นรุ่นเก่าโดยไม่มีใครรู้
    with open(os.path.join(d, "BOM.md"), "w", encoding="utf-8") as f:
        f.write(bom_md(pl, name, BOM_NOTES.get(name, "")))
    with open(os.path.join(d, "nets.md"), "w", encoding="utf-8") as f:
        f.write(nets_md(pl, name))
    real = [n for n in pl["nets"] if not n.startswith("NC_")]
    print(f"{name}: {w:.1f} x {h:.1f} mm · ชิ้นส่วน {len(pl['fps'])} · "
          f"ขา {len(r.pads)} · เน็ตจริง {len(real)} · ยังไม่เดินลาย")
    print(f"  -> {out}")
    return out


def build(name, verbose=True):
    b = N.BOARDS[name]()
    errs = N.check(b)
    if errs:
        raise SystemExit(f"netlist ของ {name} ผิด: {errs[0]}")
    pl = geom.place(b, layout.PLACE[name])
    # ตรวจการวางก่อนเดินลาย — ถูกกว่ามากที่จะรู้ตรงนี้ (ไม่ถึงวินาที)
    # แทนที่จะรอเดินลายจบแล้วอ่านจากข้อความว่าระยะห่างไม่พอ
    cl = geom.clashes(pl)
    if cl:
        raise SystemExit(f"{name}: ขาวางชนกัน {len(cl)} คู่ · เช่น "
                         f"{cl[0][0]} กับ {cl[0][1]} ห่าง {cl[0][2]} "
                         f"ต้องการ {cl[0][3]} มม.")
    w, h, x0, y0 = geom.extent(pl)
    t = time.time()
    r, errs = route_board(pl)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    files = write(d, name, pl, r, w, h, x0, y0)
    # ไฟล์ KiCad พาชื่อเน็ตไปด้วย ต่างจาก Gerber ที่มีแต่รูปทองแดง
    # EasyEDA นำเข้าแล้วตรวจการเชื่อมต่อกับรัน DRC ให้ได้ = ตัวตรวจที่ไม่ใช่ของเราเอง
    kicad_export(os.path.join(d, f"{name}.kicad_pcb"), pl, r, w, h, x0, y0)
    preview(os.path.join(d, "preview.png"), pl, r, w, h, x0, y0)
    preview(os.path.join(d, "assembly.png"), pl, r, w, h, x0, y0, assembly=True)
    with open(os.path.join(d, "nets.md"), "w", encoding="utf-8") as f:
        f.write(nets_md(pl, name))
    with open(os.path.join(d, "BOM.md"), "w", encoding="utf-8") as f:
        f.write(bom_md(pl, name, BOM_NOTES.get(name, "")))
    top = sum(1 for L, *_ in r.tracks if L == 0)
    if verbose:
        state = "ผ่าน" if not errs else f"{len(errs)} ปัญหา"
        print(f"{name:6s} {w:5.1f} x {h:5.1f} mm · ขา {len(r.pads):3d} · รู {len(r.holes):3d} · "
              f"ลายบน {top:2d} ล่าง {len(r.tracks)-top:3d} · เวีย {len(r.vias):2d} · "
              f"{time.time()-t:4.1f}s · {state}")
        for e in errs:
            print("    -", e)
    return dict(name=name, w=w, h=h, pads=len(r.pads), holes=len(r.holes),
                top=top, bot=len(r.tracks) - top, vias=len(r.vias),
                errs=errs, files=files, dir=d)


if __name__ == "__main__":
    want = sys.argv[1:] or list(N.BOARDS)
    res = [build(n) for n in want]
    bad = sum(len(x["errs"]) for x in res)
    print(f"\n{'ทุกบอร์ดผ่านการตรวจ' if not bad else f'ยังมี {bad} ปัญหา'} "
          f"· ไฟล์อยู่ที่ {OUT}")
