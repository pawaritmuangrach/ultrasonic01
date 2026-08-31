"""สร้างไฟล์ทั้งหมดของทุกบอร์ด — รันไฟล์เดียวจบ

    python pcb/build.py

ออกไฟล์ที่ pcb/out/<ชื่อบอร์ด>/ :
  *.gbr          ลายทองแดง หน้ากากประสาน ซิลค์สกรีน เส้นตัดขอบ  (ส่งโรงงาน)
  *.drl          ตำแหน่งและขนาดรูเจาะ                          (ส่งโรงงาน)
  preview.png    ภาพตรวจด้วยตา ก่อนส่งจริง
  assembly.png   ผังลงชิ้นส่วน ไว้ถือตอนบัดกรี
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


def bom_md(pl, name, extra=""):
    rows = bom_rows(pl)
    out = [f"# {name} — รายการชิ้นส่วน", "", extra, "",
           "| ค่า | ตัวถัง | จำนวน | อ้างอิง |", "|---|---|---|---|"]
    for val, fp, n, refs in rows:
        out.append(f"| **{val}** | {fp} | {n} | {refs} |")
    out.append("")
    out.append(f"รวม {sum(r[2] for r in rows)} ชิ้น")
    return "\n".join(out) + "\n"


def build(name, verbose=True):
    b = N.BOARDS[name]()
    errs = N.check(b)
    if errs:
        raise SystemExit(f"netlist ของ {name} ผิด: {errs[0]}")
    pl = geom.place(b, layout.PLACE[name])
    w, h, x0, y0 = geom.extent(pl)
    t = time.time()
    r, errs = route_board(pl)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    files = write(d, name, pl, r, w, h, x0, y0)
    preview(os.path.join(d, "preview.png"), pl, r, w, h, x0, y0)
    preview(os.path.join(d, "assembly.png"), pl, r, w, h, x0, y0, assembly=True)
    with open(os.path.join(d, "nets.md"), "w", encoding="utf-8") as f:
        f.write(nets_md(pl, name))
    with open(os.path.join(d, "BOM.md"), "w", encoding="utf-8") as f:
        f.write(bom_md(pl, name))
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
