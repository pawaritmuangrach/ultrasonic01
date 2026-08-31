"""เขียนไฟล์ Gerber (RS-274X) และไฟล์เจาะ (Excellon) — รูปแบบที่โรงงานทุกเจ้ารับ

Gerber เป็นไฟล์ข้อความล้วน ไม่ใช่รูปภาพ แต่ละบรรทัดคือคำสั่งวาด
โรงงานเอาไปทำฟิล์มแล้วกัดทองแดงตามนั้น หนึ่งไฟล์ = หนึ่งชั้น

หน่วยในไฟล์เป็นมิลลิเมตร ความละเอียด 6 ตำแหน่งทศนิยม (`%FSLAX46Y46*%`)
ละเอียดกว่าที่เครื่องจักรทำได้จริงมาก จึงไม่มีทางเป็นต้นตอของความคลาด
"""
import os

SCALE = 10 ** 6          # 1 มม. = 1,000,000 หน่วยในไฟล์


def _c(v):
    return f"{int(round(v * SCALE))}"


class Gerber:
    def __init__(self, kind):
        self.ap = {}
        self.body = []
        self.head = ["%FSLAX46Y46*%", "%MOMM*%",
                     f"%TF.FileFunction,{kind}*%", "%TF.Part,Single*%", "%LPD*%"]

    def _aper(self, shape):
        if shape not in self.ap:
            self.ap[shape] = 10 + len(self.ap)
        return self.ap[shape]

    def flash(self, x, y, dia):
        d = self._aper(("C", round(dia, 4)))
        self.body.append(f"D{d}*")
        self.body.append(f"X{_c(x)}Y{_c(y)}D03*")

    def line(self, pts, width):
        d = self._aper(("C", round(width, 4)))
        self.body.append(f"D{d}*")
        x, y = pts[0]
        self.body.append(f"X{_c(x)}Y{_c(y)}D02*")
        for x, y in pts[1:]:
            self.body.append(f"X{_c(x)}Y{_c(y)}D01*")

    def region(self, pts, dark=True):
        """พื้นที่ทึบรูปอิสระ (G36/G37) — ใช้เทแผ่นกราวด์

        Gerber วาดตามลำดับบรรทัด ของที่มาทีหลังทับของเดิม จึงต้องเทแผ่นก่อน
        แล้วค่อยวาดแป้นกับลายทับ ไม่งั้นรูโหว่ของแผ่นจะไปลบแป้นที่วาดไว้แล้ว
        """
        self.body.append("%LPD*%" if dark else "%LPC*%")
        self.body.append("G36*")
        x, y = pts[0]
        self.body.append(f"X{_c(x)}Y{_c(y)}D02*")
        for x, y in pts[1:]:
            self.body.append(f"X{_c(x)}Y{_c(y)}D01*")
        self.body.append("G37*")
        self.body.append("%LPD*%")

    def rect(self, x0, y0, x1, y1, width):
        self.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], width)

    def text(self):
        out = list(self.head)
        for (shape, size), n in self.ap.items():
            out.append(f"%ADD{n}{shape},{size:.4f}*%")
        out += self.body + ["M02*"]
        return "\n".join(out) + "\n"


def excellon(holes):
    """ไฟล์เจาะ จัดกลุ่มตามขนาดดอกสว่าน — เครื่องเปลี่ยนดอกทีเดียวต่อกลุ่ม"""
    sizes = sorted({round(d, 3) for _, _, d in holes})
    out = ["M48", "METRIC,TZ"]
    for i, d in enumerate(sizes, start=1):
        out.append(f"T{i}C{d:.3f}")
    out.append("%")
    for i, d in enumerate(sizes, start=1):
        out.append(f"T{i}")
        for x, y, dd in holes:
            if round(dd, 3) == d:
                out.append(f"X{x:.3f}Y{y:.3f}")
    out += ["T0", "M30"]
    return "\n".join(out) + "\n"


def write(outdir, name, pl, r, w, h, x0, y0):
    """เขียนไฟล์ครบชุดของบอร์ดหนึ่ง คืนรายชื่อไฟล์ที่เขียน"""
    os.makedirs(outdir, exist_ok=True)
    cu = {0: Gerber("Copper,L1,Top"), 1: Gerber("Copper,L2,Bot")}
    mask = {0: Gerber("Soldermask,Top"), 1: Gerber("Soldermask,Bot")}
    silk = Gerber("Legend,Top")
    edge = Gerber("Profile,NP")

    # เทแผ่นก่อนเสมอ ลายกับแป้นวาดทับทีหลัง (ดูเหตุผลใน Gerber.region)
    for L, net in r.pours:
        _emit_pour(cu[L], r.cop[L] == net, r.x0, r.y0, r.step)
    for L, pts, width, _net in r.tracks:
        cu[L].line(pts, width)
    for x, y, _d, dia, _n in r.pads.values():
        for L in (0, 1):
            cu[L].flash(x, y, dia)
            mask[L].flash(x, y, dia + 0.1)      # เปิดหน้ากากกว้างกว่าแป้นเล็กน้อย
    for x, y in r.vias:
        for L in (0, 1):
            cu[L].flash(x, y, 1.2)

    for mx, my, md in r.mounts:      # วงบอกรูยึด ไม่มีทองแดง ไม่ต้องบัดกรี
        silk.line([(mx - md / 2 - 0.6, my), (mx + md / 2 + 0.6, my)], 0.2)
        silk.line([(mx, my - md / 2 - 0.6), (mx, my + md / 2 + 0.6)], 0.2)
    edge.rect(x0, y0, x0 + w, y0 + h, 0.15)
    for ref, (bx0, by0, bx1, by1) in pl["bodies"].items():
        silk.rect(bx0, by0, bx1, by1, 0.15)
    # จุดบอกขา 1 ของทุกตัวถัง — ใส่ผิดด้านคือพังทันที
    for ref, fp in pl["fps"].items():
        p1 = pl["pad"][f"{ref}.1"]
        silk.line([(p1[0] - 1.6, p1[1]), (p1[0] - 1.2, p1[1])], 0.3)

    files = {
        f"{name}-F_Cu.gbr": cu[0].text(),
        f"{name}-B_Cu.gbr": cu[1].text(),
        f"{name}-F_Mask.gbr": mask[0].text(),
        f"{name}-B_Mask.gbr": mask[1].text(),
        f"{name}-F_Silkscreen.gbr": silk.text(),
        f"{name}-Edge_Cuts.gbr": edge.text(),
        f"{name}.drl": excellon(r.holes),
    }
    for fn, body in files.items():
        with open(os.path.join(outdir, fn), "w", encoding="ascii") as f:
            f.write(body)
    return sorted(files)


def _emit_pour(g, mask, x0, y0, step):
    """แปลงแผ่นทองแดงจากตารางเป็นรูปหลายเหลี่ยม

    เส้นขอบที่ได้ลากผ่าน **กลางเซลล์ขอบ** จึงหดเข้าข้างในครึ่งเซลล์เสมอ
    แปลว่าระยะห่างจากเน็ตอื่นได้เพิ่มขึ้น ไม่มีทางลดลง — ปลอดภัยเสมอ
    """
    import cv2
    import numpy as np
    m = (mask.astype(np.uint8)) * 255
    cont, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return
    for k, c in enumerate(cont):
        if len(c) < 3:
            continue
        pts = [(x0 + int(q[0][0]) * step, y0 + int(q[0][1]) * step) for q in c]
        g.region(pts + [pts[0]], dark=(hier[0][k][3] == -1))


def bom_rows(pl):
    """รวมชิ้นส่วนที่ค่าเหมือนกันเข้าด้วยกัน เรียงตามชนิด"""
    from collections import defaultdict
    g = defaultdict(list)
    for ref, val in pl["vals"].items():
        g[(val, pl["fps"][ref])].append(ref)
    rows = []
    for (val, fp), refs in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append((val, fp, len(refs), " ".join(sorted(refs))))
    return rows
