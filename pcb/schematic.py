"""วาดผังวงจร (schematic) เป็นลายเส้น สร้างจาก netlist.py โดยตรง

    python pcb/schematic.py

ทำไมต้องมี ทั้งที่มีไฟล์บอร์ดแล้ว:
  ไฟล์บอร์ดบอกว่า "ลายทองแดงวิ่งยังไง" แต่อ่านแล้วไม่รู้ว่า **วงจรทำงานยังไง**
  ผังวงจรอ่านแล้วเห็นทันทีว่าสัญญาณเดินจากหัวรับไปถึง ADC ผ่านอะไรบ้าง
  เอาไว้ให้คนอื่นตรวจ ให้ตัวเองย้อนดูตอนแก้ปัญหา และใช้ได้แม้ไม่มี EasyEDA

**ผังนี้ถูกตรวจกับ netlist.py ทุกครั้งที่สร้าง**
  ผังที่วาดด้วยมือมักเพี้ยนจากบอร์ดจริงหลังแก้อะไรไปสักพัก แล้วไม่มีใครรู้ตัว
  ที่นี่ทุกขาที่วาดต้องประกาศว่าอยู่เน็ตอะไร แล้ว verify() เทียบกับ netlist.py
  ถ้าไม่ตรงแม้ขาเดียวจะหยุดทันที ไม่ยอมสร้างไฟล์ออกมา
  แปลว่า **ถ้าไฟล์ออกมาได้ = ผังตรงกับบอร์ดจริงแน่นอน**

ออกเป็น SVG (เปิดด้วยเบราว์เซอร์ ขยายไม่แตก แก้ด้วย Inkscape ได้)
และ PNG (ไว้ดูเร็ว ๆ หรือแปะในเอกสาร)
"""
import os

import netlist as N

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "main8")

BG, INK, WIRE = "#ffffff", "#1a1a1a", "#1a5fb4"
HOT, DIM, PWR = "#c01c28", "#8a8a8a", "#2a7a2a"
LW, FONT = 2.0, "Tahoma, Arial, sans-serif"


class Sheet:
    """ผังหนึ่งแผ่น — เก็บคำสั่งวาดไว้ก่อน แล้วค่อยแปลงเป็น SVG หรือ PNG

    ที่เก็บเป็นคำสั่งกลาง ๆ แทนการเขียน SVG ตรง ๆ เพราะต้องออกสองรูปแบบ
    ถ้าเขียนแยกกันสองที่ สองที่นั้นจะค่อย ๆ เพี้ยนจากกันเองโดยไม่มีใครรู้
    """

    def __init__(self, key, w, h, title, sub=""):
        self.key, self.w, self.h = key, w, h
        self.ops = []
        self.pin = {}      # "REF.n" -> (x, y)  ขาที่วาดลงไปจริง
        self.tied = {}     # "REF.n" -> ชื่อเน็ตที่ประกาศ ใช้ตรวจกับ netlist
        # สามอย่างล่างนี้จดไว้เพื่อแปลงเป็นไฟล์ผังวงจรจริง (kicad_sch.py)
        # ภาพ SVG ใช้ ops อย่างเดียว แต่ไฟล์ผังต้องรู้ว่าอะไรเป็น "ชิ้นส่วน"
        # อะไรเป็น "สาย" อะไรเป็น "ชื่อเน็ต" ซึ่งดูจากคำสั่งวาดอย่างเดียวไม่ออก
        self.inst = []     # ชิ้นส่วน: (kind, ref, val, {เลขขา: (x, y)})
        self.wires = []    # สาย: ((x1, y1), (x2, y2))
        self.juncs = []    # จุดต่อ
        self.labels = []   # ชื่อเน็ตที่ผูกกับจุดบนสาย: (x, y, ชื่อ)
        self.text(w / 2, 44, title, 26, "middle", True)
        if sub:
            self.text(w / 2, 72, sub, 15, "middle", color=DIM)

    # ---- คำสั่งวาดพื้นฐาน
    def line(self, x1, y1, x2, y2, color=INK, w=LW):
        self.ops.append(("line", x1, y1, x2, y2, color, w))

    def rect(self, x1, y1, x2, y2, color=INK, w=LW, fill=None):
        self.ops.append(("rect", x1, y1, x2, y2, color, w, fill))

    def circle(self, x, y, r, color=INK, w=LW, fill=None):
        self.ops.append(("circ", x, y, r, color, w, fill))

    def text(self, x, y, s, size=13, anchor="start", bold=False, color=INK):
        self.ops.append(("text", x, y, s, size, anchor, bold, color))

    def poly(self, pts, color=INK, w=LW, fill=None):
        self.ops.append(("poly", pts, color, w, fill))

    # ---- สายไฟ
    def wire(self, *pts):
        """ลากสายผ่านทุกจุดที่ให้มาตามลำดับ"""
        for a, b in zip(pts, pts[1:]):
            self.line(a[0], a[1], b[0], b[1], WIRE)
            self.wires.append((tuple(a), tuple(b)))

    def dot(self, x, y):
        """จุดต่อ — สายตัดกันแล้วมีจุด = ต่อถึงกัน ไม่มีจุด = ข้ามไปเฉย ๆ"""
        self.circle(x, y, 4.5, WIRE, 0, WIRE)
        self.juncs.append((x, y))

    def gnd(self, x, y):
        """สัญลักษณ์กราวด์ — นับเป็นชื่อเน็ต GND ที่ผูกกับจุดนี้"""
        self.labels.append((x, y, "GND"))
        self.line(x, y, x, y + 14, WIRE)
        for k, w in enumerate((22, 14, 6)):
            self.line(x - w / 2, y + 14 + k * 7, x + w / 2, y + 14 + k * 7, WIRE)

    def rail(self, x1, x2, y, name, color=PWR):
        """รางไฟที่พาดยาว — ตัวรางเองคือสายเส้นหนึ่ง และมีชื่อเน็ตกำกับ"""
        self.line(x1, y, x2, y, color, 3.2)
        self.wires.append(((x1, y), (x2, y)))
        self.labels.append((x1, y, name))
        self.text(x1 - 12, y + 5, name, 15, "end", True, color)

    def netlabel(self, x, y, name, anchor="middle", at=None):
        """ชื่อเน็ต · at คือจุดบนสายที่ชื่อนี้เกาะอยู่จริงทางไฟฟ้า

        ต้องแยกจากตำแหน่งตัวอักษร เพราะตัวอักษรมักต้องเยื้องออกมาให้อ่านออก
        แต่ทางไฟฟ้ามันต้องเกาะปลายสายเป๊ะ ๆ ไม่งั้นไฟล์ผังจะได้เน็ตที่ขาดตอน
        """
        self.text(x, y, name, 13, anchor, True, WIRE)
        if at is not None:
            self.labels.append((at[0], at[1], name))

    def tie(self, net, *pins):
        """ประกาศว่าขาเหล่านี้อยู่เน็ตเดียวกัน — verify() จะเอาไปเทียบ netlist"""
        for p in pins:
            self.tied[p] = net

    # ---- สัญลักษณ์
    def _pins(self, ref, pts, kind=None, val="", anchor=None):
        for i, p in enumerate(pts, start=1):
            self.pin[f"{ref}.{i}"] = p
        if kind:
            if anchor is None:      # ค่าปริยาย: กึ่งกลางระหว่างขา
                anchor = (sum(q[0] for q in pts) / len(pts),
                          sum(q[1] for q in pts) / len(pts))
            self.inst.append((kind, ref, val,
                              {str(i): p for i, p in enumerate(pts, start=1)},
                              anchor))
        return pts

    def res(self, ref, val, x, y, vert=False, flip=False):
        """ตัวต้านทาน ยาว 80 · ขา 1 อยู่ซ้าย (แนวตั้ง = อยู่บน) เว้นแต่สั่ง flip"""
        if vert:
            self.rect(x - 13, y + 20, x + 13, y + 60)
            self.line(x, y, x, y + 20)
            self.line(x, y + 60, x, y + 80)
            self.text(x + 21, y + 36, ref, 13)
            self.text(x + 21, y + 54, val, 13, color=HOT)
            a, b = (x, y), (x, y + 80)
        else:
            self.rect(x + 20, y - 13, x + 60, y + 13)
            self.line(x, y, x + 20, y)
            self.line(x + 60, y, x + 80, y)
            self.text(x + 40, y - 21, ref, 13, "middle")
            self.text(x + 40, y + 31, val, 13, "middle", color=HOT)
            a, b = (x, y), (x + 80, y)
        return self._pins(ref, [b, a] if flip else [a, b], "res", val)

    def cap(self, ref, val, x, y, vert=False, flip=False):
        """ตัวเก็บประจุ ยาว 50 · ขา 1 อยู่ซ้าย (แนวตั้ง = อยู่บน) เว้นแต่สั่ง flip"""
        if vert:
            self.line(x, y, x, y + 20)
            self.line(x, y + 30, x, y + 50)
            self.line(x - 17, y + 20, x + 17, y + 20)
            self.line(x - 17, y + 30, x + 17, y + 30)
            self.text(x + 24, y + 18, ref, 13)
            self.text(x + 24, y + 36, val, 13, color=HOT)
            a, b = (x, y), (x, y + 50)
        else:
            self.line(x, y, x + 20, y)
            self.line(x + 30, y, x + 50, y)
            self.line(x + 20, y - 17, x + 20, y + 17)
            self.line(x + 30, y - 17, x + 30, y + 17)
            self.text(x + 25, y - 24, ref, 13, "middle")
            self.text(x + 25, y + 34, val, 13, "middle", color=HOT)
            a, b = (x, y), (x + 50, y)
        return self._pins(ref, [b, a] if flip else [a, b], "cap", val)

    def opamp(self, ref, sect, x, y, pn, pp, po):
        """ออปแอมป์หนึ่ง section · **ขาบวกอยู่บน ขาลบอยู่ล่าง**

        ปกติเขาวาดลบไว้บน แต่วงจรนี้เป็นแบบไม่กลับเฟส สายป้อนกลับจึงต้องวิ่ง
        กลับมาที่ขาลบ ถ้าวางลบไว้บนตามปกติ สายป้อนกลับกับสายสัญญาณเข้าจะตัดกัน
        สลับขึ้นล่างแล้วสายป้อนกลับอ้อมด้านล่างได้ ทั้งแผ่นจึงไม่มีสายตัดกันเลย
        """
        self.poly([(x, y - 46), (x, y + 46), (x + 82, y)], INK, LW, "#f2f5fa")
        self.text(x + 13, y - 16, "+", 21)
        self.text(x + 13, y + 32, "-", 24)
        self.text(x + 30, y + 8, sect, 15, "start", True, DIM)
        self.text(x + 26, y - 58, f"{ref}{sect}", 15, "middle", True)
        self.line(x - 20, y - 25, x, y - 25)
        self.line(x - 20, y + 25, x, y + 25)
        self.line(x + 82, y, x + 102, y)
        self.pin[f"{ref}.{pp}"] = (x - 20, y - 25)
        self.pin[f"{ref}.{pn}"] = (x - 20, y + 25)
        self.pin[f"{ref}.{po}"] = (x + 102, y)
        self.text(x - 26, y - 32, str(pp), 11, "end", False, DIM)
        self.text(x - 26, y + 18, str(pn), 11, "end", False, DIM)
        self.text(x + 108, y - 8, str(po), 11, "start", False, DIM)
        self.inst.append(("opamp", ref, sect,
                          {str(pp): self.pin[f"{ref}.{pp}"],
                           str(pn): self.pin[f"{ref}.{pn}"],
                           str(po): self.pin[f"{ref}.{po}"]}, (x, y)))
        return (self.pin[f"{ref}.{pp}"], self.pin[f"{ref}.{pn}"],
                self.pin[f"{ref}.{po}"])

    def inverter(self, ref, x, y, pin_in, pin_out):
        """หนึ่งเกตกลับสัญญาณของ 74HCT04"""
        self.poly([(x, y - 30), (x, y + 30), (x + 56, y)], INK, LW, "#f2f5fa")
        self.circle(x + 63, y, 7, INK, LW, BG)
        self.line(x - 20, y, x, y)
        self.line(x + 70, y, x + 90, y)
        self.text(x + 30, y - 40, f"ขา {pin_in} -> {pin_out}", 12, "middle",
                  False, DIM)
        self.pin[f"{ref}.{pin_in}"] = (x - 20, y)
        self.pin[f"{ref}.{pin_out}"] = (x + 90, y)
        self.inst.append(("inv", ref, f"{pin_in}/{pin_out}",
                          {str(pin_in): self.pin[f"{ref}.{pin_in}"],
                           str(pin_out): self.pin[f"{ref}.{pin_out}"]},
                          (x + 35, y)))
        return self.pin[f"{ref}.{pin_in}"], self.pin[f"{ref}.{pin_out}"]

    def header(self, ref, x, y, labels, pitch=22):
        """เฮดเดอร์ตัวเมียสำหรับเสียบ Blue Pill · ขาออกทางขวา"""
        h = (len(labels) - 1) * pitch
        self.rect(x, y - 18, x + 96, y + h + 18, INK, LW, "#f2f5fa")
        self.text(x + 48, y - 28, ref, 15, "middle", True)
        pts = []
        for i, lab in enumerate(labels):
            yy = y + i * pitch
            self.line(x + 96, yy, x + 120, yy)
            self.text(x + 8, yy + 4, str(i + 1), 11, "start", False, DIM)
            self.text(x + 88, yy + 4, lab, 12, "end")
            pts.append((x + 120, yy))
        return self._pins(ref, pts, "header", "1x20 female hdr",
                          (x + 48, y + h / 2))

    def term(self, ref, val, x, y, left=False):
        """เทอร์มินอลขันสกรู 2 ขา · ขา 1 อยู่บน ขา 2 อยู่ล่าง

        left=True ให้สายออกทางซ้าย ใช้กับตัวที่วางไว้ริมขวาของแผ่น
        ไม่งั้นสายจะต้องลากผ่านตัวกล่องเอง ซึ่งดูเหมือนสายลัดวงจร
        """
        self.rect(x, y - 34, x + 62, y + 34, INK, LW, "#fff3e3")
        for dy in (-18, 18):
            self.circle(x + 31, y + dy, 9, INK, LW, BG)
            self.line(x + 26, y + dy - 5, x + 36, y + dy + 5)
            if left:
                self.line(x, y + dy, x - 30, y + dy)
            else:
                self.line(x + 62, y + dy, x + 92, y + dy)
        self.text(x + 31, y - 44, ref, 13, "middle", True)
        self.text(x + 31, y + 57, val, 15, "middle", True, HOT)
        ex = x - 30 if left else x + 92
        return self._pins(ref, [(ex, y - 18), (ex, y + 18)], "term", val,
                          (x + 31, y))


# ---------------------------------------------------------------- แผ่นที่ 1
def sheet_power_tx():
    s = Sheet("1-power-tx", 1580, 920, "แผ่น 1/3 · ไฟเลี้ยง และภาคส่ง 3 หัว",
              "74HCT04 หนึ่งตัวมี 6 เกต ใช้เกตคู่กันต่อ 1 หัวส่ง จึงได้ 3 หัวพอดี")

    p1, p2 = s.term("JP", "5V เข้า", 60, 190)
    s.rail(240, 1210, 152, "P5V")
    s.wire(p1, (240, 172), (240, 152))
    s.dot(240, 152)
    s.wire(p2, (320, 208), (320, 262))
    s.gnd(320, 262)
    s.tie("P5V", "JP.1")
    s.tie("GND", "JP.2")

    for ref, val, x in (("CB1", "10uF", 420), ("CD1", "100nF", 520)):
        s.cap(ref, val, x, 152, vert=True)
        s.dot(x, 152)
        s.wire((x, 202), (x, 240))
        s.gnd(x, 240)
        s.tie("P5V", f"{ref}.1")
        s.tie("GND", f"{ref}.2")
    s.text(600, 200, "คาปาคู่นี้จ่ายกระแสพุ่งตอนเกตสลับสถานะ", 13, color=DIM)
    s.text(600, 222, "ตัวใหญ่เก็บพลังไว้ ตัวเล็กตอบสนองเร็ว ต้องมีทั้งคู่", 13,
           color=DIM)

    s.rect(1060, 196, 1180, 266, INK, LW, "#f2f5fa")
    s.text(1120, 228, "U1", 17, "middle", True)
    s.text(1120, 250, "SN74HCT04N", 12, "middle", False, DIM)
    s.wire((1120, 152), (1120, 196))
    s.dot(1120, 152)
    s.wire((1120, 266), (1120, 300))
    s.gnd(1120, 300)
    s.inst.append(("icpwr", "U1", "SN74HCT04N",
                   {"14": (1120, 196), "7": (1120, 266)}, (1120, 231)))
    s.tie("P5V", "U1.14")
    s.tie("GND", "U1.7")
    s.text(1196, 210, "ขา 14 = ไฟเลี้ยง", 12, color=DIM)
    s.text(1196, 258, "ขา 7 = กราวด์", 12, color=DIM)

    for i, gin, gm_a, gm_b, gout in ((1, 1, 2, 3, 4), (2, 13, 12, 11, 10),
                                     (3, 5, 6, 9, 8)):
        y = 400 + (i - 1) * 165
        s.text(60, y - 54, f"หัวส่งที่ {i}", 17, "start", True)
        ia, oa = s.inverter("U1", 250, y, gin, gm_a)
        ib, ob = s.inverter("U1", 430, y, gm_b, gout)
        s.wire((150, y), ia)
        s.netlabel(144, y - 9, f"TX{i}_IN", "end", at=(150, y))
        s.wire(oa, ib)
        # ข้อความเฉย ๆ ไม่ผูกเป็นชื่อเน็ต — ทั้งสามหัวส่งมีเน็ตกลางคนละเน็ต
        # ถ้าตั้งชื่อซ้ำกันว่า MID ทั้งสามอัน ไฟล์ผังจะลัดวงจรสามหัวเข้าด้วยกัน
        s.text(390, y - 13, f"TX{i}_MID", 12, "middle", True, DIM)
        ra, rb = s.res(f"RT{i}", "100R", 600, y)
        s.wire(ob, ra)
        ta, tb = s.term(f"T{i}", f"TX{i}", 830, y, left=True)
        s.wire(rb, (740, y), (740, y - 18), ta)
        s.wire(tb, (776, y + 18), (776, y + 58))
        s.gnd(776, y + 58)
        s.tie(f"TX{i}_IN", f"U1.{gin}")
        s.tie(f"TX{i}_MID", f"U1.{gm_a}", f"U1.{gm_b}")
        s.tie(f"TX{i}_OUT", f"U1.{gout}", f"RT{i}.1")
        s.tie(f"TX{i}_DRV", f"RT{i}.2", f"T{i}.1")
        s.tie("GND", f"T{i}.2")
        s.text(925, y - 6, f"-> หัวส่ง TX{i}", 15, "start", True, HOT)

    s.text(60, 878, "เกตสองตัวต่ออนุกรม = กลับเฟสสองครั้ง กลับมาเฟสเดิม แต่ขับ"
                    "กระแสได้เป็นสองเท่า · 100R กันกระแสพุ่งตอนเริ่มขับ "
                    "เพราะหัวส่งมีพฤติกรรมเป็นตัวเก็บประจุ", 14, color=DIM)

    s.text(1120, 400, "ไฟเลี้ยงภาครับ · กรองแยกจากไฟดิจิทัล", 17, "start", True)
    s.wire((1190, 470), (1230, 470))
    s.netlabel(1182, 466, "MCU3V3", "end", at=(1190, 470))
    ra, rb = s.res("RF", "10R", 1230, 470)
    s.wire(rb, (1420, 470))
    s.dot(1370, 470)
    s.netlabel(1430, 466, "A3V3", "start", at=(1420, 470))
    s.cap("CF", "100uF", 1370, 490, vert=True)
    s.wire((1370, 540), (1370, 576))
    s.gnd(1370, 576)
    s.tie("MCU3V3", "RF.1")
    s.tie("A3V3", "RF.2", "CF.1")
    s.tie("GND", "CF.2")
    for k, t in enumerate((
            "3.3V มาจากเรกูเลเตอร์บน Blue Pill ซึ่งเลี้ยง MCU ที่วิ่ง 72 MHz อยู่ด้วย",
            "ภาครับขยายสัญญาณ 1156 เท่า สัญญาณกวนบนรางไฟจึงถูกขยายตามไปหมด",
            "10R + 100uF คั่นไว้ · ที่ 40 kHz คาปามีอิมพีแดนซ์แค่ 0.04 โอห์ม",
            "แลกกับแรงดันตก 0.2V ที่ 20 mA เหลือ 3.1V",
            "ยังเกินขั้นต่ำ 2.5V ที่ MCP6024 ต้องการ")):
        s.text(1120, 650 + k * 24, t, 13, color=DIM)
    return s


# ---------------------------------------------------------------- แผ่นที่ 2
def rx_block(s, i, chip, sa, sb, ox, oy, notes=False):
    """วาดหนึ่งช่องรับ ณ ตำแหน่งที่กำหนด · ทั้ง 8 ช่องใช้ฟังก์ชันนี้ตัวเดียวกัน

    เขียนเป็นฟังก์ชันรับตำแหน่ง แทนที่จะวาดช่องเดียวแบบตายตัว เพราะไฟล์ผัง
    ที่จะเอาไปเปิดใน EasyEDA ต้องมีครบทั้ง 8 ช่อง ไม่งั้นผังกับบอร์ดไม่ตรงกัน
    ส่วนภาพที่ให้คนอ่านใช้ช่องเดียวก็พอ เพราะอีก 7 ช่องเหมือนกันเป๊ะ
    """
    n1, p1, o1 = N.OPA[sa]
    n2, p2, o2 = N.OPA[sb]
    U = f"U{chip}"
    SIG, FB, LEG, CY = oy, oy + 150, oy + 210, oy + 25

    k1, k2 = s.term(f"K{i}", f"RX{i}", ox + 60, oy + 18)
    s.wire(k1, (ox + 200, SIG))
    s.wire(k2, (ox + 250, oy + 36), (ox + 250, oy + 90))
    s.gnd(ox + 250, oy + 90)
    s.cap(f"C{i}I", "10nF", ox + 200, SIG)
    s.wire((ox + 250, SIG), (ox + 330, SIG))
    s.dot(ox + 330, SIG)
    s.res(f"R{i}B", "1M", ox + 330, SIG - 130, vert=True, flip=True)
    s.wire((ox + 330, SIG), (ox + 330, SIG - 50))
    s.wire((ox + 330, SIG - 130), (ox + 330, SIG - 180))
    s.netlabel(ox + 330, SIG - 190, "VREF", "middle", at=(ox + 330, SIG - 180))
    s.tie(f"RX{i}_IN", f"K{i}.1", f"C{i}I.1")
    s.tie("GND", f"K{i}.2")
    s.tie("VREF", f"R{i}B.2")

    prev = None
    for st, (pn, pp, po, dx, rf, rg, cg) in enumerate((
            (n1, p1, o1, 500, f"R{i}F1", f"R{i}G1", f"C{i}G1"),
            (n2, p2, o2, 1060, f"R{i}F2", f"R{i}G2", f"C{i}G2")), start=1):
        x0 = ox + dx
        pplus, pminus, pout = s.opamp(U, (sa, sb)[st - 1], x0, CY, pn, pp, po)
        if st == 1:
            s.wire((ox + 330, SIG), pplus)
        else:
            s.wire(prev, (x0 - 80, CY), (x0 - 80, SIG), pplus)
        fx, gx = x0 + 180, x0 - 60
        s.wire(pout, (fx, CY), (fx, FB), (fx - 80, FB))
        s.dot(fx, CY)                  # ขาออกแยกสองทาง: สเตจถัดไป กับ สายป้อนกลับ
        s.res(rf, "33k", gx + 80, FB, flip=True)
        s.wire((gx, FB), (gx, CY + 25), pminus)
        s.dot(gx, FB)
        s.res(rg, "1k", gx, LEG, vert=True)
        s.wire((gx, FB), (gx, LEG))
        s.wire((gx, LEG + 80), (gx, LEG + 110))
        s.cap(cg, "100nF", gx, LEG + 110, vert=True)
        s.wire((gx, LEG + 160), (gx, LEG + 190))
        s.gnd(gx, LEG + 190)
        if notes:
            s.text(gx + 120, FB + 52, "สายป้อนกลับ", 12, "middle", False, DIM)
            for k, t in enumerate((
                    "คาปาตัวนี้กันไฟตรงออกจากขาอัตราขยาย",
                    "ไฟตรงจึงขยาย 1 เท่า ส่วนคลื่นเสียง 34 เท่า",
                    "ถ้าไม่มี ไฟตรงจะถูกขยาย 34 เท่าจนล้นทันที")):
                s.text(gx + 105, LEG + 124 + k * 22, t, 12, color=DIM)
        prev = pout

    s.res(f"R{i}O", "1k", ox + 1280, CY)
    s.wire((ox + 1240, CY), (ox + 1280, CY))
    s.wire((ox + 1360, CY), (ox + 1440, CY))
    s.dot(ox + 1440, CY)
    s.cap(f"C{i}O", "1nF", ox + 1440, CY + 20, vert=True)
    s.wire((ox + 1440, CY + 70), (ox + 1440, CY + 100))
    s.gnd(ox + 1440, CY + 100)
    s.netlabel(ox + 1440, CY - 17, f"RX{i}_ADC", "middle",
               at=(ox + 1440, CY))

    s.tie(f"RX{i}_G", f"C{i}I.2", f"R{i}B.1", f"{U}.{p1}")
    s.tie(f"RX{i}_S1O", f"{U}.{o1}", f"R{i}F1.1", f"{U}.{p2}")
    s.tie(f"RX{i}_S1N", f"{U}.{n1}", f"R{i}F1.2", f"R{i}G1.1")
    s.tie(f"RX{i}_G1", f"R{i}G1.2", f"C{i}G1.1")
    s.tie(f"RX{i}_S2O", f"{U}.{o2}", f"R{i}F2.1", f"R{i}O.1")
    s.tie(f"RX{i}_S2N", f"{U}.{n2}", f"R{i}F2.2", f"R{i}G2.1")
    s.tie(f"RX{i}_G2", f"R{i}G2.2", f"C{i}G2.1")
    s.tie(f"RX{i}_ADC", f"R{i}O.2", f"C{i}O.1")
    s.tie("GND", f"C{i}G1.2", f"C{i}G2.2", f"C{i}O.2")


def rx_spec(i):
    """ช่องที่ i ใช้ชิปตัวไหน section ไหน — ตรรกะเดียวกับที่ netlist ใช้สร้างบอร์ด"""
    return 2 + (i - 1) // 2, *(("A", "B") if i % 2 else ("C", "D"))


def sheet_rx():
    """หนึ่งช่องรับแบบละเอียด · อีก 7 ช่องเหมือนกันทุกประการ"""
    s = Sheet("2-rx-channel", 1580, 940,
              "แผ่น 2/3 · ภาครับ 1 ช่อง (ช่อง 2-8 เหมือนกันทุกประการ)",
              "หัวรับ -> ตัดไฟตรง -> ขยาย 34 เท่า -> ขยาย 34 เท่าอีกที -> "
              "กรองความถี่สูงทิ้ง -> เข้า ADC   รวมขยาย 1156 เท่า")
    rx_block(s, 1, *rx_spec(1), 0, 380, notes=True)
    s.text(390, 185, "VREF = ครึ่งหนึ่งของไฟเลี้ยง (ดูแผ่น 3)", 13, color=DIM)
    s.text(1440, 560, "-> ขา A0 ของ STM32", 14, "middle", True, HOT)
    for k, t in enumerate((
            "10nF ที่ทางเข้า ตัดไฟตรงทิ้ง เหลือแต่คลื่นเสียง แล้ว 1M ดึงระดับ"
            "กลางไปไว้ที่ VREF เพราะวงจรใช้ไฟเลี้ยงขั้วเดียว",
            "ถ้าไม่ดึงไว้กลางทาง คลื่นครึ่งล่างจะถูกตัดหายไปทันที",
            "1k + 1nF ปลายทาง ตัดความถี่เหนือ 159 kHz ทิ้งก่อนเข้า ADC "
            "กันสัญญาณความถี่สูงย้อนกลับมาโผล่เป็นคลื่นปลอม")):
        s.text(60, 860 + k * 24, t, 13, color=DIM)
    return s


def sheet_rx_all():
    """ครบทั้ง 8 ช่อง — แผ่นนี้มีไว้ให้ไฟล์ผังที่เอาไปเปิดใน EasyEDA ครบถ้วน

    คนอ่านให้ดูแผ่น 2 ที่มีช่องเดียวพร้อมคำอธิบายแทน แผ่นนี้ยาวเกินกว่าจะอ่านสบาย
    แต่โปรแกรมต้องการครบทุกช่อง ไม่งั้นผังกับบอร์ดจะไม่ตรงกัน
    """
    s = Sheet("4-rx-all", 1580, 380 + 8 * 620,
              "ภาครับครบทั้ง 8 ช่อง (สำหรับนำเข้า EasyEDA)",
              "ทุกช่องเหมือนกันหมด ต่างแค่หมายเลขและชิปที่ใช้")
    for i in range(1, 9):
        chip, sa, sb = rx_spec(i)
        oy = 380 + (i - 1) * 620
        rx_block(s, i, chip, sa, sb, 0, oy)
        s.text(60, oy - 150, f"ช่องรับที่ {i}  ({'U%d' % chip} section {sa}+{sb})",
               18, "start", True)
    return s


# ---------------------------------------------------------------- แผ่นที่ 3
def sheet_vref_mcu():
    s = Sheet("3-vref-mcu", 1980, 1260,
              "แผ่น 3/3 · แรงดันอ้างอิง ไฟเลี้ยงออปแอมป์ และขา STM32",
              "VREF คือระดับกลางที่คลื่นเสียงแกว่งขึ้นลงรอบ ๆ")

    n, p, o = N.OPA["A"]
    s.rail(120, 700, 150, "A3V3")
    s.res("RV1", "10k", 200, 170, vert=True)
    s.wire((200, 150), (200, 170))
    s.dot(200, 150)
    s.wire((200, 250), (200, 300))
    s.dot(200, 300)
    s.res("RV2", "10k", 200, 300, vert=True)
    s.wire((200, 380), (200, 420))
    s.gnd(200, 420)
    s.netlabel(150, 296, "MID", "end", at=(200, 300))
    s.tie("A3V3", "RV1.1")
    s.tie("MID", "RV1.2", "RV2.1", f"U6.{p}")
    s.tie("GND", "RV2.2")

    pplus, pminus, pout = s.opamp("U6", "A", 380, 300, n, p, o)
    s.wire((200, 300), (360, 300), (360, 275), pplus)
    s.wire(pout, (620, 300))
    s.dot(560, 300)
    s.wire((560, 300), (560, 400), (340, 400), (340, 325), pminus)
    s.text(430, 420, "ต่อขาออกกลับเข้าขาลบตรง ๆ = ขยาย 1 เท่า", 12, color=DIM)
    s.text(430, 440, "ได้แรงดันเท่าเดิมแต่จ่ายกระแสได้ ตัวแบ่งแรงดันเปล่า ๆ "
                     "จ่ายไม่ไหว", 12, color=DIM)
    s.netlabel(660, 296, "VREF", "start", at=(620, 300))

    for ref, val, x in (("CV1", "100nF", 740), ("CB2", "10uF", 840)):
        s.wire((620, 300), (x, 300))
        s.dot(x, 300)
        s.cap(ref, val, x, 300, vert=True)
        s.wire((x, 350), (x, 386))
        s.gnd(x, 386)
        s.tie("VREF", f"{ref}.1")
        s.tie("GND", f"{ref}.2")
    s.tie("VREF", f"U6.{o}", f"U6.{n}")
    s.text(700, 236, "-> ไปที่ขา 1M ของทั้ง 8 ช่องรับ", 14, "start", True, HOT)

    s.text(120, 520, "สาม section ที่เหลือของ U6 · ผูกเป็นตัวตามไว้ ไม่ปล่อยลอย",
           17, "start", True)
    s.text(120, 548, "ออปแอมป์ที่ปล่อยขาลอยจะแกว่งเองแล้วกวนตัวอื่นในชิปเดียวกัน "
                     "จับคู่ขาเข้าออกไว้กับ VREF ให้มันนิ่ง", 13, color=DIM)
    for k, sect in enumerate("BCD"):
        sn, sp, so = N.OPA[sect]
        x = 200 + k * 330
        pp2, pn2, po2 = s.opamp("U6", sect, x, 660, sn, sp, so)
        s.wire((x - 60, 660 - 25), pp2)
        s.wire((x - 60, 660 - 25), (x - 60, 660 + 25), pn2)
        s.dot(x - 60, 660 + 25)
        s.wire(po2, (x + 140, 660), (x + 140, 730), (x - 60, 730),
               (x - 60, 660 + 25))
        s.wire((x - 60, 660 - 25), (x - 130, 660 - 25))
        s.netlabel(x - 140, 660 - 21, "VREF", "end", at=(x - 130, 635))
        s.tie("VREF", f"U6.{sn}", f"U6.{sp}", f"U6.{so}")

    s.text(120, 830, "ไฟเลี้ยงออปแอมป์ทั้ง 5 ตัว", 17, "start", True)
    for k, u in enumerate(range(2, 7)):
        x = 170 + k * 250
        s.rect(x - 50, 870, x + 50, 935, INK, LW, "#f2f5fa")
        s.text(x, 900, f"U{u}", 16, "middle", True)
        s.text(x, 922, "MCP6024", 11, "middle", False, DIM)
        s.wire((x, 858), (x, 870))
        s.wire((x, 935), (x, 950))
        s.gnd(x, 950)
        s.text(x + 55, 886, "ขา 4", 11, color=DIM)
        s.text(x + 55, 930, "ขา 11", 11, color=DIM)
        s.cap(f"CD{u}", "100nF", x + 150, 858, vert=True)
        s.wire((x, 858), (x + 150, 858))
        s.dot(x, 858)
        s.wire((x + 150, 908), (x + 150, 950))
        s.gnd(x + 150, 950)
        s.inst.append(("opwr", f"U{u}", "MCP6024-I/P",
                       {"4": (x, 870), "11": (x, 935)}, (x, 902)))
        s.tie("A3V3", f"U{u}.4", f"CD{u}.1")
        s.tie("GND", f"U{u}.11", f"CD{u}.2")
    s.rail(100, 1330, 858, "A3V3")
    s.text(120, 1016, "คาปา 100nF ทั้งห้าตัวต้องวางชิดขาชิปที่สุดเท่าที่วางได้ "
                      "· ยิ่งไกลยิ่งไม่ได้ผล เพราะสายที่ยาวขึ้นมีค่าเหนี่ยวนำ"
                      "มากพอจะกันกระแสพุ่งไม่ทัน", 13, color=DIM)

    # ---- เฮดเดอร์ Blue Pill · วาดเป็นชิ้นส่วนจริง ไม่ใช่ตาราง
    # เดิมวาดเป็นตารางซึ่งอ่านง่ายกว่า แต่ตารางไม่ใช่ชิ้นส่วน ไฟล์ผังที่ได้
    # จะไม่มีตัวเชื่อมต่อ MCU เลย แล้วผังกับบอร์ดก็ไม่ตรงกัน
    s.text(1450, 120, "ที่เสียบ STM32F103C8T6 (Blue Pill)", 18, "start", True)
    for ci, (ref, row) in enumerate((("JA", N.BP_TOP), ("JB", N.BP_BOT))):
        hy = 175 + ci * 500
        pins = s.header(ref, 1450, hy, row)
        s.text(1450, hy - 48, "แถวบน" if ci == 0 else "แถวล่าง", 13, "start",
               False, DIM)
        for i, lab in enumerate(row, start=1):
            net = N.BP_NET.get(lab)
            px, py = pins[i - 1]
            if net:
                s.wire((px, py), (px + 26, py))
                s.netlabel(px + 32, py + 4, net, "start", at=(px + 26, py))
            else:
                s.circle(px + 8, py, 6, DIM, 1.6)   # ขาที่ไม่ได้ใช้
            s.tie(net if net else f"NC_{ref}_{lab}", f"{ref}.{i}")
    s.text(1450, 1120, "วงกลมเล็ก = ขาที่ไม่ได้ต่อไปไหน", 12, color=DIM)
    for k, t in enumerate((
            "A0-A7 อยู่ติดกันบนแถวเดียว ลายจึงไม่ต้องข้ามบอร์ด",
            "B6/B7/B8 = TIM4 ช่อง 1/2/3 สร้างคลื่น 40 kHz ด้วยฮาร์ดแวร์",
            "จังหวะเป๊ะกว่าสั่งด้วยโค้ด",
            "B3/B4/B5 เลี่ยงไว้ เพราะค่าเริ่มต้นเป็นขา JTAG")):
        s.text(1450, 1150 + k * 22, t, 12, color=DIM)
    return s


# ---------------------------------------------------------------- ตรวจสอบ
def exempt_pins():
    """ขาที่ยอมให้ไม่ต้องวาด — ช่องรับที่ 2-8 ซึ่งเหมือนช่องที่ 1 ทุกประการ

    สร้างรายการด้วยตรรกะเดียวกับที่ netlist สร้างช่องรับ ไม่ใช่พิมพ์รายชื่อไว้
    ถ้าโครงช่องรับเปลี่ยน รายการนี้จะเปลี่ยนตามเอง
    """
    out = set()
    for i in range(2, 9):
        for r in (f"K{i}", f"C{i}I", f"R{i}B", f"R{i}F1", f"R{i}G1", f"C{i}G1",
                  f"R{i}F2", f"R{i}G2", f"C{i}G2", f"R{i}O", f"C{i}O"):
            out |= {f"{r}.1", f"{r}.2"}
        chip = 2 + (i - 1) // 2
        for sect in (("A", "B") if i % 2 else ("C", "D")):
            out |= {f"U{chip}.{k}" for k in N.OPA[sect]}
    return out


def verify(sheets):
    """เทียบทุกขาที่วาดกับ netlist.py · ไม่ตรงแม้ขาเดียวก็หยุด

    ตรวจสองทาง เพราะผิดได้สองแบบคนละอย่าง
      1. วาดผิด  — ขาที่วาดไปอยู่คนละเน็ตกับของจริง
      2. วาดขาด — มีขาในบอร์ดจริงที่ไม่ได้ปรากฏบนผังเลย
    ตรวจแค่ทางเดียวจะปล่อยอีกทางหลุดไปเงียบ ๆ
    """
    b = N.BOARDS["main8"]()
    real = {p: net for net, pins in b["nets"].items() for p in pins}
    drawn, bad = {}, []
    for sh in sheets:
        for pin, net in sh.tied.items():
            if real.get(pin) != net:
                bad.append(f"  {sh.key}: {pin} ผังว่า {net} "
                           f"แต่บอร์ดว่า {real.get(pin, 'ไม่มีขานี้')}")
            drawn[pin] = net
        for pin in sh.pin:
            if pin not in sh.tied:
                bad.append(f"  {sh.key}: {pin} วาดไว้แต่ไม่ได้ประกาศเน็ต")
    skip = exempt_pins()
    missing = sorted(p for p in real if p not in drawn and p not in skip)
    if bad or missing:
        for m in bad:
            print(m)
        for m in missing[:20]:
            print(f"  ขาดบนผัง: {m} (เน็ต {real[m]})")
        if len(missing) > 20:
            print(f"  ... และอีก {len(missing) - 20} ขา")
        raise SystemExit(f"ผังไม่ตรงกับ netlist · ผิด {len(bad)} · "
                         f"ขาด {len(missing)} — ไม่สร้างไฟล์")
    used = [p for p in skip if p not in drawn]
    note = f" · ยกเว้นไว้ {len(used)} ขา" if used else " · ไม่ได้ยกเว้นขาไหนเลย"
    print(f"ตรวจแล้ว: {len(drawn)} ขาบนผังตรงกับ netlist ทุกขา{note}")


# ---------------------------------------------------------------- ส่งออก
def to_svg(sh):
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sh.w}" '
         f'height="{sh.h}" viewBox="0 0 {sh.w} {sh.h}">',
         f'<rect width="{sh.w}" height="{sh.h}" fill="{BG}"/>']
    for op in sh.ops:
        k = op[0]
        if k == "line":
            _, x1, y1, x2, y2, c, w = op
            e.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="{c}" stroke-width="{w}" stroke-linecap="round"/>')
        elif k == "rect":
            _, x1, y1, x2, y2, c, w, f = op
            e.append(f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" '
                     f'height="{y2 - y1}" fill="{f or "none"}" stroke="{c}" '
                     f'stroke-width="{w}"/>')
        elif k == "circ":
            _, x, y, r, c, w, f = op
            e.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f or "none"}" '
                     f'stroke="{c}" stroke-width="{w}"/>')
        elif k == "poly":
            _, pts, c, w, f = op
            d = " ".join(f"{a},{b}" for a, b in pts)
            e.append(f'<polygon points="{d}" fill="{f or "none"}" '
                     f'stroke="{c}" stroke-width="{w}"/>')
        else:
            _, x, y, t, size, anc, bold, c = op
            t = (t.replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;"))
            e.append(f'<text x="{x}" y="{y}" font-family="{FONT}" '
                     f'font-size="{size}" fill="{c}" text-anchor="{anc}"'
                     f'{" font-weight=\"bold\"" if bold else ""}>{t}</text>')
    e.append("</svg>")
    return "\n".join(e)


def to_png(sh, path, scale=1):
    """วาดซ้ำเป็น PNG ด้วย Pillow — เผื่อคนที่เปิด SVG ไม่ได้"""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (int(sh.w * scale), int(sh.h * scale)), BG)
    d = ImageDraw.Draw(im)
    cache = {}

    def fnt(size, bold):
        key = (size, bold)
        if key not in cache:
            for name in (("tahomabd.ttf", "arialbd.ttf") if bold
                         else ("tahoma.ttf", "arial.ttf")):
                try:
                    cache[key] = ImageFont.truetype(name, int(size * scale))
                    break
                except OSError:
                    continue
            else:
                cache[key] = ImageFont.load_default()
        return cache[key]

    def S(v):
        return v * scale

    for op in sh.ops:
        k = op[0]
        if k == "line":
            _, x1, y1, x2, y2, c, w = op
            d.line([S(x1), S(y1), S(x2), S(y2)], c, max(1, int(w * scale)))
        elif k == "rect":
            _, x1, y1, x2, y2, c, w, f = op
            d.rectangle([S(x1), S(y1), S(x2), S(y2)], f, c,
                        max(1, int(w * scale)))
        elif k == "circ":
            _, x, y, r, c, w, f = op
            d.ellipse([S(x - r), S(y - r), S(x + r), S(y + r)], f,
                      None if w == 0 else c, max(1, int(w * scale)))
        elif k == "poly":
            _, pts, c, w, f = op
            xy = [(S(a), S(b)) for a, b in pts]
            d.polygon(xy, f, c, max(1, int(w * scale)))
        else:
            _, x, y, t, size, anc, bold, c = op
            d.text((S(x), S(y)), t, c, fnt(size, bold),
                   anchor={"start": "ls", "middle": "ms", "end": "rs"}[anc])
    im.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    sheets = [sheet_power_tx(), sheet_rx(), sheet_vref_mcu(), sheet_rx_all()]
    verify(sheets)
    for sh in sheets:
        for ext, fn in (("svg", lambda p: open(p, "w", encoding="utf-8")
                         .write(to_svg(sh))), ("png", lambda p: to_png(sh, p))):
            path = os.path.join(OUT, f"schematic-{sh.key}.{ext}")
            fn(path)
        print(f"  {sh.key}: {sh.w}x{sh.h} · {len(sh.ops)} เส้น/ตัวอักษร")
    print(f"เก็บที่ {OUT}")


if __name__ == "__main__":
    main()
