"""ตัวเรนเดอร์ผังสำหรับ "แผ่นไข่ปลา" (perfboard) ที่ทุกรูแยกอิสระจากกัน

ต่างจาก bb_panel.py (บอร์ดทดลอง) สองเรื่องใหญ่:
  1. ไม่มีรางไฟและไม่มีแถวที่ต่อถึงกันเอง — **ทุกรูเป็นเกาะ** ต้องเชื่อมเองทุกจุด
     ผังจึงต้องบอกชัดว่า "จุดไหนเชื่อมถึงจุดไหน" ไม่ใช่แค่ "เสียบรูไหน"
  2. มี "บัสไฟ" ที่เราสร้างเอง = ลวดทองแดงเดินยาวแล้วบัดกรีทุกรูที่ผ่าน
     (แทนรางของบอร์ดทดลอง แต่ความต้านทานต่ำกว่ามาก จึงเงียบกว่า)

พิกัด: คอลัมน์ A-Z (ตามที่พิมพ์บนแผ่น) · แถว 01-48 · อ้างอิงแบบ "H16" = คอลัมน์ H แถว 16
"""
import os

PITCH = 26                      # ระยะห่างรู (จุดต่อจุด) บนภาพ
TOP = 148                       # ขอบบนของกริดรูแรก
LEFT = 58                       # x ของคอลัมน์ A (ขวาสุด) — เลขแถวย้ายไปฝั่งขวาแล้ว
COLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

C_RES, C_CAP = ("#ffe8cc", "#f08c00"), ("#d0ebff", "#1971c2")
C_SIG, C_POS, C_NEG = "#2f9e44", "#e03131", "#343a40"
C_VREF = "#ae3ec9"
C_PAD, C_PADRING = "#ffffff", "#c9a227"


def esc(t):
    """หนี < > & ในข้อความ — ถ้าไม่หนี SVG จะพังทั้งไฟล์ (เจอมาแล้วกับ '<1 โอห์ม')"""
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _split(ref):
    """'H16' -> ('H', 16)"""
    return ref[0].upper(), int(ref[1:])


class PerfPanel:
    def __init__(self, name, title, sub, rows, cols=None, width=1180):
        self.name, self.title, self.sub = name, title, sub
        self.rows = list(rows)
        self.cols = cols or COLS[:22]
        self.W = width
        self.body = []           # วาดทับกริด
        self.top = []            # วาดบนสุด (ชิป) กันสายทับขา
        self.under = []          # วาดใต้กริด (บัส) จะได้ไม่บังรู
        self.used = set()        # รูที่ผังนี้ "ใช้จริง" (ปลายสาย/ขาอุปกรณ์) ไว้ทำผีข้ามผัง
        self.chips = []          # (x0,y0,x1,y1, ชุดพิกัดขา) ไว้ตรวจว่าไม่มีสายลอดใต้ชิป
        self.holes = {}          # รู -> ชื่ออุปกรณ์ที่ลงรูนั้น (ไข่ปลาใส่ได้ขาเดียว)
        self.rigid = set()       # ชื่อคอนเนกเตอร์ (header/terminal) ที่พันขารวมด้วยไม่ได้
        self.allow_shared = {}   # รู -> เหตุผล ที่ยอมให้ขาซ้อนกันได้ (เช่น แผ่นที่บัดกรีไปแล้ว)
        self.y = {r: TOP + i * PITCH for i, r in enumerate(self.rows)}
        n = len(self.cols)
        # แผ่นจริงเรียง A จาก **ขวา** ไปซ้าย — วาดให้ตรงกับของจริงกันนับพลาด
        self.x = {c: LEFT + (n - 1 - i) * PITCH for i, c in enumerate(self.cols)}

    # ---------- พิกัด ----------
    def p(self, ref):
        c, r = _split(ref)
        return self.x[c], self.y[r]

    # ---------- กราฟิกพื้นฐาน ----------
    def _path(self, d, colour, w=3.4, dash=None, layer=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        (layer if layer is not None else self.body).append(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{w}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')

    def _dot(self, ref, colour, r=5.5):
        x, y = self.p(ref)
        self.used.add(ref.upper())
        self.body.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{colour}"/>')

    def ghosts(self, refs):
        """วาด 'รูที่ผังอื่นจองไว้แล้ว' เป็นวงจาง — กันเดินสายทับโหนดของผังอื่น

        ผังแต่ละแผ่นโชว์เฉพาะบล็อกของตัวเอง คอลัมน์ที่ผังอื่นใช้อยู่จึงดู 'ว่าง' ทั้งที่
        มีขาอุปกรณ์เสียบอยู่จริง — เคยทำให้เดินสายผัง 3 ลงคอลัมน์ F ทับโหนด VREF ของผัง 2
        มาแล้ว (ไล่หาสาเหตุเสียเวลาหลายชั่วโมง) จึงต้องวาดไว้ทุกผัง
        """
        for ref in sorted(refs):
            c, r = _split(ref)
            if r not in self.y or c not in self.x:
                continue
            x, y = self.x[c], self.y[r]
            self.under.append(f'<circle cx="{x}" cy="{y}" r="9.5" fill="none" '
                              f'stroke="#c92a2a" stroke-width="1.6" stroke-dasharray="2 2" '
                              f'opacity="0.55"/>')

    def label(self, ref, text, dx=0, dy=-13, colour="#495057", size=11, anchor="middle"):
        x, y = self.p(ref)
        self.body.append(f'<text x="{x+dx}" y="{y+dy}" font-size="{size}" '
                         f'text-anchor="{anchor}" fill="{colour}">{esc(text)}</text>')

    # ---------- การเชื่อม ----------
    def solder(self, a, b, colour=C_SIG):
        """เชื่อมรูที่ติดกันด้วยหยดตะกั่ว (ใช้ทำ 'โหนด' ให้มีที่เสียบหลายขา)"""
        (xa, ya), (xb, yb) = self.p(a), self.p(b)
        self._path(f"M{xa},{ya} L{xb},{yb}", colour, w=9)
        self._dot(a, colour, 4); self._dot(b, colour, 4)

    def wire(self, a, b, colour=C_SIG, lane=None, side="v"):
        """สายจั๊มด้านหลังแผ่น · lane = พิกัดที่ให้สายเลี้ยวหลบ ไม่ให้ทับรูอื่น"""
        (xa, ya), (xb, yb) = self.p(a), self.p(b)
        if lane is None:
            d = f"M{xa},{ya} L{xb},{yb}"
        elif side == "v":                      # อ้อมทางแนวตั้งที่ y = lane
            d = f"M{xa},{ya} L{xa},{lane} L{xb},{lane} L{xb},{yb}"
        else:                                   # อ้อมทางแนวนอนที่ x = lane
            d = f"M{xa},{ya} L{lane},{ya} L{lane},{yb} L{xb},{yb}"
        self._path(d, colour, w=3.2, dash="7 5")
        self._dot(a, colour, 5); self._dot(b, colour, 5)

    def bus(self, col, r0, r1, colour, text, side="left"):
        """บัสไฟ = ลวดแข็งเดินตามคอลัมน์ บัดกรีทุกรูที่ผ่าน"""
        x = self.x[col]
        y0, y1 = self.y[r0], self.y[r1]
        self._path(f"M{x},{y0} L{x},{y1}", colour, w=11, layer=self.under)
        for r in self.rows:
            if r0 <= r <= r1:
                self.under.append(f'<circle cx="{x}" cy="{self.y[r]}" r="4" fill="#ffffff" '
                                  f'fill-opacity="0.55"/>')
        ty = y0 - 16
        anchor, tx = ("end", x - 10) if side == "left" else ("start", x + 10)
        self.body.append(f'<text x="{tx}" y="{ty}" font-size="12.5" font-weight="700" '
                         f'text-anchor="{anchor}" fill="{colour}">{esc(text)}</text>')

    def tap(self, ref, colour):
        """จุดที่แยกออกจากบัส"""
        self._dot(ref, colour, 6)

    # ---------- อุปกรณ์ ----------
    def part(self, a, b, text, kind, flip=False):
        """flip=True ย้ายป้ายไปอีกฝั่ง (ใช้กันป้ายทับกัน)"""
        (xa, ya), (xb, yb) = self.p(a), self.p(b)
        fill, stroke = C_RES if kind == "R" else C_CAP
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        horiz = abs(ya - yb) < 2
        w, h = (abs(xb - xa) - 16, 19) if horiz else (19, abs(yb - ya) - 16)
        self._path(f"M{xa},{ya} L{xb},{yb}", stroke, w=2.6)
        self.body.append(f'<rect x="{mx-w/2:.1f}" y="{my-h/2:.1f}" width="{w:.1f}" '
                         f'height="{h:.1f}" rx="4" fill="{fill}" stroke="{stroke}" '
                         f'stroke-width="2"/>')
        self._dot(a, stroke, 4.5); self._dot(b, stroke, 4.5)
        for ref in (a, b):
            self.holes.setdefault(ref.upper(), []).append(text or kind)
        if horiz:
            tx, ty, anchor = mx, (my + 27 if flip else my - 18), "middle"
        else:
            tx, ty, anchor = (mx - 26, my + 4, "end") if flip else (mx + 26, my + 4, "start")
        self.body.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="13" font-weight="700" '
                         f'text-anchor="{anchor}" fill="{stroke}">{esc(text)}</text>')

    # ---------- ชิป ----------
    def chip(self, col_l, row_top, name="MCP6004", npin=14, span=3):
        B = self.body
        self.body = self.top          # ชิปวาดบนสุด สายจึงไม่ทับขา
        """DIP วางตั้ง: ขา 1..n/2 ที่คอลัมน์ col_l (ไล่ลง) · ที่เหลือคอลัมน์ +span (ไล่ขึ้น)"""
        half = npin // 2
        ci = self.cols.index(col_l)
        col_r = self.cols[ci + span]
        xl, xr = self.x[col_l], self.x[col_r]
        y0 = self.y[row_top] - 15
        y1 = self.y[row_top - half + 1] + 15
        # คอลัมน์อาจกลับด้าน (A อยู่ขวา) ต้องใช้ min/max ไม่งั้นความกว้างติดลบ = ไม่วาด
        bx, bw = min(xl, xr) - 15, abs(xr - xl) + 30
        pins = {(round(xl), round(self.y[row_top - i])) for i in range(half)}
        pins |= {(round(xr), round(self.y[row_top - i])) for i in range(half)}
        self.chips.append((bx, y0, bx + bw, y1, pins))
        self.body.append(f'<rect x="{bx}" y="{y0}" width="{bw}" height="{y1-y0}" '
                         f'rx="6" fill="#2b2f36"/>')
        # รอยบากของ DIP อยู่ "ปลายฝั่งที่มีขา 1 กับขาสุดท้าย" — ในผังนี้ขา 1 อยู่แถวบนสุด
        # จึงวาดรอยบากที่ขอบบน ตรงกลางระหว่างสองแถวขา (ตรงกับที่เห็นบนตัวชิปจริง)
        self.body.append(f'<path d="M{(xl+xr)/2-11},{y0} A11,11 0 0,0 {(xl+xr)/2+11},{y0}" '
                         f'fill="#1b1e23" stroke="#5b6b7a" stroke-width="2"/>')
        self.body.append(f'<text x="{xl}" y="{y0-8}" font-size="11.5" font-weight="700" '
                         f'text-anchor="middle" fill="#f4c430">ขา 1 อยู่นี่ ↓</text>')
        self.body.append(f'<text x="{(xl+xr)/2:.0f}" y="{y1+20}" font-size="13" '
                         f'font-weight="700" text-anchor="middle" fill="#495057">{esc(name)}</text>')
        for i in range(half):
            r = row_top - i
            self.used.add(f"{col_l}{r}"); self.used.add(f"{col_r}{r}")
            self.body.append(f'<circle cx="{xl}" cy="{self.y[r]}" r="7" fill="#f4c430"/>')
            self.body.append(f'<text x="{xl}" y="{self.y[r]+4}" font-size="10" '
                             f'font-weight="700" text-anchor="middle" fill="#2b2f36">{i+1}</text>')
            pr = npin - i
            self.body.append(f'<circle cx="{xr}" cy="{self.y[r]}" r="7" fill="#f4c430"/>')
            self.body.append(f'<text x="{xr}" y="{self.y[r]+4}" font-size="10" '
                             f'font-weight="700" text-anchor="middle" fill="#2b2f36">{pr}</text>')
        self.body = B

    def header(self, refs, name, pins_txt="", side="right"):
        """pin header **ตัวผู้** N ขา บนแผ่น (สายที่มาเสียบใช้ตัวเมีย)

        refs = รายชื่อรูเรียงกัน เช่น ["F44","E44"] · วาดเป็นแถบดำครอบรูเหล่านั้น
        เหตุผลที่ใช้ header แทนบัดกรีสายตรงๆ: ถอดเปลี่ยนหัวรับ/สาย ESP32 ได้โดยไม่ต้อง
        จี้หัวแร้งซ้ำที่แผ่น ซึ่งเป็นสาเหตุที่ทำให้รอยบัดกรีเสียและลายทองแดงหลุด
        """
        self.used.update(r.upper() for r in refs)
        self.rigid.add(name)
        for r in refs:
            self.holes.setdefault(r.upper(), []).append(name)
        pts = [self.p(r) for r in refs]
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        x0, x1 = min(xs) - 13, max(xs) + 13
        y0, y1 = min(ys) - 13, max(ys) + 13
        self.body.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="4" '
                         f'fill="#1b1e23" stroke="#495057" stroke-width="2"/>')
        for (x, y) in pts:
            self.body.append(f'<rect x="{x-7}" y="{y-7}" width="14" height="14" rx="2" '
                             f'fill="#f4c430"/>')
        horiz = (max(ys) - min(ys)) < 2
        if horiz:
            tx, ty, anchor = (min(xs) + max(xs)) / 2, y0 - 9, "middle"
        elif side == "right":
            tx, ty, anchor = x1 + 10, (y0 + y1) / 2 - 4, "start"
        else:                       # ป้ายไปทางซ้าย (ใช้เมื่อ header อยู่ชิดขอบขวาของแผ่น)
            tx, ty, anchor = x0 - 10, (y0 + y1) / 2 - 4, "end"
        self.body.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="12" font-weight="700" '
                         f'text-anchor="{anchor}" fill="#343a40">{esc(name)}</text>')
        if pins_txt:
            self.body.append(f'<text x="{tx:.0f}" y="{ty+15:.0f}" font-size="11" '
                             f'text-anchor="{anchor}" fill="#868e96">{esc(pins_txt)}</text>')

    def terminal(self, refs, name, pins_txt="", side="right"):
        """เทอร์มินอลบล็อกขันสกรู (KF301 ระยะขา 5.08 mm = 2 รูพอดีบนไข่ปลา)

        ใช้กับจุดที่ "หลุดไม่ได้" โดยเฉพาะไฟเลี้ยง — ออปแอมป์ได้ไฟไม่ครบทำให้เกิดอาการ
        เพี้ยนที่ดูเหมือนวงจรผิด ไล่หาสาเหตุยาก · ขันสกรูจับสายเปลือยได้ ไม่ต้องย้ำหัว
        """
        self.used.update(r.upper() for r in refs)
        self.rigid.add(name)
        for r in refs:
            self.holes.setdefault(r.upper(), []).append(name)
        pts = [self.p(r) for r in refs]
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        x0, x1 = min(xs) - 15, max(xs) + 15
        y0, y1 = min(ys) - 16, max(ys) + 16
        self.body.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="4" '
                         f'fill="#2b6cb0" stroke="#1a4e8a" stroke-width="2"/>')
        for (x, y) in pts:                      # หัวสกรู
            self.body.append(f'<circle cx="{x}" cy="{y}" r="7.5" fill="#dfe6ee" '
                             f'stroke="#8fa3b8" stroke-width="1.5"/>')
            self.body.append(f'<path d="M{x-4.5},{y} L{x+4.5},{y}" stroke="#4a5b6d" '
                             f'stroke-width="2.2"/>')
        horiz = (max(ys) - min(ys)) < 2
        if horiz:
            tx, ty, anchor = (min(xs) + max(xs)) / 2, y0 - 9, "middle"
        elif side == "right":
            tx, ty, anchor = x1 + 10, (y0 + y1) / 2 - 4, "start"
        else:
            tx, ty, anchor = x0 - 10, (y0 + y1) / 2 - 4, "end"
        self.body.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="12" font-weight="700" '
                         f'text-anchor="{anchor}" fill="#1a4e8a">{esc(name)}</text>')
        if pins_txt:
            self.body.append(f'<text x="{tx:.0f}" y="{ty+15:.0f}" font-size="11" '
                             f'text-anchor="{anchor}" fill="#868e96">{esc(pins_txt)}</text>')

    # ---------- กล่องข้อความ (อยู่นอกแผ่นเสมอ) ----------
    def note(self, x, y, lines, colour="#f08c00", bg="#fff4e6", w=250):
        self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{20+19*len(lines)}" '
                         f'rx="8" fill="{bg}" stroke="{colour}" stroke-width="1.6"/>')
        for i, t in enumerate(lines):
            weight = "700" if i == 0 else "400"
            self.body.append(f'<text x="{x+12}" y="{y+24+19*i}" font-size="12.5" '
                             f'font-weight="{weight}" fill="#343a40">{esc(t)}</text>')

    def legend(self, x, y):
        """อธิบายชนิดเส้น — สำคัญเพราะไข่ปลาต่อสองแบบ (ตะกั่วหน้าแผ่น / สายหลังแผ่น)"""
        items = [("เส้นทึบหนา", C_NEG, "บัสลวดแข็ง บัดกรีทุกรูที่ผ่าน"),
                 ("เส้นทึบสั้น", C_SIG, "เชื่อมรูติดกันด้วยตะกั่ว = ทำโหนด"),
                 ("เส้นประ", C_SIG, "สายจั๊มด้านหลังแผ่น (ข้ามกันได้)")]
        self.body.append(f'<rect x="{x}" y="{y}" width="330" height="{22+22*len(items)}" '
                         f'rx="8" fill="#f8f9fa" stroke="#adb5bd" stroke-width="1.4"/>')
        for i, (nm, col, desc) in enumerate(items):
            yy = y + 26 + 22 * i
            dash = ' stroke-dasharray="7 5"' if "ประ" in nm else ""
            w = 9 if "หนา" in nm else 3.2
            self.body.append(f'<path d="M{x+12},{yy-4} L{x+52},{yy-4}" stroke="{col}" '
                             f'stroke-width="{w}" stroke-linecap="round"{dash}/>')
            self.body.append(f'<text x="{x+62}" y="{yy}" font-size="12" fill="#343a40">'
                             f'{esc(nm)} = {esc(desc)}</text>')

    def check_overlaps(self):
        """เตือนถ้ามีสายคนละสีวิ่งทับกันในแถว/คอลัมน์เดียวกัน

        ผังที่มีเส้นซ้อนกันทำให้คนต่อสับสนว่าจุดไหนเชื่อมกับจุดไหน — เคยหลุดไปแล้วรอบหนึ่ง
        (เส้น GND กับ VREF ทับกันที่แถวล่างสุดของผัง 1) จึงตรวจอัตโนมัติทุกครั้งที่สร้าง
        """
        import re
        segs = []
        for item in self.body + self.under:
            m = re.search(r'<path d="([^"]+)"[^>]*stroke="([^"]+)"', item)
            if not m or "dasharray" not in item:
                continue
            pts = re.findall(r"[ML](-?[0-9.]+),(-?[0-9.]+)", m.group(1))
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                if abs(y1 - y2) < 1:
                    segs.append(("h", round(y1), sorted([x1, x2]), m.group(2)))
                elif abs(x1 - x2) < 1:
                    segs.append(("v", round(x1), sorted([y1, y2]), m.group(2)))
        bad = []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                (o1, k1, r1, c1), (o2, k2, r2, c2) = segs[i], segs[j]
                if o1 == o2 and k1 == k2 and c1 != c2                         and r1[0] < r2[1] - 2 and r2[0] < r1[1] - 2:
                    bad.append((o1, k1, c1, c2))
        for o, k, c1, c2 in bad:
            print(f"   !! {self.name}: สายทับกัน ({o} ที่ {k}px) สี {c1} กับ {c2}")

        # สายคนละเส้นที่วิ่ง "แนวเดียวกัน คอลัมน์/แถวเดียวกัน สีเดียวกัน" แล้วเว้นช่องแค่
        # แถวสองแถว จะดูเป็นเส้นต่อเนื่องเส้นเดียว — ผู้ต่อจะเดินสายรวบเป็นเส้นเดียว
        # ทำให้โหนดคนละโหนดถึงกันหมด (เจอมาแล้วกับ ขา1->ขา5 กับ ขา6 ที่คอลัมน์ K)
        gap = []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                (o1, k1, r1, c1), (o2, k2, r2, c2) = segs[i], segs[j]
                if o1 != o2 or k1 != k2 or c1 != c2:
                    continue
                d = max(r2[0] - r1[1], r1[0] - r2[1])      # ระยะห่างระหว่างปลายสองท่อน
                if 0 < d <= PITCH * 2.2:
                    gap.append((o1, k1, c1, d))
        for o, k, c, d in gap:
            print(f"   !! {self.name}: สองท่อนคนละเส้นเรียงต่อกัน ({o} ที่ {k}px สี {c}) "
                  f"เว้นแค่ {d:.0f}px = ดูเป็นเส้นเดียว ให้ย้ายเส้นหนึ่งไปคอลัมน์/แถวอื่น")
        return not bad and not gap

    def check_chip_clear(self):
        """เตือนถ้ามีสายวิ่งทับตัวชิป หรือปลายสายจบอยู่ใต้ตัวชิป

        นี่คือบั๊กที่เคยทำผู้ใช้บัดกรีผิดขามาแล้ว (ส.ค. 2026 ผัง VREF ภาครับ): ตัวชิป
        วาดบน layer บนสุด สายที่ลอดใต้มันจึงหายไปจากภาพ คนต่อมองไม่เห็นว่าปลายสาย
        ไปจบที่ขาไหน เลยเดาเอาจากขาที่มองเห็น = ต่อผิดขา
        ขาชิปเองไม่นับ (จุดต่อที่ขาถูกต้องอยู่แล้ว) นับเฉพาะส่วนของเส้นที่ "ผ่านเข้าไป"
        """
        import re
        bad = []
        for item in self.body:
            m = re.search(r'<path d="([^"]+)"[^>]*stroke="([^"]+)"', item)
            if not m or "dasharray" not in item:
                continue
            pts = [(float(a), float(b))
                   for a, b in re.findall(r"[ML](-?[0-9.]+),(-?[0-9.]+)", m.group(1))]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                for x0, cy0, cx1, cy1, pins in self.chips:
                    # ตัดปลายที่อยู่ "ที่ขาชิปพอดี" ออก แล้วดูว่าส่วนที่เหลือยังกินตัวชิปไหม
                    inside = []
                    for t in [i / 40.0 for i in range(1, 40)]:
                        px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                        if x0 <= px <= cx1 and cy0 <= py <= cy1:
                            inside.append((px, py))
                    if not inside:
                        continue
                    # ยอมให้ "แตะ" ได้ถ้าปลายเส้นอยู่ที่ขาชิป และส่วนที่กินตัวชิปสั้นมาก
                    ends_on_pin = any((round(ex), round(ey)) in pins
                                      for ex, ey in ((x1, y1), (x2, y2)))
                    span = max(abs(inside[-1][0] - inside[0][0]),
                               abs(inside[-1][1] - inside[0][1]))
                    if ends_on_pin and span <= 16:
                        continue
                    bad.append((m.group(2), (x1, y1), (x2, y2), span))
        for col, a, b, span in bad:
            print(f"   !! {self.name}: สายสี {col} ลอดใต้ตัวชิป "
                  f"({a[0]:.0f},{a[1]:.0f})->({b[0]:.0f},{b[1]:.0f}) กินตัวชิป {span:.0f}px "
                  f"— ต้องให้โผล่ออกข้างชิปก่อน")
        return not bad

    def check_one_part_per_hole(self):
        """เตือนถ้ามีขาอุปกรณ์สองชิ้นลงรูเดียวกัน — ไข่ปลาใส่ได้ขาเดียวต่อรู

        เจอจริงตอนวาดผัง TX: ปลาย R อนุกรมกับขาเทอร์มินอล KF301 ไปลงรู T43 รูเดียวกัน
        บนภาพดูเหมือนต่อกันถูก แต่ประกอบจริงไม่ได้ ต้องแยกรูแล้วเชื่อมด้วยตะกั่ว
        """
        hard = []
        for h, v in sorted(self.holes.items()):
            if len(v) < 2:
                continue
            if h in self.allow_shared:
                print(f"   ~  {self.name}: รู {h} มีขาซ้อนกัน — ยกเว้นไว้โดยตั้งใจ "
                      f"({self.allow_shared[h]})")
                continue
            rigid = [n for n in v if n in self.rigid]
            if rigid:
                # คอนเนกเตอร์เป็นก้อนแข็ง พันขารวมกับมันไม่ได้ ต้องแยกรูแล้วเชื่อมตะกั่ว
                hard.append((h, v))
                print(f"   !! {self.name}: รู {h} มีทั้งคอนเนกเตอร์และขาอุปกรณ์ "
                      f"({', '.join(sorted(v))}) — ประกอบจริงไม่ได้ ต้องแยกรูแล้ว solder()")
            else:
                print(f"   -  {self.name}: รู {h} มีขา {', '.join(sorted(v))} ลงรูเดียวกัน "
                      f"— ต้องพันขาทั้งสองเข้าด้วยกันก่อนเสียบ (ทำได้ แต่ต้องรู้ก่อน)")
        return not hard

    # ---------- บันทึก ----------
    def save(self, outdir):
        self.check_overlaps()
        self.check_chip_clear()
        self.check_one_part_per_hole()
        H = TOP + len(self.rows) * PITCH + 46
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{H}" '
             f'viewBox="0 0 {self.W} {H}" font-family="Tahoma, Segoe UI, sans-serif">',
             f'<rect width="{self.W}" height="{H}" fill="#ffffff"/>',
             f'<text x="28" y="42" font-size="21" font-weight="700" fill="#1864ab">'
             f'{self.title}</text>',
             f'<text x="28" y="68" font-size="13" fill="#868e96">{self.sub}</text>']
        # แผ่นไข่ปลา
        bx0, bx1 = LEFT - 34, LEFT + (len(self.cols) - 1) * PITCH + 34
        by0, by1 = TOP - 30, TOP + (len(self.rows) - 1) * PITCH + 30
        o.append(f'<rect x="{bx0}" y="{by0}" width="{bx1-bx0}" height="{by1-by0}" rx="8" '
                 f'fill="#c98b53" fill-opacity="0.22" stroke="#b07a45" stroke-width="2"/>')
        o += self.under
        # หัวคอลัมน์
        for c in self.cols:
            o.append(f'<text x="{self.x[c]}" y="{TOP-40}" font-size="12.5" font-weight="700" '
                     f'text-anchor="middle" fill="#495057">{c}</text>')
        # เลขแถว (ช่องซ้ายสุด เว้นห่างจากคอลัมน์ A ตามที่ผู้ใช้ขอ)
        prev = None
        for r in self.rows:
            y = self.y[r]
            if prev is not None and r != prev + 1:
                o.append(f'<path d="M{bx0+10},{y-PITCH/2} L{bx1-10},{y-PITCH/2}" fill="none" '
                         f'stroke="#ced4da" stroke-width="1.5" stroke-dasharray="4 4"/>')
            rx0 = bx1 + 18
            o.append(f'<rect x="{rx0}" y="{y-11}" width="48" height="22" rx="4" '
                     f'fill="#f1f3f5"/>')
            o.append(f'<text x="{rx0+24}" y="{y+5}" font-size="12.5" text-anchor="middle" '
                     f'fill="#495057">{r:03d}</text>')
            prev = r
        # รูทั้งหมด
        for c in self.cols:
            for r in self.rows:
                o.append(f'<circle cx="{self.x[c]}" cy="{self.y[r]}" r="6.5" fill="{C_PAD}" '
                         f'stroke="{C_PADRING}" stroke-width="2"/>')
        o += self.body
        o += self.top
        o.append("</svg>")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{self.name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(o))
        return path
