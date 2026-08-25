"""Breadboard panel renderer shared by the RX and TX layout generators.

One class draws both boards. The only structural difference between them is
which edge carries the power rail: on the left-hand board the rail strip sits
on the outer (left) edge and the ESP32's pins land in column J; on the
right-hand board the rail is on the outer (right) edge and the ESP32's pins
land in column A. Everything else - column pitch, row gutter, lane routing -
is identical, so the two sets of diagrams read the same way.

Long jumpers are drawn in the blank lanes between hole columns rather than on
top of them, so a wire that merely passes a row cannot be mistaken for one that
plugs into it.
"""

import os

COL = {"A": 210, "B": 232, "C": 254, "D": 276, "E": 298,
       "F": 342, "G": 364, "H": 386, "I": 408, "J": 430}
PITCH, TOP = 34, 150

C_RES, C_CAP = ("#ffe8cc", "#f08c00"), ("#d0ebff", "#1971c2")
C_SIG, C_POS, C_NEG = "#2f9e44", "#e03131", "#343a40"


class Panel:
    def __init__(self, name, title, sub, rows, rail="left", chip_row0=None,
                 chip_name="", width=690):
        self.name, self.title, self.sub, self.rows = name, title, sub, rows
        self.rail, self.chip_row0, self.chip_name = rail, chip_row0, chip_name
        self.W = width
        self.y = {r: TOP + i * PITCH for i, r in enumerate(rows)}
        self.b = []
        self.maxy = 0          # tallest side note, so the canvas can grow
        if rail == "left":
            self.rp, self.rn = 134, 156
            self.bx0, self.bw = 110, 398
        else:
            self.rn, self.rp = 452, 474
            self.bx0, self.bw = 168, 332

    # ---------------------------------------------------------- geometry --
    def p(self, ref):
        return COL[ref[-1]], self.y[int(ref[:-1])]

    def band(self, row, below=True):
        i = self.rows.index(row)
        if below and i + 1 < len(self.rows):
            return (self.y[row] + self.y[self.rows[i + 1]]) / 2
        if not below and i > 0:
            return (self.y[row] + self.y[self.rows[i - 1]]) / 2
        return self.y[row] + (PITCH / 2 if below else -PITCH / 2)

    def dot(self, ref, colour):
        x, y = self.p(ref)
        self.b.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{colour}"/>')

    def _path(self, d, colour, w=2.6):
        self.b.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{w}" '
                      f'stroke-linejoin="round" stroke-linecap="round"/>')

    # ------------------------------------------------------------- wires --
    def wire(self, a, b, colour=C_SIG):
        ax, ay = self.p(a)
        bx, by = self.p(b)
        self._path(f"M{ax},{ay} L{bx},{by}", colour)
        self.dot(a, colour)
        self.dot(b, colour)

    def vlane(self, a, b, lane, colour=C_SIG):
        ax, ay = self.p(a)
        bx, by = self.p(b)
        self._path(f"M{ax},{ay} L{lane},{ay} L{lane},{by} L{bx},{by}", colour)
        self.dot(a, colour)
        self.dot(b, colour)

    def route(self, a, b, lane, after, below=True, colour=C_SIG):
        ax, ay = self.p(a)
        bx, by = self.p(b)
        yb = self.band(after, below)
        self._path(f"M{ax},{ay} L{lane},{ay} L{lane},{yb} L{bx},{yb} L{bx},{by}", colour)
        self.dot(a, colour)
        self.dot(b, colour)

    def to_rail(self, a, which, below=True, lane=None, after=None):
        x, y = self.p(a)
        yb = self.band(after if after is not None else int(a[:-1]), below)
        rx = self.rp if which == "+" else self.rn
        colour = C_POS if which == "+" else C_NEG
        d = (f"M{x},{y} L{x},{yb} L{rx},{yb}" if lane is None
             else f"M{x},{y} L{lane},{y} L{lane},{yb} L{rx},{yb}")
        self._path(d, colour)
        self.b.append(f'<circle cx="{rx}" cy="{yb}" r="4.5" fill="{colour}"/>')
        self.dot(a, colour)

    # -------------------------------------------------------- components --
    def _body(self, mx, my, kind, text, horiz=False):
        fill, stroke = C_RES if kind == "R" else C_CAP
        w, h = (24, 20) if kind == "R" else (20, 13)
        if horiz:
            w, h = h, w
        self.b.append(f'<rect x="{mx-w/2:.1f}" y="{my-h/2:.1f}" width="{w}" height="{h}" '
                      f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if not text:
            return
        if horiz:                      # label off to the side: a horizontal
            tx, ty = mx + 90, my + 21  # body sits on top of the chip
            self.b.append(f'<rect x="{tx-46:.1f}" y="{ty-13:.1f}" width="92" height="18" '
                          f'rx="4" fill="#ffffff" stroke="{stroke}" stroke-width="1"/>')
            self.b.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="11.5" text-anchor="middle" '
                          f'font-weight="600" fill="{stroke}">{text}</text>')
        else:
            self.b.append(f'<text x="{mx+w/2+7:.1f}" y="{my+5:.1f}" font-size="13" '
                          f'font-weight="600" fill="{stroke}">{text}</text>')

    def part(self, a, b, text, kind):
        stroke = (C_RES if kind == "R" else C_CAP)[1]
        ax, ay = self.p(a)
        bx, by = self.p(b)
        self._path(f"M{ax},{ay} L{bx},{by}", stroke, 2.4)
        self.dot(a, stroke)
        self.dot(b, stroke)
        self._body((ax + bx) / 2, (ay + by) / 2, kind, text, horiz=(ay == by))

    def part_lane(self, a, b, lane, text, kind):
        stroke = (C_RES if kind == "R" else C_CAP)[1]
        ax, ay = self.p(a)
        bx, by = self.p(b)
        self._path(f"M{ax},{ay} L{lane},{ay} L{lane},{by} L{bx},{by}", stroke, 2.4)
        self.dot(a, stroke)
        self.dot(b, stroke)
        self._body(lane, (ay + by) / 2, kind, text)

    def part_pol(self, aplus, bminus, text):
        """Electrolytic: red lead to +, black lead to -, dark band on the
        minus half of the can."""
        fill, stroke = C_CAP
        ax, ay = self.p(aplus)
        bx, by = self.p(bminus)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        self._path(f"M{ax},{ay} L{mx},{my}", C_POS, 3)
        self._path(f"M{mx},{my} L{bx},{by}", C_NEG, 3)
        w, h = 22, 20
        self.b.append(f'<rect x="{mx-w/2:.1f}" y="{my-h/2:.1f}" width="{w}" height="{h}" '
                      f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.b.append(f'<rect x="{mx-w/2+2:.1f}" y="{my+1:.1f}" width="{w-4}" '
                      f'height="{h/2-3:.1f}" rx="2" fill="#495057"/>')
        self.dot(aplus, C_POS)
        self.dot(bminus, C_NEG)
        self.b.append(f'<text x="{ax+13:.0f}" y="{ay+5:.0f}" font-size="15" font-weight="700" '
                      f'text-anchor="middle" fill="{C_POS}">+</text>')
        self.b.append(f'<text x="{bx+13:.0f}" y="{by+5:.0f}" font-size="15" font-weight="700" '
                      f'text-anchor="middle" fill="{C_NEG}">–</text>')
        self.b.append(f'<text x="{mx+w/2+7:.1f}" y="{my+5:.1f}" font-size="13" '
                      f'font-weight="600" fill="{stroke}">{text}</text>')

    def part_rail(self, a, which, text, kind, below=True):
        stroke = (C_RES if kind == "R" else C_CAP)[1]
        x, y = self.p(a)
        yb = self.band(int(a[:-1]), below)
        rx = self.rp if which == "+" else self.rn
        self._path(f"M{x},{y} L{x},{yb} L{rx},{yb}", stroke, 2.4)
        self.b.append(f'<circle cx="{rx}" cy="{yb}" r="4.5" fill="{stroke}"/>')
        self.dot(a, stroke)
        edge = 198 if self.rail == "left" else 502
        bx = (edge + x) / 2
        self._body(bx, yb, kind, "", horiz=True)
        anchor, tx = ("end", edge - 78) if self.rail == "left" else ("start", edge + 12)
        self.b.append(f'<text x="{tx}" y="{yb+5:.0f}" font-size="12" text-anchor="{anchor}" '
                      f'font-weight="600" fill="{stroke}">{text}</text>')

    def rail_cap(self, row, text, kind="C", note="", pol=False):
        """A part bridging the two rails. The rail is one node, so the row
        only says where to plug it in."""
        fill, stroke = C_RES if kind == "R" else C_CAP
        y = self.y[row]
        mid = (self.rp + self.rn) / 2
        if pol:
            self._path(f"M{self.rp},{y} L{mid},{y}", C_POS, 3)
            self._path(f"M{mid},{y} L{self.rn},{y}", C_NEG, 3)
        else:
            self._path(f"M{self.rp},{y} L{self.rn},{y}", stroke, 2.4)
        for x, c in ((self.rp, C_POS if pol else stroke), (self.rn, C_NEG if pol else stroke)):
            self.b.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{c}"/>')
        self.b.append(f'<rect x="{mid-7}" y="{y-9}" width="14" height="18" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if pol:
            sgn = 1 if self.rp > self.rn else -1
            self.b.append(f'<rect x="{mid+1 if sgn<0 else mid-6}" y="{y-7}" width="5" '
                          f'height="14" rx="2" fill="#495057"/>')
            self.b.append(f'<text x="{self.rp}" y="{y-11}" font-size="14" font-weight="700" '
                          f'text-anchor="middle" fill="{C_POS}">+</text>')
        if self.rail == "left":
            self.b.append(f'<text x="120" y="{y+5}" font-size="13" font-weight="600" '
                          f'text-anchor="end" fill="{stroke}">{text}</text>')
            if note:
                self.b.append(f'<text x="120" y="{y+21}" font-size="10.5" text-anchor="end" '
                              f'fill="#868e96">{note}</text>')
        else:
            self.b.append(f'<text x="{self.rn-26}" y="{y+5}" font-size="13" font-weight="600" '
                          f'text-anchor="end" fill="{stroke}">{text}</text>')
            if note:
                self.b.append(f'<text x="{self.rn-26}" y="{y+21}" font-size="10.5" '
                              f'text-anchor="end" fill="#868e96">{note}</text>')

    # ------------------------------------------------------------ blocks --
    def _runs(self, rows):
        runs, cur = [], []
        for r in rows:
            if cur and r != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            runs.append(cur)
        return runs

    def chip(self):
        """Only the pin rows this panel shows get drawn, in contiguous runs."""
        r0 = self.chip_row0
        shown = [r for r in self.rows if r0 <= r <= r0 + 6]
        for run in self._runs(shown):
            y0, y1 = self.y[run[0]], self.y[run[-1]]
            self.b.append(f'<rect x="{COL["E"]-13}" y="{y0-17}" width="{COL["F"]-COL["E"]+26}" '
                          f'height="{y1-y0+34}" rx="5" fill="#343a40"/>')
            if run[0] == r0:
                self.b.append(f'<path d="M{(COL["E"]+COL["F"])/2-9},{y0-17} a9,9 0 0 0 18,0" '
                              f'fill="none" stroke="#adb5bd" stroke-width="2"/>')
                self.b.append(f'<text x="{(COL["E"]+COL["F"])/2:.0f}" y="{y0-25}" font-size="12" '
                              f'font-weight="700" text-anchor="middle" fill="#343a40">'
                              f'{self.chip_name}</text>')
            for r in run:
                i, yy = r - r0, self.y[r]
                for c in (COL["E"], COL["F"]):
                    self.b.append(f'<circle cx="{c}" cy="{yy}" r="5" fill="#ffd43b"/>')
                self.b.append(f'<text x="{COL["E"]+11}" y="{yy+4}" font-size="10" '
                              f'text-anchor="middle" fill="#ffffff">{i+1}</text>')
                self.b.append(f'<text x="{COL["F"]-11}" y="{yy+4}" font-size="10" '
                              f'text-anchor="middle" fill="#ffffff">{14-i}</text>')

    def pin_marker(self, row, n):
        y = self.y[row]
        self.b.append(f'<rect x="{COL["E"]-13}" y="{y-16}" width="{COL["F"]-COL["E"]+26}" '
                      f'height="32" rx="5" fill="#343a40"/>')
        c = COL["E"] if n <= 7 else COL["F"]
        self.b.append(f'<circle cx="{c}" cy="{y}" r="5" fill="#ffd43b"/>')
        self.b.append(f'<text x="{(COL["E"]+COL["F"])/2:.0f}" y="{y+4}" font-size="11" '
                      f'text-anchor="middle" fill="#ffffff">ขา {n}</text>')

    def esp_block(self, names):
        """The DevKit's own pins, drawn as a module body in column J."""
        rows = [r for r in self.rows if r in names]
        for k, run in enumerate(self._runs(rows)):
            y0, y1 = self.y[run[0]], self.y[run[-1]]
            self.b.append(f'<rect x="{COL["J"]-15}" y="{y0-16}" width="54" '
                          f'height="{y1-y0+32}" rx="5" fill="#343a40"/>')
            for r in run:
                yy = self.y[r]
                self.b.append(f'<circle cx="{COL["J"]}" cy="{yy}" r="5" fill="#ffd43b"/>')
                self.b.append(f'<text x="{COL["J"]+12}" y="{yy+4}" font-size="10" '
                              f'fill="#ffffff">{names[r]}</text>')
            if k == 0:
                self.b.append(f'<text x="{COL["J"]+12}" y="{y0-30}" font-size="12" '
                              f'font-weight="700" text-anchor="middle" fill="#343a40">ESP32</text>')

    def esp_marker(self, row, name):
        """Right-hand board: the DevKit sits off the left edge, so its pin is
        marked on the A-E half instead of drawn as a module body."""
        y = self.y[row]
        self.b.append(f'<rect x="{COL["A"]-14}" y="{y-12}" width="{COL["E"]-COL["A"]+28}" '
                      f'height="24" rx="5" fill="#343a40"/>')
        self.b.append(f'<circle cx="{COL["A"]}" cy="{y}" r="5" fill="#ffd43b"/>')
        self.b.append(f'<text x="{COL["C"]+14}" y="{y+4}" font-size="11" text-anchor="middle" '
                      f'fill="#ffffff">ESP32 {name}</text>')

    def transducer(self, ref, text, cx=56, lead_side="left"):
        x, y = self.p(ref)
        self.b.append(f'<circle cx="{cx}" cy="{y}" r="21" fill="#f8f9fa" stroke="#495057" '
                      f'stroke-width="2"/>')
        self.b.append(f'<circle cx="{cx}" cy="{y}" r="11" fill="#dee2e6"/>')
        self.b.append(f'<text x="{cx}" y="{y-28}" font-size="11" text-anchor="middle" '
                      f'font-weight="700" fill="#495057">{text}</text>')
        s = 1 if lead_side == "left" else -1
        e = cx + s * 16
        ya = self.band(int(ref[:-1]), False)
        yb = self.band(int(ref[:-1]), True)
        self._path(f"M{e},{y-7} L{e+s*10},{y-7} L{e+s*10},{ya} L{x},{ya} L{x},{y}", C_SIG)
        self.dot(ref, C_SIG)
        self._path(f"M{e},{y+7} L{e+s*22},{y+7} L{e+s*22},{yb} L{self.rn},{yb}", C_NEG)
        self.b.append(f'<circle cx="{self.rn}" cy="{yb}" r="4.5" fill="{C_NEG}"/>')

    def note(self, x, y, lines, colour="#f08c00", bg="#fff4e6", w=160):
        self.maxy = max(self.maxy, y + 22 + 18 * len(lines))
        self.b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{22+18*len(lines)}" '
                      f'rx="6" fill="{bg}" stroke="{colour}"/>')
        for i, t in enumerate(lines):
            self.b.append(f'<text x="{x+10}" y="{y+21+i*18}" font-size="11.5" '
                          f'font-weight="{"700" if i == 0 else "400"}" fill="#212529">{t}</text>')

    def tag(self, ref, text, dx=0, dy=-15, anchor="middle", colour="#f08c00"):
        x, y = self.p(ref)
        self.b.append(f'<text x="{x+dx}" y="{y+dy}" font-size="11" text-anchor="{anchor}" '
                      f'font-weight="700" fill="{colour}">{text}</text>')

    # ------------------------------------------------------------ render --
    def svg(self):
        H = max(TOP + (len(self.rows) - 1) * PITCH + 76, self.maxy + 24)
        top, bot = TOP - 46, TOP + (len(self.rows) - 1) * PITCH + 40
        left = self.rail == "left"
        gx, ganchor = (198, "end") if left else (self.rp + 34, "start")
        dash0 = 176 if left else gx + 26
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{H}" '
             f'viewBox="0 0 {self.W} {H}" font-family="Tahoma, Segoe UI, sans-serif" fill="#343a40">',
             f'<rect width="{self.W}" height="{H}" fill="#ffffff"/>',
             f'<text x="20" y="30" font-size="17" font-weight="700" fill="#1971c2">{self.title}</text>',
             f'<text x="20" y="50" font-size="12" fill="#6c757d">{self.sub}</text>',
             f'<rect x="{self.bx0}" y="{top}" width="{self.bw}" height="{bot-top}" rx="6" '
             f'fill="#ffffff" stroke="#ced4da"/>',
             f'<rect x="308" y="{top}" width="24" height="{bot-top}" fill="#e9ecef"/>']
        for x, c in ((self.rp - 8, "#ffc9c9"), (self.rn - 8, "#a5d8ff")):
            o.append(f'<rect x="{x}" y="{top+14}" width="16" height="{bot-top-28}" rx="8" fill="{c}"/>')
        o.append(f'<rect x="202" y="{top+14}" width="104" height="{bot-top-28}" rx="8" fill="#f1f3f5"/>')
        o.append(f'<rect x="334" y="{top+14}" width="104" height="{bot-top-28}" rx="8" fill="#f1f3f5"/>')
        o.append(f'<text x="{self.rp}" y="{top+10}" font-size="13" font-weight="700" '
                 f'text-anchor="middle" fill="#e03131">+</text>')
        o.append(f'<text x="{self.rn}" y="{top+10}" font-size="13" font-weight="700" '
                 f'text-anchor="middle" fill="#1971c2">–</text>')
        for c, x in COL.items():
            o.append(f'<text x="{x}" y="{top+10}" font-size="12" text-anchor="middle" '
                     f'fill="#868e96">{c}</text>')
        prev = None
        for r in self.rows:
            y = self.y[r]
            for x in COL.values():
                o.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#ced4da"/>')
            for x in (self.rp, self.rn):
                o.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#ffffff"/>')
            o.append(f'<text x="{gx}" y="{y+4}" font-size="12" text-anchor="{ganchor}" '
                     f'fill="#6c757d">{r}</text>')
            if prev is not None and r != prev + 1:
                yy = (y + self.y[prev]) / 2
                o.append(f'<line x1="{dash0}" y1="{yy}" x2="{dash0+20}" y2="{yy}" '
                         f'stroke="#ced4da" stroke-width="1.5" stroke-dasharray="3 3"/>')
            prev = r
        o += self.b
        o.append('</svg>')
        return "\n".join(o)

    def write(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, self.name + ".svg"), "w", encoding="utf-8") as f:
            f.write(self.svg())
        return self.name + ".svg"
