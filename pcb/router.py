"""เดินลายทองแดงสองชั้นด้วยการค้นหาเส้นทางแบบแผ่คลื่น (Lee) แล้ว **ตรวจย้อน**

ตัวเดินลายผิดพลาดได้เสมอ สิ่งที่ทำให้เชื่อผลได้คือตัวตรวจท้ายไฟล์ ซึ่ง
**ไม่ดูว่าเราตั้งใจเดินยังไง แต่ดูทองแดงที่ได้จริง** แล้วถามสองคำถาม:

  1. ขาทุกขาของเน็ตเดียวกัน ต่อถึงกันจริงไหม  (ลายขาด = จับได้)
  2. ทองแดงคนละเน็ตเข้าใกล้กันเกินระยะปลอดภัยไหม  (ลายช็อต = จับได้)

ถ้าทั้งสองข้อผ่าน ลายที่ได้ใช้ผลิตได้ ไม่ต้องเชื่อใจตัวเดินลาย

**บทเรียนจากรอบแรก** เคยเก็บ 'พื้นที่จอง' เป็นเลขเน็ตเดียวต่อเซลล์ แบบใครจองก่อนได้ก่อน
ผลคือเซลล์ที่อยู่ใกล้ขาของเน็ต ก. แต่ถูกเน็ต ข. จองไว้ก่อน กลายเป็นเดินได้
แล้วลายก็ไปชิดขาของ ก. เกินระยะปลอดภัย ตัวตรวจจับได้ที่สามบอร์ด
ตอนนี้จึงคิดสิ่งกีดขวางใหม่ทุกครั้งจาก **ทองแดงจริงของเน็ตอื่น** ขยายออกตามระยะเผื่อ
ช้ากว่าเดิมแต่ถูกต้องโดยโครงสร้าง ไม่ใช่ถูกโดยบังเอิญ
"""
import numpy as np

from geom import CLEAR, TRACE, POWER, EDGE

VIA_DRILL = 0.6
VIA_PAD = 1.2


def _shift(a, di, dj):
    """เลื่อนอาเรย์แบบเติมศูนย์ ไม่วนขอบ — np.roll วนขอบ ทำให้ขอบซ้ายชนขอบขวา"""
    out = np.zeros_like(a)
    H, W = a.shape[-2:]
    si0, si1 = max(0, di), min(H, H + di)
    sj0, sj1 = max(0, dj), min(W, W + dj)
    out[..., si0:si1, sj0:sj1] = a[..., si0 - di:si1 - di, sj0 - dj:sj1 - dj]
    return out


def _disk(r_cells):
    R = int(np.ceil(r_cells))
    ii, jj = np.ogrid[-R:R + 1, -R:R + 1]
    return ii * ii + jj * jj <= r_cells * r_cells


def _dilate(mask, r_cells):
    """ขยายพื้นที่ออกทุกทิศเป็นวงกลมรัศมี r_cells

    ใช้ scipy ซึ่งเป็นภาษาซี — เขียนเป็นลูปเลื่อนอาเรย์ใน Python ได้ผลเท่ากัน
    แต่บอร์ดรวม 196x185 มม. มีเซลล์ 1.8 ล้านช่อง คูณ 81 ทิศ คูณ 90 เน็ต
    = หนึ่งหมื่นสามพันล้านครั้ง ซึ่งรอทั้งวันก็ไม่เสร็จ
    ขยายทีละชั้น ไม่ใช่ทั้งก้อน เพราะทองแดงคนละชั้นไม่ได้ชิดกันในทางกายภาพ
    """
    from scipy.ndimage import binary_dilation
    st = _disk(r_cells)
    if mask.ndim == 2:
        return binary_dilation(mask, st)
    return np.stack([binary_dilation(mask[L], st) for L in range(mask.shape[0])])


class Router:
    """ตารางทองแดงสองชั้น · ชั้น 0 = ด้านบน (top) · ชั้น 1 = ด้านล่าง (bottom)

    เก็บอาเรย์เดียวคือ cop = เลขเน็ตของทองแดงจริง
    สิ่งกีดขวางคำนวณสดจาก cop ทุกครั้งที่จะเดินเน็ตใหม่ จึงไม่มีทางเพี้ยน
    """

    def __init__(self, w, h, x0, y0, step=0.2):
        self.step, self.x0, self.y0 = step, x0, y0
        self.W, self.H = int(round(w / step)), int(round(h / step))
        self.cop = np.zeros((2, self.H, self.W), np.int16)
        # ข้ามชั้นได้เฉพาะตรงที่มีรูชุบ (ขาชิ้นส่วนหรือเวีย) เท่านั้น
        # ถ้าปล่อยให้ข้ามได้ทุกที่ ลายเน็ตเดียวกันที่บังเอิญทับกันคนละชั้น
        # จะถูกนับว่าต่อถึงกัน ทั้งที่จริงไม่ได้ต่อ = ตัวตรวจจะมองข้ามลายขาด
        self.link = np.zeros((self.H, self.W), bool)
        self.holes = []          # (x, y, drill) ไว้ทำไฟล์เจาะ
        self.vias = []
        self.tracks = []         # (layer, [(x,y)...], width, net)
        self.pours = []          # (ชั้น, เน็ต) ที่เทเป็นแผ่น
        self.mounts = []         # รูยึดสกรู ไม่ใช่ขาวงจร
        self.pads = {}           # pin -> (x, y, drill, dia, net)
        self.w_mm, self.h_mm = w, h

    # ------------------------------------------------------------ พื้นฐาน
    def _ij(self, x, y):
        return (int(round((y - self.y0) / self.step)),
                int(round((x - self.x0) / self.step)))

    def _disc(self, i, j, r_mm):
        r = int(np.ceil(r_mm / self.step))
        ii, jj = np.ogrid[-r:r + 1, -r:r + 1]
        m = ii * ii + jj * jj <= (r_mm / self.step) ** 2
        i0, i1 = max(0, i - r), min(self.H, i + r + 1)
        j0, j1 = max(0, j - r), min(self.W, j + r + 1)
        return (slice(i0, i1), slice(j0, j1),
                m[i0 - (i - r):i1 - (i - r), j0 - (j - r):j1 - (j - r)])

    def _paint_disc(self, L, i, j, r_mm, net):
        si, sj, m = self._disc(i, j, r_mm)
        self.cop[L][si, sj][m] = net

    def _edge_mask(self):
        """ห้ามทองแดงชิดขอบบอร์ด — โรงงานตัดคลาดได้ไม่กี่สิบไมครอน"""
        e = int(round(EDGE / self.step))
        m = np.zeros((2, self.H, self.W), bool)
        m[:, :e, :] = m[:, -e:, :] = True
        m[:, :, :e] = m[:, :, -e:] = True
        return m

    # ------------------------------------------------------------ วางขา
    def add_pad(self, pin, x, y, drill, dia, net):
        i, j = self._ij(x, y)
        self.pads[pin] = (x, y, drill, dia, net)
        self.holes.append((x, y, drill))
        for L in (0, 1):
            self._paint_disc(L, i, j, dia / 2, net)
        si, sj, m = self._disc(i, j, dia / 2)
        self.link[si, sj][m] = True

    def add_mount(self, x, y, drill=2.7, keepout=5.6):
        """รูยึดสกรู — ไม่ใช่ขาวงจร ต้องกันทองแดงรอบตัวไว้ให้หัวสกรู

        จองด้วยเลขเน็ต -1 ซึ่งไม่ตรงกับเน็ตไหนเลย ทุกเน็ตจึงมองว่าเป็นของคนอื่น
        และตัวตรวจการช็อตข้ามไป (มันดูเฉพาะเลขเน็ตที่มากกว่า 0)
        """
        i, j = self._ij(x, y)
        self.mounts.append((x, y, drill))
        self.holes.append((x, y, drill))
        for L in (0, 1):
            self._paint_disc(L, i, j, keepout / 2, -1)

    # ------------------------------------------------------------ เดินลาย
    def route(self, net, pins, width=TRACE):
        """ต่อขาทุกขาของเน็ตเข้าด้วยกัน คืนรายชื่อขาที่เดินไม่ถึง

        **เป้าหมายต้องตัดก้อนทองแดงของขาตัวเองออกก่อน** ไม่งั้นการค้นหาจะจบทันที
        ตั้งแต่ก้าวแรก เพราะก้าวออกจากขาไปหนึ่งเซลล์ก็ยังอยู่บนแป้นของขานั้นเอง
        ซึ่งเป็นทองแดงเน็ตเดียวกัน = เข้าเงื่อนไข 'ถึงเป้าหมาย' แล้ว
        อาการคือได้ลายยาวสองเซลล์ทุกเส้น และทุกขารายงานว่าต่อไม่ถึง
        """
        pts = [self._ij(*self.pads[p][:2]) for p in pins]
        # คิดสิ่งกีดขวางครั้งเดียวต่อเน็ต ไม่ใช่ต่อขา — ระหว่างเดินเน็ตเดียวกัน
        # ทองแดงของเน็ตอื่นไม่เปลี่ยน ที่งอกเป็นของเน็ตเราเอง ซึ่งไม่กีดขวางตัวเอง
        free = self._free_mask(net, width)
        todo = list(range(1, len(pts)))
        for _ in range(len(pts) + 2):
            lab = self._labels(net)
            base = int(lab[:, pts[0][0], pts[0][1]].max())
            todo = [k for k in range(1, len(pts))
                    if int(lab[:, pts[k][0], pts[k][1]].max()) != base]
            if not todo:
                return []
            moved = False
            for k in todo:
                own = int(lab[:, pts[k][0], pts[k][1]].max())
                goal = (self.cop == net) & (lab != own)
                if not goal.any():
                    continue
                path = self._bfs(pts[k], goal, free)
                if path is not None:
                    self._paint(path, net, width)
                    lab = self._labels(net)     # ทองแดงเปลี่ยน ต้องแบ่งก้อนใหม่
                    moved = True
            if not moved:
                break
        # สรุปจาก **สภาพจริงตอนจบ** ไม่ใช่จาก todo ของรอบสุดท้าย
        # todo ถูกคำนวณตอนต้นรอบ ถ้าระหว่างรอบนั้นต่อสำเร็จ ก็ยังค้างอยู่ในลิสต์
        # อาการคือรายงานว่าขาหลุด 12 ขา ทั้งที่วัดแล้วทุกขาอยู่ก้อนเดียวกันหมด
        lab = self._labels(net)
        base = int(lab[:, pts[0][0], pts[0][1]].max())
        return [pins[k] for k in range(1, len(pts))
                if int(lab[:, pts[k][0], pts[k][1]].max()) != base]

    def _free_mask(self, net, width):
        """เซลล์ที่เดินได้ = ไม่ใกล้ทองแดงของเน็ตอื่นเกินระยะเผื่อ และไม่ชิดขอบ"""
        foreign = (self.cop != 0) & (self.cop != net)
        # เผื่ออีกหนึ่งช่องตาราง เพราะทั้งการวางลายและการตรวจปัดเศษลงตารางเหมือนกัน
        # ถ้าคิดระยะพอดีเป๊ะ ผลจะออกมาขาดไปครึ่งช่องแล้วตัวตรวจฟ้อง (เจอที่บอร์ด
        # vref และ rx2 อย่างละจุด ห่างกัน 0.39 มม. จากเกณฑ์ 0.40)
        r = (width / 2 + CLEAR) / self.step + 1.0
        return ~(_dilate(foreign, r) | self._edge_mask())

    def _labels(self, nid):
        """แบ่งทองแดงของเน็ตหนึ่งเป็นก้อน ๆ ที่ต่อถึงกันจริง

        แบ่งทีละชั้นด้วย scipy แล้วค่อยเชื่อมข้ามชั้น **เฉพาะตรงที่มีรูชุบ**
        เชื่อมข้ามชั้นแบบเหมารวมไม่ได้ เพราะลายเน็ตเดียวกันที่ทับกันคนละชั้น
        โดยไม่มีรู ไม่ได้ต่อถึงกันจริง ถ้าเหมารวมจะมองข้ามลายขาด
        """
        from scipy.ndimage import label
        mask = self.cop == nid
        lab = np.zeros((2, self.H, self.W), np.int32)
        n = 0
        for L in (0, 1):
            l2, k = label(mask[L])
            lab[L] = np.where(l2 > 0, l2 + n, 0)
            n += k
        parent = list(range(n + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        both = self.link & mask[0] & mask[1]
        for x, y in set(zip(lab[0][both].tolist(), lab[1][both].tolist())):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        root = np.array([find(k) for k in range(n + 1)], np.int32)
        root[0] = 0
        return root[lab]

    def _bfs(self, start, goal, free):
        """แผ่คลื่นทั้งหน้าพร้อมกันด้วย numpy

        ลอง **ชั้นเดียวก่อน** ทั้งบนและล่าง ถ้าไปถึงได้ก็ไม่ต้องเจาะเวียเลย
        ค่อยยอมใช้สองชั้นเมื่อชั้นเดียวไปไม่ถึง — ได้ผลเหมือนใส่ค่าปรับให้เวีย
        แต่เร็วกว่ามาก เพราะไม่ต้องทำคิวลำดับความสำคัญทีละเซลล์ใน Python
        (แบบทีละเซลล์ใช้เวลา 319 วินาทีต่อบอร์ด แบบนี้ไม่ถึงวินาที)
        """
        # ลองด้านบนก่อน แล้วค่อยด้านล่าง — ด้านล่างสงวนไว้ให้แผ่นกราวด์
        # บอร์ดสองชั้นที่มีทั้งอนาล็อกและดิจิทัล วิธีมาตรฐานคือให้ชั้นหนึ่งเป็น
        # กราวด์เกือบเต็มแผ่น กระแสกลับจะได้วิ่งใต้ลายสัญญาณตรง ๆ วงเล็กที่สุด
        # ถ้าเทกราวด์ทั้งสองด้าน จะไม่เหลือที่ให้ VREF กับไฟแอนะล็อกวิ่งไปทั้งสี่มุม
        for layers in ((0,), (1,), (0, 1)):
            path = self._wave(start, goal, free, layers)
            if path is not None:
                return path
        return None

    def _wave(self, start, goal, free, layers, pad=150):
        """แผ่คลื่นเฉพาะในกรอบรอบจุดเริ่มกับเป้าหมาย ไม่กวาดทั้งบอร์ด

        เน็ตส่วนใหญ่เชื่อมของที่อยู่ใกล้กัน แต่การแผ่คลื่นเดิมทำงานบนอาเรย์
        เต็มบอร์ด 1.8 ล้านเซลล์ทุกก้าว บอร์ดรวมจึงใช้เวลา 31 นาทีและยังไม่เสร็จ
        ตัดให้เหลือเฉพาะกรอบที่เกี่ยวข้อง (เผื่อขอบ 30 มม.) เร็วขึ้นหลายสิบเท่า
        ถ้าหาทางในกรอบไม่เจอ ค่อยขยายเป็นทั้งบอร์ดแล้วลองใหม่
        """
        gi = np.nonzero(goal.any(0).any(1))[0]
        gj = np.nonzero(goal.any(0).any(0))[0]
        if len(gi) == 0:
            return None
        i0 = max(0, min(int(gi[0]), start[0]) - pad)
        i1 = min(self.H, max(int(gi[-1]), start[0]) + pad + 1)
        j0 = max(0, min(int(gj[0]), start[1]) - pad)
        j1 = min(self.W, max(int(gj[-1]), start[1]) + pad + 1)
        if (i1 - i0) * (j1 - j0) < self.H * self.W:
            sub = self._wave_in((start[0] - i0, start[1] - j0),
                                goal[:, i0:i1, j0:j1], free[:, i0:i1, j0:j1],
                                layers, i0, j0)
            if sub is not None:
                return sub
        return self._wave_in(start, goal, free, layers, 0, 0)

    def _wave_in(self, start, goal, free, layers, oi, oj):
        si, sj = start
        H, W = goal.shape[1], goal.shape[2]
        use = np.zeros((2, H, W), bool)
        for L in layers:
            use[L] = True
        allowed = (free | goal) & use
        seen = np.zeros((2, H, W), bool)
        dist = np.full((2, H, W), -1, np.int32)
        link = self.link[oi:oi + H, oj:oj + W]
        for L in layers:
            seen[L, si, sj] = True
            dist[L, si, sj] = 0
        cur = seen.copy()
        two = len(layers) == 2
        d = 0
        while cur.any():
            hits = cur & goal
            hits[:, si, sj] = False
            if hits.any():
                L, i, j = np.argwhere(hits)[0]
                p = self._back(dist, (int(L), int(i), int(j)), two, link)
                return None if p is None else [(a, b + oi, c + oj) for a, b, c in p]
            nxt = (_shift(cur, 1, 0) | _shift(cur, -1, 0)
                   | _shift(cur, 0, 1) | _shift(cur, 0, -1))
            if two:
                nxt |= cur[::-1] & link[None, :, :]
            nxt &= allowed & ~seen
            if not nxt.any():
                return None
            d += 1
            dist[nxt] = d
            seen |= nxt
            cur = nxt
        return None

    def _back(self, dist, cell, two, link):
        """ไล่ย้อนจากจุดที่ถึงเป้าหมาย ลงตามระยะที่บันทึกไว้ทีละก้าว"""
        H, W = dist.shape[1], dist.shape[2]
        L, i, j = cell
        path = [(L, i, j)]
        d = int(dist[L, i, j])
        while d > 0:
            cand = [(L, i + 1, j), (L, i - 1, j), (L, i, j + 1), (L, i, j - 1)]
            if two and link[i, j]:
                cand.append((L ^ 1, i, j))
            for nL, ni, nj in cand:
                if (0 <= ni < H and 0 <= nj < W
                        and dist[nL, ni, nj] == d - 1):
                    L, i, j, d = nL, ni, nj, d - 1
                    path.append((L, i, j))
                    break
            else:
                return None
        return path[::-1]

    def _paint(self, path, net, width):
        r = width / 2
        run, last = [], path[0][0]
        for L, i, j in path:
            if L != last:
                self._flush(run, last, net, width)
                x, y = self.x0 + j * self.step, self.y0 + i * self.step
                self.vias.append((x, y))
                self.holes.append((x, y, VIA_DRILL))
                for LL in (0, 1):
                    self._paint_disc(LL, i, j, VIA_PAD / 2, net)
                si, sj, m = self._disc(i, j, VIA_PAD / 2)
                self.link[si, sj][m] = True
                run, last = [(i, j)], L
            run.append((i, j))
            self._paint_disc(L, i, j, r, net)
        self._flush(run, last, net, width)

    def _flush(self, run, layer, net, width):
        if len(run) < 2:
            return
        pts = [(self.x0 + j * self.step, self.y0 + i * self.step) for i, j in run]
        self.tracks.append((layer, _simplify(pts), width, net))

    # ------------------------------------------------------------ เทกราวด์
    def pour(self, net, layer, thermal=True):
        """เทกราวด์เป็นแผ่นเต็มชั้น แทนการลากเป็นเส้น

        วงจรนี้ขยาย 1,156 เท่า กระแสกลับของช่องหนึ่งที่วิ่งผ่านลายกราวด์ยาว ๆ
        จะสร้างแรงดันตกคร่อมที่ช่องอื่นอ่านเป็นสัญญาณ = ครอสทอล์ก
        แผ่นกราวด์ให้กระแสกลับวิ่งใต้ลายสัญญาณตรง ๆ วงจึงเล็กที่สุดเท่าที่เป็นไปได้
        (โปรเจกต์นี้เคยวัดแล้วว่าเพิ่มสายกราวด์สั้นเส้นเดียวลด noise ได้ 3.3 เท่า)

        thermal = เว้นคูรอบขากราวด์ เหลือซี่เชื่อมสี่ทิศ ถ้าเชื่อมเต็มหน้า
        แผ่นทองแดงจะดูดความร้อนจนหัวแร้งธรรมดาบัดกรีขาไม่ติด
        """
        free = self._free_mask(net, 0.0)[layer]
        self.cop[layer][free & (self.cop[layer] == 0)] = net
        self.pours.append((layer, net))
        if not thermal:
            return
        gap_in, ring_w, spoke = 0.15, 0.55, 0.35
        for _pin, (x, y, _d, dia, n) in self.pads.items():
            if n != net:
                continue
            i, j = self._ij(x, y)
            r_in, r_out = dia / 2 + gap_in, dia / 2 + gap_in + ring_w
            R = int(np.ceil(r_out / self.step))
            ii, jj = np.ogrid[-R:R + 1, -R:R + 1]
            d2 = ii * ii + jj * jj
            sp = spoke / self.step
            ring = ((d2 > (r_in / self.step) ** 2) & (d2 <= (r_out / self.step) ** 2)
                    & ~((np.abs(ii) <= sp) | (np.abs(jj) <= sp)))
            i0, i1 = max(0, i - R), min(self.H, i + R + 1)
            j0, j1 = max(0, j - R), min(self.W, j + R + 1)
            sub = ring[i0 - (i - R):i1 - (i - R), j0 - (j - R):j1 - (j - R)]
            win = self.cop[layer][i0:i1, j0:j1]
            win[sub & (win == net)] = 0

    # ------------------------------------------------------------ ตรวจย้อน
    def verify(self, nets, names):
        errs = []
        for nid, pins in nets.items():
            comp = self._component(nid, self._ij(*self.pads[pins[0]][:2]))
            for p in pins[1:]:
                i, j = self._ij(*self.pads[p][:2])
                if not (comp[0, i, j] or comp[1, i, j]):
                    errs.append(f"เน็ต {names[nid]}: {p} ต่อไม่ถึงขาอื่น")
        # ตรวจระยะห่างโดยขยายทองแดงของแต่ละเน็ตออกไปครึ่งเกณฑ์ แล้วดูว่าไปทับ
        # ทองแดงเน็ตอื่นไหม — ได้ผลเท่ากับไล่เทียบทุกคู่เซลล์ แต่เร็วกว่ามาก
        from scipy.ndimage import binary_dilation
        st = _disk(CLEAR / self.step)
        for L in (0, 1):
            a = self.cop[L]
            for nid in np.unique(a):
                if nid <= 0:
                    continue
                near = binary_dilation(a == nid, st)
                bad = near & (a > 0) & (a != nid)
                if bad.any():
                    i, j = np.argwhere(bad)[0]
                    errs.append(
                        f"ชั้น {'top' if L == 0 else 'bottom'}: "
                        f"{names[int(nid)]} กับ {names[int(a[i, j])]} "
                        f"ใกล้กันเกิน {CLEAR} มม. ที่ "
                        f"({self.x0 + j * self.step:.1f}, "
                        f"{self.y0 + i * self.step:.1f})")
                    return errs
        return errs

    def _component(self, nid, start):
        """ก้อนทองแดงที่ต่อถึงขานั้นจริง — ตัวตรวจใช้ตอบว่า 'ลายขาดไหม'"""
        lab = self._labels(nid)
        own = int(lab[:, start[0], start[1]].max())
        return (lab == own) & (own > 0)


def _simplify(pts):
    """ทิ้งจุดกลางที่อยู่บนเส้นตรงเดียวกัน ให้ไฟล์ Gerber ไม่บวม"""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        if (b[0] - a[0]) * (c[1] - a[1]) != (b[1] - a[1]) * (c[0] - a[0]):
            out.append(b)
    out.append(pts[-1])
    return out


POWER_NETS = {"GND", "+5V", "3V3", "VREF", "A3V3", "MCU3V3", "P5V"}


def route_board(pl):
    """เดินลายทั้งบอร์ด คืน (router, ปัญหาที่พบ)

    เดินเน็ตสัญญาณก่อน แล้วค่อยไฟ เพราะเน็ตไฟมีขาเยอะและอ้อมได้
    ถ้าเดินไฟก่อน มันจะกินทางจนสัญญาณหาทางไม่เจอ
    """
    from geom import extent
    w, h, x0, y0 = extent(pl)
    r = Router(w, h, x0, y0)
    ids = {net: k for k, net in enumerate(pl["nets"], start=1)}
    names = {k: n for n, k in ids.items()}
    for mx, my in ((x0 + 3.5, y0 + 3.5), (x0 + w - 3.5, y0 + 3.5),
                   (x0 + 3.5, y0 + h - 3.5), (x0 + w - 3.5, y0 + h - 3.5)):
        r.add_mount(mx, my)
    for net, pins in pl["nets"].items():
        for p in pins:
            x, y, d, dia = pl["pad"][p]
            r.add_pad(p, x, y, d, dia, ids[net])
    # ลำดับ: เน็ตสัญญาณก่อน แล้วค่อยไฟ
    # เคยลองกลับลำดับให้ไฟเดินก่อน (เพื่อแก้ VREF ของบอร์ดรวมที่ต้องวิ่งไกล)
    # ผลคือบอร์ดรวมดีขึ้นแต่ rx2 พังแทน 24 จุด เพราะลายไฟไปตัดแผ่นกราวด์แตก
    # ไม่มีลำดับเดียวที่ดีกับทุกบอร์ด จึงคงของเดิมที่ผ่านสามบอร์ดไว้ก่อน
    sig = sorted(set(pl["nets"]) - POWER_NETS, key=lambda n: len(pl["nets"][n]))
    seq = sig + [n for n in ("A3V3", "VREF", "MCU3V3", "3V3", "P5V", "+5V")
                 if n in pl["nets"]]
    fails = []
    for net in seq:
        wtr = POWER if net in POWER_NETS else TRACE
        fails += [f"เน็ต {net}: เดินไม่ถึง {p}"
                  for p in r.route(ids[net], pl["nets"][net], wtr)]
    # กราวด์ทำท้ายสุดและทำเป็นแผ่น ไม่ใช่เส้น — เหตุผลอยู่ใน Router.pour
    if "GND" in pl["nets"]:
        g = ids["GND"]
        # เทเฉพาะด้านล่าง ให้เป็นแผ่นกราวด์เกือบเต็ม ส่วนด้านบนเป็นชั้นสัญญาณ
        r.pour(g, 1)
        # ขากราวด์ที่แผ่นไปไม่ถึง (โดนลายอื่นตัดขาด) ค่อยลากเส้นเสริมให้
        fails += [f"เน็ต GND: เดินไม่ถึง {p}"
                  for p in r.route(g, pl["nets"]["GND"], POWER)]
    errs = fails + r.verify({ids[n]: p for n, p in pl["nets"].items()}, names)
    return r, errs
