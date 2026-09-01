#!/usr/bin/env python3
"""หน้าจอกลางที่ใช้ร่วมกัน — **แสดงเฉพาะข้อมูลดิบ** ไม่ปรับปรุงสัญญาณใด ๆ

ใช้ทั้งตอนอัด dataset (record.py) และตอนตรวจระบบ (check.py) เพื่อให้เห็นภาพเดียวกันเสมอ

สิ่งที่แสดง = สิ่งที่บันทึกลงไฟล์เป๊ะ ๆ ไม่มีอะไรมาคั่นกลาง:
  * ภาพ depth ดิบจากกล้อง (มิลลิเมตร) ไล่สีตามระยะเท่านั้น
  * ค่า ADC ดิบทั้ง 4 ช่อง (0-4095) วาดแบบ min-max ต่อพิกเซลอย่างออสซิลโลสโคป
  * เฉลยจากกล้อง (ระยะ/มุมของช่องที่ใกล้ที่สุด) ซึ่งเป็นค่าที่จะถูกบันทึกเป็น label

**ไม่มี**: การกรองความถี่ · การหาเปลือกคลื่น (envelope) · การหายอด · SNR · ระยะจากเสียง
สิ่งเหล่านั้นเป็นการ "ตีความ" ข้อมูล ย้ายไปทำตอนวิเคราะห์/เทรนแทน จะได้เปลี่ยนวิธี
ย้อนหลังได้โดยไม่ต้องเก็บใหม่ และหน้าจอนี้จะไม่หลอกตาด้วยสัญญาณที่ถูกแต่งมาแล้ว

ผลพลอยได้: ไม่มี FFT ในเส้นทางแสดงผลอีกต่อไป จึงตัดความเสี่ยง heap corruption
ที่เคยเกิดตอน numpy FFT ทำงานพร้อมกับ OpenNI คนละเธรดไปด้วย

ข้อความบนภาพเป็นอังกฤษ เพราะ OpenCV ไม่มีฟอนต์ไทย (ไทยจะออกมาเป็น ???)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

C = 343.0
T0_US = 1220.0               # ใช้แค่ทำป้ายแกนระยะ ไม่ได้แตะข้อมูล

DW, DH = 480, 360            # ขนาดภาพ depth บนหน้าจอ
TW, TH = 500, 112            # ขนาดกราฟต่อช่อง
OV_H, HIST_GAP = 132, 46     # กราฟประวัติแบบเลื่อน (แผงล่างสุด) + ระยะเว้นจากแผงบน
PAD, TOP = 12, 104
W = PAD * 3 + DW + TW + 34
_STACK = TH * 4 + HIST_GAP + OV_H + 28
H = TOP + max(DH, _STACK) + 74

BG, PANEL = (22, 24, 29), (34, 38, 46)
CH_COL = [(90, 220, 90), (250, 200, 90), (90, 190, 250), (200, 130, 250)]
OKC, BADC, WARNC = (110, 230, 130), (90, 90, 245), (80, 200, 250)
VIS_MIN_MM, VIS_MAX_MM = 0, 2000     # ไล่สี 0-200 cm · ค่า 0 (อ่านไม่ได้) วาดดำแยก

# ตำแหน่งจริงของหัวรับบนแผ่น plate_mini — RX สี่ตัวที่มุมทั้งสี่ TX อยู่กลาง
#   TL = GPIO35    TR = GPIO34
#          TX GPIO18
#   BL = GPIO32    BR = GPIO33
# เบสไลน์ 110 มม. ทั้งสองแกน (ของเดิมเรียงเป็นเส้นนอน แยกบน-ล่างไม่ได้เลย)
# เรียงตามลำดับอ่านหนังสือ: สองกราฟบน = หัวรับแถวบน สองกราฟล่าง = แถวล่าง
POS = {35: (0, "TOP LEFT"), 34: (1, "TOP RIGHT"),
       32: (2, "BOTTOM LEFT"), 33: (3, "BOTTOM RIGHT")}


def channel_order(pins):
    """คืน [(ดัชนีในข้อมูล, ป้ายชื่อ), ...] เรียงจากซ้ายสุดไปขวาสุด

    ข้อมูลใน counts เรียงตามลำดับขาที่ตั้งไว้ (34,35,32,33) ซึ่งไม่ตรงกับตำแหน่งจริง
    ฟังก์ชันนี้จึงจับคู่ดัชนีข้อมูลกับตำแหน่งบนแผ่นให้ ถ้าเจอขาที่ไม่รู้จักจะใช้
    ลำดับเดิมและป้าย GPIO ตามปกติ
    """
    out = []
    for i, p_ in enumerate(pins):
        rank, name = POS.get(int(p_), (100 + i, f"GPIO{p_}"))
        out.append((rank, i, name))
    out.sort()
    return [(i, name) for _, i, name in out]


def measure(depth, ping, gate=(40.0, 200.0)):
    """สรุปค่าจากข้อมูลดิบล้วน — ไม่มีการกรองหรือแปลงสัญญาณใด ๆ"""
    from labels import depth_to_profile, bin_angles
    out = {"valid": float(np.count_nonzero(depth)) / depth.size,
           "cam_cm": None, "cam_deg": None, "bin": None,
           "counts": None, "rate": None, "amps": None,
           "in_range": False, "gate": gate}
    dist, ok = depth_to_profile(depth)
    if ok.any():
        kb = int(np.argmin(np.where(ok, dist, 1e9)))
        out["bin"] = kb
        out["cam_cm"] = float(dist[kb]) / 10.0
        out["cam_deg"] = -float(bin_angles()[kb])   # กลับเครื่องหมายให้ตรงมุมคนหน้ากล้อง
        out["in_range"] = gate[0] <= out["cam_cm"] <= gate[1]
    if ping is not None:
        c = ping["counts"]
        out["rate"] = float(ping["rate"])
        out["counts"] = c
        # แอมพลิจูดดิบรายช่อง (นับจากค่ากลางของช่องนั้น) — สถิติตรง ๆ ไม่ใช่การกรอง
        out["amps"] = [float(np.max(np.abs(c[i].astype(np.float32)
                                           - np.median(c[i])))) * 3.3 / 4095 * 1000
                       for i in range(c.shape[0])]
    return out


def _depth_panel(img, depth, m):
    import cv2
    norm = (np.clip(depth.astype(np.float32), VIS_MIN_MM, VIS_MAX_MM) - VIS_MIN_MM) \
        / (VIS_MAX_MM - VIS_MIN_MM)
    vis = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth == 0] = 0                       # 0 = อ่านไม่ได้ ให้เป็นสีดำ
    vis = cv2.resize(vis, (DW, DH), interpolation=cv2.INTER_NEAREST)
    x0, y0 = PAD, TOP
    if m["bin"] is not None:
        from labels import N_BINS
        bw = DW / N_BINS
        bx = int(x0 + (m["bin"] + 0.5) * bw)
        cv2.rectangle(vis, (int(m["bin"] * bw), 0),
                      (int((m["bin"] + 1) * bw), DH - 1), (255, 255, 255), 1)
        img[y0:y0 + DH, x0:x0 + DW] = vis
        cv2.line(img, (bx, y0), (bx, y0 + DH), (255, 255, 255), 1)
        cv2.putText(img, f"{m['cam_cm']:.0f}cm {m['cam_deg']:+.0f}deg",
                    (max(x0 + 4, bx - 60), y0 + DH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        img[y0:y0 + DH, x0:x0 + DW] = vis
        cv2.putText(img, "no depth", (x0 + 16, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BADC, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + DW, y0 + DH), (70, 76, 86), 1)
    cv2.putText(img, f"DEPTH raw {depth.shape[1]}x{depth.shape[0]} mm   "
                     f"valid {m['valid']:.0%}   colour 0-200 cm",
                (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                OKC if m["valid"] > 0.25 else BADC, 1, cv2.LINE_AA)


def _traces(img, m, pins):
    """วาดค่า ADC ดิบล้วน — min/max ต่อพิกเซลแบบออสซิลโลสโคป

    ต้องวาดแบบ min-max เพราะข้อมูล ~870 จุดลงพื้นที่ 500 พิกเซล ถ้าสุ่มเลือกจุดมาวาด
    คลื่นความถี่สูงจะหายไป การเก็บทั้งค่าสูงสุดและต่ำสุดของทุกช่วงพิกเซลทำให้เห็น
    ความสูงจริงของทุกลูกคลื่น โดยไม่ได้ดัดแปลงข้อมูลเลย
    """
    import cv2
    x0 = PAD * 2 + DW
    cv2.putText(img, "ULTRASONIC  raw ADC counts, full scale 0-4095   no filtering, no scaling",
                (x0, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (170, 175, 185), 1, cv2.LINE_AA)
    if m["counts"] is None:
        cv2.rectangle(img, (x0, TOP), (x0 + TW, TOP + TH * 4), PANEL, -1)
        cv2.putText(img, "no ping", (x0 + 20, TOP + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BADC, 2, cv2.LINE_AA)
        return
    cnt, rate = m["counts"], m["rate"]
    nch, n = cnt.shape
    order = channel_order(pins)
    # **แกนตั้งตรึงที่ 0-4095 เสมอ** — ไม่ปรับสเกล ไม่เลื่อนศูนย์กลาง
    # ความสูงบนจอ = ค่าจาก ADC จริง ๆ อ่านเทียบกันข้ามช่องและข้ามเฟรมได้ตรง ๆ
    FULL = 4095.0
    for slot, (ci, name) in enumerate(order):
        y0 = TOP + slot * TH
        yh = TH - 8
        col = CH_COL[slot % 4]
        cv2.rectangle(img, (x0, y0), (x0 + TW, y0 + yh), PANEL, -1)
        for lvl in (0, 1024, 2048, 3072, 4095):
            gy = int(y0 + yh - lvl / FULL * (yh - 1))
            cv2.line(img, (x0, gy), (x0 + TW, gy), (46, 50, 58), 1)
            cv2.putText(img, str(lvl), (x0 + TW + 3, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (95, 100, 110), 1, cv2.LINE_AA)
        raw = cnt[ci].astype(np.float32)
        if raw.size >= TW:
            k_ = raw.size // TW
            blk = raw[:k_ * TW].reshape(TW, k_)
            lo_v, hi_v = blk.min(1), blk.max(1)
        else:
            idxs = np.linspace(0, raw.size - 1, TW).astype(int)
            lo_v = hi_v = raw[idxs]
        # เติมแท่ง min-max ด้วยหน้ากากทั้งผืนทีเดียว ไม่วนเรียก cv2.line ทีละพิกเซล
        # (วนแบบเดิม 500 ครั้ง x 4 ช่อง กินเวลา ~8 ms/เฟรม ซึ่งทำให้คาบยิงหลุดจาก 50 ms)
        y_lo = np.clip((yh - lo_v / FULL * (yh - 1)).astype(np.int32), 0, yh - 1)
        y_hi = np.clip((yh - hi_v / FULL * (yh - 1)).astype(np.int32), 0, yh - 1)
        rows = np.arange(yh)[:, None]
        mask = (rows >= y_hi[None, :]) & (rows <= y_lo[None, :])
        sub = img[y0:y0 + yh, x0:x0 + TW]
        sub[mask] = col
        cv2.putText(img, name, (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"GPIO{pins[ci]}   min {int(raw.min())}  max {int(raw.max())}",
                    (x0 + 78, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150, 155, 165), 1, cv2.LINE_AA)

    # ---- แผงล่างสุด: ประวัติแบบเลื่อน ----
    # วาดในฟังก์ชัน render() แทน เพราะต้องใช้บัฟเฟอร์ที่ข้ามเฟรม
    # แกนระยะ = การแปลงหน่วยของแกนเวลา (index / rate) ไม่ได้แตะข้อมูล
    yb = TOP + TH * 4
    far = (n / rate * 1e6 - T0_US) * 1e-6 * C / 2 * 100
    for cm in range(0, int(far) + 1, 25):
        idx = (T0_US + 2 * cm / 100 / C * 1e6) * 1e-6 * rate
        x = int(x0 + idx / max(n - 1, 1) * TW)
        if not (x0 <= x <= x0 + TW):
            continue
        cv2.line(img, (x, TOP), (x, yb - 10), (48, 52, 60), 1)
        cv2.putText(img, str(cm), (x - 8, yb + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 126, 136), 1, cv2.LINE_AA)
    cv2.putText(img, f"x = distance (cm), t0={T0_US:.0f}us    "
                     f"{n} samples @ {rate/1000:.1f} kHz/ch",
                (x0, yb + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (110, 116, 126), 1, cv2.LINE_AA)


class History:
    """เก็บช่วงค่าดิบ (ต่ำสุด-สูงสุด) ของทุกช่องย้อนหลัง ไว้วาดเป็นกราฟเลื่อน

    เก็บทั้ง min และ max ของเฟรมนั้น ไม่ใช่ค่าเดียว เพราะ "ค่าดิบทั้งหมด ณ ขณะนั้น"
    คือช่วงที่สัญญาณกวาดไปทั้งหมดในเฟรม ย่อเหลือค่าเดียวจะทิ้งข้อมูลไปครึ่งหนึ่ง
    """

    def __init__(self, width=TW):
        from collections import deque
        self.cols = deque(maxlen=width)      # ซ้ายสุด = ใหม่สุด

    def push(self, m):
        if m.get("counts") is None:
            return
        self.cols.appendleft([(int(c.min()), int(c.max())) for c in m["counts"]])


def _history_panel(img, hist, pins):
    import cv2
    x0 = PAD * 2 + DW
    oy = TOP + TH * 4 + HIST_GAP
    cv2.rectangle(img, (x0, oy), (x0 + TW, oy + OV_H), PANEL, -1)
    cv2.putText(img, "HISTORY  raw min-max per channel   (newest on the LEFT)",
                (x0, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (170, 175, 185), 1, cv2.LINE_AA)
    FULL = 4095.0
    for lvl in (0, 1024, 2048, 3072, 4095):
        gy = int(oy + OV_H - lvl / FULL * (OV_H - 1))
        cv2.line(img, (x0, gy), (x0 + TW, gy), (46, 50, 58), 1)
        cv2.putText(img, str(lvl), (x0 + TW + 3, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (95, 100, 110), 1, cv2.LINE_AA)
    if hist is not None and hist.cols:
        cols = list(hist.cols)[:TW]
        xs = (x0 + np.arange(len(cols))).astype(np.int32)   # ซ้ายสุด = ใหม่สุด
        # วาดเป็นเส้นขอบบน/ล่างของแต่ละช่อง ไม่ใช่แท่งทึบ ไม่งั้นช่องที่วาดทีหลัง
        # จะบังช่องก่อนหน้าจนเห็นสีเดียว
        for slot, (ci, _n) in enumerate(channel_order(pins)):
            if ci >= len(cols[0]):
                continue
            for pick in (1, 0):                             # 1 = max, 0 = min
                v = np.array([c[ci][pick] for c in cols], np.float32)
                ys = (oy + OV_H - v / FULL * (OV_H - 1)).astype(np.int32)
                cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False,
                              CH_COL[slot % 4], 1, cv2.LINE_AA)
        n = len(hist.cols)
        cv2.putText(img, f"<- newest      {n} frames", (x0 + 6, oy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 156, 166), 1, cv2.LINE_AA)
    lx = x0 + 6
    for slot, (_i, name) in enumerate(channel_order(pins)):
        cv2.putText(img, name, (lx, oy + OV_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, CH_COL[slot % 4], 1, cv2.LINE_AA)
        lx += cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] + 14


_CANVAS = None


def render(depth, ping, m, pins, title, lines, foot="", sub="", hist=None):
    """คืนภาพ BGR พร้อมแสดง — lines = [(ข้อความ, สี)] แถวสถานะบนสุด

    **ใช้ผืนภาพเดิมซ้ำทุกครั้ง** ไม่จองใหม่ — ภาพขนาด 832x1050x3 คือ 2.6 MB
    ถ้าจองใหม่ทุกเฟรมที่ 20 fps = 52 MB/วินาที ทำให้ GC ทำงานถี่มาก และ GC ที่วิ่ง
    พร้อมกับ OpenNI ในอีกเธรดทำให้เกิด heap corruption (เจอมาแล้ว)
    => ผู้เรียกต้องใช้ภาพทันที (imshow/imwrite) ห้ามเก็บอ้างอิงไว้ข้ามเฟรม
    """
    import cv2
    global _CANVAS
    if _CANVAS is None or _CANVAS.shape != (H, W, 3):
        _CANVAS = np.empty((H, W, 3), np.uint8)
    img = _CANVAS
    img[:] = BG
    cv2.putText(img, title, (PAD, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (210, 215, 225), 1, cv2.LINE_AA)
    x = PAD
    for txt, col in lines:
        cv2.putText(img, txt, (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)
        x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 22
    if sub:
        cv2.putText(img, sub, (PAD, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (130, 136, 146), 1, cv2.LINE_AA)
    _depth_panel(img, depth, m)
    _traces(img, m, pins)
    _history_panel(img, hist, pins)

    # แถบล่าง = ค่าที่จะถูกบันทึกจริง ไม่มีค่าที่คำนวณต่อยอด
    y = TOP + max(DH, _STACK) + 26
    cv2.rectangle(img, (PAD, y - 22), (W - PAD, y + 22), PANEL, -1)
    cam = f"{m['cam_cm']:.0f} cm" if m["cam_cm"] is not None else "--"
    deg = f"{m['cam_deg']:+.0f} deg" if m["cam_deg"] is not None else "--"
    if m["counts"] is not None:
        rng_txt = "  ".join(f"{int(m['counts'][i].min())}-{int(m['counts'][i].max())}"
                            for i, _n in channel_order(pins))
    else:
        rng_txt = "--"
    for i, (lab, val, col) in enumerate((
            ("label dist", cam, (250, 250, 250)),
            ("label angle", deg, (250, 250, 250)),
            ("depth valid", f"{m['valid']:.0%}",
             OKC if m["valid"] > 0.4 else WARNC),
            ("raw min-max  L->R", rng_txt, (200, 206, 216)))):
        bx = PAD + 10 + i * 190
        cv2.putText(img, lab, (bx, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (140, 146, 156), 1, cv2.LINE_AA)
        fs = 0.44 if lab.startswith("raw") else 0.58
        cv2.putText(img, val, (bx, y + 16), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    col, 2 if fs > 0.5 else 1, cv2.LINE_AA)
    # ป้ายบอกสถานะ ไม่ใช่ตัวกรอง — ระบบบันทึกทุกเฟรมเสมอ
    if m["cam_cm"] is None:
        badge, bcol = "REC  no depth label", WARNC
    elif not m["in_range"]:
        badge, bcol = "REC  label beyond sonar", WARNC
    else:
        badge, bcol = "REC  in range", OKC
    tw = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    cv2.rectangle(img, (W - PAD - tw - 24, y - 19), (W - PAD - 4, y + 19), bcol, -1)
    cv2.putText(img, badge, (W - PAD - tw - 14, y + 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 22, 26), 2, cv2.LINE_AA)
    if foot:
        cv2.putText(img, foot, (PAD, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 126, 136), 1, cv2.LINE_AA)
    return img
