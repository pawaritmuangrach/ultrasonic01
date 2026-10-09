"""หน้าจอดูโมเดลรอบสาม — กลุ่มจุดไล่เฉดตามระยะ

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 1 : INTERFACE   (ไฟล์นี้ทั้งไฟล์)
═══════════════════════════════════════════════════════════════════════

ไฟล์นี้ **ไม่มีโค้ด ML เลย** มีแต่การวาดภาพและการอ่านปุ่ม
ตัวโมเดลอยู่ที่ model3.py · การเทรนอยู่ที่ train_map3.py
แยกกันเพื่อให้อ่านทีละเรื่อง และเปลี่ยนหน้าจอได้โดยไม่แตะโมเดล

    python car/live_map2.py --port COM6 --no-cam       ดูสด เสียงล้วน
    python car/live_map2.py --name mix --section 4      เล่นย้อนจากที่อัดไว้

ซ้าย = กล้อง (ความจริง) · ขวา = เสียงล้วน (โมเดล) วาดบนตารางเดียวกัน
สีฟ้า = ใกล้ · สีม่วง = ไกล (ช่วงที่เซ็นเซอร์เห็นคือ 40-200 ซม.)

**ถ้าไม่เปิดกล้อง ช่องซ้ายจะกลายเป็นแผงเช็คคลื่นดิบสี่ช่อง** (กด c สลับได้ทุกเมื่อ)
มีไว้เพราะภาพขวาที่ดูมั่วตอบไม่ได้ว่าเป็นที่โมเดลหรือที่วงจร — แผงซ้ายตอบให้
ช่องไหนเสียจะขึ้นแดงพร้อมบอกว่าเสียแบบไหน ดู chan_health() ว่าตัดสินจากอะไร
"""
import argparse
import gc
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import mapdata2 as MD
from mapdata2 import GW, GH, NEAR_CM, FAR_CM
from rig2 import PINS, NAMES

HERE = Path(__file__).resolve().parent
W, H = 1600, 900
PAD = 26
PANEL = (700, 525)
BG = (16, 18, 22)
LINE = (52, 57, 66)
DIM = (128, 134, 146)
TRUE = (235, 240, 248)
WARN = (60, 210, 250)
GOOD = (120, 230, 140)
BAD = (86, 86, 246)
FULL = 4095.0                # ADC 12 บิต — สเกลคงที่ ไม่ปรับตามสัญญาณ


def _text(img, s, xy, sc=0.45, col=DIM, th=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def depth_lut():
    """ตารางสี ฟ้า(ใกล้) -> ม่วง(ไกล) · ใช้ COLORMAP_COOL ซึ่งไล่แบบนั้นอยู่แล้ว"""
    import cv2
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_COOL)[0]


def cloud(img, x0, y0, w, h, occ, dep, lut, thr=0.5, title="", sub=""):
    """วาดกลุ่มจุด · สีบอกระยะ · ขนาดบอกความมั่นใจ

    วาดด้วยการเติมสี่เหลี่ยมลงอาเรย์แล้วค่อยขยาย แทนการวาดวงกลมทีละจุด
    เพราะตอนตาราง 80x60 มี 4,800 ช่อง วาดทีละวงกลมช้าเกินไป
    ตอนนี้ 40x30 เหลือ 1,200 ช่องแล้วแต่วิธีนี้ก็ยังเร็วกว่าอยู่ดี
    """
    import cv2
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + h), LINE, 1)
    m = occ >= thr
    if not m.any():
        _text(img, title, (x0, y0 - 26), 0.62, TRUE, 2)
        _text(img, sub, (x0, y0 - 8), 0.42, DIM)
        return
    t = np.clip((dep - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1)
    small = np.zeros((GH, GW, 3), np.uint8)
    small[m] = lut[(t[m] * 255).astype(np.uint8)]
    # ความมั่นใจ -> ความสว่าง จุดที่โมเดลลังเลจะจางลง ไม่ใช่หายไปเฉย ๆ
    conf = np.clip((occ - thr) / max(1.0 - thr, 1e-6), 0, 1)
    small = (small * (0.45 + 0.55 * conf)[:, :, None]).astype(np.uint8)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # เว้นร่องระหว่างช่องให้ดูเป็น 'จุด' ไม่ใช่แผ่นทึบ
    cw, ch = w // GW, h // GH
    if cw >= 3 and ch >= 3:
        big[ch - 1::ch, :] = BG
        big[:, cw - 1::cw] = BG
    img[y0:y0 + h, x0:x0 + w] = big
    _text(img, title, (x0, y0 - 26), 0.62, TRUE, 2)
    _text(img, sub, (x0, y0 - 8), 0.42, DIM)
    _text(img, f"{int(m.sum())} dots", (x0 + w - 78, y0 + h + 16), 0.42, DIM)


def legend(img, x0, y0, w, lut):
    import cv2
    bar = lut[np.linspace(0, 255, w).astype(int)][None]
    img[y0:y0 + 16, x0:x0 + w] = np.repeat(bar, 16, 0)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + 16), LINE, 1)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        _text(img, f"{NEAR_CM + (FAR_CM - NEAR_CM) * f:.0f}",
              (int(x0 + f * w) - 10, y0 + 32), 0.4, DIM)
    _text(img, "near", (x0 - 2, y0 - 6), 0.42, DIM)
    _text(img, "far   (cm)", (x0 + w - 62, y0 - 6), 0.42, DIM)


def chan_health(counts, ref=None):
    """สภาพรายช่องจากคลื่นดิบ

    ref = (dc, scale) ที่โมเดลใช้ทำ normalize ตอนเทรน อ่านจากเช็คพอยต์

    **ระดับพักเทียบกับ DC ที่โมเดลเทรนมา ไม่ใช่ค่ากลางของสี่ช่อง**
    เวอร์ชันแรกเทียบกับค่ากลางของสี่ช่อง แล้วพังบนบอร์ดจริงทันที:
    COM6 อ่านได้ 2095 / 782 / 1904 / 482 → ค่ากลาง 1343 ตกอยู่ระหว่างสองกลุ่ม
    ทุกช่องห่างจากมันเกิน 200 เลยโดนฟ้องว่าเสียทั้งสี่ รวมถึง BL ที่ดีเกือบเป๊ะ
    แค่เสียสองช่องก็หลอกได้แล้ว ไม่ต้องถึงสาม

    DC ของโมเดลไม่ใช่ค่าคงที่ของวงจรที่ฝังไว้ มันอ่านจากเช็คพอยต์ เทรนใหม่ก็ตามเอง
    และเป็นคำถามที่ถูกสำหรับจอนี้พอดี: **"คลื่นที่เข้าไปหน้าตาเหมือนที่โมเดลเคยเห็นไหม"**
    เกณฑ์คือห่างเกินครึ่งหนึ่งของ scale — prep() ตัดสัญญาณที่ dc±scale ทิ้ง
    ระดับพักที่เลื่อนไปครึ่งช่วงแล้วแปลว่าครึ่งหนึ่งของคลื่นถูกตัดหายก่อนถึงโมเดล
    ไม่มีโมเดล (ref=None) ค่อยถอยไปใช้ค่ากลางของสี่ช่อง

    **ความแรงก็พังแบบเดียวกัน** ถ้าเทียบกันเอง: บน COM6 ทุกช่องเกนสูงพร้อมกัน
    BL แกว่ง 654 ขณะที่ข้อมูลเทรนแกว่งแค่ ~105 แต่เทียบกับเพื่อนที่แรงพอกันก็ดูปกติ
    ทางที่ถูกคือวัด **"โมเดลเห็นคลื่นกี่ %"** = สัดส่วนตัวอย่างที่อยู่ใน dc±scale
    ไม่โดน prep() ตัดทิ้ง ข้อมูลเทรนได้ 99.8-100% ทุกช่อง ต่ำกว่า 90% คือผิดปกติ

    ธงที่ใช้ตัดสิน เรียงจากร้ายไปเบา:
      CLIP   ชนขอบบนหรือล่างของ ADC -> ขยายแรงไป ข้อมูลตรงยอดหายไปแล้ว
      DEAD   แทบไม่ขยับ -> สายหลุด ไมค์เสีย หรือออปแอมป์ไม่ทำงาน
      DC OFF ระดับพักห่างจากที่โมเดลเทรนมา -> บัส VREF ของช่องนั้นผิด
      hot    โมเดลเห็นคลื่นไม่ถึง 90% (ไม่มีโมเดล: แรงกว่าเพื่อน 2.5 เท่า) -> เกนสูงเกิน
      weak   แกว่งไม่ถึงหนึ่งในสี่ของ scale (ไม่มีโมเดล: เบากว่าเพื่อนมาก) -> เกนต่ำ
    """
    v = counts.astype(np.float32)
    dc = np.median(v, 1)
    # ใช้เปอร์เซ็นไทล์แทนค่าสุดขั้ว จุดหลงตัวเดียวจะได้ไม่ตัดสินทั้งช่อง
    sw = np.percentile(v, 99.5, 1) - np.percentile(v, 0.5, 1)
    clip = ((v <= 8) | (v >= FULL - 8)).mean(1)
    msw = max(float(np.median(sw)), 1.0)
    if ref is not None:
        mid, scale = float(ref[0]), float(ref[1])
        tol = 0.5 * scale
        seen = (np.abs(v - mid) < scale).mean(1)
        is_hot = seen < 0.90
        is_weak = sw < 0.25 * scale
    else:
        mid, tol = float(np.median(dc)), 200.0
        seen = np.full(len(dc), np.nan)
        is_hot = sw > 2.5 * msw
        is_weak = sw < 0.35 * msw
    out = []
    for i in range(len(dc)):
        if clip[i] > 1e-3:
            st, col = "CLIP", BAD
        elif sw[i] < 15:
            st, col = "DEAD", BAD
        elif abs(dc[i] - mid) > tol:
            # มาก่อน hot เสมอ: ช่องที่ DC หนีไปไกลโมเดลก็เห็น 0% อยู่แล้ว
            # แต่ต้นเหตุคือ VREF ไม่ใช่เกน ป้ายต้องชี้ไปที่ต้นเหตุ
            st, col = "DC OFF", BAD
        elif is_hot[i]:
            st, col = "hot", BAD if ref is not None else WARN
        elif is_weak[i]:
            st, col = "weak", WARN
        else:
            st, col = "ok", GOOD
        out.append({"dc": float(dc[i]), "swing": float(sw[i]),
                    "clip": float(clip[i]), "seen": float(seen[i]),
                    "state": st, "col": col,
                    "ref": mid, "has_ref": ref is not None})
    return out


def chan_panel(img, x0, y0, w, h, counts, hl):
    """คลื่นดิบสี่ช่องซ้อนกัน · **ทั้งสี่ช่องใช้สเกลเดียวกันเสมอ**

    จุดสำคัญคือสเกลร่วม ไม่ใช่สเกลคงที่ ถ้าปล่อยให้แต่ละช่องปรับสเกลของตัวเอง
    ช่องที่เงียบสนิทกับช่องที่ดังปกติจะวาดออกมาสูงเท่ากัน ซึ่งกลบอาการที่
    ต้องการดูพอดี พอใช้สเกลร่วม ช่องที่เบากว่าเพื่อนจะแบนกว่าเพื่อนให้เห็น

    หน้าต่างกว้าง 2.5 เท่าของการแกว่งกลาง (อย่างน้อย +-300) ไม่ใช่ 0-4095 เต็ม
    เพราะสัญญาณจริงแกว่งแค่ ~150 จาก 4095 ถ้ากางเต็มสเกลจะเหลือเส้นบาง ๆ
    มองไม่ออกว่าเป็นคลื่นหรือเส้นตรง · ช่องที่หลุดกรอบ (DC เพี้ยนหรือ CLIP)
    จะอัดแน่นอยู่ที่ขอบ ซึ่งเป็นสัญญาณที่อ่านง่ายอยู่แล้วว่ามันผิด
    """
    import cv2
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + h), LINE, 1)
    rh = h // 4
    # กึ่งกลางกรอบ = ระดับเดียวกับที่ chan_health ใช้ตัดสิน (DC ของโมเดลถ้ามี)
    # ช่องที่ลอยหนีเส้นประคือช่องที่โดนฟ้อง DC OFF พอดี ภาพกับป้ายจะได้ตรงกัน
    mid = hl[0]["ref"]
    half = max(300.0, 2.5 * max(float(np.median([d["swing"] for d in hl])), 1.0))
    lo_v, span_v = mid - half, 2 * half
    for i in range(4):
        ry, d = y0 + i * rh, hl[i]
        if i:
            cv2.line(img, (x0, ry), (x0 + w, ry), LINE, 1)
        s = counts[i].astype(np.float32)
        k = max(len(s) // w, 1)
        b = s[:len(s) // k * k].reshape(-1, k)
        lo, hi = b.min(1), b.max(1)
        pad, n = 22, len(lo)
        span = rh - 2 * pad

        def ypos(v):
            return np.clip(ry + rh - pad - (v - lo_v) / span_v * span,
                           ry + 2, ry + rh - 2).astype(int)

        # แถบจาก min ถึง max ของแต่ละคอลัมน์ = เห็นเปลือกคลื่นโดยไม่ต้องวาดทุกจุด
        ylo, yhi = ypos(lo), ypos(hi)
        yy = np.arange(ry, ry + rh)[:, None]
        img[ry:ry + rh, x0:x0 + n][(yy >= yhi[None, :]) & (yy <= ylo[None, :])] = d["col"]
        # เส้นอ้างอิง = ระดับพักของค่ากลาง ช่องที่ลอยหนีจะเห็นว่าไม่อยู่บนเส้น
        img[int(ypos(np.array([mid]))[0]), x0:x0 + w:6] = LINE
        # แถบมืดรองข้อความ — ช่องที่ CLIP จะถมสีเต็มแถว ป้ายจะอ่านไม่ออกถ้าไม่รอง
        band = img[ry + 2:ry + 26, x0:x0 + w]
        band[:] = (band * 0.25).astype(np.uint8)
        _text(img, f"{NAMES[i]}  GPIO{PINS[i]}", (x0 + 8, ry + 18), 0.46, TRUE, 1)
        _text(img, f"dc {d['dc']:.0f}   swing {d['swing']:.0f}"
                   + (f"   model sees {d['seen']*100:.0f}%" if d["has_ref"] else "")
                   + (f"   clipped {d['clip']*100:.1f}%" if d["clip"] > 1e-3 else ""),
              (x0 + 132, ry + 18), 0.42, DIM)
        _text(img, d["state"], (x0 + w - 76, ry + 18), 0.52, d["col"], 2)
    what = (f"level the model was trained on ({mid:.0f})" if hl[0]["has_ref"]
            else "median resting level of the four")
    _text(img, f"shared scale  {mid - half:.0f} - {mid + half:.0f} counts "
               f"(of 0-{FULL:.0f})   dotted line = {what}",
          (x0 + 8, y0 + h - 7), 0.4, DIM)


def stat(img, x, y, k, v, col=TRUE):
    _text(img, k, (x, y), 0.42, DIM)
    _text(img, v, (x, y + 26), 0.72, col, 2)


def render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps, banner, foot, thr,
           echo=None, cam_on=True, hl=None, counts=None):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    _text(img, "DEPTH MAP  -  from sound alone", (PAD, 44), 0.86, TRUE, 2)
    _text(img, f"ML model, {sc.get('params', 0):,} params  |  no hand-written DSP  |  "
               f"grid {GW}x{GH}  |  position and range work - pose does not",
          (PAD, 70), 0.46, DIM)
    if banner:
        txt, col = banner
        cv2.rectangle(img, (W - 430, 26), (W - PAD, 62), col, -1)
        _text(img, txt, (W - 416, 51), 0.56, (12, 14, 18), 2)

    pw, ph = PANEL
    top = 128
    if cam_on:
        cloud(img, PAD, top, pw, ph, occ_t.astype(float), dep_t, lut, 0.5,
              "CAMERA   ground truth",
              "reduced to the same grid the model predicts, so the two are comparable")
    elif hl is not None:
        # ไม่มีกล้อง = ช่องซ้ายว่าง เอามาโชว์คลื่นดิบแทน ซึ่งมีประโยชน์กว่า
        # ช่องว่าง ๆ มาก: ถ้าภาพขวาดูมั่ว จะได้รู้ทันทีว่าเป็นที่โมเดลหรือที่วงจร
        bad = [NAMES[i] for i, d in enumerate(hl) if d["col"] is BAD]
        chan_panel(img, PAD, top, pw, ph, counts, hl)
        _text(img, "SENSORS   raw waveform, all four", (PAD, top - 26), 0.62, TRUE, 2)
        # ข้อความบนจอเป็นอังกฤษทั้งหมด — ฟอนต์ Hershey ของ OpenCV ไม่มีตัวไทย
        _text(img, ("all four channels look healthy" if not bad else
                    "FAULTY: " + ", ".join(bad)
                    + "   -   the map on the right cannot be trusted until fixed"),
              (PAD, top - 8), 0.42, BAD if bad else DIM)
    else:
        cloud(img, PAD, top, pw, ph, occ_t.astype(float), dep_t, lut, 0.5,
              "CAMERA   off",
              "running on sound alone - nothing to compare against")
    cloud(img, PAD + pw + 60, top, pw, ph, occ_p, dep_p, lut, thr,
          "SOUND   model prediction",
          f"a dot where the model is over {thr:.0%} sure - no camera data used")
    legend(img, PAD, top + ph + 46, pw, lut)

    yb = top + ph + 128
    cv2.line(img, (PAD, yb - 16), (W - PAD, yb - 16), LINE, 1)
    if cam_on:
        stat(img, PAD, yb, "frames scored", f"{run['n']}")
        stat(img, PAD + 200, yb, "overlap with truth", f"{run['iou']:.3f}",
             GOOD if run["iou"] > sc.get("mean_iou", 1) else WARN)
        stat(img, PAD + 430, yb, "mean depth error", f"{run['mae']:.0f} cm")
        # ความสูงเงา = ท่าทาง · ตัวเลขนี้คือคำถามหลักของรอบนี้
        stat(img, PAD + 650, yb, "shadow height  cam / sound",
             f"{run['h_true']:.0f} / {run['h_pred']:.0f}",
             GOOD if abs(run["h_true"] - run["h_pred"]) < 6 else WARN)
        stat(img, PAD + 960, yb, "guessing the average gets",
             f"{sc.get('mean_iou', float('nan')):.3f}", DIM)
    else:
        # ไม่มีกล้อง = ไม่มีอะไรให้เทียบ · โชว์เลข 0.000 ไว้จะเข้าใจผิดว่าโมเดลพลาดหมด
        stat(img, PAD, yb, "shadow height  from sound", f"{run['h_pred']:.0f}")
        stat(img, PAD + 280, yb, "scored on held-out data",
             f"IoU {sc.get('iou', float('nan')):.3f}", DIM)
        stat(img, PAD + 560, yb, "depth error then",
             f"{sc.get('mae_cm', float('nan')):.1f} cm", DIM)
        stat(img, PAD + 800, yb, "pose R2 then",
             f"{sc.get('h_r2', float('nan')):+.3f}", DIM)
    if echo is not None:
        stat(img, PAD + 1230, yb, "echo strength", f"{echo:.0f} mV",
             TRUE if echo >= 60 else WARN)
    stat(img, PAD + 1430, yb, "fps", f"{fps:.1f}")
    _text(img, foot, (PAD, H - 14), 0.42, (104, 110, 120))
    return img


def scores(occ_p, dep_p, occ_t, dep_t, thr):
    p = occ_p >= thr
    iou = float((p & occ_t).sum()) / max(int((p | occ_t).sum()), 1)
    mae = float(np.abs(dep_p[occ_t] - dep_t[occ_t]).mean()) if occ_t.any() else None
    return iou, mae


def height_of(mask):
    """ความสูงเงา · ใช้นิยามเดียวกับตอนเทรน เลขบนจอจะได้เทียบกันได้"""
    return MD.shadow_height(mask)


def banner_for(held):
    """ป้ายบอกว่าเฟรมที่กำลังดูอยู่ โมเดลเคยเห็นตอนเทรนหรือยัง

    รอบสามกันทั้ง section ไว้วัดผล ทั้ง section จึงเป็นอย่างใดอย่างหนึ่งไปเลย
    (รอบสองกันแค่ท้ายของทุกช่วง ป้ายจึงต้องเปลี่ยนกลางทาง)
    """
    if held is None:
        return None
    return (("HELD OUT  -  never seen in training", GOOD) if held
            else ("TRAINED ON THIS  -  score inflated", WARN))


def run_loop(src, pr, thr, fps_target, foot, live=False, state=None):
    import cv2
    lut = depth_lut()
    sc = dict(pr.score)
    sc["params"] = pr.params
    run = {"n": 0, "iou": 0.0, "mae": 0.0, "h_true": 0.0, "h_pred": 0.0}
    # นับแยกสองชุด: ทุกเฟรม กับ เฉพาะเฟรมที่ไม่เคยเห็นตอนเทรน
    # เลขที่เชื่อถือได้คือชุดหลัง ชุดแรกมีข้อมูลที่โมเดลจำได้ปนอยู่
    acc = {"all": [0, 0.0, 0.0, 0.0, 0.0, 0], "held": [0, 0.0, 0.0, 0.0, 0.0, 0]}
    hist = deque(maxlen=60)
    bn = None
    paused = False
    show_ch = False          # c = สลับช่องซ้ายระหว่างกล้องกับคลื่นดิบ
    t0 = time.time()
    for ping, depth, held in src:
        bn = banner_for(held)
        cnt = MD.read_counts(ping)
        hl = chan_health(cnt, ref=(pr.dc, pr.scale))
        occ_p, dep_p = pr.push(ping)
        if depth is None:
            occ_t = np.zeros((GH, GW), bool)
            dep_t = np.zeros((GH, GW), np.float32)
        else:
            # แคชเก็บภาพเฉลยที่ย่อแล้ว ส่วนโฟลเดอร์ดิบเก็บ PNG เต็มความละเอียด
            small = depth if depth.shape == (GH, GW) else MD.shrink(depth)
            occ_t = small > 0
            dep_t = small.astype(np.float32) / 10.0
            iou, mae = scores(occ_p, dep_p, occ_t, dep_t, thr)
            ht, hp = height_of(occ_t), height_of(occ_p >= thr)
            for key in ("all",) + (("held",) if held else ()):
                v = acc[key]
                v[0] += 1
                v[1] += iou
                v[3] += ht
                v[4] += hp
                if mae is not None:
                    v[2] += mae
                    v[5] += 1
            v = acc["held"] if acc["held"][0] else acc["all"]
            run = {"n": v[0], "iou": v[1] / v[0], "mae": v[2] / max(v[5], 1),
                   "h_true": v[3] / v[0], "h_pred": v[4] / v[0],
                   "honest": bool(acc["held"][0])}
        hist.append(time.time())
        fps = (len(hist) - 1) / max(hist[-1] - hist[0], 1e-6) if len(hist) > 1 else 0.0
        have_cam = (state["cam"] if state else True) and not show_ch
        frame = render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps, bn, foot, thr,
                       echo=max(pr.amps) if live else None,
                       cam_on=have_cam, hl=hl, counts=cnt)
        cv2.imshow("map2", frame)
        wait = 0 if paused else (1 if live else max(1, int(1000 / fps_target)))
        k = cv2.waitKey(wait) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("c"):
            show_ch = not show_ch
        if k == ord("s"):
            n = f"cloud2_{int(time.time())}.png"
            cv2.imwrite(n, frame)
            print(f"เซฟ {n}")
    cv2.destroyAllWindows()
    run["seen_n"] = acc["all"][0] - acc["held"][0]
    return run, time.time() - t0


def replay_src(name, section, test_section, held_only=False):
    """เล่นย้อนข้อมูลที่อัดไว้ · บอกด้วยว่า section นี้โมเดลเคยเห็นตอนเทรนไหม

    รอบสามกัน **ทั้ง section** ไว้วัดผล ต่างจากรอบสองที่กันแค่ท้ายของทุกช่วง
    ทั้ง section จึงเป็นอย่างใดอย่างหนึ่งไปเลย ไม่ใช่ครึ่ง ๆ กลาง ๆ
    """
    import cv2
    d = f"{name}_s{section}"
    fs = MD.frame_files(d)
    if not fs:
        sys.exit(f"ไม่พบข้อมูลที่ {MD.DATA / d}")
    held = section == test_section
    tag = ("ไม่เคยเห็นตอนเทรน — คะแนนที่เห็นคือของจริง" if held
           else "**โมเดลเคยเห็น section นี้ตอนเทรน คะแนนจะดูดีเกินจริง**")
    print(f"เล่นย้อน {d} · {len(fs)} เฟรม · {tag}")
    if held_only and not held:
        sys.exit(f"--held-only ต้องใช้กับ section {test_section} เท่านั้น")
    for u, p in fs:
        z = np.load(u)
        yield ({"counts": z["counts"], "pins": z["pins"]},
               cv2.imread(str(p), cv2.IMREAD_UNCHANGED), held)


def cache_src(section, test_section, held_only=False):
    """เล่นย้อนจากแคชที่ train_map3 สร้างไว้ — ใช้เมื่อโฟลเดอร์ดิบไม่อยู่แล้ว

    โฟลเดอร์ mix_s1..s5 ถูกลบไปแล้ว เหลือแต่แคช ซึ่งเก็บทั้งคลื่นดิบและภาพเฉลย
    ที่ย่อเป็นตารางเป้าหมายไว้ครบ จึงเล่นย้อนได้เหมือนเดิมทุกอย่าง ต่างกันแค่
    ภาพเฉลยย่อมาแล้ว ไม่ต้องย่อซ้ำ (run_loop ดูจากรูปทรงว่าย่อมาแล้วหรือยัง)

    แคชเก็บคลื่นตามลำดับ PINS มาตรฐานอยู่แล้ว เพราะ read_counts ทำตอนสร้างแคช
    """
    if not MD.CACHE.exists():
        sys.exit(f"ไม่มีทั้งโฟลเดอร์ดิบและแคชที่ {MD.CACHE}")
    cn = np.load(MD.CACHE / "counts.npy", mmap_mode="r")
    dm = np.load(MD.CACHE / "dmm.npy", mmap_mode="r")
    sec = np.load(MD.CACHE / "sec.npy")
    idx = np.nonzero(sec == section)[0]
    if not len(idx):
        sys.exit(f"แคชไม่มี section {section} (มี {sorted(set(sec.tolist()))})")
    held = section == test_section
    tag = ("ไม่เคยเห็นตอนเทรน — คะแนนที่เห็นคือของจริง" if held
           else "**โมเดลเคยเห็น section นี้ตอนเทรน คะแนนจะดูดีเกินจริง**")
    print(f"เล่นย้อนจากแคช section {section} · {len(idx)} เฟรม · {tag}")
    if held_only and not held:
        sys.exit(f"--held-only ต้องใช้กับ section {test_section} เท่านั้น")
    for i in idx:
        yield ({"counts": np.asarray(cn[i]), "pins": list(PINS)},
               np.asarray(dm[i]), held)


def live_src(a, state):
    """โหมดสด — ลำดับสำคัญ: ปิด GC ก่อนแตะกล้อง เปิดคืนหลังปิดกล้องแล้ว

    a.no_cam = ใช้เสียงล้วน ไม่เปิดกล้องเลย ซึ่งคือเป้าหมายจริงของงานนี้
    กล้องมีไว้เทียบว่าโมเดลถูกแค่ไหนเท่านั้น ไม่ได้เป็นส่วนหนึ่งของการทำงาน
    ปิดกล้องแล้วเบาลง ไม่ต้องพึ่ง OpenNI และไม่มีปัญหาเธรดกล้องชนกับ torch
    """
    from sync4 import Sync4
    from record import _warmup
    pr = a.pr
    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()
    print("ซ้อมเส้นทางคำนวณก่อนเปิดกล้อง ...", flush=True)
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    pr.warmup(us.samples)
    cam = th = None
    if not a.no_cam:
        from astra import Astra
        from record import DepthThread
        print(f"เปิดกล้อง depth {w}x{h} ...", flush=True)
        gc.disable()
        try:
            cam = Astra(want_rgb=False, depth_size=(w, h))
            th = DepthThread(cam, 1)
            th.start()
            time.sleep(0.6)
        except Exception as e:
            # กล้องเปิดไม่ได้ไม่ใช่เหตุให้เลิก — โมเดลไม่ได้ใช้กล้องอยู่แล้ว
            # กล้องมีไว้วาดเฉลยเทียบเท่านั้น ถอยไปโหมดเสียงล้วนต่อได้เลย
            # และต้องเปิด GC คืนตรงนี้ เพราะ finally ข้างล่างยังไม่เริ่มทำงาน
            gc.enable()
            cam = th = None
            state["cam"] = False
            print(f"  !! เปิดกล้องไม่ได้ ({e})", flush=True)
            print("  ไปต่อแบบเสียงล้วน — ช่องซ้ายจะว่างเพราะไม่มีเฉลยให้เทียบ",
                  flush=True)
    else:
        print("เสียงล้วน ไม่เปิดกล้อง", flush=True)
    print("เปิดหน้าต่างแล้ว — ยืนหน้าเซ็นเซอร์ได้เลย", flush=True)
    try:
        while True:
            ping = us.ping()
            got = th.get() if th else None
            yield ping, (got[1] if got else None), None
    finally:
        if th:
            th.stop_flag = True
            try:
                cam.close()
            except Exception:
                pass
            gc.enable()
        try:
            us.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", help="ต่อเซ็นเซอร์จริง เช่น COM5")
    ap.add_argument("--name", default="mix")
    ap.add_argument("--section", type=int, default=4)
    ap.add_argument("--pins", default="35,34,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=55.0)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--thr", type=float, default=None)
    # รอบสามป้อน 5 การยิงเข้าโมเดลพร้อมกันอยู่แล้ว การเกลี่ยผลลัพธ์ทีหลัง
    # จึงไม่จำเป็นอีก และจะทำให้ภาพหน่วงโดยไม่ได้อะไรเพิ่ม
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--held-only", action="store_true",
                    help="ข้ามเฟรมที่โมเดลเคยเห็นตอนเทรน ดูเฉพาะของจริง")
    ap.add_argument("--no-cam", action="store_true",
                    help="โหมดสดแบบเสียงล้วน ไม่เปิดกล้อง")
    a = ap.parse_args()

    if MD.ROUND != "r3":
        sys.exit(f"หน้าจอนี้ใช้กับโมเดลรอบ 3 เท่านั้น "
                 f"(ตอนนี้ MAP2_ROUND={MD.ROUND}) · "
                 f"โมเดลรอบเก่าโครงสร้างต่างกันจนโหลดข้ามกันไม่ได้")
    from model3 import MapPredictor
    pr = MapPredictor(smooth=a.smooth)
    s = pr.score
    if a.thr is None:
        a.thr = float(s.get("thr", 0.5))
        print(f"  ใช้จุดตัด {a.thr:.2f} (ค่าที่ดีที่สุดตอนเทรน)")
    print(f"โมเดล {pr.params:,} พารามิเตอร์ · ตาราง {pr.gw}x{pr.gh} · "
          f"{pr.stack} การยิงต่อภาพ · กันทั้ง section {pr.test_section} ไว้วัดผล")
    print(f"  IoU {s.get('iou', float('nan')):.3f} "
          f"(ทายค่าเฉลี่ยได้ {s.get('mean_iou', float('nan')):.3f}) · "
          f"ระยะพลาด {s.get('mae_cm', float('nan')):.1f} ซม. "
          f"(ทายค่าเฉลี่ยได้ {s.get('mean_mae', float('nan')):.1f}) · "
          f"สูงเงา R2 {s.get('h_r2', float('nan')):+.3f}")
    if s.get("h_r2", 0.0) <= 0.02:
        print("  ** ท่าทางยังทายไม่ได้ดีกว่าการเดาค่าเฉลี่ย **")
        print("  รูปร่างบนจอบอกตำแหน่งกับระยะได้ แต่ยังบอกท่าทางไม่ได้")

    import cv2  # noqa: F401
    # ต้องเป็นอังกฤษ — cv2.putText ใช้ฟอนต์ Hershey ซึ่งวาดตัวไทยไม่ได้ ออกมาเป็น ?
    foot = "q quit  ·  space pause  ·  c camera / raw sensors  ·  s save png"
    if a.port:
        a.pr = pr
        # ใช้ dict ร่วมกัน เพราะ generator เริ่มทำงานทีหลัง run_loop
        # ถ้าส่งค่าไปตรง ๆ แล้วกล้องเปิดไม่ได้ทีหลัง จอจะยังเขียนว่ามีกล้องอยู่
        state = {"cam": not a.no_cam}
        run, el = run_loop(live_src(a, state), pr, a.thr, a.fps, foot,
                           live=True, state=state)
    else:
        if MD.frame_files(f"{a.name}_s{a.section}"):
            src = replay_src(a.name, a.section, pr.test_section, a.held_only)
        else:
            print(f"ไม่มีโฟลเดอร์ดิบ {a.name}_s{a.section} — ใช้แคชตอนเทรนแทน")
            src = cache_src(a.section, pr.test_section, a.held_only)
        run, el = run_loop(src, pr, a.thr, a.fps, foot)
    if run["n"]:
        what = ("เฉพาะเฟรมที่ไม่เคยเห็นตอนเทรน" if run.get("honest")
                else "ทุกเฟรม (มีเฟรมที่โมเดลเคยเห็นปนอยู่ เลขจึงดูดีเกินจริง)")
        print(f"\nสรุป {run['n']} เฟรม ใน {el:.0f} วินาที · {what}")
        print(f"  IoU {run['iou']:.3f} · ระยะพลาด {run['mae']:.1f} ซม. · "
              f"สูงเงา กล้อง {run['h_true']:.1f} เทียบ เสียง {run['h_pred']:.1f} ช่อง")
        if run.get("seen_n"):
            print(f"  (ข้ามอีก {run['seen_n']} เฟรมที่โมเดลเคยเห็นตอนเทรน "
                  f"ไม่เอามาคิดคะแนน)")


if __name__ == "__main__":
    main()
