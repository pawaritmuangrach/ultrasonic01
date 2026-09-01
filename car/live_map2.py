"""หน้าจอดูโมเดลรอบสอง — กลุ่มจุด 80x60 ไล่เฉดตามระยะ

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 1 : INTERFACE   (ไฟล์นี้ทั้งไฟล์)
═══════════════════════════════════════════════════════════════════════

ไฟล์นี้ **ไม่มีโค้ด ML เลย** มีแต่การวาดภาพและการอ่านปุ่ม
ตัวโมเดลอยู่ที่ model2.py · การเทรนอยู่ที่ train_map2.py
แยกกันเพื่อให้อ่านทีละเรื่อง และเปลี่ยนหน้าจอได้โดยไม่แตะโมเดล

    python car/live_map2.py --port COM5                ดูสด
    python car/live_map2.py --name pose --section 1     เล่นย้อนจากที่อัดไว้

ซ้าย = กล้อง (ความจริง) · ขวา = เสียงล้วน (โมเดล) วาดบนตารางเดียวกัน 80x60
สีฟ้า = ใกล้ · สีม่วง = ไกล (ช่วงที่เซ็นเซอร์เห็นคือ 40-200 ซม.)
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


def _text(img, s, xy, sc=0.45, col=DIM, th=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def depth_lut():
    """ตารางสี ฟ้า(ใกล้) -> ม่วง(ไกล) · ใช้ COLORMAP_COOL ซึ่งไล่แบบนั้นอยู่แล้ว"""
    import cv2
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_COOL)[0]


def cloud(img, x0, y0, w, h, occ, dep, lut, thr=0.5, title="", sub=""):
    """วาดกลุ่มจุด · สีบอกระยะ · ขนาดบอกความมั่นใจ

    ที่ตาราง 80x60 จุดเล็กลงกว่ารอบแรกมาก จึงวาดด้วยการเติมสี่เหลี่ยมลงอาเรย์
    แล้วค่อยขยาย แทนการวาดวงกลมทีละจุด — 4,800 ช่องถ้าวาดทีละวงกลมช้าเกินไป
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


def stat(img, x, y, k, v, col=TRUE):
    _text(img, k, (x, y), 0.42, DIM)
    _text(img, v, (x, y + 26), 0.72, col, 2)


def render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps, banner, foot, thr,
           echo=None, cam_on=True):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    _text(img, "DEPTH MAP  -  from sound alone", (PAD, 44), 0.86, TRUE, 2)
    _text(img, f"ML model, {sc.get('params', 0):,} params  |  no hand-written DSP  |  "
               f"grid {GW}x{GH}  |  square array, so height is real",
          (PAD, 70), 0.46, DIM)
    if banner:
        txt, col = banner
        cv2.rectangle(img, (W - 430, 26), (W - PAD, 62), col, -1)
        _text(img, txt, (W - 416, 51), 0.56, (12, 14, 18), 2)

    pw, ph = PANEL
    top = 128
    cloud(img, PAD, top, pw, ph, occ_t.astype(float), dep_t, lut, 0.5,
          "CAMERA   ground truth" if cam_on else "CAMERA   off",
          "reduced to the same grid the model predicts, so the two are comparable"
          if cam_on else "running on sound alone - nothing to compare against")
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

    ต้องดูรายเฟรม ไม่ใช่รายช่วง เพราะการแบ่งแบบใหม่กัน 25% ท้ายของทุกช่วง
    ไว้วัดผล เล่นย้อนช่วงหนึ่งจึงผ่านทั้งข้อมูลที่เคยเห็นและไม่เคยเห็น
    ถ้าติดป้ายเดียวทั้งช่วงจะโกหกไปครึ่งหนึ่งของเวลา
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
    t0 = time.time()
    for ping, depth, held in src:
        bn = banner_for(held)
        occ_p, dep_p = pr.push(ping)
        if depth is None:
            occ_t = np.zeros((GH, GW), bool)
            dep_t = np.zeros((GH, GW), np.float32)
        else:
            small = MD.shrink(depth)
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
        cv2.imshow("map2", render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps,
                                  bn, foot, thr,
                                  echo=max(pr.amps) if live else None,
                                  cam_on=state["cam"] if state else True))
        wait = 0 if paused else (1 if live else max(1, int(1000 / fps_target)))
        k = cv2.waitKey(wait) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            n = f"cloud2_{int(time.time())}.png"
            cv2.imwrite(n, render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps,
                                  bn, foot, thr))
            print(f"เซฟ {n}")
    cv2.destroyAllWindows()
    run["seen_n"] = acc["all"][0] - acc["held"][0]
    return run, time.time() - t0


def replay_src(name, section, tail, held_only=False):
    """เล่นย้อนข้อมูลที่อัดไว้ · บอกไปด้วยว่าเฟรมไหนโมเดลเคยเห็นตอนเทรน

    ตอนเทรนกัน tail ท้ายของทุกช่วงไว้วัดผล เฟรมท้าย ๆ จึงเป็นของจริงที่ไม่เคยเห็น
    ส่วนเฟรมต้น ๆ โมเดลเคยเห็นแล้ว คะแนนตรงนั้นจะดูดีเกินจริง
    """
    import cv2
    d = f"{name}_s{section}"
    fs = MD.frame_files(d)
    if not fs:
        sys.exit(f"ไม่พบข้อมูลที่ {MD.DATA / d}")
    cut = int(len(fs) * (1.0 - tail))
    print(f"เล่นย้อน {d} · {len(fs)} เฟรม · "
          f"{cut} เฟรมแรกโมเดลเคยเห็นตอนเทรน · "
          f"{len(fs) - cut} เฟรมท้ายไม่เคยเห็น (ใช้วัดผลจริง)")
    for i, (u, p) in enumerate(fs):
        if held_only and i < cut:
            continue
        z = np.load(u)
        yield ({"counts": z["counts"], "pins": z["pins"]},
               cv2.imread(str(p), cv2.IMREAD_UNCHANGED), i >= cut)


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
    ap.add_argument("--name", default="pose")
    ap.add_argument("--section", type=int, default=1)
    ap.add_argument("--pins", default="35,34,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=55.0)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--held-only", action="store_true",
                    help="ข้ามเฟรมที่โมเดลเคยเห็นตอนเทรน ดูเฉพาะของจริง")
    ap.add_argument("--no-cam", action="store_true",
                    help="โหมดสดแบบเสียงล้วน ไม่เปิดกล้อง")
    a = ap.parse_args()

    from model2 import MapPredictor
    pr = MapPredictor(smooth=a.smooth)
    s = pr.score
    if a.thr is None:
        a.thr = float(s.get("thr", 0.5))
        print(f"  ใช้จุดตัด {a.thr:.2f} (ค่าที่ดีที่สุดตอนเทรน)")
    how = (f"กัน {pr.tail:.0%} ท้ายของทุกช่วงไว้วัดผล" if pr.split == "time"
           else f"กันทั้งช่วง s{pr.holdout} ไว้วัดผล")
    print(f"โมเดล {pr.params:,} พารามิเตอร์ · {how}")
    print(f"  IoU {s.get('iou', float('nan')):.3f} · "
          f"ระยะพลาด {s.get('mae_cm', float('nan')):.1f} ซม. · "
          f"สูงเงา R2 {s.get('h_r2', float('nan')):+.3f} "
          f"(เพดานจากตำแหน่ง {s.get('pos_h_r2', float('nan')):+.3f})")

    import cv2  # noqa: F401
    foot = "q ออก · space หยุด/เล่นต่อ · s เซฟภาพ"
    if a.port:
        a.pr = pr
        # ใช้ dict ร่วมกัน เพราะ generator เริ่มทำงานทีหลัง run_loop
        # ถ้าส่งค่าไปตรง ๆ แล้วกล้องเปิดไม่ได้ทีหลัง จอจะยังเขียนว่ามีกล้องอยู่
        state = {"cam": not a.no_cam}
        run, el = run_loop(live_src(a, state), pr, a.thr, a.fps, foot,
                           live=True, state=state)
    else:
        run, el = run_loop(replay_src(a.name, a.section, pr.tail, a.held_only),
                           pr, a.thr, a.fps, foot)
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
