"""หน้าจอดูโมเดลแผนที่ depth — กลุ่มจุดไล่เฉดตามระยะ ไม่มีป้ายซ้าย/กลาง/ขวา

    python car/live_map.py --port COM5             ดูสด
    python car/live_map.py --name walk --section 5 เล่นย้อนจากที่อัดไว้

ซ้าย = กล้อง (ความจริง) · ขวา = เสียงล้วน (โมเดล) วาดบน **ตารางเดียวกัน 40x30**
เพื่อให้เทียบกันได้ตรง ๆ ไม่ใช่เอาภาพกล้องเต็มความละเอียดมาข่มภาพที่โมเดลทายได้

สีฟ้า = ใกล้ · สีม่วง = ไกล (ช่วงที่เซ็นเซอร์เห็นคือ 40-200 ซม.)
"""
import argparse
import gc
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import mapdata as MD
from mapdata import GW, GH, NEAR_CM, FAR_CM

HERE = Path(__file__).resolve().parent
W, H = 1560, 880
PAD = 26
PANEL = (700, 520)
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
    """ตารางสี ฟ้า(ใกล้) -> ม่วง(ไกล)

    ใช้ COLORMAP_COOL ของ OpenCV ซึ่งไล่จากฟ้าไปม่วงพอดีอยู่แล้ว
    ไม่ปั้นเองเพราะการไล่สีที่ตาคนเห็นว่า 'เท่ากัน' ทำเองให้ดีได้ยาก
    """
    import cv2
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_COOL)[0]


def cloud(img, x0, y0, w, h, occ, dep, lut, thr=0.5, title="", sub=""):
    """วาดกลุ่มจุด ช่องไหนมั่นใจว่ามีวัตถุก็ลงจุด · สีบอกระยะ · **ขนาดบอกความมั่นใจ**

    ให้ขนาดสื่อความมั่นใจ เพราะถ้าวาดทุกจุดเท่ากันหมด ขอบของก้อนจะดูคมชัด
    ราวกับโมเดลรู้แน่ ทั้งที่ตรงนั้นมันลังเลอยู่ — ภาพจะโกหกมากกว่าที่โมเดลโกหก
    """
    import cv2
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + h), LINE, 1)
    sx, sy = w / GW, h / GH
    rmax = max(2.0, min(sx, sy) * 0.46)
    ys, xs = np.nonzero(occ >= thr)
    for gy, gx in zip(ys, xs):
        t = (dep[gy, gx] - NEAR_CM) / (FAR_CM - NEAR_CM)
        c = lut[int(np.clip(t, 0, 1) * 255)]
        conf = (occ[gy, gx] - thr) / max(1.0 - thr, 1e-6)
        r = int(round(rmax * (0.45 + 0.55 * float(np.clip(conf, 0, 1)))))
        cv2.circle(img, (int(x0 + (gx + 0.5) * sx), int(y0 + (gy + 0.5) * sy)), max(r, 1),
                   (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    _text(img, title, (x0, y0 - 26), 0.62, TRUE, 2)
    _text(img, sub, (x0, y0 - 8), 0.42, DIM)
    _text(img, f"{int((occ >= thr).sum())} dots", (x0 + w - 74, y0 + h + 16), 0.42, DIM)


def legend(img, x0, y0, w, lut):
    """แถบสีบอกว่าสีไหนคือระยะเท่าไร — ไม่มีอันนี้ ภาพสวยก็อ่านไม่ออก"""
    import cv2
    bar = lut[np.linspace(0, 255, w).astype(int)][None]
    img[y0:y0 + 16, x0:x0 + w] = np.repeat(bar, 16, 0)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + 16), LINE, 1)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        cm = NEAR_CM + (FAR_CM - NEAR_CM) * f
        _text(img, f"{cm:.0f}", (int(x0 + f * w) - 10, y0 + 32), 0.4, DIM)
    _text(img, "near", (x0 - 2, y0 - 6), 0.42, DIM)
    _text(img, "far   (cm)", (x0 + w - 62, y0 - 6), 0.42, DIM)


def stat(img, x, y, k, v, col=TRUE):
    _text(img, k, (x, y), 0.42, DIM)
    _text(img, v, (x, y + 26), 0.72, col, 2)


def render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps, banner, foot, thr, echo=None):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    _text(img, "DEPTH MAP  -  from sound alone", (PAD, 44), 0.86, TRUE, 2)
    _text(img, f"ML model, {sc['params']:,} params  |  no hand-written DSP  |  "
               f"grid {GW}x{GH}  |  cyan = near, purple = far", (PAD, 70), 0.46, DIM)
    if banner:
        txt, col = banner
        cv2.rectangle(img, (W - 430, 26), (W - PAD, 62), col, -1)
        _text(img, txt, (W - 416, 51), 0.56, (12, 14, 18), 2)

    pw, ph = PANEL
    top = 128
    cloud(img, PAD, top, pw, ph, occ_t.astype(float), dep_t, lut, 0.5,
          "CAMERA   ground truth",
          "reduced to the same grid the model predicts, so the two are comparable")
    x2 = PAD + pw + 60
    cloud(img, x2, top, pw, ph, occ_p, dep_p, lut, thr,
          "SOUND   model prediction",
          f"a dot where the model is over {thr:.0%} sure - no camera data used")

    legend(img, PAD, top + ph + 46, pw, lut)

    yb = top + ph + 130
    cv2.line(img, (PAD, yb - 16), (W - PAD, yb - 16), LINE, 1)
    stat(img, PAD, yb, "frames scored", f"{run['n']}")
    stat(img, PAD + 200, yb, "overlap with truth", f"{run['iou']:.3f}",
         GOOD if run["iou"] > sc["mean_iou"] else WARN)
    stat(img, PAD + 430, yb, "mean depth error", f"{run['mae']:.0f} cm")
    stat(img, PAD + 650, yb, "guessing the average gets", f"{sc['mean_iou']:.3f}", DIM)
    stat(img, PAD + 920, yb, "template told the answer", f"{sc['tmpl_iou']:.3f}", DIM)
    if echo is not None:
        # โหมดสดต้องเห็นค่านี้ ไม่งั้นแยกไม่ออกระหว่าง 'ไม่มีอะไรอยู่ตรงนั้น'
        # กับ 'เซ็นเซอร์ไม่ได้ยินอะไรเลย' — สองอย่างนี้หน้าจอออกมาเหมือนกัน
        stat(img, PAD + 1180, yb, "echo strength", f"{echo:.0f} mV",
             TRUE if echo >= 60 else WARN)
    stat(img, PAD + 1390, yb, "fps", f"{fps:.1f}")
    _text(img, foot, (PAD, H - 14), 0.42, (104, 110, 120))
    return img


def scores(occ_p, dep_p, occ_t, dep_t, thr):
    p = occ_p >= thr
    u = int((p | occ_t).sum())
    iou = float((p & occ_t).sum()) / max(u, 1)
    mae = float(np.abs(dep_p[occ_t] - dep_t[occ_t]).mean()) if occ_t.any() else None
    return iou, mae


def banner_for(held):
    if held is None:
        return None
    return (("TEST SECTION  -  never seen", GOOD) if held
            else ("TRAINED ON THIS  -  score inflated", WARN))


def run_loop(src, pr, held, thr, fps_target, foot, live=False):
    """ลูปกลางที่ใช้ทั้งโหมดสดและโหมดเล่นย้อน

    src เป็น generator ที่คืน (ping, depth_mm เต็มภาพ หรือ None)
    """
    import cv2
    lut = depth_lut()
    sc = dict(pr.score)
    sc["params"] = pr.params
    for k in ("mean_iou", "tmpl_iou"):
        sc.setdefault(k, float("nan"))
    run = {"n": 0, "iou": 0.0, "mae": 0.0}
    si, sm, sn = 0.0, 0.0, 0
    hist = deque(maxlen=60)
    bn = banner_for(held)
    paused = False
    t0 = time.time()
    for ping, depth in src:
        occ_p, dep_p = pr.push(ping)
        if depth is None:
            occ_t = np.zeros((GH, GW), bool)
            dep_t = np.zeros((GH, GW), np.float32)
        else:
            small = MD.shrink(depth)
            occ_t = small > 0
            dep_t = small.astype(np.float32) / 10.0
            iou, mae = scores(occ_p, dep_p, occ_t, dep_t, thr)
            si += iou
            sn += 1
            if mae is not None:
                sm += mae
            run = {"n": sn, "iou": si / sn, "mae": sm / max(sn, 1)}
        hist.append(time.time())
        fps = (len(hist) - 1) / max(hist[-1] - hist[0], 1e-6) if len(hist) > 1 else 0.0
        img = render(occ_p, dep_p, occ_t, dep_t, lut, sc, run, fps, bn, foot, thr,
                     echo=max(pr.amps) if live else None)
        cv2.imshow("map", img)
        # โหมดสดไม่ต้องหน่วง จังหวะถูกกำหนดด้วยคาบยิงของเซ็นเซอร์อยู่แล้ว (50 ms)
        # หน่วงซ้ำ = คูณสองครั้ง เคยทำให้เหลือ 0.9 fps เพราะส่ง fps_target=1 มาหาร
        wait = 0 if paused else (1 if live else max(1, int(1000 / fps_target)))
        k = cv2.waitKey(wait) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            n = f"cloud_{int(time.time())}.png"
            cv2.imwrite(n, img)
            print(f"เซฟ {n}")
    cv2.destroyAllWindows()
    return run, time.time() - t0


def replay_src(name, section):
    import cv2
    d = f"{name}_s{section}"
    fs = MD.frame_files(d)
    if not fs:
        sys.exit(f"ไม่พบข้อมูลที่ {MD.DATA / d}")
    print(f"เล่นย้อน {d} · {len(fs)} เฟรม")
    for u, p in fs:
        z = np.load(u)
        yield ({"counts": z["counts"], "pins": z["pins"]},
               cv2.imread(str(p), cv2.IMREAD_UNCHANGED))


def live_src(a):
    """โหมดสด — ลำดับการเปิดสำคัญมาก ดูเหตุผลใน mapmodel.MapPredictor.warmup"""
    from astra import Astra
    from sync4 import Sync4
    from record import DepthThread, _warmup
    pr = a.pr
    w, h = (int(v) for v in a.size.lower().split("x"))
    print(f"เปิดเซ็นเซอร์ {a.port} ...", flush=True)
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    us.ping()
    print("ซ้อมเส้นทางคำนวณก่อนเปิดกล้อง ...", flush=True)
    _warmup(Path(HERE) / "data", nsamp=us.samples, rate=us.rate)
    pr.warmup(us.samples)
    print(f"เปิดกล้อง depth {w}x{h} ...", flush=True)
    # **ปิด GC ก่อนแตะกล้อง ไม่ใช่หลัง** — การเปิดสตรีมกับการปิดสตรีม
    # ก็เป็นช่วงที่ OpenNI ทำงานในเธรด native เหมือนกัน ถ้า GC วิ่งตอนนั้น
    # ได้ access violation เหมือนกัน เจอมาแล้วทั้งตอน create_depth_stream
    # และตอน oniStreamStop
    gc.disable()
    cam = Astra(want_rgb=False, depth_size=(w, h))
    th = DepthThread(cam, 1)
    th.start()
    time.sleep(0.6)
    print("เปิดหน้าต่างแล้ว — ยืนหน้าเซ็นเซอร์ได้เลย", flush=True)
    try:
        while True:
            ping = us.ping()
            got = th.get()          # คืน (เวลา, ภาพ) หรือ None ถ้ากล้องยังไม่ส่งเฟรมแรก
            yield ping, (got[1] if got else None)
    finally:
        th.stop_flag = True
        for fn in (cam.close, us.close):
            try:
                fn()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", help="ต่อเซ็นเซอร์จริง เช่น COM5")
    ap.add_argument("--name", default="walk", help="ชื่อชุดตอนเล่นย้อน")
    ap.add_argument("--section", type=int, default=5)
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--max-cm", type=float, default=200.0)
    ap.add_argument("--period-ms", type=float, default=50.0)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--thr", type=float, default=None,
                    help="มั่นใจเกินเท่าไรถึงลงจุด (ไม่ใส่ = ใช้ค่าที่ดีที่สุดจากตอนเทรน)")
    ap.add_argument("--smooth", type=int, default=5, help="เกลี่ยย้อนหลังกี่เฟรม")
    a = ap.parse_args()

    from mapmodel import MapPredictor
    pr = MapPredictor(smooth=a.smooth)
    s = pr.score
    if a.thr is None:
        # จุดตัดที่วัดแล้วว่าให้ผลดีที่สุดตอนเทรน ดีกว่าเดา 0.5 ลอย ๆ
        a.thr = float(s.get("thr", 0.5))
        print(f"  ใช้จุดตัด {a.thr:.2f} (ค่าที่ดีที่สุดตอนเทรน)")
    print(f"โมเดล {pr.params:,} พารามิเตอร์ · กันช่วง s{pr.holdout} ไว้ตอนเทรน")
    print(f"  ผลกับช่วงที่กันไว้: IoU {s.get('iou', float('nan')):.3f} · "
          f"ระยะพลาด {s.get('mae_cm', float('nan')):.1f} ซม.")
    print(f"  เกณฑ์เทียบ: ทายค่าเฉลี่ย {s.get('mean_iou', float('nan')):.3f} · "
          f"แม่แบบรู้ตำแหน่ง {s.get('tmpl_iou', float('nan')):.3f}")

    import cv2  # noqa: F401  ให้ import ก่อนเปิดกล้อง
    foot = "q quit  |  space pause  |  s save"
    if a.port:
        a.pr = pr
        run, el = run_loop(live_src(a), pr, None, a.thr, a.fps, foot, live=True)
    else:
        held = (pr.holdout == a.section)
        run, el = run_loop(replay_src(a.name, a.section), pr, held, a.thr, a.fps, foot)
    if run["n"]:
        print(f"\nสรุป {run['n']} เฟรม ใน {el:.0f} วินาที: "
              f"IoU {run['iou']:.3f} · ระยะพลาดเฉลี่ย {run['mae']:.1f} ซม.")


if __name__ == "__main__":
    main()
