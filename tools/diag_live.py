"""ทดลองทางกายภาพแบบสด — ดูตัวเลขเปลี่ยนต่อหน้า และบันทึกทุกปิงพร้อมป้ายเฟส

    python tools/diag_live.py --port COM6

ทำไมต้องมีเครื่องมือนี้
  ทดลองแบบ "ครอบโฟมแล้วดูว่าหายไหม" ตัดสินด้วยตาจากกราฟคลื่นไม่ได้ และถ้าเก็บข้อมูลแยก
  คนละช่วงเวลาจะไม่รู้ว่าตอนเก็บโฟมอยู่ที่เดิมหรือเปล่า เครื่องมือนี้ให้คุณเห็นเลขเปลี่ยน
  สด และบันทึกทุกปิงกับป้ายเฟสไว้วิเคราะห์ทีหลัง

**ยิงสลับกับไม่ยิงทุกปิง** (ปิงคู่ = ยิงปกติ · ปิงคี่ = คำสั่ง q เก็บคลื่นโดยไม่ยิง)
  เพราะการถอดสาย/ครอบโฟมเปลี่ยนทั้ง "สัญญาณจากเสียงยิง" และ "นอยส์ของวงจรรับ"
  ถ้าวัดแยกคนละช่วงเวลาจะแยกสองอย่างนี้ไม่ได้ ที่นี่ทั้งสองอย่างถูกวัดในเวลาเดียวกัน

ตัวเลขที่โชว์ (ค่า RMS ของคลื่นหลังหักระดับพัก เฉลี่ยทั้งสี่ช่อง)
  FAR    ช่วง 200-500 ซม. ของปิงที่ยิง  = โครงสร้างไกลที่กำลังสงสัย
  NEAR   ช่วง 13-80 ซม. ของปิงที่ยิง    = หางของพัลส์ตรง
  NOISE  RMS ทั้งเฟรมของปิงที่ไม่ยิง    = นอยส์ล้วน ๆ ของวงจรรับ ณ ตอนนั้น

กดเลขเพื่อบอกว่าตอนนี้กำลังทำอะไร (ป้ายถูกบันทึกกับทุกปิง) กด q เพื่อจบ
  1 เปิดปกติ (TX ต่ออยู่)   2 ครอบหัวรับ 4 ตัว   3 ครอบ TX   4 ถอดสาย TX ออก
  5 หมุนแผ่น 90 องศา   6 เงยขึ้นเพดาน   7 ยกลอยห่างโต๊ะ   0 อื่น ๆ

ตอนจบพิมพ์ตารางสรุป **ต่อช่วงที่ต่อเนื่อง** (เสียบ -> ถอด -> เสียบ เห็นเป็นสามแถว) ตัด
SETTLE_S วินาทีแรกของแต่ละช่วงทิ้ง เพราะมือยังอยู่ในฉาก
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "car"))
sys.path.insert(0, HERE)

C, T0 = 343.0, 1220e-6
LABELS = {1: "OPEN", 2: "RX COVERED", 3: "TX COVERED", 4: "TX WIRE OFF",
          5: "ROTATED 90", 6: "TILT UP", 7: "LIFTED", 0: "OTHER"}
SETTLE_S = 4.0
FAR_CM, NEAR_CM = (200.0, 500.0), (13.0, 80.0)
CH = ("TL", "TR", "BL", "BR")


def features(counts, dist):
    """(FAR, NEAR, CLIP%) ของหนึ่งเฟรมที่ยิง หรือหลายเฟรมที่ซ้อนกันเป็น (m, 4, n)"""
    x = counts.astype(np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    mf = (dist >= FAR_CM[0]) & (dist < FAR_CM[1])
    mn = (dist >= NEAR_CM[0]) & (dist < NEAR_CM[1])
    clip = 100.0 * float(((counts <= 2) | (counts >= 4093)).mean())
    return (float(np.sqrt((x[..., mf] ** 2).mean())),
            float(np.sqrt((x[..., mn] ** 2).mean())), clip)


def noise_rms(counts):
    x = counts.astype(np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    return float(np.sqrt((x ** 2).mean()))


def tone(quiet, rate):
    """เส้นสเปกตรัมเด่นสุดของนอยส์ (kHz หลังพับ) กับสัดส่วนกำลังใน ±0.5 kHz ของยอดนั้น

    นอยส์ขาวปกติได้ราว 3% (ถ้าแผ่เรียบ) · ค่าสูงมากคือมีสัญญาณแคบ ๆ กวนอยู่
    ความถี่ที่ได้เป็นความถี่ **หลังพับ** ที่อัตราสุ่ม ความถี่จริงอาจเป็น f, rate-f, rate+f ..."""
    x = quiet.astype(np.float32)
    x = x - x.mean(-1, keepdims=True)
    w = np.hanning(x.shape[-1])
    S = (np.abs(np.fft.rfft(x * w, axis=-1)) ** 2).mean(axis=tuple(range(x.ndim - 1)))
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / rate)
    i = int(np.argmax(S[5:])) + 5
    frac = float(S[(f > f[i] - 500) & (f < f[i] + 500)].sum() / S.sum())
    return f[i] / 1e3, 100.0 * frac


def repeat_corr(fire, dist):
    """เฟรมที่ยิงซ้ำกันแค่ไหนที่ช่วง >150 ซม. (สหสัมพันธ์ของซองกับค่าเฉลี่ยของเฟรมอื่น)

    เอคโค่จริงของเสียงยิงตำแหน่งนิ่ง ได้ใกล้ 1 · นอยส์สุ่มได้ใกล้ 0
    ไม่ขึ้นกับระดับนอยส์ จึงเทียบข้ามสถานะที่นอยส์ต่างกันได้"""
    x = fire.astype(np.float32)
    y = np.abs(x - np.median(x, axis=-1, keepdims=True)).mean(1)
    ker = np.ones(31, np.float32) / 31
    y = np.array([np.convolve(r, ker, "same") for r in y])[:, dist >= 150]
    if len(y) < 4:
        return float("nan")
    return float(np.mean([np.corrcoef(y[k], np.delete(y, k, 0).mean(0))[0, 1]
                          for k in range(len(y))]))


class Grabber(threading.Thread):
    """อ่านเซ็นเซอร์ในเธรดแยก หน้าต่างจะได้ไม่ค้างตอนเฟรมหาย (ดู car/check.py)

    สลับยิง/ไม่ยิงทุกปิงที่ได้ข้อมูลครบ (ปิงที่เสียไม่นับ เพื่อให้สองชนิดมีจำนวนเท่ากัน)"""

    def __init__(self, us):
        super().__init__(daemon=True)
        self.us, self.lock, self.items, self.stop_flag = us, threading.Lock(), [], False

    def run(self):
        fire = True
        while not self.stop_flag:
            try:
                p = self.us.ping(fire=fire)
            except Exception:
                time.sleep(0.2)
                continue
            if p is not None:
                with self.lock:
                    self.items.append((time.time(), p["counts"], fire))
                fire = not fire

    def take(self):
        with self.lock:
            out, self.items = self.items, []
        return out


def draw(cv2, hist, label, cur, base, n):
    W, H = 980, 600
    img = np.full((H, W, 3), (22, 24, 29), np.uint8)

    def put(s, xy, sc=0.5, col=(200, 205, 215), th=1):
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    put("PHYSICAL TEST - live  (fire / listen alternate)", (20, 34), 0.8, (235, 240, 248), 2)
    put(f"phase: {LABELS.get(label, '?')}", (20, 72), 0.9, (110, 230, 130), 2)
    for i, (name, v, b) in enumerate((("FAR 200-500 cm", cur[0], base[0]),
                                      ("NEAR 13-80 cm", cur[1], base[1]),
                                      ("NOISE (no fire)", cur[2], base[2]))):
        x = 20 + i * 320
        put(name, (x, 112), 0.5, (150, 156, 166))
        put(f"{v:6.0f}", (x, 160), 1.3, (250, 250, 250), 2)
        if b:
            put(f"{v / b:5.2f} x CONNECTED", (x, 190), 0.6,
                (110, 230, 130) if 0.8 <= v / b <= 1.25 else (80, 200, 250), 1)
    put(f"pings {n}", (860, 160), 0.5, (110, 116, 126))

    gx, gy, gw, gh = 20, 215, 940, 310
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (34, 38, 46), -1)
    lo, hi = np.log10(8.0), np.log10(2500.0)

    def yy(v):
        return int(gy + gh - (np.log10(max(v, 1.0)) - lo) / (hi - lo) * gh)

    for v in (10, 30, 100, 300, 1000):
        cv2.line(img, (gx, yy(v)), (gx + gw, yy(v)), (50, 54, 62), 1)
        put(str(v), (gx + gw - 38, yy(v) - 3), 0.38, (100, 106, 116))
    pts = hist[-gw // 4:]
    for k, p in enumerate(pts):
        x = gx + k * 4
        cv2.rectangle(img, (x, gy + gh - 6), (x + 3, gy + gh), (90, 90, 90 + 20 * (p[3] % 7)), -1)
    for idx, col in ((0, (250, 200, 90)), (1, (90, 190, 250)), (2, (110, 230, 130))):
        line = np.array([[gx + k * 4, yy(p[idx])] for k, p in enumerate(pts)], np.int32)
        if len(line) > 1:
            cv2.polylines(img, [line.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
    put("FAR (orange)   NEAR (blue)   NOISE (green)   log scale", (gx + 6, gy + 18), 0.42,
        (150, 156, 166))
    put("1 connected  2 cover RX  3 cover TX  4 TX wire off  5 rotate  6 tilt up  7 lifted  q finish",
        (20, H - 20), 0.45, (120, 126, 136))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--max-cm", type=float, default=500.0)
    ap.add_argument("--out", default=None, help="ไฟล์ .npz ที่บันทึก (ค่าเริ่มต้นใน car/data/)")
    a = ap.parse_args()

    import cv2
    from sync4 import Sync4
    us = Sync4(port=a.port, pins="35,34,32,33", max_cm=a.max_cm, period_ms=None, verbose=True)
    n = us.samples
    dist = (np.arange(n) / us.rate - T0) * C / 2 * 100
    grab = Grabber(us)
    grab.start()

    label = 1
    rec_c, rec_t, rec_l, rec_f, hist = [], [], [], [], []
    far = near = noise = 0.0
    base = None
    ref_fire, ref_noise = [], []
    print("เปิดหน้าต่างแล้ว · กดเลขเปลี่ยนเฟส · q จบ", flush=True)
    try:
        while True:
            for t, c, fired in grab.take():
                rec_c.append(c), rec_t.append(t), rec_l.append(label), rec_f.append(fired)
                if fired:
                    far, near, _ = features(c, dist)
                else:
                    noise = noise_rms(c)
                hist.append((far, near, noise, label))
                if label == 1:                     # ค่าอ้างอิง = ช่วงที่ต่อ TX ตามปกติ
                    (ref_fire if fired else ref_noise).append(
                        (far, near) if fired else noise)
                    if len(ref_fire) >= 6 and len(ref_noise) >= 6:
                        base = (float(np.mean([r[0] for r in ref_fire[2:]])),
                                float(np.mean([r[1] for r in ref_fire[2:]])),
                                float(np.mean(ref_noise[2:])))
            cv2.imshow("diag", draw(cv2, hist, label, (far, near, noise),
                                    base or (0, 0, 0), len(rec_c)))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("0") <= k <= ord("7"):
                label = k - ord("0")
                print(f"  -> เฟส {LABELS[label]}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        grab.stop_flag = True
        out = a.out or os.path.join(os.path.dirname(HERE), "car", "data",
                                    time.strftime("diag_live_%Y%m%d_%H%M%S.npz"))
        if rec_c:
            np.savez_compressed(out, counts=np.stack(rec_c), t=np.array(rec_t),
                                label=np.array(rec_l), fired=np.array(rec_f), rate=us.rate)
            print(f"\nบันทึก {len(rec_c)} ปิง -> {out}")
            summary(np.stack(rec_c), np.array(rec_t), np.array(rec_l), np.array(rec_f),
                    dist, us.rate)
        us.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(0)


def summary(counts, t, lab, fired, dist, rate):
    """ตารางต่อช่วงต่อเนื่อง · ตัด SETTLE_S วินาทีแรกของแต่ละช่วงทิ้ง"""
    runs, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            runs.append((s, i))
            s = i
    print(f"\n{'#':2} {'เฟส':12}{'ยิง':>5}{'ไม่ยิง':>7}{'FAR':>8}{'NEAR':>8}"
          f"{'ซ้ำได้':>8}  {'นอยส์ รายช่อง TL TR BL BR':26}{'เส้นเด่น kHz (%กำลัง)':>22}")
    for k, (a, b) in enumerate(runs):
        idx = np.arange(a, b)[(t[a:b] - t[a]) >= SETTLE_S]
        fi, qi = idx[fired[idx]], idx[~fired[idx]]
        if len(fi) < 3 or len(qi) < 3:
            print(f"{k:<2} {LABELS.get(lab[a], '?'):12} สั้นเกินไป ({b - a} ปิง)")
            continue
        far, near, _ = features(counts[fi], dist)
        q = counts[qi].astype(np.float32)
        q = q - np.median(q, axis=-1, keepdims=True)
        chn = np.sqrt((q ** 2).mean(axis=(0, 2)))
        fk, fp = tone(counts[qi], rate)
        print(f"{k:<2} {LABELS.get(lab[a], '?'):12}{len(fi):5d}{len(qi):7d}{far:8.1f}{near:8.1f}"
              f"{repeat_corr(counts[fi], dist):8.2f}  "
              f"{' '.join('%5.0f' % v for v in chn):26}{'%.2f (%.0f%%)' % (fk, fp):>22}")


if __name__ == "__main__":
    main()
