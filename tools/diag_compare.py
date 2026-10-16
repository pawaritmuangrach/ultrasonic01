"""วิเคราะห์และเทียบไฟล์ที่ diag_capture.py เก็บไว้ (เสียบ TX / ถอด TX)

    python tools/diag_compare.py                  เทียบ 3 ไฟล์ล่าสุดใน car/data/
    python tools/diag_compare.py a.npz b.npz ...  เทียบตามลำดับที่ใส่

แต่ละไฟล์วัด 3 ช่วงของคลื่น โดยทุกช่วงมีค่าของปิงที่ "ยิง" และปิงที่ "ไม่ยิง" ในเวลาเดียวกัน
  PULSE  -15..13 ซม.   พัลส์ตรงจากไดรเวอร์/TX (ช่วง ~0.4-2 ms แรก)
  NEAR   13..80 ซม.    หางของพัลส์
  FAR    200..500 ซม.  โครงสร้างไกลที่กำลังสงสัย

  ส่วนเกิน = sqrt(ยิง^2 - ไม่ยิง^2)  คือสัญญาณที่ **เกิดเพราะสั่งยิง** หลังหักนอยส์ของวงจรรับ
  ในสถานะเดียวกันออกแล้ว (นอยส์สุ่มเป็นอิสระต่อสัญญาณ จึงหักแบบกำลังสองได้)
  จุดนี้สำคัญ: ถอดสายแล้วนอยส์เปลี่ยน ถ้าเทียบเฉพาะ "ยิง" จะปนกัน

  ซ้ำได้  ความนิ่งของตำแหน่งโครงสร้างไกลระหว่างเฟรมที่ยิง (1 = นิ่งเป๊ะ · ~0 = สุ่ม)
          ไม่ขึ้นกับระดับนอยส์ จึงเป็นตัวชี้ขาดว่า "ยังมีเอคโค่ของเสียงยิงไหม"
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from diag_live import repeat_corr, tone, T0, C        # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "car", "data")
SKIP = 2                       # ทิ้งปิงแรกของแต่ละชนิด (เซ็นเซอร์เพิ่งเริ่มจับจังหวะ)
WINDOWS = {"PULSE": (-15.0, 13.0), "NEAR": (13.0, 80.0), "FAR": (200.0, 500.0)}


def win_rms(x, dist, lo, hi):
    x = x.astype(np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    m = (dist >= lo) & (dist < hi)
    return float(np.sqrt((x[..., m] ** 2).mean()))


def excess(fire, quiet):
    return float(np.sqrt(max(fire * fire - quiet * quiet, 0.0)))


def analyze(path):
    d = np.load(path, allow_pickle=True)
    counts, fired, rate = d["counts"], d["fired"].astype(bool), float(d["rate"])
    dist = (np.arange(counts.shape[-1]) / rate - T0) * C / 2 * 100
    f = counts[fired][SKIP:]
    q = counts[~fired][SKIP:]
    out = {"path": path, "tx": (str(d["tx"]) + ("+" + str(d["tag"]) if "tag" in d.files and str(d["tag"]) else ""))
          if "tx" in d.files else "?",
           "n_fire": len(f), "n_quiet": len(q)}
    for name, (lo, hi) in WINDOWS.items():
        a, b = win_rms(f, dist, lo, hi), win_rms(q, dist, lo, hi)
        out[name] = (a, b, excess(a, b))
    out["repeat"] = repeat_corr(f, dist)
    qq = q.astype(np.float32)
    qq = qq - np.median(qq, axis=-1, keepdims=True)
    out["noise_ch"] = np.sqrt((qq ** 2).mean(axis=(0, 2)))
    out["tone"] = tone(q, rate)
    out["clip"] = 100.0 * float(((f <= 2) | (f >= 4093)).mean())
    return out


def show(results):
    print(f"{'ไฟล์':34}{'TX':>12}{'ยิง/ไม่ยิง':>12}")
    for r in results:
        print(f"{os.path.basename(r['path'])[:34]:34}{r['tx']:>12}"
              f"{r['n_fire']:>6}/{r['n_quiet']:<5}")
    print("\nแต่ละช่อง: ยิง | ไม่ยิง | ส่วนเกิน (นับ ADC RMS เฉลี่ยทั้งสี่ช่อง)")
    print(f"{'#':2}{'TX':>12} " + "".join(f"{w:>26}" for w in WINDOWS) + f"{'ซ้ำได้':>9}{'ชน%':>6}")
    for k, r in enumerate(results):
        cells = "".join(f"{r[w][0]:8.0f} |{r[w][1]:6.0f} |{r[w][2]:6.0f}   " for w in WINDOWS)
        print(f"{k:<2}{r['tx']:>12} {cells}{r['repeat']:9.2f}{r['clip']:6.1f}")
    print("\nนอยส์ตอนไม่ยิง (RMS) รายช่อง และเส้นสเปกตรัมเด่น (ความถี่หลังพับ · % ของกำลังนอยส์ในยอดนั้น)")
    print(f"{'#':2}{'TX':>12}{'TL':>8}{'TR':>8}{'BL':>8}{'BR':>8}{'เส้นเด่น':>24}")
    for k, r in enumerate(results):
        n = r["noise_ch"]
        print(f"{k:<2}{r['tx']:>12}{n[0]:8.0f}{n[1]:8.0f}{n[2]:8.0f}{n[3]:8.0f}"
              f"{'%.2f kHz (%.0f%%)' % r['tone']:>24}")
    if len(results) > 1:
        print("\nส่วนเกิน เทียบกับไฟล์แรก (1.00 = เท่ากัน)")
        b = results[0]
        for k, r in enumerate(results[1:], 1):
            print(f"  #{k} ({r['tx']}) " + "  ".join(
                f"{w} {r[w][2] / b[w][2]:5.2f}" if b[w][2] > 1 else f"{w}   -"
                for w in WINDOWS))


def main():
    files = sys.argv[1:]
    if not files:
        # เรียงตามเวลาที่เก็บ (ท้ายชื่อไฟล์) ไม่ใช่ตามตัวอักษร ไม่งั้น "off" มาก่อน "on" เสมอ
        # แล้วอัตราส่วน "เทียบไฟล์แรก" จะเทียบกับสถานะที่ผิด
        files = sorted(glob.glob(os.path.join(DATA, "diag_tx-*.npz")),
                       key=lambda f: "_".join(os.path.basename(f).split("_")[-2:]))[-3:]
    if not files:
        sys.exit("ไม่มีไฟล์ diag_tx-*.npz ใน car/data/ · เก็บด้วย tools/diag_capture.py ก่อน")
    show([analyze(p) for p in files])


if __name__ == "__main__":
    main()
