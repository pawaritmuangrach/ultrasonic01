"""ตรวจแกนระยะของเซ็นเซอร์ด้วยกล้อง: เอคโค่ของคนอยู่ตรงระยะที่กล้องบอกไหม

    python tools/calib_range.py car/data/calib_s1          โฟลเดอร์ที่ record.py อัดไว้
    python tools/calib_range.py --cache                    แคชรอบ 3 (ไว้ทดสอบวิธีวัดกับข้อมูลเก่า)

ปัญหาที่แก้: บนกราฟดิบ เอคโค่ของคนที่ 1-2 ม. จมอยู่ในหางของพัลส์ตรง (ยาวถึง ~1 ม.) และปนกับ
เอคโค่ห้องที่นิ่งอยู่ ดูด้วยตาไม่ได้ว่าอันไหนคือคน และแกนระยะ (T0 = 1220 us) คาลิเบรตไว้กับ
ฮาร์ดแวร์รุ่นเก่า ไม่เคยตรวจกับฮาร์ดแวร์ปัจจุบัน

วิธี
  1. ซองสัญญาณต่อเฟรม (เฉลี่ยสี่ช่อง)
  2. **ลบพื้นหลังนิ่ง** = ซองมัธยฐานข้ามทุกเฟรมของช่วง ซึ่งรวมหางพัลส์ตรง เอคโค่ห้อง และไฟฟ้า
     ที่ซ้ำทุกเฟรม เหลือเฉพาะสิ่งที่เคลื่อนที่ = คน (ใช้ได้เมื่อคนเดินเปลี่ยนระยะระหว่างอัด)
  3. ยอดของส่วนที่เหลือในช่วง 40-200 ซม. เทียบกับระยะใกล้สุดที่กล้องอ่านได้ (cam_near_cm)
  4. ถดถอย: ระยะจากเสียง = a x ระยะกล้อง + b · b คือออฟเซ็ตของแกน a ควรใกล้ 1
     ถ้า b ไม่ใช่ศูนย์ แก้ T0 ใหม่ตามที่พิมพ์บอก

ข้อจำกัด: กล้องกับอาเรย์ไม่ได้อยู่จุดเดียวกัน (ออฟเซ็ตจริงไม่รู้) และ "ระยะใกล้สุด" ของกล้องคือจุด
ที่ใกล้ที่สุดของตัวคน ซึ่งเสียงอาจสะท้อนจากส่วนอื่น (หน้าอก/ท้อง) ที่ลึกกว่าไปหลายเซนติเมตร
จึงคาดว่าจะเห็น b เป็นบวกเล็กน้อยแม้ระบบถูกต้อง
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C, T0_US = 343.0, 1220.0
LO_CM, HI_CM = 40.0, 200.0
MIN_PEAK_SNR = 3.0            # ยอดต้องสูงกว่านอยส์ของส่วนที่เหลือกี่เท่าถึงเชื่อ


def envelope(x, k=31):
    """ซอง = |x - dc| เฉลี่ยเคลื่อนที่ เฉลี่ยทุกช่อง · x (m, 4, n)"""
    x = x.astype(np.float32)
    y = np.abs(x - np.median(x, axis=-1, keepdims=True)).mean(1)
    ker = np.ones(k, np.float32) / k
    return np.array([np.convolve(r, ker, "same") for r in y])


def load_folder(folder, step):
    fs = sorted(glob.glob(os.path.join(folder, "us_*.npz")))[::step]
    if not fs:
        sys.exit(f"ไม่พบไฟล์ us_*.npz ใน {folder}")
    cnt, near, rate = [], [], None
    for f in fs:
        z = np.load(f)
        cnt.append(z["counts"])
        near.append(float(z["cam_near_cm"]) if "cam_near_cm" in z.files else np.nan)
        rate = rate or float(z["rate"])
    return np.stack(cnt), np.array(near), rate


def load_cache(step):
    root = os.path.join(os.path.dirname(HERE), "car", "data", "_map2_cache_r3_40x30")
    cn = np.load(os.path.join(root, "counts.npy"), mmap_mode="r")
    dm = np.load(os.path.join(root, "dmm.npy"), mmap_mode="r")
    idx = np.arange(0, len(cn), step)
    cnt = np.asarray(cn[idx])
    near = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        d = np.asarray(dm[i]).ravel()
        d = d[d > 0]
        if len(d) > 20:
            near[j] = np.percentile(d, 5) / 10.0       # เหมือน mapdata2.shape_stats
    return cnt, near, 66313.0


def theil_sen(x, y):
    """ถดถอยเส้นตรงแบบทนค่าผิดปกติ (มัธยฐานของความชันทุกคู่ สุ่มคู่ถ้ามากเกิน)"""
    rng = np.random.default_rng(0)
    n = len(x)
    i = rng.integers(0, n, 20000)
    j = rng.integers(0, n, 20000)
    ok = np.abs(x[i] - x[j]) > 5
    s = np.median((y[i][ok] - y[j][ok]) / (x[i][ok] - x[j][ok]))
    return float(s), float(np.median(y - s * x))


def analyze(cnt, near, rate):
    n = cnt.shape[-1]
    dist = (np.arange(n) / rate - T0_US * 1e-6) * C / 2 * 100
    env = envelope(cnt)
    bg = np.median(env, axis=0)
    res = env - bg
    win = (dist >= LO_CM) & (dist < HI_CM)
    noise = np.median(np.abs(res[:, win] - np.median(res[:, win], axis=1, keepdims=True)))
    out = []
    for k in range(len(env)):
        if not (np.isfinite(near[k]) and 55.0 <= near[k] <= 190.0):
            continue
        r = res[k, win]
        p = int(np.argmax(r))
        if r[p] < MIN_PEAK_SNR * max(noise, 1e-6):
            continue
        out.append((near[k], float(dist[win][p]), float(r[p] / max(noise, 1e-6))))
    return np.array(out), dist, bg, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--step", type=int, default=0, help="สุ่มเฟรมทุกกี่เฟรม (ค่าเริ่มต้น: อัตโนมัติ)")
    a = ap.parse_args()
    if not a.folder and not a.cache:
        sys.exit("ระบุโฟลเดอร์ หรือ --cache")
    if a.cache:
        cnt, near, rate = load_cache(a.step or 5)
    else:
        nfiles = len(glob.glob(os.path.join(a.folder, "us_*.npz")))
        cnt, near, rate = load_folder(a.folder, a.step or max(1, nfiles // 1500))
    print(f"{len(cnt)} เฟรม · มีระยะกล้อง {int(np.isfinite(near).sum())} เฟรม")
    pts, dist, bg, noise = analyze(cnt, near, rate)
    if len(pts) < 30:
        sys.exit(f"เหลือเฟรมที่เห็นยอดชัดแค่ {len(pts)} เฟรม — ข้อมูลน้อยหรือคนไม่ได้เปลี่ยนระยะพอ "
                 f"(ต้องเดินเข้า-ออกให้ครอบคลุม 60-190 ซม.)")
    cam, snd, snr = pts[:, 0], pts[:, 1], pts[:, 2]
    a_, b_ = theil_sen(cam, snd)
    err = snd - cam
    print(f"\nเฟรมที่เห็นยอดของคนชัด (≥{MIN_PEAK_SNR:g}x นอยส์): {len(pts)} · ช่วงระยะกล้อง "
          f"{cam.min():.0f}-{cam.max():.0f} ซม.")
    print(f"สหสัมพันธ์ระยะเสียง-ระยะกล้อง: {np.corrcoef(cam, snd)[0, 1]:+.2f}")
    print(f"ถดถอยทนค่าผิดปกติ: ระยะเสียง = {a_:.2f} x ระยะกล้อง {b_:+.1f} ซม.")
    print(f"ความคลาด (เสียง - กล้อง): กลาง {np.median(err):+.1f} ซม. · MAE {np.mean(np.abs(err)):.1f} ซม. · "
          f"อยู่ในช่วง ±15 ซม. {100 * np.mean(np.abs(err) < 15):.0f}% ของเฟรม")
    print("\nแยกตามช่วงระยะกล้อง:")
    for lo, hi in ((55, 90), (90, 120), (120, 150), (150, 190)):
        m = (cam >= lo) & (cam < hi)
        if m.sum() >= 5:
            print(f"  {lo:3d}-{hi:3d} ซม.  {m.sum():5d} เฟรม · เสียงกลาง {np.median(snd[m]):6.1f} ซม. "
                  f"(เทียบกล้อง {np.median(cam[m]):6.1f}) · ความคลาดกลาง {np.median(err[m]):+6.1f} ซม.")
    dt_us = 2 * np.median(err) / 100 / C * 1e6
    print(f"\nถ้าความคลาดกลางเป็นออฟเซ็ตของแกนล้วน ๆ T0 ควรเป็น {T0_US + dt_us:.0f} us (ปัจจุบัน {T0_US:.0f})")
    print("  (อย่าเปลี่ยนก่อนดูตารางแยกช่วง: ถ้าความคลาดไล่ตามระยะ ปัญหาอยู่ที่ความเร็วเสียง/สเกล ไม่ใช่ T0)")


if __name__ == "__main__":
    main()
