"""ตรวจ dataset ที่เก็บมา — ดูว่าอัลตราซาวด์กับ label จากกล้องสอดคล้องกันไหม

ต้องรันทุกครั้งหลังเก็บฉากใหม่ ก่อนเอาไปเทรน ถ้าความสัมพันธ์ไม่โผล่ในตารางนี้
NN ก็จะหาไม่เจอเหมือนกัน — เสียเวลาเทรนเปล่า

    python car/inspect_data.py                  ตรวจทุกฉากใน car/data/
    python car/inspect_data.py --scene test_01  ตรวจฉากเดียว
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from labels import depth_to_profile, bin_angles, profile_to_text    # noqa: E402
from scope_view import bandpass, envelope, counts_to_volts          # noqa: E402

C = 343.0
DATA = Path(HERE) / "data"


def echo_features(counts, rate):
    """ดึงลักษณะเด่นของเอคโค่ช่องแรก: ความแรง, ระยะที่ยอด, SNR

    ใช้แค่ดูความสอดคล้องด้วยตา ตอนเทรนจริง NN จะกินคลื่นดิบเข้าไปเอง
    """
    v = counts_to_volts(counts[0].astype(float))
    e = envelope(bandpass(v - v.mean(), rate, 25e3, 60e3))
    floor = float(np.median(e[int(0.85 * len(e)):]))
    # ข้ามพัลส์ตรง/ring-down ช่วงต้น แล้วหายอดที่แรงที่สุด
    skip = int(6e-4 * rate)
    k = skip + int(np.argmax(e[skip:]))
    snr = 20 * np.log10(e[k] / floor) if floor > 0 else 0.0
    rng_cm = (k / rate) * C / 2 * 100
    return e[k] * 1e3, rng_cm, snr, len(e) / rate * C / 2 * 100


def inspect(scene_dir):
    import cv2
    us_files = sorted(glob.glob(str(scene_dir / "us_*.npz")))
    if not us_files:
        print(f"  (ไม่มีข้อมูลใน {scene_dir.name})")
        return None
    rows = []
    for u in us_files:
        idx = Path(u).stem.split("_")[1]
        d_path = scene_dir / f"depth_{idx}.png"
        if not d_path.exists():
            continue
        z = np.load(u)
        amp, rng, snr, maxrng = echo_features(z["counts"], float(z["rate"]))
        depth = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
        dist, valid = depth_to_profile(depth)
        nearest = float(dist[valid].min()) if valid.any() else 0.0
        rows.append((idx, amp, rng, snr, nearest, dist, valid, maxrng))

    print(f"\n=== {scene_dir.name} · {len(rows)} ตัวอย่าง ===")
    print(f"{'#':>6} {'แรง(mV)':>9} {'ยอดที่':>9} {'SNR':>7} {'ใกล้สุดจากกล้อง':>16}")
    for idx, amp, rng, snr, near, *_ in rows:
        print(f"{idx:>6} {amp:8.1f} {rng:7.0f}cm {snr:6.1f}dB "
              f"{near/10 if near else 0:13.0f}cm")

    a = np.array([r[1] for r in rows])          # ความแรงเอคโค่
    n = np.array([r[4] for r in rows]) / 10.0   # ระยะใกล้สุดจากกล้อง cm
    ok = n > 0
    print(f"\nช่วงบันทึกอัลตราซาวด์ {rows[0][7]:.0f} cm · "
          f"ระยะจากกล้อง {n[ok].min():.0f}..{n[ok].max():.0f} cm")
    if ok.sum() >= 3 and n[ok].std() > 1:
        r = float(np.corrcoef(a[ok], n[ok])[0, 1])
        print(f"สหสัมพันธ์ ความแรงเอคโค่ กับ ระยะจากกล้อง = {r:+.2f}")
        print("  (ควรเป็นลบ: ใกล้ = เสียงกลับแรง · ถ้าใกล้ 0 แปลว่าฉากนี้ยังไม่มีสัญญาณให้เรียน)")
    else:
        print("ระยะในฉากนี้แทบไม่เปลี่ยน — ต้องเก็บหลายระยะจึงจะเห็นความสัมพันธ์")
    print(f"\nตัวอย่างโปรไฟล์เชิงมุม (label ที่ NN จะเรียน):")
    for idx, _, _, _, _, dist, valid, _ in rows[:3]:
        print(f"  {idx}: {profile_to_text(dist, valid)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None)
    a = ap.parse_args()
    scenes = ([DATA / a.scene] if a.scene
              else sorted(p for p in DATA.glob("*") if p.is_dir()))
    if not scenes:
        sys.exit(f"ไม่พบฉากใน {DATA}")
    total = 0
    for s in scenes:
        rows = inspect(s)
        total += len(rows) if rows else 0
    print(f"\nรวมทุกฉาก {total} ตัวอย่าง")


if __name__ == "__main__":
    main()
