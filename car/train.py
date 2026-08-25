"""เทรนโมเดลแรก: คลื่นอัลตราซาวด์ -> "อะไรอยู่ข้างหน้า" (Stage A)

โมเดลแรกทำนายสองค่าที่เรียนได้จริงก่อน: ระยะ กับ มุม ของสิ่งที่ใกล้ที่สุด
(โปรไฟล์ 9 ช่องเต็มไว้ทีหลังตอนข้อมูลเยอะ ตอนนี้มีหลักสิบตัวอย่าง จะ overfit)

เทียบกับ baseline "เดาค่าเฉลี่ย" เสมอ ถ้าโมเดลไม่ชนะ baseline = ยังไม่มีอะไรให้เรียน

    python car/train.py                 ใช้ทุกฉากใน car/data/ (ยกเว้น test_*)
    python car/train.py --scenes d70    ระบุฉากเอง
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
from features import distance_features, angle_features             # noqa: E402
from labels import depth_to_profile, bin_angles, near_object_count  # noqa: E402

DATA = Path(HERE) / "data"


def load_all(scenes):
    """คืน (Xd, Xa, y, grp) — feature ระยะ, feature มุม (แยกกัน), เป้า, ฉาก

    แยก feature ตามเป้า: ระยะใช้เวลาเอคโค่ · มุมใช้ TDOA — ปนกันแล้วต่างฝ่ายกลายเป็น
    noise ของอีกฝ่าย (envelope เต็มทำมุมแย่ลง, direction feature ทำระยะสับสน)
    """
    import cv2
    Xd, Xa, y, grp = [], [], [], []
    ang = bin_angles()
    skipped_multi = 0
    for si, sc in enumerate(scenes):
        for u in sorted(glob.glob(str(sc / "us_*.npz"))):
            i = Path(u).stem.split("_")[1]
            dp = sc / f"depth_{i}.png"
            if not dp.exists():
                continue
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            dist, valid = depth_to_profile(depth)
            if not valid.any():
                continue
            if near_object_count(depth) > 1:      # กำกวม (ท่อ + ขา/พื้นหลังใกล้) — ข้าม
                skipped_multi += 1
                continue
            kb = int(np.argmin(np.where(valid, dist, 1e9)))
            z = np.load(u)
            counts, rate = z["counts"], float(z["rate"])
            Xd.append(distance_features(counts, rate))
            Xa.append(angle_features(counts, rate))
            y.append([dist[kb] / 10.0, ang[kb]])     # [ระยะ cm, มุม deg]
            grp.append(si)
    # กันไฟล์จำนวนช่องไม่ตรงกัน (เช่น 2 ช่องปนกับ 4 ช่อง = feature ยาวไม่เท่ากัน)
    from collections import Counter
    da = Counter(len(x) for x in Xa).most_common(1)[0][0]
    dd = Counter(len(x) for x in Xd).most_common(1)[0][0]
    ok = [k for k in range(len(Xa)) if len(Xa[k]) == da and len(Xd[k]) == dd]
    if skipped_multi:
        print(f"** ข้าม {skipped_multi} เฟรมที่มีของใกล้หลายจุด (ท่อ+ขา/พื้นหลัง = label กำกวม)")
    if len(ok) < len(Xa):
        print(f"** ข้าม {len(Xa) - len(ok)} ตัวอย่างที่ช่องไม่ครบ (feature ยาวไม่ตรง)")
    Xd = np.array([Xd[k] for k in ok], np.float32)
    Xa = np.array([Xa[k] for k in ok], np.float32)
    y = np.array([y[k] for k in ok], np.float32)
    grp = np.array([grp[k] for k in ok])
    return Xd, Xa, y, grp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--val-frac", type=float, default=0.3)
    a = ap.parse_args()

    scenes = ([DATA / s for s in a.scenes] if a.scenes
              else sorted(p for p in DATA.glob("*")
                          if p.is_dir() and not p.name.startswith("test")))
    Xd, Xa, y, grp = load_all(scenes)
    print(f"โหลด {len(y)} ตัวอย่าง จาก {len(scenes)} ฉาก · "
          f"feature ระยะ {Xd.shape[1]} · มุม {Xa.shape[1]} มิติ")
    if len(y) < 20:
        sys.exit("ตัวอย่างน้อยเกินไป เก็บเพิ่มก่อน")

    # ---- วัดผลแบบหมุนกันทีละฉาก (leave-one-scene-out) ----
    # ทำไมไม่กันฉากกลางฉากเดียว: ถ้าเก็บระยะห่างเท่าๆ กัน ค่าเฉลี่ยของฉากที่เหลือจะ
    # ตรงกับฉากกลางพอดี -> baseline "เดาค่าเฉลี่ย" แม่นเองโดยไม่ต้องรู้อะไร (เคยได้ 1.4 cm)
    # หมุนทดสอบทุกฉากจึงวัดได้ทั้งช่วงกลาง (interpolation) และช่วงปลาย (extrapolation)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    alphas = np.logspace(-1, 4, 20)
    uniq = np.unique(grp)
    names = ["ระยะ (cm)", "มุม (deg)"]
    feat_for = [Xd, Xa]

    if len(uniq) < 2:
        sys.exit("ต้องมีอย่างน้อย 2 ฉากถึงจะวัดผลแบบซื่อสัตย์ได้")

    scene_dist = {t: float(np.median(y[grp == t, 0])) for t in uniq}
    order = sorted(uniq, key=lambda t: scene_dist[t])

    print("หมุนทดสอบทีละฉาก (เทรนจากฉากที่เหลือ) — ตัวเลขคือ MAE\n\n")
    print(f"{'ฉากที่กัน':<10} {'ระยะจริง':>9} | {'ระยะ base':>10} {'ระยะ โมเดล':>11} "
          f"{'ระยะ ฟิสิกส์':>12} | {'มุม base':>9} {'มุม โมเดล':>10}")
    acc = {k: [] for k in ("db", "dm", "dp", "ab", "am")}
    for t in order:
        te = grp == t
        tr = ~te
        res = []
        for j in range(2):
            X = feat_for[j]
            sc_ = StandardScaler().fit(X[tr])
            base = np.abs(y[te, j] - y[tr, j].mean()).mean()
            m = RidgeCV(alphas=alphas).fit(sc_.transform(X[tr]), y[tr, j])
            mae = np.abs(m.predict(sc_.transform(X[te])) - y[te, j]).mean()
            res += [base, mae]
        # ฟิสิกส์: ระยะจากยอดเอคโค่ร่วมตรงๆ ไม่ผ่านการเทรนเลย
        # (Xd คอลัมน์ 0 = ระยะของยอดร่วม · คอลัมน์ที่เหลือคือความแรงรายช่อง)
        phys = np.abs(Xd[te][:, 0] - y[te, 0]).mean()
        acc["db"].append(res[0]); acc["dm"].append(res[1]); acc["dp"].append(phys)
        acc["ab"].append(res[2]); acc["am"].append(res[3])
        print(f"{scenes[t].name:<10} {scene_dist[t]:>7.0f}cm | {res[0]:>10.1f} {res[1]:>11.1f} "
              f"{phys:>12.1f} | {res[2]:>9.1f} {res[3]:>10.1f}")

    d_b, d_m, d_p = (np.mean(acc[k]) for k in ("db", "dm", "dp"))
    a_b, a_m = np.mean(acc["ab"]), np.mean(acc["am"])
    print(f"\n{'เฉลี่ยทุกฉาก':<10} {'':>9} | {d_b:>10.1f} {d_m:>11.1f} {d_p:>12.1f} "
          f"| {a_b:>9.1f} {a_m:>10.1f}")

    def verdict(base, val):
        return "✓" if val < base * 0.85 else ("~" if val < base else "✗")

    print(f"""
สรุป:
  ระยะ  ฟิสิกส์ {d_p:.1f} cm {verdict(d_b, d_p)}  ·  โมเดล {d_m:.1f} cm {verdict(d_b, d_m)}  (เดาค่าเฉลี่ย {d_b:.1f} cm)
  มุม   โมเดล {a_m:.1f}° {verdict(a_b, a_m)}  (เดาค่าเฉลี่ย {a_b:.1f}°)

  ✓ = ชนะการเดาชัด (>15%) · ~ = ชนะนิดหน่อย · ✗ = ไม่ชนะ
  ระยะควรใช้ 'ฟิสิกส์' (คำนวณจากเวลาเอคโค่) · ML มีไว้ทำ 'มุม' ที่คำนวณตรงๆ ไม่ได้
  {len(y)} ตัวอย่าง จาก {len(uniq)} ระยะ""")


if __name__ == "__main__":
    main()
