#!/usr/bin/env python3
"""เทรนจากข้อมูลที่อัดต่อเนื่อง (record.py) — กัน 1 ช่วงไว้ทดสอบ เทรนด้วยที่เหลือ

ทำไมต้องแบ่งตาม "ช่วง" ไม่ใช่สุ่มแบ่งทีละเฟรม:
  เฟรมที่ติดกันในเวลาแทบเหมือนกันทุกอย่าง ถ้าสุ่มแบ่ง เฟรมข้างเคียงของเฟรมทดสอบ
  จะไปอยู่ในชุดเทรน = โมเดล "เคยเห็นคำตอบมาแล้ว" ตัวเลขจะสวยเกินจริงมาก
  แบ่งตามช่วงเวลาที่อัดคนละรอบจึงเป็นการวัดที่ซื่อสัตย์

    python car/train_sections.py --name rec            กัน s4 ไว้ทดสอบ
    python car/train_sections.py --name rec --test 2   กัน s2 ไว้ทดสอบ
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import DATA                    # noqa: E402
import features as F                      # noqa: E402


def load_both(scenes, keep_multi):
    """อ่านไฟล์รอบเดียว แล้วคำนวณ feature ทั้งสองโหมด (ยอดร่วม / แยกคู่)

    อ่านรอบเดียวเพราะถอดรหัส PNG + คำนวณเอนเวโลปเป็นงานหนัก ทำสองรอบเสียเวลาเปล่า
    """
    import glob
    import cv2
    from labels import depth_to_profile, bin_angles, near_object_count
    ang = bin_angles()
    out = {"common": ([], []), "pair": ([], [])}
    y, grp, snr, skip_multi = [], [], [], 0
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
            if not keep_multi and near_object_count(depth) > 1:
                skip_multi += 1
                continue
            z = np.load(u)
            counts, rate = z["counts"], float(z["rate"])
            for mode, pp in (("common", False), ("pair", True)):
                out[mode][0].append(F.distance_features(counts, rate, per_pair=pp))
                out[mode][1].append(F.angle_features(counts, rate, per_pair=pp))
            # เก็บ SNR ของยอดไว้ด้วย — ใช้ดูว่าเฟรมที่เสียงกลับอ่อนทำให้ผลแย่แค่ไหน
            envs = [F.envelope_of(counts[c], rate) for c in range(counts.shape[0])]
            kk, _, ref = F.common_peak(envs, rate)
            lo = max(1, int((F.T0_US + 2 * F.GATE_MIN_CM / 100 / F.C * 1e6) * 1e-6 * rate))
            hi = min(len(envs[ref]),
                     int((F.T0_US + 2 * F.GATE_MAX_CM / 100 / F.C * 1e6) * 1e-6 * rate))
            snr.append(float(envs[ref][kk]) / max(float(np.median(envs[ref][lo:hi])), 1e-12))
            kb = int(np.argmin(np.where(valid, dist, 1e9)))
            y.append([dist[kb] / 10.0, ang[kb]])
            grp.append(si)
    if skip_multi:
        print(f"** ข้าม {skip_multi} เฟรมที่มีของใกล้หลายจุด (ใช้ --keep-multi ถ้าไม่ต้องการ)")
    # ความยาว feature ของ "ระยะ" กับ "มุม" ไม่เท่ากันอยู่แล้ว (5 vs 6 มิติ)
    # ต้องหาความยาวมาตรฐานแยกกันทีละอย่าง ไม่ใช่รวมเป็นเซ็ตเดียว
    from collections import Counter
    want = {(m, j): Counter(len(v) for v in out[m][j]).most_common(1)[0][0]
            for m in out for j in (0, 1)}
    keep = [k for k in range(len(y))
            if all(len(out[m][j][k]) == want[(m, j)] for m in out for j in (0, 1))]
    if len(keep) < len(y):
        print(f"** ข้าม {len(y) - len(keep)} ตัวอย่างที่ช่องไม่ครบ")
    res = {m: (np.array([out[m][0][k] for k in keep], np.float32),
               np.array([out[m][1][k] for k in keep], np.float32)) for m in out}
    return (res, np.array([y[k] for k in keep], np.float32),
            np.array([grp[k] for k in keep]), np.array([snr[k] for k in keep], np.float32))


def fit_eval(Xd, Xa, y, tr, te):
    """คืน dict ของ MAE — baseline (เดาค่าเฉลี่ย) · โมเดล · ฟิสิกส์"""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    alphas = np.logspace(-1, 4, 20)
    out = {}
    for j, (X, key) in enumerate(((Xd, "d"), (Xa, "a"))):
        sc = StandardScaler().fit(X[tr])
        out[key + "_base"] = float(np.abs(y[te, j] - y[tr, j].mean()).mean())
        m = RidgeCV(alphas=alphas).fit(sc.transform(X[tr]), y[tr, j])
        out[key + "_model"] = float(np.abs(m.predict(sc.transform(X[te])) - y[te, j]).mean())
    # ฟิสิกส์: ระยะจากเวลาเอคโค่ตรง ๆ ไม่ผ่านการเทรนเลย (Xd คอลัมน์ 0)
    out["d_phys"] = float(np.abs(Xd[te][:, 0] - y[te, 0]).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="rec")
    ap.add_argument("--keep-multi", action="store_true",
                    help="ไม่กรองเฟรมที่มีของใกล้หลายจุด (ใช้เมื่อ 'คน' คือเป้าเอง)")
    ap.add_argument("--test", type=int, default=None,
                    help="เลขช่วงที่กันไว้ทดสอบ (ไม่ระบุ = ช่วงสุดท้าย)")
    a = ap.parse_args()

    secs = sorted(DATA.glob(f"{a.name}_s*"))
    secs = [p for p in secs if p.is_dir()]
    if len(secs) < 2:
        sys.exit(f"ต้องมีอย่างน้อย 2 ช่วง — เจอ {len(secs)} "
                 f"(มองหา {DATA}/{a.name}_s*)")

    feats, y, grp, snr = load_both(secs, a.keep_multi)
    Xd, Xa = feats["common"]
    print(f"\nโหลด {len(y)} ตัวอย่าง จาก {len(secs)} ช่วง · "
          f"feature ระยะ {Xd.shape[1]} · มุม {Xa.shape[1]} มิติ")
    for i, p in enumerate(secs):
        n = int((grp == i).sum())
        if n:
            d = y[grp == i, 0]
            print(f"   {p.name:<14} {n:5d} ตัวอย่าง · ระยะ {d.min():.0f}-{d.max():.0f} cm "
                  f"(กลาง {np.median(d):.0f})")
        else:
            print(f"   {p.name:<14}     0 ตัวอย่าง  ← ใช้ไม่ได้")

    uniq = sorted(set(grp.tolist()))
    if len(uniq) < 2:
        sys.exit("เหลือช่วงที่ใช้ได้ไม่ถึง 2 — เก็บเพิ่มก่อน")

    ti = (a.test - 1) if a.test else uniq[-1]
    if ti not in uniq:
        sys.exit(f"ช่วงทดสอบ s{ti+1} ไม่มีข้อมูล")
    te, tr = (grp == ti), (grp != ti)
    tr_names = ", ".join(secs[i].name for i in uniq if i != ti)

    print(f"\n=== แบบที่ขอ: เทรนด้วย {tr_names} · ทดสอบด้วย {secs[ti].name} ===")
    r = fit_eval(Xd, Xa, y, tr, te)
    print(f"  เทรน {int(tr.sum())} ตัวอย่าง · ทดสอบ {int(te.sum())} ตัวอย่าง\n")
    print(f"  {'':<8}{'เดาค่าเฉลี่ย':>13}{'โมเดล':>10}{'ฟิสิกส์':>11}")
    print(f"  {'ระยะ':<8}{r['d_base']:>11.1f} cm{r['d_model']:>8.1f} cm"
          f"{r['d_phys']:>9.1f} cm")
    print(f"  {'มุม':<8}{r['a_base']:>12.1f}°{r['a_model']:>9.1f}°{'—':>11}")

    # หมุนกันทีละช่วงด้วย — ช่วงเดียวอาจบังเอิญง่ายหรือยากผิดปกติ
    print(f"\n=== ตรวจซ้ำแบบหมุนกันทุกช่วง (leave-one-section-out) ===")
    print(f"  {'กันช่วง':<14}{'ระยะ base':>11}{'ระยะ โมเดล':>12}{'ระยะ ฟิสิกส์':>13}"
          f"{'มุม base':>10}{'มุม โมเดล':>11}")
    acc = []
    for i in uniq:
        rr = fit_eval(Xd, Xa, y, grp != i, grp == i)
        acc.append(rr)
        print(f"  {secs[i].name:<14}{rr['d_base']:>11.1f}{rr['d_model']:>12.1f}"
              f"{rr['d_phys']:>13.1f}{rr['a_base']:>10.1f}{rr['a_model']:>11.1f}")
    mean = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
    print(f"  {'เฉลี่ย':<14}{mean['d_base']:>11.1f}{mean['d_model']:>12.1f}"
          f"{mean['d_phys']:>13.1f}{mean['a_base']:>10.1f}{mean['a_model']:>11.1f}")

    print("\n=== เทียบวิธีหายอด: 'ยอดร่วมทั้ง 4 ช่อง' vs 'แยกคู่' ===")
    print("  (เป้าเคลื่อนที่ควรใช้ 'แยกคู่' เพราะ 4 ช่องยิงสองรอบห่างกัน ~190 ms)")
    print(f"  {'วิธี':<12}{'ระยะ โมเดล':>12}{'ระยะ ฟิสิกส์':>13}{'มุม โมเดล':>11}")
    best = None
    for mode, label in (("common", "ยอดร่วม"), ("pair", "แยกคู่")):
        Xd_, Xa_ = feats[mode]
        rs = [fit_eval(Xd_, Xa_, y, grp != i, grp == i) for i in uniq]
        mm = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
        print(f"  {label:<12}{mm['d_model']:>12.1f}{mm['d_phys']:>13.1f}{mm['a_model']:>11.1f}")
        if best is None or mm["a_model"] < best[1]["a_model"]:
            best = (label, mm)
    print(f"  -> วิธีที่ให้มุมแม่นกว่า: **{best[0]}**")

    print("\n=== ผลแยกตามความแรงเอคโค่ (SNR ของยอด) ===")
    print("  เสียงกลับอ่อน = ยอดที่เจออาจไม่ใช่ตัวเป้า แต่เป็นเสียงขยะในห้อง")
    Xd_, Xa_ = feats["common"]
    edges = np.quantile(snr, [0, .25, .5, .75, 1.0])
    print(f"  {'ช่วง SNR':<18}{'จำนวน':>7}{'ระยะ ฟิสิกส์':>14}{'มุม โมเดล':>12}")
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (snr >= lo_) & (snr <= hi_)
        if m.sum() < 10:
            continue
        dphys = float(np.abs(Xd_[m][:, 0] - y[m, 0]).mean())
        # มุม: เทรนจากทุกเฟรม แต่วัดผลเฉพาะเฟรมในช่วงนี้ (หมุนกันทีละช่วงเหมือนเดิม)
        errs = []
        for i in uniq:
            tem = (grp == i) & m
            if tem.sum() < 3:
                continue
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler
            trm = grp != i
            scl = StandardScaler().fit(Xa_[trm])
            mo = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(scl.transform(Xa_[trm]), y[trm, 1])
            errs.append(np.abs(mo.predict(scl.transform(Xa_[tem])) - y[tem, 1]))
        amae = float(np.concatenate(errs).mean()) if errs else float("nan")
        print(f"  {lo_:6.1f} - {hi_:7.1f}{int(m.sum()):>10}{dphys:>11.1f} cm{amae:>11.1f}°")

    def verdict(base, val):
        return "✓" if val < base * 0.85 else ("~" if val < base else "✗")

    print(f"""
สรุป (ใช้ค่าเฉลี่ยจากการหมุนทุกช่วง):
  ระยะ  ฟิสิกส์ {mean['d_phys']:.1f} cm {verdict(mean['d_base'], mean['d_phys'])}  ·  """
          f"""โมเดล {mean['d_model']:.1f} cm {verdict(mean['d_base'], mean['d_model'])}  """
          f"""(เดาค่าเฉลี่ย {mean['d_base']:.1f} cm)
  มุม   โมเดล {mean['a_model']:.1f}° {verdict(mean['a_base'], mean['a_model'])}  """
          f"""(เดาค่าเฉลี่ย {mean['a_base']:.1f}°)

  ✓ = ชนะการเดาชัด (>15%) · ~ = ชนะนิดหน่อย · ✗ = ไม่ชนะ""")


if __name__ == "__main__":
    main()
