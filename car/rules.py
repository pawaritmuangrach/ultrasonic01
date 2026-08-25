#!/usr/bin/env python3
"""กฎแบบเขียนมือ (rule base) — บอกทิศของเป้าจากความแรงเอคโค่ โดยไม่ใช้ ML

ใช้กับข้อมูลที่อัดด้วย `record.py` โดยเดินเปลี่ยนตำแหน่งไปมาในช่วงเดียวกัน
เฉลยมาจากกล้อง depth ทีละเฟรม (จุดศูนย์กลางของก้อนเป้า) ไม่ใช่จากชื่อช่วง

    python car/rules.py --name walk --test 1      กันช่วงที่ 1 ไว้ทดสอบ
    python car/rules.py --name walk --refresh     บังคับอ่านไฟล์ดิบใหม่

หลักการ — ตัวเลขเดียว ไม่มีการเทรนแบบ ML:
    เสียงสะท้อนเข้าหัวรับที่อยู่ใกล้เป้ากว่าย่อมแรงกว่า เทียบสองฝั่งก็บอกทิศได้

        log4 = ln( (ซ้ายสุด + ซ้าย + 1) / (ขวา + ขวาสุด + 1) )

    **ใช้ ln ของอัตราส่วน ไม่ใช่ (a-b)/(a+b)** เพราะการตอบสนองของหัวรับเป็นตัวคูณ
    (beam pattern คือเกน) ln จึงเป็นตัวที่แปรตามมุมแบบเส้นตรง ส่วน (a-b)/(a+b)
    อิ่มตัวที่ +-1 แล้วบีบช่วงปลายทิ้ง — วัดจริงกับ walk 5 ช่วง: log4 ดีกว่าทุกเกณฑ์
    **รวมทั้งสี่ช่อง** ดีกว่าใช้คู่นอกอย่างเดียว (MAE 6.2 เทียบกับ 7.2 องศา)

    +1 ใน ln กันหารศูนย์ตอนช่องหนึ่งเงียบสนิท (พื้นเสียงรบกวนราว 10 mV)

ทำไมต้องเกลี่ยตามเวลา:
    เสียงสะท้อนจากคนเป็นแบบ specular — ขยับตัวนิดเดียวความแรงเปลี่ยน 40 เท่า
    (วัดจริง: 28 mV ถึง 1173 mV ที่ระยะเท่ากัน) มัธยฐาน 9 เฟรม (0.6 วินาที)
    ตัดการกระพริบนี้ทิ้งโดยไม่ทำให้ขอบการเดินเบลอ  MAE 6.2 -> 4.8 องศา

ทำไมใช้ความแรง ไม่ใช่ TDOA:
    TDOA แม่นกว่าในทฤษฎี แต่ที่ 66 kHz ความละเอียดเวลา 15 us และต้องหายอดให้ตรงกัน
    ทุกช่องก่อน ซึ่งพังทันทีเมื่อเสียงกลับอ่อน — กฎง่ายที่ทนทานดีกว่ากฎแม่นที่พังบ่อย

**ไม่ล็อกทิศทางไว้ล่วงหน้า** เครื่องหมายของ log4 เทียบกับมุมกล้องหาจากข้อมูลเอง
ถ้าสายหัวรับสลับหรือมุมกล้องกลับด้าน จะเห็นทันทีจากตารางความแรงรายช่อง
ไม่ใช่ไปโผล่เป็นผลลัพธ์ที่ผิดแบบเงียบ ๆ
"""
import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import features as F                                    # noqa: E402

DATA = Path(HERE) / "data"
PINS = [34, 33, 32, 35]                 # ซ้ายสุด, ซ้าย, ขวา, ขวาสุด (ตามตำแหน่งจริง)
NAME = {34: "FAR LEFT", 33: "LEFT", 32: "RIGHT", 35: "FAR RIGHT"}
COLS = "sec idx deg cam cov rng a34 a33 a32 a35 n34 n33 n32 n35".split()


# ------------------------------------------------------------------ ดึงคุณลักษณะ
def extract(name, refresh=False):
    """อ่านไฟล์ดิบทุกเฟรม -> ตาราง (ช่วง, ลำดับ, มุมกล้อง, ระยะกล้อง, สัดส่วนที่เห็น,
    ระยะเสียง, ยอด 4 ช่อง, พื้นเสียงรบกวน 4 ช่อง) แล้วเก็บแคชไว้ เพราะการหาเปลือกคลื่น
    ทั้ง 4325 เฟรมใช้เวลาราว 10 นาที แต่การลองปรับเกณฑ์ควรใช้เวลาไม่ถึงวินาที"""
    import cv2
    from labels import target_angle

    secs = sorted(p for p in DATA.glob(f"{name}_s*") if p.is_dir())
    if len(secs) < 2:
        sys.exit(f"ต้องมีอย่างน้อย 2 ช่วง — เจอ {len(secs)} (มองหา {DATA}/{name}_s*)")
    cache = DATA / f"_{name}_cache.npz"
    stamp = np.array([len(secs)] + [len(list(s.glob("us_*.npz"))) for s in secs])
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        if list(z["cols"]) == COLS and np.array_equal(z["stamp"], stamp):
            return z["rows"], [p.name for p in secs]
        print("** ข้อมูลเปลี่ยนไปจากตอนทำแคช — อ่านไฟล์ดิบใหม่")

    rows = []
    for si, sc in enumerate(secs):
        fs = sorted(glob.glob(str(sc / "us_*.npz")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
        for f in fs:
            tag = Path(f).stem.split("_")[1]        # เก็บเลข 0 นำหน้าไว้ ชื่อไฟล์เติมศูนย์
            dp = sc / f"depth_{tag}.png"
            if not dp.exists():
                continue
            i = int(tag)
            lab = target_angle(cv2.imread(str(dp), cv2.IMREAD_UNCHANGED))
            z = np.load(f)
            c, rate = z["counts"], float(z["rate"])
            idx = {int(p): j for j, p in enumerate(z["pins"])}
            envs = [F.envelope_of(c[j], rate) for j in range(c.shape[0])]
            k, rng, _ = F.common_peak(envs, rate)
            tail = int(0.9 * len(envs[0]))
            deg, cam, cov = lab if lab else (np.nan, np.nan, 0.0)
            rows.append([si, i, deg, cam, cov, rng]
                        + [float(envs[idx[p]][k]) * 1e3 for p in PINS]
                        + [float(np.median(envs[idx[p]][tail:])) * 1e3 for p in PINS])
        print(f"  อ่าน {sc.name}: {len(rows)} เฟรม", flush=True)
    r = np.array(rows, np.float32)
    np.savez_compressed(cache, rows=r, cols=np.array(COLS), stamp=stamp)
    print(f"  เก็บแคชไว้ที่ {cache.name} — ครั้งหน้าจะเร็วขึ้นมาก")
    return r, [p.name for p in secs]


def median_filter(x, sec, idx, keep, k):
    """มัธยฐานเลื่อนภายในแต่ละช่วง เรียงตามลำดับเฟรมจริง ใช้เฉพาะเฟรมที่ผ่านเกณฑ์

    แยกทำทีละช่วงเพื่อไม่ให้ค่าจากช่วงเทรนรั่วข้ามมาปนช่วงทดสอบ"""
    if k <= 1:
        return x.copy()
    out, h = x.copy(), k // 2
    for s in np.unique(sec):
        j = np.nonzero((sec == s) & keep)[0]
        if j.size < k:
            continue
        j = j[np.argsort(idx[j])]
        w = x[j]
        pad = np.r_[np.repeat(w[0], h), w, np.repeat(w[-1], h)]
        out[j] = [np.median(pad[i:i + k]) for i in range(w.size)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="walk")
    ap.add_argument("--test", type=int, default=1,
                    help="ช่วงที่กันไว้ทดสอบ นับจาก 1 (ที่เหลือใช้เทรน)")
    ap.add_argument("--min-amp", type=float, default=80.0,
                    help="ยอดขั้นต่ำ mV — ต่ำกว่านี้ทิศเชื่อไม่ได้ (พื้นเสียงราว 10 mV)")
    ap.add_argument("--min-cov", type=float, default=0.10,
                    help="สัดส่วนภาพขั้นต่ำที่เป้าต้องกิน จึงจะเชื่อมุมจากกล้อง")
    ap.add_argument("--smooth", type=int, default=9, help="มัธยฐานกี่เฟรม (1 = ไม่เกลี่ย)")
    ap.add_argument("--zones", default="-12,4", help="ขอบโซนเป็นองศา")
    ap.add_argument("--refresh", action="store_true", help="ไม่ใช้แคช อ่านไฟล์ดิบใหม่")
    a = ap.parse_args()

    r, names = extract(a.name, a.refresh)
    c = {k: i for i, k in enumerate(COLS)}
    sec, idx = r[:, c["sec"]], r[:, c["idx"]]
    deg, cam, cov, rng = (r[:, c[k]] for k in ("deg", "cam", "cov", "rng"))
    A = r[:, [c[f"a{p}"] for p in PINS]]
    N = r[:, [c[f"n{p}"] for p in PINS]]
    edges = [float(v) for v in a.zones.split(",")]
    zname = (["ซ้าย", "กลาง", "ขวา"] if len(edges) == 2
             else ["ซ้าย"] + [f"โซน{i + 2}" for i in range(len(edges) - 1)] + ["ขวา"])

    ti = a.test - 1
    if not (0 <= ti < len(names)):
        sys.exit(f"--test ต้องอยู่ในช่วง 1..{len(names)}")

    # ---------------------------------------------------------------- คัดเฟรม
    seen = np.isfinite(deg)
    inframe = seen & (cov >= a.min_cov)
    keep = inframe & (A.max(1) >= a.min_amp)
    print(f"\nข้อมูล {a.name}: {len(names)} ช่วง {len(deg)} เฟรม")
    print(f"  กล้องไม่เห็นเป้า      : {int((~seen).sum()):5d}")
    print(f"  เป้าเกือบหลุดเฟรม     : {int((seen & ~inframe).sum()):5d} "
          f"(เห็นน้อยกว่า {a.min_cov:.0%} ของแถบ — มุมจะติดขอบ)")
    print(f"  เสียงกลับอ่อนเกินไป   : {int((inframe & ~keep).sum()):5d} "
          f"(ยอดต่ำกว่า {a.min_amp:.0f} mV)")
    print(f"  เหลือใช้ได้           : {int(keep.sum()):5d}  ({keep.mean():.0%})")
    if keep.sum() < 200:
        sys.exit("เหลือเฟรมน้อยเกินไป — ลด --min-amp หรือ --min-cov")
    print(f"  มุมจากกล้อง  : {deg[keep].min():+.0f} .. {deg[keep].max():+.0f} องศา "
          f"(+ = เป้าอยู่ทางขวาของภาพ)")
    print(f"  ระยะจากกล้อง : {cam[keep].min():.0f} .. {cam[keep].max():.0f} cm")

    d = np.abs(rng[keep] - cam[keep])
    print(f"\nเสียงได้ยินเป้าจริงไหม — เทียบระยะที่วัดได้กับกล้อง")
    print(f"  กล้อง {cam[keep].mean():.0f} cm · เสียง {rng[keep].mean():.0f} cm · "
          f"ต่างกันเฉลี่ย {d.mean():.1f} cm · ตรงกันในระยะ 15 cm {np.mean(d < 15):.0%}")

    # ---------------------------------------------------------------- ตารางตรวจสาย
    print(f"\nความแรงมัธยฐาน (mV) แยกตามมุมกล้อง — ใช้ตรวจว่าสายต่อถูกด้าน")
    print(f"  {'มุม':>6}{'เฟรม':>7}" + "".join(f"{NAME[p]:>11}" for p in PINS))
    ab = np.round(deg / 8.0) * 8
    for v in np.unique(ab[keep]):
        m = keep & (ab == v)
        if m.sum() < 25:
            continue
        print(f"  {v:>+6.0f}{int(m.sum()):>7}"
              + "".join(f"{np.median(A[m, j]):>11.0f}" for j in range(4)))
    print(f"  {'พื้นเสียง':>6}{'':>7}"
          + "".join(f"{np.median(N[keep, j]):>11.1f}" for j in range(4)))

    # ---------------------------------------------------------------- ฟีเจอร์
    L, R = A[:, 0] + A[:, 1], A[:, 2] + A[:, 3]
    raw = np.log((L + 1.0) / (R + 1.0))
    x = median_filter(raw, sec, idx, keep, a.smooth)
    tr, te = keep & (sec != ti), keep & (sec == ti)
    if te.sum() < 50:
        sys.exit(f"ช่วงทดสอบ {names[ti]} เหลือเฟรมน้อยเกินไป ({int(te.sum())})")
    print(f"\nเทรนด้วย {', '.join(names[i] for i in range(len(names)) if i != ti)}"
          f" · ทดสอบด้วย {names[ti]}   ({int(tr.sum())}/{int(te.sum())} เฟรม)")
    print(f"  ความสัมพันธ์ log4 กับมุมกล้อง : {np.corrcoef(x[keep], deg[keep])[0, 1]:+.3f}"
          f"   (ก่อนเกลี่ย {np.corrcoef(raw[keep], deg[keep])[0, 1]:+.3f})")

    # ---------------------------------------------------------------- กฎ 1: องศา
    co, *_ = np.linalg.lstsq(np.c_[x[tr], np.ones(int(tr.sum()))], deg[tr], rcond=None)
    pred = np.c_[x[te], np.ones(int(te.sum()))] @ co
    mae = float(np.abs(pred - deg[te]).mean())
    base = float(np.abs(deg[te] - deg[tr].mean()).mean())
    print(f"\nกฎ 1 — ทำนายมุมเป็นองศา")
    print(f"  มุม = {co[0]:+.2f} * log4 {co[1]:+.2f}")
    print(f"  ผิดเฉลี่ย {mae:.1f} องศา   (เดาค่าเฉลี่ยผิด {base:.1f} องศา  "
          f"-> ดีขึ้น {(1 - mae / base) * 100:.0f}%)")

    # ---------------------------------------------------------------- กฎ 2: โซน
    cuts = sorted((v - co[1]) / co[0] for v in edges)
    zt = np.searchsorted(edges, deg[te])
    zp = np.searchsorted(edges, pred)
    acc = float((zt == zp).mean())
    maj = float(max((zt == v).mean() for v in np.unique(zt)))
    print(f"\nกฎ 2 — แบ่งเป็น {len(zname)} โซน "
          f"(ขอบที่ {', '.join(f'{v:+.0f}' for v in edges)} องศา)")
    seq = zname if co[0] > 0 else zname[::-1]
    print(f"     log4 < {cuts[0]:+.3f} -> {seq[0]}")
    for i in range(len(cuts) - 1):
        print(f"     log4 {cuts[i]:+.3f} .. {cuts[i + 1]:+.3f} -> {seq[i + 1]}")
    print(f"     log4 > {cuts[-1]:+.3f} -> {seq[-1]}")
    print(f"  ถูก {acc:.0%}  (เดาโซนที่เจอบ่อยสุดได้ {maj:.0%})")
    w = max(len(v) for v in zname) + 5
    print(f"\n  {'จริง \\ ทาย':<12}" + "".join(f"{v:>{w}}" for v in zname))
    for i in range(len(zname)):
        row = [int(((zt == i) & (zp == j)).sum()) for j in range(len(zname))]
        print(f"  {zname[i]:<12}" + "".join(f"{v:>{w}}" for v in row)
              + f"   ({row[i] / max(sum(row), 1):.0%})")
    last = len(zname) - 1
    lr = int(((zt == 0) & (zp == last)).sum() + ((zt == last) & (zp == 0)).sum())
    print(f"\n  สลับซ้ายเป็นขวา (ผิดร้ายแรง) : {lr} เฟรม จาก {int(te.sum())} "
          f"({lr / max(int(te.sum()), 1):.1%})")

    # ---------------------------------------------------------------- เก็บกฎไว้ใช้สด
    # เทรนซ้ำด้วย **ทุกช่วง** สำหรับใช้งานจริง — ตัวเลขข้างบนคือผลที่วัดอย่างซื่อสัตย์
    # จากช่วงที่กันไว้ ส่วนกฎที่เอาไปใช้จริงควรได้เห็นข้อมูลให้มากที่สุด
    ca, *_ = np.linalg.lstsq(np.c_[x[keep], np.ones(int(keep.sum()))],
                             deg[keep], rcond=None)
    rule = DATA / f"_{a.name}_rule.json"
    import json
    rule.write_text(json.dumps({
        "pins": PINS, "slope": float(ca[0]), "intercept": float(ca[1]),
        "min_amp": a.min_amp, "smooth": a.smooth, "zones": edges,
        "zone_names": zname, "trained_on": names, "n_frames": int(keep.sum()),
        "holdout": names[ti], "mae_deg": mae, "baseline_deg": base,
        "zone_acc": acc, "zone_base": maj,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nเก็บกฎไว้ที่ {rule.name} (เทรนซ้ำด้วยทุกช่วง {int(keep.sum())} เฟรม: "
          f"มุม = {ca[0]:+.2f} * log4 {ca[1]:+.2f})")
    print(f"  ลองยืนดูผลสด ๆ ได้ที่:  python car/predict.py --port COM5")

    good = mae < base * 0.8 and acc > maj + 0.15
    print(f"""
สรุป: ตัวเลขเดียว ไม่มี ML ไม่มีพารามิเตอร์ซ่อน — พอดีเส้นตรง 2 ตัวจากชุดเทรนเท่านั้น
  {'ใช้ได้' if good else 'ยังไม่ชนะการเดาอย่างชัดเจน'} — มุมผิด {mae:.1f} องศา (เดา {base:.1f}) · โซนถูก {acc:.0%} (เดา {maj:.0%})
  การเกลี่ย {a.smooth} เฟรมหน่วงคำตอบราว {a.smooth / 2 / 15:.1f} วินาที ที่ 15 fps
  ถ้าต้องแม่นกว่านี้ ต้องใช้รูปคลื่นทั้งเส้น ไม่ใช่ยอดค่าเดียว -> car/train_sections.py""")


if __name__ == "__main__":
    main()
