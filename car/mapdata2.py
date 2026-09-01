"""ชั้นข้อมูลรอบสอง — อาเรย์สี่เหลี่ยม · ตาราง 80x60

ต่างจาก mapdata.py (รอบแรก) สองอย่าง:
  * ลำดับช่องเป็น TL TR BL BR ตามตำแหน่งจริงบน plate_mini ไม่ใช่ซ้ายไปขวา
  * ตารางละเอียดขึ้นเป็น 80x60 (ย่อจาก 320x240 ด้วยบล็อก 4x4)

**80x60 ไม่ได้เพิ่มความละเอียดจริง** เบสไลน์ 110 มม. ที่ 40 kHz แยกได้ 4.5 องศา
ซึ่งกินราว 6 ช่องของตารางนี้ ที่ละเอียดขึ้นคือความเนียนของภาพ ไม่ใช่ข้อมูล
สิ่งที่หวังว่าจะได้เพิ่มจริงคือ **ท่าทาง** ซึ่งเป็นการเปลี่ยนแปลงระดับ 40-60 องศา
"""
import json
from pathlib import Path

import numpy as np

from rig2 import PINS, FOV_H_DEG, FOV_V_DEG

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = DATA / "_map2_cache"

GW, GH, BLK = 80, 60, 4
NEAR_CM, FAR_CM = 40.0, 200.0

SETS = ([f"pose_s{i}" for i in range(1, 6)] + [f"free_s{i}" for i in range(1, 4)])


def shrink(depth_mm):
    """ย่อภาพ depth เป็นระยะหน่วย mm ขนาด GH x GW · 0 = ไม่มีวัตถุในช่วงที่เห็น

    ใช้มัธยฐานเฉพาะพิกเซลที่อ่านได้ในแต่ละบล็อก ไม่ใช่ค่าเฉลี่ยทั้งบล็อก
    เพราะกล้องคืน 0 ตรงที่อ่านไม่ได้ ถ้าเฉลี่ยรวม 0 เข้าไปด้วย
    ขอบของคนจะถูกดึงให้ดูใกล้กว่าความจริง
    """
    d = depth_mm.reshape(GH, BLK, GW, BLK).transpose(0, 2, 1, 3)
    mm = d.reshape(GH, GW, BLK * BLK).astype(np.float32)
    ok = (mm >= NEAR_CM * 10) & (mm <= FAR_CM * 10)
    n = ok.sum(2)
    mm[~ok] = np.inf
    mm.sort(axis=2)
    med = np.take_along_axis(mm, np.minimum(n // 2, BLK * BLK - 1)[:, :, None], 2)[:, :, 0]
    keep = n >= (BLK * BLK) // 4
    return np.where(keep, med, 0.0).astype(np.uint16)


def frame_files(name):
    d = DATA / name
    out = []
    for u in sorted(d.glob("us_*.npz")):
        p = d / f"depth_{u.stem.split('_')[1]}.png"
        if p.exists():
            out.append((u, p))
    return out


def read_counts(z):
    """เรียงช่องเป็น TL TR BL BR เสมอ ไม่ว่าไฟล์จะบันทึกมาลำดับไหน"""
    at = {int(v): k for k, v in enumerate(z["pins"])}
    return np.stack([z["counts"][at[p]] for p in PINS])


def shape_stats(d):
    """ตัวเลขบอกรูปร่างของเงาในภาพย่อ — ใช้ตรวจว่าท่าทางแยกจากตำแหน่งจริงไหม

    คืน (มุมองศา, ระยะ cm, กว้างกี่ช่อง, สูงกี่ช่อง, จำนวนช่องที่มีวัตถุ)
    กว้าง/สูง วัดจากช่วงที่ครอบคลุม 90% ของช่องที่มีวัตถุ ไม่ใช่ค่าสุดขั้ว
    เพราะพิกเซลหลงตัวเดียวไม่ควรทำให้ตัวเลขกระโดด
    """
    m = d > 0
    if int(m.sum()) < 20:
        return None
    ys, xs = np.nonzero(m)
    near = float(np.percentile(d[m], 5)) / 10.0
    sel = m & (d < (near + 40.0) * 10)          # เอาเฉพาะผิวหน้า หนาไม่เกิน 40 ซม.
    if int(sel.sum()) < 20:
        return None
    sy, sx = np.nonzero(sel)
    deg = float((sx.mean() / (GW - 1) - 0.5) * FOV_H_DEG)
    w = float(np.percentile(sx, 95) - np.percentile(sx, 5))
    h = float(np.percentile(sy, 95) - np.percentile(sy, 5))
    return deg, near, w, h, float(sel.sum())


def build(names=None, verbose=True):
    import cv2
    names = list(names or SETS)
    CACHE.mkdir(parents=True, exist_ok=True)
    pairs = [(n, f) for n in names for f in frame_files(n)]
    if not pairs:
        raise SystemExit("ไม่พบข้อมูลดิบเลย")
    N = len(pairs)
    if verbose:
        print(f"พบ {N:,} เฟรม จาก {len(names)} ชุด — กำลังอ่าน", flush=True)
    counts = np.lib.format.open_memmap(CACHE / "counts.npy", "w+", np.uint16, (N, 4, 871))
    dmm = np.lib.format.open_memmap(CACHE / "dmm.npy", "w+", np.uint16, (N, GH, GW))
    sec = np.zeros(N, np.uint8)
    src = np.zeros(N, np.uint8)
    fam = sorted({n.rsplit("_s", 1)[0] for n in names})
    for i, (name, (u, p)) in enumerate(pairs):
        z = np.load(u)
        c = read_counts(z)
        if c.shape[1] != 871:
            continue
        counts[i] = c
        dmm[i] = shrink(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        sec[i] = int(name.rsplit("_s", 1)[1])
        src[i] = fam.index(name.rsplit("_s", 1)[0])
        if verbose and i % 2000 == 0:
            print(f"  {i:6,}/{N:,}", flush=True)
    counts.flush()
    dmm.flush()
    np.save(CACHE / "sec.npy", sec)
    np.save(CACHE / "src.npy", src)
    (CACHE / "meta.json").write_text(json.dumps(
        {"n": N, "sets": names, "fam": fam, "grid": [GW, GH], "blk": BLK,
         "near_cm": NEAR_CM, "far_cm": FAR_CM, "pins": list(PINS)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"เก็บแคชที่ {CACHE}")
    return N


def load(mmap=True):
    if not (CACHE / "meta.json").exists():
        raise SystemExit(f"ยังไม่มีแคชที่ {CACHE} — สร้างก่อนด้วย:\n"
                         f"  python car/mapdata2.py")
    m = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    return (np.load(CACHE / "counts.npy", mmap_mode="r" if mmap else None),
            np.load(CACHE / "dmm.npy"),
            np.load(CACHE / "sec.npy"), np.load(CACHE / "src.npy"), m)


if __name__ == "__main__":
    build()


def shadow_height(mask, lo=5.0, hi=95.0):
    """ความสูงของเงาเป็นจำนวนช่อง — ตัวแทนของ "ท่าทาง"

    ย่อตัวแล้วเตี้ยลง กางแขนยกแขนแล้วสูงขึ้น เป็นตัวเลขเดียวที่จับท่าทางได้ตรงสุด

    **ใช้เปอร์เซ็นไทล์ 5-95 ของแถวที่มีวัตถุ ไม่ใช่แถวบนสุดลบแถวล่างสุด**
    ถ้าใช้ค่าสุดขั้ว จุดหลงจุดเดียวที่ขอบบนกับขอบล่างก็ดันความสูงเต็มจอทันที
    แล้วตัวเลขจะอิ่มตัวอยู่ที่ค่าสูงสุดแทบทุกเฟรม จนแยกท่าทางไม่ออกเลย
    ถ่วงน้ำหนักด้วยจำนวนช่องในแต่ละแถว แถวที่มีวัตถุจาง ๆ จึงไม่มีน้ำหนักเท่าลำตัว
    """
    w = mask.sum(1).astype(np.float64)
    tot = w.sum()
    if tot < 1:
        return 0.0
    c = np.cumsum(w) / tot * 100.0
    ys = np.arange(len(w))
    a = float(np.interp(lo, c, ys))
    b = float(np.interp(hi, c, ys))
    return max(b - a, 0.0)
