"""ชั้นข้อมูลของโมเดลแผนที่ depth

`train_nn.py` ทำนาย **มุมเดียว** ไฟล์นี้เตรียมข้อมูลให้ทำนาย **ภาพ depth ทั้งภาพ**
คือย่อภาพจากกล้อง 320x240 เหลือ 40x30 แล้วเก็บคู่กับคลื่นดิบ 4 ช่อง

ย่อด้วยบล็อก 8x8 ไม่ใช่เพราะสวย แต่เพราะกำลังแยกแยะของอาเรย์นี้หยาบกว่านั้นมาก
(ระยะห่างเสา 12 ซม. ที่ 40 kHz แยกได้ราว 4 องศา = ราว 7 พิกเซลของภาพ 40 ช่อง)
ทำนายละเอียดกว่านี้คือทำนายลม
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = DATA / "_map_cache"

PINS = (34, 33, 32, 35)          # เรียงตามตำแหน่งจริงบนบอร์ด ซ้ายสุด -> ขวาสุด
GW, GH, BLK = 40, 30, 8          # ภาพที่ทำนาย 40x30 ย่อจาก 320x240
NEAR_CM, FAR_CM = 40.0, 200.0    # ช่วงที่เซ็นเซอร์เห็นจริง นอกช่วงนี้ = ไม่มีข้อมูล
FOV_H_DEG = 58.4

# ชุดที่มี depth ครบและใช้เทรนได้ · เลข _sN ท้ายชื่อคือช่วงที่ใช้แบ่ง train/test
SETS = ([f"grid_s{i}" for i in range(1, 6)]
        + [f"walk_s{i}" for i in range(1, 6)]
        + [f"walk2_s{i}" for i in range(1, 4)])


def shrink(depth_mm):
    """ย่อภาพ depth เป็นระยะหน่วย mm ขนาด GH x GW · 0 = ไม่มีวัตถุในช่วงที่เห็น

    ใช้ **มัธยฐานเฉพาะพิกเซลที่อ่านได้** ในแต่ละบล็อก ไม่ใช่ค่าเฉลี่ยทั้งบล็อก
    เพราะกล้องคืน 0 ตรงที่อ่านไม่ได้ (ขอบวัตถุ ผิวมันวาว ที่ที่ IR ไปไม่ถึง)
    ถ้าเฉลี่ยรวม 0 เข้าไปด้วย ขอบของคนจะถูกดึงให้ดูใกล้กว่าความจริง
    """
    d = depth_mm.reshape(GH, BLK, GW, BLK).transpose(0, 2, 1, 3)
    mm = d.reshape(GH, GW, BLK * BLK).astype(np.float32)
    ok = (mm >= NEAR_CM * 10) & (mm <= FAR_CM * 10)
    n = ok.sum(2)
    # ดันค่าที่ใช้ไม่ได้ไปท้ายแถวด้วย inf แล้วหยิบตัวกลางของ 'เฉพาะที่ใช้ได้'
    # เลี่ยง nanmedian ที่ทั้งช้าและเตือนรัว ๆ เวลาทั้งบล็อกอ่านไม่ได้
    mm[~ok] = np.inf
    mm.sort(axis=2)
    med = np.take_along_axis(mm, np.minimum(n // 2, BLK * BLK - 1)[:, :, None], 2)[:, :, 0]
    keep = n >= (BLK * BLK) // 4     # ต้องอ่านได้อย่างน้อย 1 ใน 4 ของบล็อก
    return np.where(keep, med, 0.0).astype(np.uint16)


def frame_files(name):
    """คู่ (คลื่น, ภาพ depth) ของชุดหนึ่ง เรียงตามเวลา"""
    d = DATA / name
    out = []
    for u in sorted(d.glob("us_*.npz")):
        p = d / f"depth_{u.stem.split('_')[1]}.png"
        if p.exists():
            out.append((u, p))
    return out


def read_counts(z):
    """เรียงช่องให้เป็นซ้าย->ขวาเสมอ ไม่ว่าไฟล์จะบันทึกมาลำดับไหน"""
    at = {int(v): k for k, v in enumerate(z["pins"])}
    return np.stack([z["counts"][at[p]] for p in PINS])


def build(names=None, verbose=True):
    """อ่านข้อมูลดิบทั้งหมดครั้งเดียว เก็บเป็น .npy ให้เทรนรอบต่อ ๆ ไปเปิดได้ทันที

    เก็บแยกไฟล์แทน npz ก้อนเดียว เพราะ counts ก้อนใหญ่ (~200 MB) จะได้ memmap
    ตอนเทรนโดยไม่ต้องโหลดเข้าแรมทั้งก้อน
    """
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
        if c.shape[1] != 871:            # ชุดเก่าบางชุดยาวไม่เท่ากัน
            continue
        counts[i] = c
        dmm[i] = shrink(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        sec[i] = int(name.rsplit("_s", 1)[1])
        src[i] = fam.index(name.rsplit("_s", 1)[0])
        if verbose and i % 2000 == 0:
            print(f"  {i:6,}/{N:,}", flush=True)
    counts.flush(); dmm.flush()
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
    """คืน (counts, dmm, sec, src, meta) — counts เป็น memmap ถ้าไม่สั่งเป็นอย่างอื่น"""
    if not (CACHE / "meta.json").exists():
        raise SystemExit(f"ยังไม่มีแคชที่ {CACHE} — สร้างก่อนด้วย:\n"
                         f"  python car/mapdata.py")
    m = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    mode = "r" if mmap else None
    return (np.load(CACHE / "counts.npy", mmap_mode=mode),
            np.load(CACHE / "dmm.npy"),
            np.load(CACHE / "sec.npy"), np.load(CACHE / "src.npy"), m)


if __name__ == "__main__":
    build()


def pos_from_map(d):
    """ตำแหน่งเป้าจากภาพย่อ — คืน (องศา, ระยะ cm) หรือ None

    วัดจาก **จุดศูนย์กลางของผิวหน้าที่ใกล้ที่สุด** ไม่ใช่พิกเซลที่ใกล้ที่สุดจุดเดียว
    เคยพลาดตรงนี้มาแล้วตอนทำ rules.py: การหยิบจุดใกล้สุดจุดเดียวทำให้ป้ายกระโดด
    ไปมาเวลาคนกางแขน จนโมเดลเรียนอะไรไม่ได้เลย
    """
    m = d > 0
    if int(m.sum()) < 8:
        return None
    near = float(np.percentile(d[m], 5)) / 10.0
    sel = m & (d < (near + 40.0) * 10)      # เอาเฉพาะผิวหน้า หนาไม่เกิน 40 ซม.
    if int(sel.sum()) < 8:
        return None
    cols = np.nonzero(sel)[1]
    return float((cols.mean() / (GW - 1) - 0.5) * FOV_H_DEG), near
