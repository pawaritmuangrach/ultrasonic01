"""รวมแคชหลายรอบเป็นแคชเดียว — ใช้บนเครื่องเทรนที่ส่งข้อมูลขึ้นไปทีละส่วน

    MAP2_ROUND=r5 python car/merge_cache.py r5dev r5d      สร้างแคช r5 = r5dev + r5d

ผลต้องเหมือนการสร้างแคช r5 จากข้อมูลดิบทุกไบต์ (ตรวจด้วย --check กับแคชที่สร้างจากข้อมูลดิบ)
src (ดัชนีตระกูลชื่อชุด) ถูกคำนวณใหม่จากรายชื่อตระกูลที่เรียงแล้วของแคชรวม
เหมือนที่ mapdata2.build ทำ ไม่ใช่ต่อเลขเดิมของแต่ละแคช
"""
import json
import os
import sys

import numpy as np

import mapdata2 as MD


def merge(parts):
    caches = [MD.DATA / f"_map2_cache_{p}_{MD.GW}x{MD.GH}" for p in parts]
    metas = [json.loads((c / "meta.json").read_text(encoding="utf-8")) for c in caches]
    sets = [s for m in metas for s in m["sets"]]
    if sets != list(MD.SETS):
        raise SystemExit(f"ชุดที่รวมได้ {sets} ไม่ตรงกับ {MD.ROUND}: {list(MD.SETS)}")
    fam = sorted({s.rsplit("_s", 1)[0] for s in sets})
    N = sum(m["n"] for m in metas)
    MD.CACHE.mkdir(parents=True, exist_ok=True)
    counts = np.lib.format.open_memmap(MD.CACHE / "counts.npy", "w+", np.uint16, (N, 4, 871))
    dmm = np.lib.format.open_memmap(MD.CACHE / "dmm.npy", "w+", np.uint16, (N, MD.GH, MD.GW))
    sec = np.zeros(N, np.uint8)
    src = np.zeros(N, np.uint8)
    i = 0
    for c, m in zip(caches, metas):
        n = m["n"]
        counts[i:i + n] = np.load(c / "counts.npy", mmap_mode="r")
        dmm[i:i + n] = np.load(c / "dmm.npy", mmap_mode="r")
        sec[i:i + n] = np.load(c / "sec.npy")
        old = np.load(c / "src.npy")
        # แปลงดัชนีตระกูลเดิม -> ดัชนีในรายชื่อรวม (เฟรมที่ถูกข้ามมี sec=0 คงไว้ตามเดิม)
        remap = np.array([fam.index(f) for f in m["fam"]], np.uint8)
        src[i:i + n] = np.where(sec[i:i + n] > 0, remap[old], 0)
        i += n
    counts.flush()
    dmm.flush()
    np.save(MD.CACHE / "sec.npy", sec)
    np.save(MD.CACHE / "src.npy", src)
    meta = dict(metas[0], n=N, sets=sets, fam=fam)
    (MD.CACHE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"รวม {parts} -> {MD.CACHE.name} · {N:,} เฟรม · ตระกูล {fam}")


def check(other):
    """เทียบแคชที่รวมได้กับแคชอีกที่หนึ่ง (เช่น ที่สร้างจากข้อมูลดิบ) ทุกไฟล์ทุกไบต์"""
    bad = []
    for f in ("counts.npy", "dmm.npy", "sec.npy", "src.npy"):
        a = np.load(MD.CACHE / f, mmap_mode="r")
        b = np.load(os.path.join(other, f), mmap_mode="r")
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(f)
    print("ตรงกันทุกไบต์" if not bad else f"!! ไม่ตรง: {bad}")
    return not bad


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:                      # เขียนที่อื่นแทนที่แคชของรอบนี้ (ใช้ตอนทดสอบ)
        k = args.index("--out")
        from pathlib import Path
        MD.CACHE = Path(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--check" in args:
        k = args.index("--check")
        other = args[k + 1]
        args = args[:k] + args[k + 2:]
        merge(args)
        sys.exit(0 if check(other) else 1)
    merge(args)
