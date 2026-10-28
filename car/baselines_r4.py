"""คู่เทียบที่ไม่ใช่ deep learning — ใช้วัดว่าโครงข่ายประสาทได้อะไรเพิ่มจริง

    python car/baselines_r4.py          ผลลง car/data/r4_runs/baseline_<ชื่อ>.json

แบ่งข้อมูลแบบเดียวกับ train_r4.py ทุกอย่าง (เทรน s1-s4 · เลือก s5 · รายงาน s6)
และเลือกค่าตั้ง (alpha, k, จุดตัด) จาก s5 เท่านั้น

  mean   ภาพเฉลี่ยของชุดเทรน ไม่ใช้เสียงเลย = เพดานล่างที่ทุกวิธีต้องชนะ
  ridge  ถดถอยเชิงเส้น (scikit-learn) จากซองคลื่นไปทุกช่องของภาพ
  knn    หาเฟรมเทรนที่คลื่นคล้ายที่สุด k เฟรมแล้วเอาภาพมาเฉลี่ย

คุณลักษณะ (feature) = ซอง |คลื่น - จุดกลาง| ของการยิงล่าสุด บีบลอการิทึม
เฉลี่ยทีละ 8 จุด (ทุก ~1.4 ซม. ของระยะ) ทั้ง 4 ช่อง = 436 ตัวเลขต่อเฟรม
"""
import os
os.environ.setdefault("MAP2_ROUND", "r4")

import json         # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402

import mapdata2 as MD                              # noqa: E402
from mapdata2 import NEAR_CM, FAR_CM               # noqa: E402
from train_r4 import best_thr, metrics, OUT, split_mask   # noqa: E402
import argparse     # noqa: E402

BIN = 8
PREFIX = "baseline"
HAS_TEST = True


def features(counts, dc):
    x = np.abs(counts.astype(np.float32) - dc)
    n = x.shape[-1] // BIN * BIN
    x = x[..., :n].reshape(*x.shape[:-1], n // BIN, BIN).mean(-1)
    return np.log1p(x).reshape(len(x), -1)


def run(name, P_v, D_v, P_t, D_t, occ_v, dep_v, occ_t, dep_t, extra, t0):
    thr = best_thr(P_v, occ_v)
    res = {"name": f"{PREFIX}_{name}", "val": metrics(P_v, D_v, occ_v, dep_v, thr),
           "test": metrics(P_t, D_t, occ_t, dep_t, thr) if HAS_TEST else None,
           "minutes": (time.time() - t0) / 60, "args": extra, "round": MD.ROUND}
    (OUT / f"{PREFIX}_{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    v, t = res["val"], res["test"]
    print(f"  {name:6} เลือก IoU {v['iou']:.3f} ระยะ {v['mae_cm']:5.1f} R2 {v['h_r2']:+.3f}"
          + (f" · รายงาน IoU {t['iou']:.3f} ระยะ {t['mae_cm']:5.1f} R2 {t['h_r2']:+.3f}" if t else "")
          + f" · {extra}", flush=True)


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="1,2,3,4")
    ap.add_argument("--val", default="5")
    ap.add_argument("--test", default="6", help="ว่าง = ไม่มีชุดรายงาน (รอบเลือกสูตร)")
    ap.add_argument("--prefix", default="baseline")
    ap.add_argument("--target", default="scene", choices=("scene", "person"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counts, dmm, sec, src, meta = MD.load()
    counts = np.asarray(counts)
    if a.target == "person":
        from person_target import person_dmm
        dmm = person_dmm(dmm, sec, src)
    tr, va = (np.nonzero(split_mask(x, sec, src, meta["fam"]))[0] for x in (a.train, a.val))
    te = np.nonzero(split_mask(a.test, sec, src, meta["fam"]))[0] if a.test else va
    global PREFIX, HAS_TEST
    HAS_TEST = bool(a.test)
    PREFIX = a.prefix
    dc = float(np.median(counts[tr[::40]]))
    occ = dmm > 0
    dep = dmm.astype(np.float32) / 10.0
    z = np.clip((dep - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1)
    occ_v, dep_v, occ_t, dep_t = occ[va], dep[va], occ[te], dep[te]
    print(f"เทรน {len(tr):,} · เลือก {len(va):,} · รายงาน {len(te):,} เฟรม")

    # ---- ภาพเฉลี่ย
    t0 = time.time()
    mo = occ[tr].mean(0)
    md = np.where(occ[tr].sum(0) > 0, (dep[tr] * occ[tr]).sum(0) / np.maximum(occ[tr].sum(0), 1),
                  (NEAR_CM + FAR_CM) / 2)
    rep = lambda a_, n: np.repeat(a_[None], n, 0)                      # noqa: E731
    run("mean", rep(mo, len(va)), rep(md, len(va)), rep(mo, len(te)), rep(md, len(te)),
        occ_v, dep_v, occ_t, dep_t, "ไม่ใช้เสียง", t0)

    X = features(counts, dc)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    Yo = occ.reshape(len(occ), -1).astype(np.float32)
    # ระยะในช่องว่างไม่มีความหมาย เติมด้วยระยะเฉลี่ยของช่องนั้น จะได้ไม่ดึงเส้นถดถอยไปทางศูนย์
    Yz = np.where(occ, z, ((md - NEAR_CM) / (FAR_CM - NEAR_CM))[None]).reshape(len(z), -1)
    shape = (GH_, GW_) = occ.shape[1:]

    # ---- ridge: เลือก alpha จาก s5
    t0 = time.time()
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        mo_r = Ridge(alpha=alpha).fit(Xs[tr], Yo[tr])
        Pv = np.clip(mo_r.predict(Xs[va]), 0, 1).reshape(-1, *shape)
        sc_v = metrics(Pv, rep(md, len(va)), occ_v, dep_v, best_thr(Pv, occ_v))["iou"]
        if best is None or sc_v > best[0]:
            best = (sc_v, alpha)
    alpha = best[1]
    mo_r = Ridge(alpha=alpha).fit(Xs[tr], Yo[tr])
    md_r = Ridge(alpha=alpha).fit(Xs[tr], Yz[tr])
    pz = lambda idx: NEAR_CM + (FAR_CM - NEAR_CM) * np.clip(md_r.predict(Xs[idx]), 0, 1)  # noqa: E731
    run("ridge", np.clip(mo_r.predict(Xs[va]), 0, 1).reshape(-1, *shape), pz(va).reshape(-1, *shape),
        np.clip(mo_r.predict(Xs[te]), 0, 1).reshape(-1, *shape), pz(te).reshape(-1, *shape),
        occ_v, dep_v, occ_t, dep_t, f"alpha={alpha:g}", t0)

    # ---- kNN บน PCA 50 มิติ: เลือก k จาก s5
    t0 = time.time()
    pca = PCA(50, random_state=0).fit(Xs[tr])
    Ztr, Zva, Zte = pca.transform(Xs[tr]), pca.transform(Xs[va]), pca.transform(Xs[te])
    nn_ = NearestNeighbors(n_neighbors=100).fit(Ztr)
    _, Iv = nn_.kneighbors(Zva)
    _, It = nn_.kneighbors(Zte)
    o_tr, d_tr = occ[tr].astype(np.float32), dep[tr]

    def knn_pred(I, k):
        o = o_tr[I[:, :k]]
        P = o.mean(1)
        D = (d_tr[I[:, :k]] * o).sum(1) / np.maximum(o.sum(1), 1e-6)
        return P, np.where(o.sum(1) > 0, D, md[None])

    best = None
    for k in (5, 10, 20, 50, 100):
        Pv, _ = knn_pred(Iv, k)
        sc_v = metrics(Pv, md[None].repeat(len(va), 0), occ_v, dep_v, best_thr(Pv, occ_v))["iou"]
        if best is None or sc_v > best[0]:
            best = (sc_v, k)
    k = best[1]
    Pv, Dv = knn_pred(Iv, k)
    Pt, Dt = knn_pred(It, k)
    run("knn", Pv, Dv, Pt, Dt, occ_v, dep_v, occ_t, dep_t, f"k={k} PCA50", t0)


if __name__ == "__main__":
    main()
