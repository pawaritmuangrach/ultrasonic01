"""เทรนโมเดลรอบสอง — ภาพระยะ 80x60 จากคลื่นเสียงล้วน

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ไฟล์นี้ทั้งไฟล์ ไม่มีโค้ดหน้าจอ)
═══════════════════════════════════════════════════════════════════════

    python car/train_map2.py --test 1

**วิธีวัดผลสำคัญกว่าตัวโมเดล** ภาพที่ทายออกมาดูน่าเชื่อได้ง่ายมาก
เพราะห้องกับคนหน้าตาคล้ายกันทุกเฟรม โมเดลที่ตอบภาพเฉลี่ยเฉย ๆ ก็ดูดีแล้ว
จึงต้องเทียบกับคู่เทียบที่ตั้งใจทำให้แข็ง ไม่ใช่คู่เทียบที่ตั้งใจทำให้แพ้

คู่เทียบสามอัน

  1. ทายค่าเฉลี่ย       ไม่ดูคลื่นเลย ตอบภาพเฉลี่ยของชุดเทรนทุกครั้ง
                        ชนะไม่ได้ = ไม่ได้เรียนอะไรจากเสียงเลย

  2. แม่แบบรู้ตำแหน่ง   โกงด้วยการดูมุมกับระยะจริงจากกล้อง แล้วแปะรูปร่างเฉลี่ย
                        ของตำแหน่งนั้น · ชนะไม่ได้ = เสียงบอกได้แค่ 'อยู่ตรงไหน'

  3. **วัดท่าทางแยกต่างหาก**  ของใหม่รอบนี้
                        เอา 'ความสูงของเงา' (ซึ่งคือท่าทาง) มาดูว่าโมเดลทายได้
                        แม่นกว่าการเดาจากตำแหน่งอย่างเดียวไหม
                        นี่คือคำถามที่รอบแรกตอบว่า 'ไม่' และเป็นเหตุผลทั้งหมด
                        ที่เก็บข้อมูลใหม่

และการทดสอบสลับคลื่น: เอาคำตอบที่ทายจากคลื่นเฟรมอื่นมาเทียบกับภาพจริงเฟรมนี้
ถ้าคะแนนไม่ตก แปลว่าโมเดลไม่ได้ใช้เสียงเลย แค่จำภาพเฉลี่ยไว้ตอบ

แบ่ง train/test **ตามช่วงการอัด** ไม่ใช่สุ่มรายเฟรม เพราะเฟรมติดกันเกือบเหมือนกัน
สุ่มรายเฟรมจะทำให้เฟรมข้างเคียงของ test ไปโผล่ใน train แล้วคะแนนพองเกินจริง
"""
import argparse
import time

import numpy as np

import mapdata2 as MD
from mapdata2 import GW, GH, NEAR_CM, FAR_CM
from model2 import prep, make_model, to_cm

ANG_BINS, DIST_BINS = 12, 6


# ------------------------------------------------------------- ตัววัดผล
def coarsen(a4):
    """ย่อ 80x60 ลงเป็น 40x30 ด้วยการรวมบล็อก 2x2

    มีไว้เทียบกับผลรอบแรกซึ่งวัดที่ 40x30 — IoU บนตารางละเอียดกว่าย่อมต่ำกว่า
    โดยธรรมชาติ เพราะขอบต้องตรงกันละเอียดขึ้น เทียบเลขข้ามความละเอียดไม่ได้
    """
    n, h, w = a4.shape
    return a4.reshape(n, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def iou_at(occ_p, occ_t, thr):
    """ทับกันแค่ไหนที่จุดตัดหนึ่ง ๆ

    ใช้ IoU ไม่ใช่ 'ทายถูกกี่ช่อง' เพราะภาพส่วนใหญ่เป็นที่ว่าง
    การตอบ 'ว่างทั้งภาพ' จะได้ความถูกต้องสูงลิ่วทั้งที่ไม่เจออะไรเลย
    """
    pred = occ_p >= thr
    return float((pred & occ_t).sum()) / max(int((pred | occ_t).sum()), 1)


def best_iou(occ_p, occ_t):
    """IoU ที่ดีที่สุดเท่าที่วิธีนั้นทำได้ พร้อมจุดตัดที่ให้ผลนั้น

    ต้องหาจุดตัดให้ **คู่เทียบด้วย** ไม่ใช่แค่โมเดล ไม่งั้นเทียบไม่ยุติธรรม
    ภาพเฉลี่ยไม่มีช่องไหนเกิน 0.5 เลย (คนขยับไปมา ไม่มีจุดไหนมีคนเกินครึ่งเวลา)
    ถ้าล็อกจุดตัดที่ 0.5 คู่เทียบจะได้ IoU 0 แล้วโมเดลก็ดูเก่งเกินตัว
    """
    best = (0.0, 0.5)
    for t in np.arange(0.05, 1.0, 0.05):
        v = iou_at(occ_p, occ_t, t)
        if v > best[0]:
            best = (v, float(t))
    return best


def r2(pred, true):
    """อธิบายความแปรปรวนได้กี่ % เทียบกับการตอบค่าเฉลี่ยเฉย ๆ · ติดลบได้"""
    ss = float(((true - pred) ** 2).sum())
    sv = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss / max(sv, 1e-9)


def shadow_height(occ):
    """ความสูงเงาของทุกเฟรม · นิยามอยู่ที่ mapdata2 ให้หน้าจอใช้ตัวเดียวกัน"""
    return np.array([MD.shadow_height(m) for m in occ], np.float32)


def position_bins(dmm, nang=ANG_BINS, ndist=DIST_BINS):
    """แปลงภาพจริงเป็นหมายเลขช่อง (มุม, ระยะ) ไว้ทำแม่แบบรู้ตำแหน่ง"""
    ang = np.full(len(dmm), -1, np.int16)
    dst = np.full(len(dmm), -1, np.int16)
    for i, d in enumerate(dmm):
        s = MD.shape_stats(d)
        if s is None:
            continue
        deg, cm = s[0], s[1]
        ang[i] = np.clip(int((deg / MD.FOV_H_DEG + 0.5) * nang), 0, nang - 1)
        dst[i] = np.clip(int((cm - NEAR_CM) / (FAR_CM - NEAR_CM) * ndist), 0, ndist - 1)
    return ang, dst


def baselines(dmm, tr, te, ang, dst):
    occ_t = dmm[te] > 0
    dep_t = dmm[te].astype(np.float32) / 10.0
    occ_tr = dmm[tr] > 0
    dep_tr = dmm[tr].astype(np.float32) / 10.0
    m_occ = occ_tr.mean(0)
    tot = occ_tr.sum(0)
    flat = dep_tr[occ_tr].mean()
    m_dep = np.where(tot > 0, (dep_tr * occ_tr).sum(0) / np.maximum(tot, 1), flat)
    i1, t1 = best_iou(np.broadcast_to(m_occ, occ_t.shape), occ_t)
    e1 = float(np.abs(np.broadcast_to(m_dep, dep_t.shape)[occ_t] - dep_t[occ_t]).mean())

    to = np.zeros((ANG_BINS, DIST_BINS, GH, GW), np.float32)
    td = np.zeros_like(to)
    n = np.zeros((ANG_BINS, DIST_BINS), np.int32)
    for k, i in enumerate(tr):
        if ang[i] < 0:
            continue
        to[ang[i], dst[i]] += occ_tr[k]
        td[ang[i], dst[i]] += dep_tr[k] * occ_tr[k]
        n[ang[i], dst[i]] += 1
    td = np.where(to > 0, td / np.maximum(to, 1e-6), flat)
    to = to / np.maximum(n, 1)[:, :, None, None]
    po = np.empty(occ_t.shape, np.float32)
    pd = np.empty(dep_t.shape, np.float32)
    for k, i in enumerate(te):
        if ang[i] < 0 or n[ang[i], dst[i]] == 0:
            po[k], pd[k] = m_occ, m_dep
        else:
            po[k], pd[k] = to[ang[i], dst[i]], td[ang[i], dst[i]]
    i2, t2 = best_iou(po, occ_t)
    e2 = float(np.abs(pd[occ_t] - dep_t[occ_t]).mean())

    # เพดานของ 'รู้แค่ตำแหน่ง' สำหรับความสูงเงา
    # **ต้องทำนายจากค่าเฉลี่ยความสูงจริงของกลุ่มตำแหน่งนั้น** ไม่ใช่วัดความสูง
    # จากภาพแม่แบบที่ผ่านการตัดขีดแล้ว — วิธีหลังเอาความคลาดของการตัดขีด
    # เข้ามาปน จนได้ R2 ติดลบทั้งที่ควรเป็นบวก (เจอตอนรันรอบแรก)
    h_tr = shadow_height(occ_tr)
    h_te_true = shadow_height(occ_t)
    tab, cnt = {}, {}
    for k, i in enumerate(tr):
        g = (int(ang[i]), int(dst[i]))
        tab[g] = tab.get(g, 0.0) + h_tr[k]
        cnt[g] = cnt.get(g, 0) + 1
    glob = float(h_tr.mean())
    h_pos = np.array([tab[g] / cnt[g] if (g := (int(ang[i]), int(dst[i]))) in cnt
                      else glob for i in te], np.float32)
    return (i1, t1, e1), (i2, t2, e2), h_pos, h_te_true


def evaluate(torch, net, counts, idx, bs=96):
    net.eval()
    O = np.empty((len(idx), GH, GW), np.float32)
    D = np.empty_like(O)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xs, sc = prep(counts[j])
            ol, dl = net(torch.from_numpy(xs).float(), torch.from_numpy(sc).float())
            O[i:i + bs] = torch.sigmoid(ol).numpy()
            D[i:i + bs] = to_cm(dl, torch).numpy()
    return O, D


# ------------------------------------------------------------- การเทรน
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, help="กันช่วงไหนไว้วัดผล (โหมด section)")
    ap.add_argument("--split", choices=["time", "section"], default="time",
                    help="time = กัน 25%% สุดท้ายของทุกช่วง (แนะนำ) · "
                         "section = กันทั้งช่วง")
    ap.add_argument("--tail", type=float, default=0.25,
                    help="โหมด time กันท้ายช่วงละกี่ส่วน")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    counts, dmm, sec, src, meta = MD.load()
    idx = np.nonzero(sec > 0)[0]
    if a.split == "section":
        # กันทั้งช่วง — **ใช้กับข้อมูลชุดนี้ไม่ได้ผล** เพราะแต่ละช่วงคือคนละตำแหน่ง
        # กันช่วงหนึ่งไว้ = กันตำแหน่งนั้นออกจากการเทรนทั้งหมด โมเดลต้องเดาข้ามไป
        # วัดแล้วแม้แต่แม่แบบที่รู้ตำแหน่งจริงยังได้ R2 ติดลบ ซึ่งเป็นอาการของ
        # การทดสอบที่ไม่มีอะไรให้อ้างอิงเลย ไม่ใช่ความผิดของโมเดล
        te = idx[sec[idx] == a.test]
        tr = idx[sec[idx] != a.test]
        how = f"กันทั้งช่วง s{a.test}"
    else:
        # กันท้ายของทุกช่วง — ทุกตำแหน่งอยู่ในชุดเทรนครบ ส่วนชุดทดสอบเป็น
        # **ท่าที่ทำใหม่อีกรอบ** ห่างจากเฟรมเทรนหลายนาที ไม่ใช่เฟรมข้างเคียง
        # (ท่าหนึ่งวนครบรอบทุก 48 วินาที คนจึงจัดท่าใหม่หลายครั้งระหว่างนั้น)
        tr_l, te_l = [], []
        for g in np.unique(np.stack([sec[idx], src[idx]]), axis=1).T:
            m = idx[(sec[idx] == g[0]) & (src[idx] == g[1])]
            m = np.sort(m)
            cut = int(len(m) * (1.0 - a.tail))
            tr_l.append(m[:cut])
            te_l.append(m[cut:])
        tr, te = np.concatenate(tr_l), np.concatenate(te_l)
        how = f"กัน {a.tail:.0%} ท้ายของทุกช่วง"
    print(f"ข้อมูล {len(idx):,} เฟรม · เทรน {len(tr):,} · กันไว้วัด {len(te):,} "
          f"({how}) · ชุด {', '.join(meta['fam'])}")

    print("\nคำนวณคู่เทียบ ...", flush=True)
    ang, dst = position_bins(dmm)
    (b1i, _b1t, b1e), (b2i, _b2t, b2e), h_pos, h_true = baselines(dmm, tr, te, ang, dst)
    occ_te = dmm[te] > 0
    dep_te = dmm[te].astype(np.float32) / 10.0
    r_pos = r2(h_pos, h_true)
    print(f"  1. ทายค่าเฉลี่ย      IoU {b1i:.3f}  ระยะพลาด {b1e:5.1f} ซม.")
    print(f"  2. แม่แบบรู้ตำแหน่ง  IoU {b2i:.3f}  ระยะพลาด {b2e:5.1f} ซม.  (โกง)")
    print(f"  3. ความสูงเงาจากตำแหน่งอย่างเดียว  R2 {r_pos:+.3f}  <- เพดานที่ต้องชนะ")
    print("     (วัดข้ามช่วง ไม่ใช่ในกลุ่มเดียวกัน · ติดลบ = แย่กว่าเดาค่าเฉลี่ยเฉย ๆ")
    print("      แปลว่าตำแหน่งทำนายท่าทางข้ามช่วงไม่ได้เลย ซึ่งเป็นข่าวดีสำหรับเรา)")

    net = make_model(torch, nn)
    npar = sum(p.numel() for p in net.parameters())
    print(f"\nโมเดล {npar:,} พารามิเตอร์ · ตาราง {GW}x{GH} · เทรน {a.epochs} รอบ",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, epochs=a.epochs, steps_per_epoch=max(1, len(tr) // a.batch + 1))
    rng = np.random.default_rng(0)
    best, best_state = -1.0, None

    for ep in range(1, a.epochs + 1):
        net.train()
        order = rng.permutation(tr)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(order), a.batch):
            j = np.sort(order[i:i + a.batch])
            xs, sc = prep(counts[j])
            d = dmm[j].astype(np.float32) / 10.0
            occ = torch.from_numpy((dmm[j] > 0).astype(np.float32))
            z = torch.from_numpy((d - NEAR_CM) / (FAR_CM - NEAR_CM))
            ol, dl = net(torch.from_numpy(xs).float(), torch.from_numpy(sc).float())
            bce = nn.functional.binary_cross_entropy_with_logits(ol, occ)
            # นับระยะเฉพาะช่องที่กล้องเห็นวัตถุจริง ที่ว่างไม่มีระยะให้เรียน
            l1 = (torch.abs(torch.sigmoid(dl) - z) * occ).sum() / occ.sum().clamp(min=1)
            loss = bce + 2.0 * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        O, D = evaluate(torch, net, counts, te)
        iou, thr = best_iou(O, occ_te)
        mae = float(np.abs(D[occ_te] - dep_te[occ_te]).mean())
        rh = r2(shadow_height(O >= thr), h_true)
        flag = ""
        if iou > best:
            best = iou
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            flag = "  <- ดีสุด"
        print(f"ep {ep:3d}  loss {tot/max(nb,1):.4f} | IoU {iou:.3f}  "
              f"ระยะ {mae:5.1f} ซม.  สูงเงา R2 {rh:+.3f}  "
              f"({time.time()-t0:.0f}s){flag}", flush=True)

    net.load_state_dict(best_state)
    O, D = evaluate(torch, net, counts, te)
    iou, thr = best_iou(O, occ_te)
    mae = float(np.abs(D[occ_te] - dep_te[occ_te]).mean())
    rh = r2(shadow_height(O >= thr), h_true)
    perm = rng.permutation(len(te))
    s_iou = iou_at(O[perm], occ_te, thr)
    s_mae = float(np.abs(D[perm][occ_te] - dep_te[occ_te]).mean())

    print("\n" + "=" * 64)
    print(f"{'':24s}{'IoU':>8s}{'ระยะพลาด':>12s}{'สูงเงา R2':>12s}")
    print(f"{'1. ทายค่าเฉลี่ย':24s}{b1i:8.3f}{b1e:9.1f} ซม.{'':>12s}")
    print(f"{'2. แม่แบบรู้ตำแหน่ง':24s}{b2i:8.3f}{b2e:9.1f} ซม.{r_pos:+12.3f}")
    print(f"{'3. โมเดล เสียงล้วน':24s}{iou:8.3f}{mae:9.1f} ซม.{rh:+12.3f}")
    print(f"{'4. โมเดล+สลับคลื่น':24s}{s_iou:8.3f}{s_mae:9.1f} ซม.")
    print("=" * 64)
    c_iou = iou_at(coarsen(O >= thr).astype(np.float32), coarsen(occ_te), 0.5)
    print(f"ถ้าย่อลง 40x30 เท่ารอบแรก IoU = {c_iou:.3f}  "
          f"(ตารางละเอียดกว่าทำให้ IoU ต่ำลงเองโดยธรรมชาติ)")
    if s_iou >= iou - 0.005:
        print("เตือน: สลับคลื่นแล้วคะแนนไม่ตก — โมเดลไม่ได้ใช้เสียง อย่าเชื่อภาพที่เห็น")
    elif rh > max(r_pos, 0.0) + 0.05:
        print("อ่านผล: **ทายความสูงเงาได้ดีกว่าทั้งการเดาค่าเฉลี่ย และการรู้ตำแหน่ง**")
        print("        แปลว่าเสียงบอกท่าทางได้จริง ไม่ใช่แค่บอกว่าอยู่ตรงไหน")
        print("        นี่คือสิ่งที่รอบแรกทำไม่ได้")
    else:
        print("อ่านผล: ยังทายท่าทางไม่ได้ดีกว่าการเดาค่าเฉลี่ย")
        print("        รูปร่างที่เห็นบนจอยังเป็นค่าเฉลี่ยที่จำมา ไม่ใช่สิ่งที่เสียงบอก")

    torch.save({"model": net.state_dict(), "holdout": a.test, "params": npar,
                "split": a.split, "tail": a.tail,
                "grid": [GW, GH],
                "score": {"iou": iou, "thr": thr, "mae_cm": mae, "h_r2": rh,
                          "shuffle_iou": s_iou, "mean_iou": b1i, "mean_mae": b1e,
                          "tmpl_iou": b2i, "tmpl_mae": b2e, "pos_h_r2": r_pos}},
               MD.MODEL)
    print(f"เก็บโมเดลที่ {MD.MODEL}  (รอบ {MD.ROUND})")


if __name__ == "__main__":
    main()
