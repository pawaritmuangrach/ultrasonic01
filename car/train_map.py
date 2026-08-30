"""เทรนโมเดลที่แปลงคลื่นเสียงเป็นภาพ depth — ML ล้วน ไม่มี DSP ที่เขียนมือ

**วิธีวัดผลสำคัญกว่าตัวโมเดล** ภาพ depth ที่ทายออกมาดูน่าเชื่อได้ง่ายมาก
เพราะห้องกับคนหน้าตาคล้ายกันทุกเฟรม โมเดลที่ตอบ 'ภาพเฉลี่ย' เฉย ๆ ก็ดูดีแล้ว
ไฟล์นี้จึงเทียบกับเกณฑ์สองอันเสมอ:

  1. ทายค่าเฉลี่ย     ไม่ดูคลื่นเลย ตอบภาพเฉลี่ยของชุดเทรนทุกครั้ง
                      ชนะอันนี้ไม่ได้ = ไม่ได้เรียนอะไรจากเสียงเลย

  2. แม่แบบรู้ตำแหน่ง  โกงด้วยการดูมุมและระยะจริงจากกล้อง แล้วแปะรูปร่างเฉลี่ย
                      ของตำแหน่งนั้นลงไป
                      ชนะอันนี้ไม่ได้ = เสียงบอกได้แค่ 'อยู่ตรงไหน' ไม่ได้บอก 'รูปร่างยังไง'

เกณฑ์ที่ 2 คือคำถามที่แพงที่สุดของโปรเจกต์ ถ้าชนะมันไม่ได้ ภาพสวย ๆ ที่เห็น
ก็เป็นแค่รูปร่างเฉลี่ยที่ถูกเลื่อนไปวางถูกที่ ไม่ใช่การ 'มองเห็น' ด้วยเสียงจริง ๆ

แบ่ง train/test **ตามช่วงการอัด** ไม่ใช่สุ่มรายเฟรม เพราะเฟรมติดกันเกือบเหมือนกัน
สุ่มรายเฟรมจะทำให้เฟรมข้างเคียงของ test ไปโผล่ใน train แล้วคะแนนพองเกินจริง
"""
import argparse
import time
import numpy as np

import mapdata as MD
from mapdata import GW, GH, NEAR_CM, FAR_CM
from mapmodel import prep, make_model, to_cm

ANG_BINS, DIST_BINS = 12, 6


def iou_at(occ_p, occ_t, thr):
    """ทับกันแค่ไหนที่จุดตัดหนึ่ง ๆ

    ใช้ IoU ไม่ใช่ 'ทายถูกกี่ช่อง' เพราะภาพส่วนใหญ่เป็นที่ว่าง
    การตอบ 'ว่างทั้งภาพ' จะได้ความถูกต้องสูงลิ่วทั้งที่ไม่เจออะไรเลย
    """
    pred = occ_p >= thr
    return float((pred & occ_t).sum()) / max(int((pred | occ_t).sum()), 1)


def best_iou(occ_p, occ_t):
    """IoU ที่ดีที่สุดเท่าที่วิธีนั้นทำได้ พร้อมจุดตัดที่ให้ผลนั้น

    ต้องหาจุดตัดให้ **ทุกวิธีรวมทั้งเกณฑ์เทียบ** ไม่งั้นเทียบไม่ยุติธรรม:
    ภาพเฉลี่ยไม่มีช่องไหนเกิน 0.5 เลย (คนเดินไปมา ไม่มีจุดไหนมีคนยืนเกินครึ่งเวลา)
    ถ้าล็อกจุดตัดไว้ที่ 0.5 เกณฑ์นี้จะได้ IoU 0.000 ซึ่งเป็นคู่เทียบที่อ่อนเกินจริง
    แล้วโมเดลก็จะดูเก่งเกินตัว
    """
    best = (0.0, 0.5)
    for t in np.arange(0.05, 1.0, 0.05):
        v = iou_at(occ_p, occ_t, t)
        if v > best[0]:
            best = (v, float(t))
    return best


def iou_mae(occ_p, dep_p, occ_t, dep_t):
    """คืน (IoU ที่ดีที่สุด, จุดตัดที่ให้ผลนั้น, ระยะพลาด ซม.)"""
    iou, thr = best_iou(occ_p, occ_t)
    mae = float(np.abs(dep_p[occ_t] - dep_t[occ_t]).mean()) if occ_t.any() else float("nan")
    return iou, thr, mae


def position_bins(dmm, nang=ANG_BINS, ndist=DIST_BINS):
    """แปลงภาพจริงเป็นหมายเลขช่อง (มุม, ระยะ) ไว้ทำแม่แบบรู้ตำแหน่ง

    รับจำนวนช่องเป็นพารามิเตอร์ เพื่อให้ check_map.py ไล่ความละเอียดได้
    โดยไม่ต้องไปแก้ตัวแปรระดับโมดูลของไฟล์นี้ซึ่งมองไม่เห็นจากจุดที่เรียก
    """
    ang = np.full(len(dmm), -1, np.int16)
    dst = np.full(len(dmm), -1, np.int16)
    for i, d in enumerate(dmm):
        p = MD.pos_from_map(d)
        if p is None:
            continue
        deg, cm = p
        ang[i] = np.clip(int((deg / MD.FOV_H_DEG + 0.5) * nang), 0, nang - 1)
        dst[i] = np.clip(int((cm - NEAR_CM) / (FAR_CM - NEAR_CM) * ndist), 0, ndist - 1)
    return ang, dst


def baselines(dmm, tr, te, ang, dst):
    """คืนคะแนนของเกณฑ์เทียบสองอัน"""
    occ_t = dmm[te] > 0
    dep_t = dmm[te].astype(np.float32) / 10.0
    occ_tr = dmm[tr] > 0
    dep_tr = dmm[tr].astype(np.float32) / 10.0

    # --- 1. ทายค่าเฉลี่ย: ภาพเดียว ใช้ตอบทุกเฟรม
    m_occ = occ_tr.mean(0)
    tot = occ_tr.sum(0)
    flat = dep_tr[occ_tr].mean()
    m_dep = np.where(tot > 0, (dep_tr * occ_tr).sum(0) / np.maximum(tot, 1), flat)
    a = iou_mae(np.broadcast_to(m_occ, occ_t.shape),
                np.broadcast_to(m_dep, occ_t.shape), occ_t, dep_t)

    # --- 2. แม่แบบรู้ตำแหน่ง: เฉลี่ยแยกตามช่อง (มุม, ระยะ) จริงจากกล้อง
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
            po[k], pd[k] = m_occ, m_dep          # ช่องที่ชุดเทรนไม่เคยเจอ
        else:
            po[k], pd[k] = to[ang[i], dst[i]], td[ang[i], dst[i]]
    b = iou_mae(po, pd, occ_t, dep_t)
    return a, b


def evaluate(torch, net, counts, idx, bs=128):
    """คืน (occ prob, depth cm) ของทุกเฟรมใน idx"""
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


def report(rows, note):
    print("\n" + "=" * 62)
    for name, iou, mae, tail in rows:
        print(f"{name:24s}{iou:7.3f}{mae:9.1f} ซม.   {tail}")
    print("=" * 62)
    print(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, help="กันช่วงไหนไว้วัดผล (1-5)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    counts, dmm, sec, src, meta = MD.load()
    idx = np.nonzero(sec > 0)[0]        # sec = 0 คือเฟรมที่อ่านไม่สำเร็จตอนสร้างแคช
    te = idx[sec[idx] == a.test]
    tr = idx[sec[idx] != a.test]
    print(f"ข้อมูล {len(idx):,} เฟรม · เทรน {len(tr):,} · กันไว้วัด {len(te):,} "
          f"(ช่วง s{a.test} ของทุกชุด)")
    print(f"ชุดที่ใช้: {', '.join(meta['fam'])}")

    print("\nคำนวณเกณฑ์เทียบ ...", flush=True)
    ang, dst = position_bins(dmm)
    (b1_iou, b1_thr, b1_mae), (b2_iou, b2_thr, b2_mae) = baselines(dmm, tr, te, ang, dst)
    print(f"  1. ทายค่าเฉลี่ย      IoU {b1_iou:.3f} (จุดตัด {b1_thr:.2f})  "
          f"ระยะพลาด {b1_mae:5.1f} ซม.")
    print(f"  2. แม่แบบรู้ตำแหน่ง  IoU {b2_iou:.3f} (จุดตัด {b2_thr:.2f})  "
          f"ระยะพลาด {b2_mae:5.1f} ซม.  (โกง)")

    net = make_model(torch, nn)
    npar = sum(p.numel() for p in net.parameters())
    print(f"\nโมเดล {npar:,} พารามิเตอร์ · เทรน {a.epochs} รอบ", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, epochs=a.epochs, steps_per_epoch=max(1, len(tr) // a.batch + 1))
    rng = np.random.default_rng(0)
    occ_te = dmm[te] > 0
    dep_te = dmm[te].astype(np.float32) / 10.0
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
            # นับระยะเฉพาะช่องที่ **กล้องเห็นวัตถุจริง** ที่ว่างไม่มีระยะให้เรียน
            l1 = (torch.abs(torch.sigmoid(dl) - z) * occ).sum() / occ.sum().clamp(min=1)
            loss = bce + 2.0 * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        O, D = evaluate(torch, net, counts, te)
        iou, thr, mae = iou_mae(O, D, occ_te, dep_te)
        flag = ""
        if iou > best:
            best = iou
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            flag = "  <- ดีสุด"
        print(f"ep {ep:3d}  loss {tot/max(nb,1):.4f} | IoU {iou:.3f}"
              f"  ระยะพลาด {mae:5.1f} ซม.  ({time.time()-t0:.0f}s){flag}", flush=True)

    net.load_state_dict(best_state)
    O, D = evaluate(torch, net, counts, te)
    iou, thr, mae = iou_mae(O, D, occ_te, dep_te)
    at_half = iou_at(O, occ_te, 0.5)   # จุดตัดที่หน้าจอใช้จริง

    # สลับคลื่น: เอาผลที่ทายจากคลื่นของเฟรมอื่นมาเทียบกับภาพจริงของเฟรมนี้
    # ถ้าคะแนนไม่ตก แปลว่าโมเดลไม่ได้ใช้เสียงเลย แค่จำภาพเฉลี่ยไว้ตอบ
    perm = rng.permutation(len(te))
    s_iou, s_thr, s_mae = iou_mae(O[perm], D[perm], occ_te, dep_te)

    if s_iou >= iou - 0.005:
        note = "เตือน: สลับคลื่นแล้วคะแนนไม่ตก — โมเดลไม่ได้ใช้เสียง อย่าเชื่อภาพที่เห็น"
    elif iou <= b1_iou + 0.005:
        note = "เตือน: ชนะการทายค่าเฉลี่ยไม่ได้ — ยังไม่ได้เรียนอะไรจากเสียง"
    elif iou <= b2_iou:
        note = ("อ่านผล: เสียงบอก 'อยู่ตรงไหน' ได้จริง "
                "แต่ยังไม่เกินการแปะรูปร่างเฉลี่ยลงตำแหน่งที่ถูก")
    else:
        note = "อ่านผล: ชนะแม้แต่แม่แบบที่รู้ตำแหน่งจริง — เสียงบอกมากกว่าแค่ตำแหน่ง"

    report([("1. ทายค่าเฉลี่ย", b1_iou, b1_mae, f"จุดตัด {b1_thr:.2f}"),
            ("2. แม่แบบรู้ตำแหน่ง", b2_iou, b2_mae, f"จุดตัด {b2_thr:.2f}  (โกง: ดูกล้อง)"),
            ("3. โมเดล เสียงล้วน", iou, mae, f"จุดตัด {thr:.2f}"),
            ("4. โมเดล+สลับคลื่น", s_iou, s_mae, "(ต้องแย่กว่าข้อ 3)")], note)
    print(f"หน้าจอใช้จุดตัด 0.50 ซึ่งได้ IoU {at_half:.3f}")

    torch.save({"model": net.state_dict(), "holdout": a.test, "params": npar,
                "score": {"iou": iou, "thr": thr, "iou_at_half": at_half,
                          "mae_cm": mae, "shuffle_iou": s_iou,
                          "mean_iou": b1_iou, "mean_mae": b1_mae,
                          "tmpl_iou": b2_iou, "tmpl_mae": b2_mae}},
               MD.DATA / "_map_model.pt")
    print(f"เก็บโมเดลที่ {MD.DATA / '_map_model.pt'}")


if __name__ == "__main__":
    main()
