"""เทรนโมเดลรอบสาม — BCE + Dice + Huber + ความสูงเงา

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ไฟล์นี้ทั้งไฟล์)
═══════════════════════════════════════════════════════════════════════

    python car/train_map3.py --epochs 20

**แบ่งข้อมูลด้วยการกันทั้ง section** ซึ่งรอบสองทำไม่ได้
รอบสองหนึ่งช่วงคือหนึ่งตำแหน่ง กันช่วงไว้ทดสอบจึงเท่ากับกันตำแหน่งนั้น
ออกจากการเทรนทั้งหมด ต้องเลี่ยงไปแบ่งตามเวลาแทน
รอบสามทุกช่วงครอบคลุมเหมือนกัน กันทั้งช่วงได้ตรง ๆ เป็นการทดสอบที่สะอาดกว่า

**loss มีสี่ส่วน แต่ละส่วนคุมคนละเรื่อง**
    BCE     ความถูกต้องรายช่อง — เกรเดียนต์นิ่ง ละเอียด
    Dice    รูปร่างทับกันจริง — กันไม่ให้ทายจาง ๆ กว้าง ๆ เอาตัวรอด
    Huber   ระยะ — ทำตัวเหมือน MSE ตอนผิดน้อย เหมือน L1 ตอนผิดมาก
    ความสูง ท่าทาง — **สิ่งที่รอบก่อนวัดแต่ไม่เคยสอน**

ยังวัดด้วยคู่เทียบชุดเดิมทุกครั้ง เพื่อให้รู้ว่าโมเดลเก่งจริงหรือแค่ดูเหมือนเก่ง
"""
import argparse
import time

import numpy as np

import mapdata2 as MD
import model3 as M3
from mapdata2 import GW, GH, NEAR_CM, FAR_CM
from model3 import prep, make_model, to_cm
from train_map2 import best_iou, iou_at, r2, position_bins


def shadow_height(occ):
    """ความสูงเงาแบบที่ใช้ **รายงานผล** · นิยามเดียวกับที่หน้าจอใช้"""
    return np.array([MD.shadow_height(m) for m in occ], np.float32)


def soft_height(p, torch):
    """ความสูงเงาแบบที่ใช้ **สอนโมเดล** · ต้องหาอนุพันธ์ได้

    ตัวที่ใช้รายงานผลใช้เปอร์เซ็นไทล์ ซึ่งเป็นฟังก์ชันขั้นบันได หาอนุพันธ์ไม่ได้
    เอามาใส่ใน loss ตรง ๆ ไม่ได้ ต้องมีตัวแทนที่ลื่นไหล

    ใช้ส่วนเบี่ยงเบนของแถวถ่วงน้ำหนักด้วยจำนวนช่องในแถวนั้น
    ก้อนสี่เหลี่ยมสูง h มีส่วนเบี่ยงเบน h/sqrt(12) = 0.289h
    คูณกลับด้วย 3.464 จึงได้ตัวเลขที่สเกลใกล้เคียงความสูงจริง
    """
    b, h, w = p.shape
    wt = p.sum(2) + 1e-6                          # น้ำหนักของแต่ละแถว
    y = torch.arange(h, device=p.device, dtype=p.dtype)[None]
    tot = wt.sum(1, keepdim=True)
    mu = (wt * y).sum(1, keepdim=True) / tot
    var = (wt * (y - mu) ** 2).sum(1, keepdim=True) / tot
    return 3.4641 * torch.sqrt(var.clamp(min=1e-6)).squeeze(1)


def baselines(dmm, tr, te, ang, dst):
    """คู่เทียบสามแบบ · ทุกแบบได้จุดตัดที่ดีที่สุดของตัวเอง ไม่งั้นเทียบไม่ยุติธรรม"""
    occ_tr, occ_te = dmm[tr] > 0, dmm[te] > 0
    z_tr = np.clip((dmm[tr] / 10.0 - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1)
    z_te = np.clip((dmm[te] / 10.0 - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1)

    mean_occ = occ_tr.mean(0)
    mean_z = (z_tr * occ_tr).sum(0) / np.maximum(occ_tr.sum(0), 1)
    m_iou, _ = best_iou(np.repeat(mean_occ[None], len(te), 0), occ_te)
    m_mae = float(np.abs((mean_z[None] - z_te)[occ_te]).mean()
                  * (FAR_CM - NEAR_CM))

    # แม่แบบที่ **โกง** ด้วยการรู้ตำแหน่งจริงจากกล้อง — เพดานของ 'รู้แค่ตำแหน่ง'
    tab, cnt = {}, {}
    for k, i in enumerate(tr):
        g = (int(ang[i]), int(dst[i]))
        if g not in tab:
            tab[g] = [np.zeros_like(mean_occ, np.float64),
                      np.zeros_like(mean_z, np.float64)]
            cnt[g] = 0
        tab[g][0] += occ_tr[k]
        tab[g][1] += z_tr[k] * occ_tr[k]
        cnt[g] += 1
    T_occ = np.stack([tab.get((int(ang[i]), int(dst[i])),
                              [mean_occ * cnt.get((0, 0), 1)])[0]
                      / max(cnt.get((int(ang[i]), int(dst[i])), 1), 1)
                      if (int(ang[i]), int(dst[i])) in tab else mean_occ
                      for i in te])
    t_iou, _ = best_iou(T_occ, occ_te)
    return m_iou, m_mae, t_iou


def evaluate(torch, net, counts, ends, dc, scale, stack, bs=48):
    net.eval()
    O = np.empty((len(ends), GH, GW), np.float32)
    D = np.empty_like(O)
    with torch.no_grad():
        for i in range(0, len(ends), bs):
            e = ends[i:i + bs]
            w = prep(M3.gather(counts, e, stack), dc, scale)
            ol, dl = net(torch.from_numpy(w).float())
            O[i:i + bs] = torch.sigmoid(ol).numpy()
            D[i:i + bs] = to_cm(dl, torch).numpy()
    return O, D


def main():
    import torch
    from torch import nn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", type=int, default=4,
                    help="section ที่ใช้รายงานผล · **ไม่แตะเลยระหว่างเทรน**")
    ap.add_argument("--val-tail", type=float, default=0.15,
                    help="ใช้ท้าย section เทรนกี่ส่วนเป็นชุดเลือกเช็คพอยต์")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stack", type=int, default=M3.STACK)
    ap.add_argument("--w-depth", type=float, default=2.0)
    ap.add_argument("--w-height", type=float, default=0.5)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument("--huber", type=float, default=0.1,
                    help="เส้นแบ่งว่าผิดแค่ไหนถือว่ามาก · 0.1 = 16 ซม.")
    a = ap.parse_args()
    torch.manual_seed(0)

    counts, dmm, sec, src, meta = MD.load()
    ends = M3.stack_index(sec, src, a.stack)
    te = ends[sec[ends] == a.test]
    # **section 4 ไม่ถูกแตะเลยจนกว่าจะจบการเทรน**
    # ชุดเลือกเช็คพอยต์เอามาจากท้าย section เทรน ไม่ใช่กินทั้ง section
    # เพราะกินทั้ง section ทำให้ข้อมูลเทรนหายไป 25% ซึ่งรอบก่อนเห็นแล้วว่าเจ็บ
    # ชุดนี้อยู่เซสชันเดียวกับชุดเทรนจึงไม่เหมาะจะใช้รายงาน แต่ใช้จัดอันดับ
    # ว่ารอบไหนดีกว่ารอบไหนได้ ซึ่งเป็นงานที่เบากว่ามาก
    rest = ends[sec[ends] != a.test]
    tr_l, va_l = [], []
    for sv in np.unique(sec[rest]):
        m = np.sort(rest[sec[rest] == sv])
        cut = int(len(m) * (1.0 - a.val_tail))
        tr_l.append(m[:cut])
        va_l.append(m[cut:])
    tr, va = np.concatenate(tr_l), np.concatenate(va_l)
    dc, scale = M3.norm_params(counts, tr)
    print(f"ตาราง {GW}x{GH} · {a.stack} การยิงต่อภาพ")
    print(f"เทรน {len(tr):,} · เลือกเช็คพอยต์ {len(va):,} "
          f"({a.val_tail:.0%} ท้ายของ section เทรน) · "
          f"รายงาน {len(te):,} (s{a.test} ไม่แตะระหว่างเทรน)")
    print(f"ปรับค่าจาก section เทรนเท่านั้น: จุดกลาง {dc:.0f} · สเกล {scale:.0f}")

    print("\nคำนวณคู่เทียบ ...")
    ang, dst = position_bins(dmm)
    m_iou, m_mae, t_iou = baselines(dmm, tr, te, ang, dst)
    print(f"  1. ทายค่าเฉลี่ย      IoU {m_iou:.3f}  ระยะพลาด {m_mae:5.1f} ซม.")
    print(f"  2. แม่แบบรู้ตำแหน่ง  IoU {t_iou:.3f}  (โกง)")

    net = make_model(torch, nn, GH, GW, a.stack)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = max(1, len(tr) // a.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=nb)
    occ_v, dep_v = dmm[va] > 0, dmm[va].astype(np.float32) / 10.0
    h_val = shadow_height(occ_v)
    occ_t = dmm[te] > 0
    z_te = np.clip((dmm[te] / 10.0 - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1)
    # เฉลยระยะเป็นเซนติเมตร · to_cm คืนค่าเป็นเซนติเมตรอยู่แล้ว เทียบกันตรง ๆ
    # เคยพลาดตรงนี้: หารด้วย 10 ซ้ำเหมือนมันเป็นมิลลิเมตร ทำให้รายงานว่า
    # พลาด 115 ซม. ทั้งที่จริง 10 ซม. — คู่เทียบไม่โดนเพราะคิดในหน่วย 0-1 ตลอด
    dep_te = dmm[te].astype(np.float32) / 10.0
    h_true = shadow_height(occ_t)
    print(f"\nโมเดล {npar:,} พารามิเตอร์ · เทรน {a.epochs} รอบ")

    P_IOU = MD.MODEL
    P_R2 = MD.MODEL.with_name(MD.MODEL.stem + "_r2.pt")
    best = {"iou": -9e9, "r2": -9e9}
    hist = []
    for ep in range(1, a.epochs + 1):
        net.train()
        t0 = time.time()
        perm = np.random.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), a.batch):
            e = tr[perm[i:i + a.batch]]
            w = torch.from_numpy(prep(M3.gather(counts, e, a.stack),
                                      dc, scale)).float()
            occ = torch.from_numpy((dmm[e] > 0).astype(np.float32))
            z = torch.from_numpy(np.clip(
                (dmm[e] / 10.0 - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1).astype(np.float32))
            ol, dl = net(w)
            p = torch.sigmoid(ol)

            bce = nn.functional.binary_cross_entropy_with_logits(ol, occ)
            # Dice มองทั้งภาพพร้อมกัน · ทายจาง ๆ กว้าง ๆ จะได้ตัวหารใหญ่แต่เศษไม่โต
            dice = 1 - (2 * (p * occ).sum((1, 2)) + 1) / \
                       (p.sum((1, 2)) + occ.sum((1, 2)) + 1)
            # ระยะวัดเฉพาะช่องที่มีวัตถุจริง · ระยะในช่องว่างไม่มีความหมาย
            err = (torch.sigmoid(dl) - z).abs()
            hub = torch.where(err <= a.huber, 0.5 * err ** 2,
                              a.huber * (err - 0.5 * a.huber))
            hub = (hub * occ).sum() / occ.sum().clamp(min=1)
            hgt = (soft_height(p, torch) -
                   soft_height(occ, torch)).abs().mean() / GH

            loss = (bce + a.w_dice * dice.mean() + a.w_depth * hub
                    + a.w_height * hgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())

        # วัดด้วย section เลือกเช็คพอยต์เท่านั้น · section รายงานยังไม่แตะ
        O, D = evaluate(torch, net, counts, va, dc, scale, a.stack)
        iou, thr = best_iou(O, occ_v)
        mae = float(np.abs(D - dep_v)[occ_v].mean())
        rh = r2(shadow_height(O >= thr), h_val)
        # **เก็บสองเช็คพอยต์แยกกัน** ไม่รวมคะแนนเป็นตัวเดียว
        # รอบก่อนใช้ IoU + R2 ซึ่งพัง เพราะ IoU อยู่ช่วง 0.3-0.45
        # แต่ R2 แกว่งถึง -4.2 ผลรวมจึงถูก R2 กลืนจนไปเลือกรอบที่ยังเทรนไม่ทัน
        # แยกเก็บแล้ววัดทั้งคู่ตอนจบ จะได้เห็นว่าเลือกเพื่อท่าทางแลกมาด้วยอะไร
        hist.append((ep, iou, mae, rh))
        snap = {"model": net.state_dict(), "grid": (GW, GH),
                "stack": a.stack, "norm": (dc, scale), "params": npar,
                "test": a.test, "epoch": ep,
                "val": {"iou": iou, "thr": thr, "mae_cm": mae, "h_r2": rh},
                "score": {"mean_iou": m_iou, "mean_mae": m_mae,
                          "tmpl_iou": t_iou}}
        mark = ""
        if iou > best["iou"]:
            best["iou"] = iou
            torch.save(snap, P_IOU)
            mark += "  <-IoU"
        if rh > best["r2"]:
            best["r2"] = rh
            torch.save(snap, P_R2)
            mark += "  <-R2"
        print(f"ep {ep:3d}  loss {tot/nb:.4f} | IoU {iou:.3f}  ระยะ {mae:5.1f} ซม."
              f"  สูงเงา R2 {rh:+.3f}  ({time.time()-t0:.0f}s){mark}")

    # ---- ถึงตรงนี้ค่อยแตะ section รายงาน · ครั้งเดียว ตอนจบ
    print("\n" + "=" * 72)
    print(f"{'':26}{'IoU':>8}{'ระยะพลาด':>12}{'สูงเงา R2':>12}")
    print(f"{'1. ทายค่าเฉลี่ย':26}{m_iou:8.3f}{m_mae:9.1f} ซม.")
    print(f"{'2. แม่แบบรู้ตำแหน่ง':26}{t_iou:8.3f}")

    out = {}
    for tag, path in (("เลือกด้วย IoU", P_IOU), ("เลือกด้วย R2", P_R2)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        O, D = evaluate(torch, net, counts, te, dc, scale, a.stack)
        iou_te, thr_te = best_iou(O, occ_t)
        mae_te = float(np.abs(D - dep_te)[occ_t].mean())
        rh_te = r2(shadow_height(O >= thr_te), h_true)
        # สลับคลื่น: จับคู่คลื่นกับเฉลยผิดคู่ · ถ้าคะแนนไม่ตก แปลว่าไม่ได้ใช้เสียง
        sh = np.random.permutation(len(te))
        s_iou = iou_at(O[sh] >= thr_te, occ_t, 0.5)
        s_mae = float(np.abs(D[sh] - dep_te)[occ_t].mean())
        ck["score"].update({"iou": iou_te, "thr": thr_te, "mae_cm": mae_te,
                            "h_r2": rh_te, "shuffle_iou": s_iou,
                            "shuffle_mae": s_mae})
        torch.save(ck, path)
        out[tag] = (ck, iou_te, mae_te, rh_te, s_iou, s_mae)
        v = ck["val"]
        print(f"{'3. โมเดล · ' + tag:26}{iou_te:8.3f}{mae_te:9.1f} ซม."
              f"{rh_te:+12.3f}   (รอบ {ck['epoch']} · "
              f"ชุดเลือก IoU {v['iou']:.3f} R2 {v['h_r2']:+.3f})")
        print(f"{'   สลับคลื่น':26}{s_iou:8.3f}{s_mae:9.1f} ซม.")
    print("=" * 72)

    a_iou, a_r2 = out["เลือกด้วย IoU"], out["เลือกด้วย R2"]
    print(f"เลือกเพื่อท่าทางแลกมาด้วย IoU {a_iou[1]-a_r2[1]:+.3f} "
          f"และระยะ {a_r2[2]-a_iou[2]:+.1f} ซม. "
          f"เพื่อ R2 ที่ดีขึ้น {a_r2[3]-a_iou[3]:+.3f}")
    if max(a_iou[3], a_r2[3]) > 0:
        print("อ่านผล: มีเช็คพอยต์ที่ทายท่าทางได้ดีกว่าการเดาค่าเฉลี่ย")
    else:
        print("อ่านผล: ไม่มีเช็คพอยต์ไหนทายท่าทางได้ดีกว่าการเดาค่าเฉลี่ย")
    print(f"เก็บที่ {P_IOU.name} และ {P_R2.name}")


if __name__ == "__main__":
    main()
