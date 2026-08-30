"""ตรวจสอบว่าโมเดลแผนที่ 'เห็น' อะไรจริง ๆ — แยกสิ่งที่เรียนได้ ออกจากสิ่งที่จำมา

รันหลังเทรนเสร็จ:  python car/check_map.py

ตอบสามคำถามที่ภาพสวย ๆ ตอบให้ไม่ได้:

  1. ชนะคู่เทียบที่รู้ตำแหน่งจริงไหม — และคู่เทียบนั้นต้อง **ละเอียดพอ** ด้วย
     ถ้าแบ่งช่องหยาบ คู่เทียบจะอ่อนเกินจริงแล้วโมเดลดูเก่งเกินตัว
     จึงไล่ตั้งแต่หยาบไปละเอียดจนคู่เทียบแทบเป็น 'ตัวรู้ตำแหน่งสมบูรณ์แบบ'

  2. รูปร่างที่เห็นมาจากการได้ยิน หรือแค่จากการรู้ว่าอยู่ตรงไหน — วัดเทียบกับ
     เพดานของ 'รู้ตำแหน่งจริงอย่างเดียว' ถ้าโมเดลไม่เกินเพดานนั้น รูปร่างที่เห็น
     ก็อธิบายได้หมดด้วยตำแหน่ง ไม่ใช่สิ่งที่เสียงมองเห็น

  3. แต่ละชุดข้อมูลต่างกันแค่ไหน — grid ถ่ายใกล้เห็นแค่ลำตัว walk2 เห็นคนเต็มตัว
     ตัวเลขรวมก้อนเดียวจะกลบความต่างนี้
"""
import numpy as np

import mapdata as MD
from mapdata import GW, GH
from train_map import iou_at, best_iou, evaluate, position_bins


def r2(pred, true):
    """อธิบายความแปรปรวนได้กี่ % เทียบกับการตอบค่าเฉลี่ยเฉย ๆ

    ติดลบได้ และถ้าติดลบแปลว่าแย่กว่าการตอบค่าเฉลี่ย
    """
    ss = ((true - pred) ** 2).sum()
    sv = ((true - true.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - float(ss / max(float(sv), 1e-9))


def template(dmm, tr, te, nang, ndist):
    """คู่เทียบที่ 'รู้ตำแหน่งจริงจากกล้อง' แล้วแปะรูปร่างเฉลี่ยของตำแหน่งนั้น"""
    ang, dst = position_bins(dmm, nang, ndist)
    occ_tr = dmm[tr] > 0
    dep_tr = dmm[tr].astype(np.float32) / 10.0
    flat = dep_tr[occ_tr].mean()
    to = np.zeros((nang, ndist, GH, GW), np.float32)
    td = np.zeros_like(to)
    n = np.zeros((nang, ndist), np.int32)
    for k, i in enumerate(tr):
        if ang[i] < 0:
            continue
        to[ang[i], dst[i]] += occ_tr[k]
        td[ang[i], dst[i]] += dep_tr[k] * occ_tr[k]
        n[ang[i], dst[i]] += 1
    td = np.where(to > 0, td / np.maximum(to, 1e-6), flat)
    to = to / np.maximum(n, 1)[:, :, None, None]
    m_occ, m_dep = occ_tr.mean(0), np.full((GH, GW), flat, np.float32)
    po = np.empty((len(te), GH, GW), np.float32)
    pd = np.empty_like(po)
    for k, i in enumerate(te):
        if ang[i] < 0 or n[ang[i], dst[i]] == 0:
            po[k], pd[k] = m_occ, m_dep
        else:
            po[k], pd[k] = to[ang[i], dst[i]], td[ang[i], dst[i]]
    occ_t = dmm[te] > 0
    iou, _ = best_iou(po, occ_t)
    mae = float(np.abs(pd[occ_t] - dmm[te].astype(np.float32)[occ_t] / 10.0).mean())
    return iou, mae, int((n > 0).sum()), float(n[n > 0].mean())


def cond_r2(dmm, tr, te, keys, axis):
    """ทายโปรไฟล์จาก **ของจริงที่บอกให้ฟรี** ได้ R^2 เท่าไร

    keys คือสิ่งที่ยอมให้รู้ (มุมจริง ระยะจริง ชุดที่อัด) แล้วตอบค่าเฉลี่ยของกลุ่มนั้น
    เป็น **เพดาน** ของคำว่า 'รู้แค่นี้ก็ทำได้เท่านี้' ถ้าโมเดลได้ไม่เกินเพดาน
    แปลว่าสิ่งที่เห็นบนจออธิบายได้หมดด้วยของที่บอกให้ฟรี ไม่ใช่สิ่งที่เสียงมองเห็น
    """
    ptr = (dmm[tr] > 0).sum(axis).astype(np.float32)
    pte = (dmm[te] > 0).sum(axis).astype(np.float32)
    tab, cnt = {}, {}
    for k, i in enumerate(tr):
        g = tuple(int(x[i]) for x in keys)
        tab[g] = tab.get(g, 0) + ptr[k]
        cnt[g] = cnt.get(g, 0) + 1
    glob = ptr.mean(0)
    pred = np.stack([tab[g] / cnt[g] if (g := tuple(int(x[i]) for x in keys)) in tab
                     else glob for i in te])
    return r2(pred, pte)


def main():
    import torch
    from mapmodel import make_model

    counts, dmm, sec, src, meta = MD.load()
    ck = torch.load(MD.DATA / "_map_model.pt", map_location="cpu", weights_only=False)
    hold = int(ck["holdout"])
    idx = np.nonzero(sec > 0)[0]
    te = idx[sec[idx] == hold]
    tr = idx[sec[idx] != hold]
    net = make_model(torch, torch.nn)
    net.load_state_dict(ck["model"])
    torch.set_num_threads(4)
    print(f"โมเดล {ck['params']:,} พารามิเตอร์ · กันช่วง s{hold} ไว้ "
          f"({len(te):,} เฟรม) · เทรนจาก {len(tr):,} เฟรม\n")

    O, D = evaluate(torch, net, counts, te)
    occ_t = dmm[te] > 0
    dep_t = dmm[te].astype(np.float32) / 10.0
    m_iou, m_thr = best_iou(O, occ_t)
    m_mae = float(np.abs(D[occ_t] - dep_t[occ_t]).mean())

    print("1) คู่เทียบที่รู้ตำแหน่งจริง — ยิ่งแบ่งช่องละเอียด ยิ่งเป็นคู่แข่งที่แข็ง")
    print(f"   {'ชอง (มุม x ระยะ)':20s}{'ชองที่ใช้':>12s}{'เฟรม/ชอง':>12s}"
          f"{'IoU':>9s}{'ระยะพลาด':>12s}")
    for na, nd in ((8, 4), (12, 6), (20, 8), (32, 12)):
        iou, mae, used, per = template(dmm, tr, te, na, nd)
        print(f"   {na:>3d} x {nd:<14d}{used:>12,}{per:>12.0f}{iou:>9.3f}{mae:>9.1f} ซม.")
    print(f"   {'โมเดล (เสียงลวน)':20s}{'':>12s}{'':>12s}{m_iou:>9.3f}{m_mae:>9.1f} ซม.")
    print("   -> รูปร่างแพ้ทุกความละเอียด แต่ระยะชนะทุกความละเอียด")

    print("\n2) รูปร่างที่เห็น มาจากการได้ยิน หรือแค่จากการรู้ว่าอยู่ตรงไหน")
    print("   ยิ่งบอกของจริงให้ฟรีมาก เพดานยิ่งสูง · โมเดลไม่รู้อะไรฟรีเลย")
    ang, dst = position_bins(dmm, 20, 8)
    pred_occ = O >= m_thr
    for lab, axis, ladder in (
            ("แนวนอน (ซ้าย-ขวา)", 1, [("รู้มุมจริง", [ang]),
                                       ("รู้มุม+ระยะ+ชุด", [ang, dst, src])]),
            ("แนวตั้ง (บน-ล่าง)", 2, [("รู้ระยะจริง", [dst]),
                                      ("รู้ระยะ+ชุด", [dst, src]),
                                      ("รู้ระยะ+ชุด+มุม", [dst, src, ang])])):
        mo = r2(pred_occ.sum(axis).astype(np.float32), occ_t.sum(axis).astype(np.float32))
        print(f"   {lab}   โมเดล R^2 = {mo:+.3f}")
        for name, keys in ladder:
            print(f"      เพดานถ้า {name:18s} {cond_r2(dmm, tr, te, keys, axis):+.3f}")
    print("   แนวตั้งที่ดูเหมือนโมเดลเก่ง ไม่ได้แปลว่าเสียงแยกบน-ล่างได้ — เสา 4 ต้น")
    print("   เรียงแนวนอน แยกบน-ล่างไม่ได้ในทางฟิสิกส์ ที่ทายได้เพราะความสูงในภาพ")
    print("   เกาะไปกับระยะและมุม (ยิ่งใกล้ยิ่งเต็มเฟรม อยู่ริมก็โดนตัด)")

    print("\n3) แยกตามชุดข้อมูล (แต่ละชุดถ่ายไม่เหมือนกัน)")
    st = src[te]
    for i, f in enumerate(meta["fam"]):
        m = st == i
        if not m.any():
            continue
        iou = iou_at(O[m], occ_t[m], m_thr)
        mae = float(np.abs(D[m][occ_t[m]] - dep_t[m][occ_t[m]]).mean())
        print(f"   {f:7s} {int(m.sum()):>6,} เฟรม   IoU {iou:.3f}   ระยะพลาด {mae:5.1f} ซม.")

    print(f"\nจุดตัดที่ดีที่สุดของโมเดลคือ {m_thr:.2f} "
          f"(หน้าจอตั้งไว้ 0.50 ได้ {iou_at(O, occ_t, 0.5):.3f})")


if __name__ == "__main__":
    main()
