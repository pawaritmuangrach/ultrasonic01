"""ตรวจชุดข้อมูลรอบสอง **ก่อนเทรน** — ท่าทางแยกออกจากตำแหน่งจริงหรือยัง

    python car/check_data2.py

รอบแรกเทรนแล้วพบทีหลังว่าโมเดลเรียนรูปร่างไม่ได้เลย เพราะในข้อมูล
**ตำแหน่งอธิบายรูปร่างได้เกือบหมดอยู่แล้ว** กว่าจะรู้ก็เสียเวลาเทรนไปหลายชั่วโมง

ไฟล์นี้ตอบคำถามนั้นก่อนเทรน ใช้เวลาไม่กี่วินาที ถ้าตัวเลขบอกว่ายังไม่แยก
ก็ไม่ต้องเทรน กลับไปเก็บข้อมูลใหม่ถูกกว่า

วิธีวัด: เอา **ความสูงและความกว้างของเงา** (ซึ่งคือท่าทาง) มาลองทำนายจาก
**มุมกับระยะ** (ซึ่งคือตำแหน่ง) ถ้าทำนายได้แม่น แปลว่าท่าทางไม่ได้เป็นอิสระ
โมเดลก็จะไม่มีอะไรใหม่ให้เรียน
"""
import numpy as np

import mapdata2 as M


def r2(pred, true):
    ss = float(((true - pred) ** 2).sum())
    sv = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss / max(sv, 1e-9)


def predict_from_bins(keys, target, nb=12):
    """ทำนาย target จากค่าเฉลี่ยของกลุ่มที่แบ่งตาม keys — เพดานของ 'รู้แค่นี้'"""
    idx = np.zeros(len(target), np.int64)
    mul = 1
    for k in keys:
        lo, hi = np.percentile(k, [1, 99])
        b = np.clip(((k - lo) / max(hi - lo, 1e-9) * nb).astype(int), 0, nb - 1)
        idx += b * mul
        mul *= nb
    pred = np.zeros_like(target)
    for g in np.unique(idx):
        m = idx == g
        pred[m] = target[m].mean()
    return pred


def main():
    counts, dmm, sec, src, meta = M.load()
    print(f"เฟรมทั้งหมด {len(dmm):,} · ชุด {', '.join(meta['fam'])} · "
          f"ตาราง {meta['grid'][0]}x{meta['grid'][1]}\n")

    rows = []
    for i, d in enumerate(dmm):
        s = M.shape_stats(d)
        rows.append((np.nan,) * 5 if s is None else s)
    a = np.array(rows, np.float32)
    ok = ~np.isnan(a[:, 0])
    deg, near, w, h, area = (a[ok, k] for k in range(5))
    S, R = sec[ok], src[ok]
    print(f"เฟรมที่กล้องเห็นเป้า {ok.sum():,} ({ok.mean():.0%})\n")

    print("1) ข้อมูลกระจายพอไหม")
    for nm, v, u in (("มุม", deg, "องศา"), ("ระยะ", near, "ซม."),
                     ("กว้างเงา", w, "ช่อง"), ("สูงเงา", h, "ช่อง")):
        print(f"   {nm:9s} {v.mean():7.1f} +- {v.std():5.1f} {u:5s} "
              f"· ช่วง {np.percentile(v,2):6.1f} ถึง {np.percentile(v,98):6.1f}")

    print("\n2) ท่าทางเปลี่ยนขณะตำแหน่งคงที่ไหม  (ดูส่วนเบี่ยงเบนภายในช่วงเดียวกัน)")
    print("   ช่วงหนึ่ง = ยืนที่เดียว ถ้าสูง/กว้างยังแกว่ง แปลว่าท่าเปลี่ยนจริง")
    for s in np.unique(S):
        m = S == s
        if m.sum() < 50:
            continue
        fam = meta["fam"][int(R[m][0])]
        print(f"   s{s} ({fam:5s}) n={m.sum():5d} · ระยะ sd {near[m].std():4.1f} ซม. "
              f"· สูง sd {h[m].std():4.1f} ช่อง · กว้าง sd {w[m].std():4.1f} ช่อง")

    print("\n3) **คำถามชี้ขาด** ทำนายรูปร่างจากตำแหน่งอย่างเดียวได้แม่นแค่ไหน")
    print("   ยิ่งแม่น = ท่าทางยิ่งไม่เป็นอิสระ = โมเดลไม่มีอะไรใหม่ให้เรียน")
    for nm, tgt in (("สูงเงา", h), ("กว้างเงา", w), ("พื้นที่เงา", area)):
        a1 = r2(predict_from_bins([deg], tgt), tgt)
        a2 = r2(predict_from_bins([deg, near], tgt), tgt)
        a3 = r2(predict_from_bins([deg, near, S.astype(np.float32)], tgt), tgt)
        print(f"   {nm:11s} จากมุม {a1:+.3f} · จากมุม+ระยะ {a2:+.3f} · "
              f"จากมุม+ระยะ+ช่วง {a3:+.3f}")

    print("\n4) เทียบกับรอบแรก")
    print("   รอบแรกโมเดลอยู่ใต้เพดาน 'รู้แค่ตำแหน่ง' ทั้งสองแกน จึงเรียนรูปร่างไม่ได้")
    print("   รอบนี้ตัวเลขข้อ 3 ยิ่งต่ำ ยิ่งแปลว่ามีที่ให้เรียนเรื่องรูปร่างจริง")
    hi = r2(predict_from_bins([deg, near], h), h)
    if hi > 0.75:
        print("   -> ยังสูงมาก ท่าทางแทบไม่เป็นอิสระ **ไม่ควรเทรน** กลับไปเก็บใหม่")
    elif hi > 0.45:
        print("   -> ปานกลาง เทรนได้แต่คาดหวังได้จำกัด")
    else:
        print("   -> ต่ำพอ ท่าทางเป็นอิสระจากตำแหน่งจริง เทรนได้")


if __name__ == "__main__":
    main()
