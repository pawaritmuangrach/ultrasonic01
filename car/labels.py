"""แปลงภาพความลึกจากกล้อง เป็น label ที่รถใช้จริง — "อะไรอยู่ตรงไหนข้างหน้า"

รถไม่ต้องการ depth map 640x480 มันต้องการคำตอบเดียว: **ในแต่ละทิศข้างหน้า
สิ่งกีดขวางที่ใกล้ที่สุดอยู่ไกลเท่าไหร่** นั่นคือโปรไฟล์เชิงมุม (polar profile)
ซึ่งเอาไปสั่งเลี้ยว/หยุดได้ตรงๆ และเป็นรูปแบบที่อัลตราซาวด์ 2-4 ช่องมีทางทำนายได้จริง
(ทำนาย depth map เต็มภาพจากเสียงไม่กี่ช่องเป็นงานที่เกินตัวเซ็นเซอร์)

    depth 640x480  ->  ระยะที่ใกล้ที่สุด ในแต่ละช่องมุม N ช่อง  (+ mask ว่าช่องไหนเชื่อได้)

ทำไมเอา "ใกล้ที่สุด" ไม่ใช่ค่าเฉลี่ย: สำหรับการหลบสิ่งกีดขวาง สิ่งที่ใกล้ที่สุดคือ
สิ่งที่จะชน ค่าเฉลี่ยจะถูกพื้นหลังที่ไกลดึงจนสิ่งกีดขวางบางๆ หายไป

ทำไมตัดเฉพาะแถบกลางแนวตั้ง: อัลตราซาวด์มองเป็นกรวยแคบในแนวตั้งเช่นกัน ถ้าเอา
ทั้งภาพ พื้นและเพดานจะเข้ามาปนเป็น label ที่เสียงไม่เคยได้ยิน
"""
import numpy as np

# Astra Pro: มุมมองของ depth ประมาณ 58.4 องศาแนวนอน x 45.5 แนวตั้ง
FOV_H_DEG = 58.4
FOV_V_DEG = 45.5

N_BINS = 9             # ช่องมุม ครอบคลุมมุมมองกล้อง -> ช่องละ ~6.5 องศา
V_BAND = 0.40          # ใช้แถบกลางแนวตั้ง 40% ของภาพ (ตัดพื้น/เพดานทิ้ง)
MIN_MM, MAX_MM = 400, 2500     # นอกช่วงนี้ไม่เชื่อ (ต่ำกว่า = จุดบอดกล้อง)
MIN_PIX = 40           # ต้องมีพิกเซลที่ใช้ได้อย่างน้อยเท่านี้ จึงจะเชื่อช่องนั้น


def depth_to_profile(depth, n_bins=N_BINS, v_band=V_BAND,
                     min_mm=MIN_MM, max_mm=MAX_MM, min_pix=MIN_PIX):
    """คืน (dist_mm[n_bins], valid[n_bins])

    dist_mm = ระยะของสิ่งที่ใกล้ที่สุดในช่องมุมนั้น (หน่วยมิลลิเมตร)
    valid   = True ถ้าช่องนั้นมีพิกเซลพอให้เชื่อได้ (ใช้เป็น mask ตอนคิด loss)
    """
    h, w = depth.shape
    r0 = int(h * (0.5 - v_band / 2))
    r1 = int(h * (0.5 + v_band / 2))
    band = depth[r0:r1, :]

    dist = np.zeros(n_bins, np.float32)
    valid = np.zeros(n_bins, bool)
    edges = np.linspace(0, w, n_bins + 1).astype(int)
    for i in range(n_bins):
        col = band[:, edges[i]:edges[i + 1]]
        ok = col[(col >= min_mm) & (col <= max_mm)]
        if ok.size >= min_pix:
            # เปอร์เซ็นไทล์ที่ 5 ไม่ใช่ min เพราะ min ไวต่อพิกเซลเสียตัวเดียว
            dist[i] = _percentile5(ok)
            valid[i] = True
    return dist, valid


def _percentile5(a):
    """เปอร์เซ็นไทล์ที่ 5 แบบ linear interpolation — ให้ค่าเท่า np.percentile(a, 5) เป๊ะ

    ทำไมไม่ใช้ np.percentile ตรงๆ: ข้างในมันเรียก np.unique ซึ่ง **lazy-import numpy.ma**
    ครั้งแรกที่ถูกเรียก การโหลดโมดูลกลางคันตอนที่ OpenNI (native) ทำงานอยู่ ทำให้โปรเซส
    ล้มระดับ native เงียบๆ (เจอจริงตอนเก็บ dataset — faulthandler ชี้จุดนี้)
    np.partition ไม่แตะเส้นทางนั้นและเร็วกว่าด้วย (O(n) ไม่ต้องเรียงทั้งชุด)
    """
    k = (a.size - 1) * 0.05
    lo = int(k)                      # floor
    hi = min(lo + 1, a.size - 1)
    part = np.partition(a, (lo, hi))
    v = float(part[lo])
    if hi > lo:
        v += (float(part[hi]) - v) * (k - lo)
    return v


def bin_angles(n_bins=N_BINS, fov_deg=FOV_H_DEG):
    """มุมกลางของแต่ละช่อง หน่วยองศา (ลบ = ซ้าย, บวก = ขวา)"""
    edges = np.linspace(-fov_deg / 2, fov_deg / 2, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def profile_to_text(dist, valid, n_bins=N_BINS):
    """พิมพ์โปรไฟล์เป็นบรรทัดเดียวไว้ดูเร็วๆ"""
    ang = bin_angles(n_bins)
    parts = []
    for a, d, v in zip(ang, dist, valid):
        parts.append(f"{a:+5.1f}°:{d/10:5.0f}cm" if v else f"{a:+5.1f}°:  --- ")
    return " ".join(parts)


def near_object_count(depth, band_cm=20, n_bins=N_BINS):
    """จำนวน 'ก้อนของที่ใกล้พอๆ กับตัวใกล้สุด' ที่แยกกันในแนวมุม
    1 = วัตถุเดียว (เช่นท่อ) · >1 = มีของใกล้หลายจุด (เช่นติดขาคนเก็บ) → ควรข้ามเฟรมนั้น
    ใช้กรอง label ที่กำกวม (ไม่รู้ว่าเลเบลท่อหรือขา)"""
    import numpy as np
    dist, valid = depth_to_profile(depth, n_bins=n_bins)
    if not valid.any():
        return 0
    nearest = float(np.min(np.where(valid, dist, 1e9)))
    near = valid & (dist < nearest + band_cm * 10)
    groups, prev = 0, False
    for b in near:
        if b and not prev:
            groups += 1
        prev = bool(b)
    return groups

# ---------------------------------------------------------------- เฉลยของเป้าเดี่ยว
PERSON_THICK_MM = 400      # ความ "หนา" ของเป้าที่ยอมนับเป็นก้อนเดียวกัน (คนหนาราว 30 cm)
MIN_BLOB_PIX = 200         # พิกเซลขั้นต่ำจึงจะเชื่อว่าเห็นเป้า


def target_angle(depth, v_band=V_BAND, min_mm=MIN_MM, max_mm=MAX_MM,
                 thick_mm=PERSON_THICK_MM, min_pix=MIN_BLOB_PIX, fov_deg=FOV_H_DEG):
    """มุมของ **จุดศูนย์กลางเป้า** — คืน (deg, dist_cm, coverage) หรือ None ถ้าไม่เห็นเป้า

    deg      : + = เป้าอยู่ทางขวาของภาพ, - = ทางซ้าย  (0 = กลางภาพ)
    dist_cm  : ระยะของเป้า
    coverage : สัดส่วนพิกเซลที่เป้ากินในแถบกลาง — ต่ำมาก (<10%) แปลว่าเป้าเกือบหลุดเฟรม
               แล้วมุมจะติดขอบ ไม่ใช่ตำแหน่งจริง

    **ทำไมไม่ใช้ช่องมุมที่ใกล้ที่สุด** (depth_to_profile + argmin):
        คนที่ยืนห่าง 60 cm กินภาพ 5-7 ช่องมุม ที่ระยะ *เท่ากันหมด* ต่างกันแค่ 1-3 cm
        ตามความโค้งของลำตัว argmin จึงเลือกช่องแบบสุ่มไปมาในกลุ่มนั้น ได้เฉลยที่
        กระโดดข้ามภาพทั้ง ๆ ที่คนยืนนิ่ง (วัดจริงกับ walk_s1: เฉลยเด้ง +-26 องศา
        ขณะยืนนิ่ง ทำให้กฎทายทิศแพ้การเดามั่ว)
        จุดศูนย์กลางของก้อนเป้าไม่มีปัญหานี้ เพราะเฉลี่ยทั้งลำตัว
    """
    h, w = depth.shape
    band = depth[int(h * (0.5 - v_band / 2)):int(h * (0.5 + v_band / 2)), :]
    ok = (band >= min_mm) & (band <= max_mm)
    if int(ok.sum()) < min_pix:
        return None
    v = band[ok]
    near = _percentile5(v)
    # เป้า = พิกเซลที่ลึกใกล้เคียงตัวที่ใกล้สุด (ตัดพื้นหลังทิ้ง)
    m = ok & (band < near + thick_mm)
    if int(m.sum()) < min_pix:
        return None
    cols = np.nonzero(m)[1]
    deg = (cols.mean() / (w - 1) - 0.5) * fov_deg
    return float(deg), float(near) / 10.0, float(m.sum()) / m.size
