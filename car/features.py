"""แปลงคลื่นดิบหนึ่งปิง เป็นเวกเตอร์ feature ให้ NN

ยังไม่ป้อนคลื่นดิบ 40 kHz ตรงๆ ด้วยเหตุผลเชิงข้อมูล: มีตัวอย่างหลักสิบ แต่คลื่นดิบ
มีหลักพันจุดต่อช่อง โมเดลจะจำ noise แทนที่จะเรียน ย่อเป็นเอนเวโลปก่อน = เก็บข้อมูล
"เสียงกลับแรงแค่ไหน ที่ระยะไหน" ซึ่งเป็นแกนหลักของ "อะไรอยู่ข้างหน้า" ไว้ครบ
(เฟส/TDOA ละเอียดหายไป แต่นั่นไว้ใช้ทีหลังตอนมีข้อมูลเยอะและช่องเยอะ)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from scope_view import bandpass, envelope, counts_to_volts   # noqa: E402

C = 343.0
N_BINS = 32           # ย่อเอนเวโลปเหลือ 32 จุดต่อช่อง (~7 cm/จุด) — 128 จุดเดิมมิติเยอะไป
                      # ทำให้โมเดลจำ noise (overfit หนักตอนข้ามระยะ) 32 พอเก็บรูปเอคโค่แล้ว
R_MIN_CM, R_MAX_CM = 30, 250

# เวลาศูนย์ของระบบ: หักเวลายิง (เบิร์สต์ 16 รอบ = 400 µs) + หน่วงเอนเวโลป/ฟิลเตอร์ +
# ออฟเซ็ตทริกเกอร์ ค่านี้ **คาลิเบรตกับกล้อง** จาก 454 ตัวอย่าง 3 ระยะ (d70/d100/d130):
# ค่าเดิม 435 µs ทำให้ US อ่านเกินจริง +13 cm ทุกระยะ · แก้เป็น 1220 แล้ว US ตรงกับกล้อง
# ภายใน 20 cm ถึง 93% (จาก 78%) และ MAE 20 → 9.7 cm
# (ค่าดีที่สุดรายฉากคือ 1110/1210/1300 — ไล่ขึ้นตามระยะเล็กน้อย ใช้ค่ากลางพอ)
T0_US = 1220.0

# ช่วงระยะที่ยอมให้ค้นหาเอคโค่ — กันไม่ให้ argmax ไปเกาะเสียงสะท้อนใกล้ๆ (ขอบโต๊ะ/ขาตั้ง/
# ring-down) ที่บางเฟรมดังกว่าเป้า เคยเห็นอ่านได้ 26 cm ทั้งที่เป้าอยู่ 130 cm
# สำคัญกับ "มุม" ด้วย เพราะหน้าต่างหา TDOA อิงตำแหน่งยอดนี้ ยอดผิด = มุมผิดตาม
GATE_MIN_CM, GATE_MAX_CM = 40.0, 200.0

# เป้าเคลื่อนที่ (เช่นคนเดิน) ต้องให้แต่ละคู่ช่องหายอดของตัวเอง — ดู angle_features
PER_PAIR = False


def _cm_to_index(cm, rate):
    return int((T0_US + 2 * cm / 100 / C * 1e6) * 1e-6 * rate)


def echo_peak(e, rate, lo_cm=GATE_MIN_CM, hi_cm=GATE_MAX_CM):
    """คืน (index, ระยะ cm) ของยอดเอคโค่ที่แรงสุด ภายในช่วงระยะที่กำหนด"""
    i_lo = max(1, _cm_to_index(lo_cm, rate))
    i_hi = min(len(e), _cm_to_index(hi_cm, rate))
    if i_hi <= i_lo:
        i_lo, i_hi = 1, len(e)
    k = i_lo + int(np.argmax(e[i_lo:i_hi]))
    return k, (k / rate * 1e6 - T0_US) * 1e-6 * C / 2 * 100


def common_peak(envs, rate, lo_cm=GATE_MIN_CM, hi_cm=GATE_MAX_CM):
    """หายอดเอคโค่ **ร่วมของทุกช่อง** โดยเชื่อช่องที่พื้นเสียงรบกวนต่ำที่สุด

    เป้าอยู่ที่ระยะเดียว ทุกช่องจึงควรอ่านที่หน้าต่างเดียวกัน · ถ้าปล่อยให้แต่ละช่อง
    argmax เอง ช่องที่สายยาว/รบกวนสูงจะไปเกาะ "ยอดปลอม" จากสัญญาณรบกวน
    วัดจริงกับ 790 ตัวอย่าง: แต่ละช่องหาเอง MAE 20.6 cm (ถูก 57%) เทียบกับ
    ใช้ยอดร่วม MAE 8.2 cm (ถูก 94%)

    เลือกช่องอ้างอิงจากพื้นเสียงต่ำสุดแทนการฮาร์ดโค้ด เพื่อให้ปรับตามฮาร์ดแวร์เอง
    (ตอนนี้ ch0/C140 เงียบสุดทุกเฟรม เพราะสายสั้นกว่าช่องอื่น)

    คืน (index, ระยะ cm, ช่องอ้างอิงที่เลือก)
    """
    n = min(len(e) for e in envs)
    i_lo = max(1, _cm_to_index(lo_cm, rate))
    i_hi = min(n, _cm_to_index(hi_cm, rate))
    if i_hi <= i_lo:
        i_lo, i_hi = 1, n
    nf = [max(float(np.median(e[i_lo:i_hi])), 1e-12) for e in envs]
    ref = int(np.argmin(nf))
    k = i_lo + int(np.argmax(envs[ref][i_lo:i_hi]))
    return k, (k / rate * 1e6 - T0_US) * 1e-6 * C / 2 * 100, ref


F0 = 40e3                 # ความถี่ทำงานของหัวส่ง/หัวรับ


def signal_band(rate):
    """ย่านที่ต้องกรอง — ขึ้นกับอัตราสุ่ม เพราะเราสุ่มต่ำกว่า Nyquist ได้ (bandpass sampling)

    อ่าน 4 ช่องพร้อมกัน ESP32 ให้ได้แค่ ~66 kHz/ช่อง ซึ่งต่ำกว่า 2x40 kHz
    คลื่น 40 kHz จึง "พับ" ลงมาปรากฏที่ |40k - n x rate| — ข้อมูลไม่หาย เพราะสัญญาณเรา
    เป็นแถบแคบ การพับความถี่จึงเป็นแค่การเลื่อนย่าน เปลือกคลื่นและ TDOA ยังอยู่ครบ
    แต่ **ต้องกรองที่ย่านใหม่** ถ้ายังกรอง 25-60 kHz จะได้แต่ศูนย์ (เกิน Nyquist ไปแล้ว)
    """
    nyq = rate / 2.0
    if F0 < nyq * 0.95:                    # สุ่มเร็วพอ ใช้ย่านจริงตามปกติ
        return 25e3, min(60e3, nyq * 0.95)
    f_alias = abs(F0 - round(F0 / rate) * rate)
    half = min(8e3, f_alias * 0.6, (nyq - f_alias) * 0.85)
    return max(1e3, f_alias - half), min(nyq * 0.97, f_alias + half)


def envelope_of(counts_ch, rate):
    """คลื่นดิบหนึ่งช่อง -> เอนเวโลปในย่านสัญญาณ (เลือกย่านตามอัตราสุ่มอัตโนมัติ)"""
    v = counts_to_volts(counts_ch.astype(float))
    lo, hi = signal_band(rate)
    return envelope(bandpass(v - v.mean(), rate, lo, hi))


def waveform_features(counts, rate):
    """คืนเวกเตอร์ feature ของหนึ่งปิง

    สองส่วน:
      1. เอนเวโลปย่อของทุกช่อง (N_BINS ต่อช่อง) — เก็บ "แรงแค่ไหน ที่ระยะไหน" = ระยะ
      2. feature ข้ามช่อง — เก็บ "ทิศ" ซึ่งอยู่ที่ความต่างระหว่างช่อง ไม่ใช่ในช่องเดียว
         ฟิสิกส์: มุม ∝ TDOA ระหว่างช่อง และอัตราส่วนความแรงสองช่อง
    """
    envs = []
    for ch in range(counts.shape[0]):
        v = counts_to_volts(counts[ch].astype(float))
        envs.append(envelope(bandpass(v - v.mean(), rate, 25e3, 60e3)))

    edges = np.linspace(R_MIN_CM, R_MAX_CM, N_BINS + 1)
    feats = []
    for e in envs:
        t_us = (np.arange(len(e)) / rate * 1e6) - T0_US
        rng_cm = t_us * 1e-6 * C / 2 * 100
        idx = np.digitize(rng_cm, edges) - 1
        binned = np.zeros(N_BINS, np.float32)
        for b in range(N_BINS):
            m = idx == b
            if m.any():
                binned[b] = e[m].max()
        feats.append(binned * 1e3)

    # --- feature ข้ามช่อง = ทิศ ทำต่อทุกคู่เบสไลน์ ---
    # 2 ช่อง: คู่เดียว (A140-C140, เบสไลน์ 242mm)
    # 4 ช่อง: สองคู่ (A140-C140 242mm ละเอียด · A80-C80 139mm หยาบไม่กำกวม)
    #   = coarse-to-fine ที่วางแผนไว้ คู่หยาบช่วยแก้ cycle ambiguity ของคู่ละเอียด
    for p in range(0, len(envs) - 1, 2):
        feats.append(_pair_direction_features(envs[p], envs[p + 1], rate))

    return np.concatenate(feats)


def _pair_direction_features(e0, e1, rate, k=None):
    """feature บอกทิศจากหนึ่งคู่เบสไลน์: log อัตราส่วน + TDOA + ระยะยอด

    k = index ของยอดเอคโค่ที่จะใช้เป็นศูนย์กลางหน้าต่าง (ปกติมาจาก common_peak)
    ถ้าไม่ส่งมาจะหายอดจาก e0 เอง (ของเดิม ใช้กับโค้ดเก่า)
    """
    if k is None:
        k, _ = echo_peak(e0, rate)
    w = int(3e-4 * rate)
    lo, hi = max(0, k - w), k + w
    a0 = float(e0[lo:hi].max()); a1 = float(e1[lo:hi].max())
    ratio = np.log((a1 + 1e-6) / (a0 + 1e-6))              # ทิศ (ขนาด)
    seg0 = e0[lo:hi] - e0[lo:hi].mean()
    seg1 = e1[lo:hi] - e1[lo:hi].mean()
    corr = np.correlate(seg1, seg0, mode="full")
    lag = (np.argmax(corr) - (len(seg0) - 1)) / rate * 1e6   # ทิศ (TDOA, µs)
    return np.array([ratio * 20, lag, k / rate * 1e6 * 1e-3], np.float32)


def distance_features(counts, rate, per_pair=None):
    """feature สำหรับ 'ระยะ' — ระยะของยอดร่วม + ความแรงที่ยอดนั้นของแต่ละช่อง

    ใช้ยอด **ร่วม** (ดู common_peak) ไม่ใช่ให้แต่ละช่องหายอดเอง เพราะเป้าอยู่ระยะเดียว
    และช่องที่รบกวนสูงจะไปเกาะยอดปลอม (MAE 20.6 -> 8.2 cm)
    """
    per_pair = PER_PAIR if per_pair is None else per_pair
    envs = [envelope_of(counts[ch], rate) for ch in range(counts.shape[0])]
    if per_pair:
        # เป้าเคลื่อนที่: ใช้ระยะจากคู่แรก (ยิงรอบแรก) และวัดความแรงของแต่ละคู่
        # ที่ยอดของคู่ตัวเอง ไม่งั้นคู่ที่สองอ่านความแรงผิดตำแหน่ง
        out = []
        for p in range(0, len(envs) - 1, 2):
            pair = [envs[p], envs[p + 1]]
            kp, rngp, _ = common_peak(pair, rate)
            if not out:
                out.append(rngp)
            out += [float(e[kp]) * 1e3 for e in pair]
        return np.array(out, np.float32)
    k, rng, _ = common_peak(envs, rate)
    out = [rng] + [float(e[k]) * 1e3 for e in envs]
    return np.array(out, np.float32)


def angle_features(counts, rate, per_pair=None):
    """feature สำหรับ 'มุม' — TDOA + อัตราส่วนความแรง ต่อคู่เบสไลน์ (3 ต่อคู่)

    มุมอยู่ที่ความต่างระหว่างช่อง ไม่ใช่ในช่องเดียว จึงใช้เฉพาะ feature ข้ามช่อง
    ไม่ปนเอนเวโลปในช่อง (ซึ่งเป็น noise สำหรับมุม)

    ทุกคู่ใช้ **หน้าต่างจากยอดร่วม** เดียวกัน (common_peak) — เดิมคู่ที่สองใช้ยอดของ
    ch2 ซึ่งเป็นช่องที่รบกวนสูงสุด หน้าต่างจึงผิดบ่อยและมุมของเบสไลน์สั้นใช้ไม่ได้
    """
    per_pair = PER_PAIR if per_pair is None else per_pair
    envs = [envelope_of(counts[ch], rate) for ch in range(counts.shape[0])]
    if per_pair:
        # เป้าเคลื่อนที่: 4 ช่องต้องยิงสองรอบห่างกัน ~190 ms คนเดิน 0.5 m/s ขยับ 9.5 cm
        # ซึ่งเกินหน้าต่างวิเคราะห์ (±5 cm) -> คู่ที่สองพลาดเอคโค่ถ้าใช้ยอดของคู่แรก
        # ให้แต่ละคู่หายอดของตัวเอง (ภายในคู่ ADC สุ่มพร้อมกัน ห่างแค่ 1.25 µs จึงตรงเสมอ)
        feats = []
        for p in range(0, len(envs) - 1, 2):
            pair = [envs[p], envs[p + 1]]
            kp, _, _ = common_peak(pair, rate)
            feats.append(_pair_direction_features(pair[0], pair[1], rate, kp))
        return np.concatenate(feats)
    k, _, _ = common_peak(envs, rate)
    feats = [_pair_direction_features(envs[p], envs[p + 1], rate, k)
             for p in range(0, len(envs) - 1, 2)]
    return np.concatenate(feats)



def load_scene(scene_dir):
    """คืน (X[n, feat], us_range[n], meta) จากโฟลเดอร์ฉากหนึ่ง"""
    import glob
    import cv2
    us = sorted(glob.glob(os.path.join(scene_dir, "us_*.npz")))
    X, idxs = [], []
    for u in us:
        i = os.path.basename(u).split("_")[1].split(".")[0]
        if not os.path.exists(os.path.join(scene_dir, f"depth_{i}.png")):
            continue
        z = np.load(u)
        X.append(waveform_features(z["counts"], float(z["rate"])))
        idxs.append(i)
    return np.array(X, np.float32), idxs
