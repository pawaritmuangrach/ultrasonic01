"""Physics features with repo-grade calibration:
- bandpass at the aliased 40 kHz band (bandpass sampling, rate~66kHz)
- T0 = 1220 us system delay (repo-calibrated against camera)
- common peak: shared echo window across channels using quietest channel as reference
- GCC-PHAT cross-correlation between all 6 channel pairs (learned-angle gold)
"""
import numpy as np
from scipy.signal import hilbert

C_CM_S = 34300.0
F0 = 40e3
T0_US = 1220.0
GATE_MIN_CM, GATE_MAX_CM = 40.0, 200.0
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def signal_band(rate):
    nyq = rate / 2.0
    if F0 < nyq * 0.95:
        return 25e3, min(60e3, nyq * 0.95)
    f_alias = abs(F0 - round(F0 / rate) * rate)
    half = min(8e3, f_alias * 0.6, (nyq - f_alias) * 0.85)
    return max(1e3, f_alias - half), min(nyq * 0.97, f_alias + half)


def bandpass(x, rate):
    from scipy.signal import butter, filtfilt
    lo, hi = signal_band(rate)
    b, a = butter(4, [lo / (rate / 2), hi / (rate / 2)], btype="band")
    return filtfilt(b, a, x)


def envelopes(counts, rate):
    """counts (4,N) float -> list of envelopes in the signal band."""
    out = []
    for ch in range(counts.shape[0]):
        v = counts[ch] - counts[ch].mean()
        try:
            v = bandpass(v, rate)
        except Exception:
            pass
        out.append(np.abs(hilbert(v)))
    return out


def _gate_idx(rate, n):
    lo = max(1, int((T0_US + 2 * GATE_MIN_CM / 34300 * 1e6) * 1e-6 * rate))
    hi = min(n, int((T0_US + 2 * GATE_MAX_CM / 34300 * 1e6) * 1e-6 * rate))
    if hi <= lo:
        lo, hi = 1, n
    return lo, hi


def index_to_cm(k, rate):
    return (k / rate * 1e6 - T0_US) * 1e-6 * C_CM_S / 2    # cm (C in cm/s, no x100)


def common_peak(envs, rate):
    """Repo trick: shared window via quietest channel -> (peak_idx, dist_cm, ref_ch)."""
    n = min(len(e) for e in envs)
    lo, hi = _gate_idx(rate, n)
    nf = [max(float(np.median(e[lo:hi])), 1e-12) for e in envs]
    ref = int(np.argmin(nf))
    k = lo + int(np.argmax(envs[ref][lo:hi]))
    return k, index_to_cm(k, rate), ref


# ระยะห่างหัวรับตามแกน X (cm) -> เพดาน TDOA ทางกายภาพของแต่ละคู่
def _rx_x():
    from .config import RECEIVERS, CHANNEL_NAMES
    return [RECEIVERS[n]["xy"][0] for n in CHANNEL_NAMES]


def max_lag_us(pair):
    """TDOA มากที่สุดที่คู่นี้เป็นไปได้ = ระยะห่างหัวรับ / ความเร็วเสียง
    (เกิดตอนเป้าอยู่ในแนวเดียวกับเส้นเชื่อมหัวรับพอดี) ค่าที่เกินนี้เป็นไปไม่ได้"""
    xs = _rx_x()
    return abs(xs[pair[0]] - xs[pair[1]]) / C_CM_S * 1e6


def pair_tdoa(e0, e1, rate, k, pair, win_us=1000):
    """TDOA ระหว่างเปลือกคลื่นสองช่อง -> (ไมโครวินาที, ความเชื่อมั่น 0-1)

    เขียนแทน gcc_phat เดิมซึ่งวัดจริงแล้วใช้ไม่ได้ (walk_s1: 60% ของเฟรมได้ 0 ทั้ง 6 คู่
    ที่เหลือได้ค่าถึง ±782 us ทั้งที่อาเรย์กว้าง 12 cm รองรับได้แค่ ±350 us) สามสาเหตุ:

    1. **PHAT whitening บนเปลือกคลื่น** — PHAT หารด้วยขนาดสเปกตรัมเพื่อให้ยอดคมขึ้น
       ซึ่งได้ผลกับสัญญาณที่มีแบนด์กว้าง แต่เปลือกคลื่นเรียบมาก การหารแบบนั้นจึงไป
       ขยายสัญญาณรบกวนความถี่สูงที่ไม่มีข้อมูลหน่วงเวลาอยู่เลย ยอดเลยไปโผล่มั่ว
       -> ใช้สหสัมพันธ์ไขว้แบบปกติ (normalized) ไม่ whiten
    2. **ไม่จำกัดช่วง lag** — เดิม zero-pad เป็น 128 จุดแล้วหา argmax ทั้งเส้น ทำให้
       lag ออกมาได้ถึง ±64 แซมเปิล (±965 us) ทั้งที่หน้าต่างที่ตัดมาซ้อนกันแค่ ±19
       -> ค้นเฉพาะช่วง ±max_lag_us ของคู่นั้น ค่าที่เป็นไปไม่ได้จึงเกิดไม่ได้
    3. **ไม่มีการประมาณย่อยแซมเปิล** — ที่ 66 kHz หนึ่งแซมเปิล = 15 us ซึ่งหยาบมาก
       เทียบกับช่วงทั้งหมด ±73 us ของคู่ที่ชิดกัน -> ใส่พาราโบลาหาจุดยอด

    คืนความเชื่อมั่นมาด้วย (ค่าสหสัมพันธ์ที่จุดยอด) เพราะเฟรมที่เสียงกลับอ่อน
    ค่าที่ได้ไม่ควรถูกเชื่อเท่ากับเฟรมที่ชัด
    """
    L = int(np.ceil(max_lag_us(pair) * 1e-6 * rate)) + 1
    w = int(win_us * 1e-6 * rate)
    lo, hi = max(0, k - w), min(len(e0), len(e1), k + w)
    a = np.asarray(e0[lo:hi], float)
    b = np.asarray(e1[lo:hi], float)
    if len(a) < 3 * L + 8:            # หน้าต่างต้องยาวพอให้เลื่อนได้เต็มช่วง
        return 0.0, 0.0
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0, 0.0
    lags = np.arange(-L, L + 1)
    cc = np.empty(len(lags))
    for j, m in enumerate(lags):
        if m >= 0:
            x, y = a[m:], (b[:len(b) - m] if m else b)
        else:
            x, y = a[:len(a) + m], b[-m:]
        cc[j] = float(np.dot(x, y)) / (na * nb)
    j = int(np.argmax(cc))
    off = 0.0
    if 0 < j < len(cc) - 1:
        y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            off = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return float((lags[j] + off) / rate * 1e6), float(max(cc[j], 0.0))


def gcc_phat(e0, e1, rate, k, win_us=300):
    """เก็บชื่อเดิมไว้ให้โค้ดเก่าเรียกได้ — เรียกต่อไปยังตัวที่แก้แล้ว"""
    return pair_tdoa(e0, e1, rate, k, (0, 1), win_us=max(win_us, 1000))[0]


def physics_features(npz_path):
    """-> dict with everything cheap and physical."""
    d = np.load(npz_path)
    counts = d["counts"].astype(float)
    rate = float(d["rate"])
    envs = envelopes(counts, rate)

    k, dist_cm, ref = common_peak(envs, rate)
    amps = np.array([e[max(0, k - 20):k + 20].max() for e in envs])
    noise = np.array([np.median(e[_gate_idx(rate, len(e))[0]:_gate_idx(rate, len(e))[1]])
                      for e in envs])
    snr = amps / np.maximum(noise, 1e-12)
    _t = [pair_tdoa(envs[a], envs[b], rate, k, (a, b)) for a, b in PAIRS]
    tdoa = np.array([v for v, _ in _t])
    tconf = np.array([c for _, c in _t])
    # หารด้วยเพดานของคู่ตัวเอง -> อยู่ในช่วง -1..+1 เท่ากันทุกคู่
    # ของเดิมป้อนเป็นไมโครวินาทีดิบ (ถึง ±782) เข้าชั้น Linear ที่ฟีเจอร์อื่นอยู่แถว 0-25
    # ตัวที่ใหญ่กว่าเป็นสิบเท่าจะครอบงำการเรียนรู้ช่วงต้น
    tnorm = np.clip(tdoa / np.array([max_lag_us(p) for p in PAIRS]), -1.0, 1.0)

    # range-profile (128 bins over gate) per channel, normalized
    prof = range_profile_from_envs(envs, rate)
    return {
        "rate": rate, "dist_cm": dist_cm, "ref": ref,
        "amps": (amps / max(amps.max(), 1e-9)).astype(np.float32),
        "snr": snr.astype(np.float32),
        "tdoa_us": tdoa.astype(np.float32),          # ไมโครวินาทีจริง ไว้ดู/ดีบัก
        "tdoa": tnorm.astype(np.float32),            # ตัวที่ป้อนโมเดล (-1..1)
        "tdoa_conf": tconf.astype(np.float32),
        "prof": prof,
        "envs": envs, "k": k,
    }


def range_profile_from_envs(envs, rate, bins=128):
    n = min(len(e) for e in envs)
    lo, hi = _gate_idx(rate, n)
    r = np.array([index_to_cm(i, rate) for i in range(lo, hi)])
    edges = np.linspace(GATE_MIN_CM, GATE_MAX_CM, bins + 1)
    idx = np.clip(np.searchsorted(edges, r) - 1, 0, bins - 1)
    out = np.zeros((len(envs), bins), np.float32)
    cnt = np.zeros(bins)
    np.add.at(cnt, idx, 1)
    for ch, e in enumerate(envs):
        np.add.at(out[ch], idx, e[lo:hi])
    out /= np.maximum(cnt, 1)[None, :]
    mx = out.max(axis=1, keepdims=True)
    return out / np.maximum(mx, 1e-9)


def tof_cm_simple(npz_path):
    """Backward-compatible ToF with ignore zone (used by old grid pipeline)."""
    f = physics_features(npz_path)
    return np.full(4, f["dist_cm"], np.float32) + f["tdoa_us"] * 0.0343 / 2
