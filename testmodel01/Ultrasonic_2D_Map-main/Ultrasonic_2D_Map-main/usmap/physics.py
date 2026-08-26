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


def gcc_phat(e0, e1, rate, k, win_us=300):
    """GCC-PHAT delay between two envelopes around shared peak window.
    Returns delay in microseconds (positive = e1 later than e0)."""
    w = max(3, int(win_us * 1e-6 * rate))
    lo, hi = max(0, k - w), min(min(len(e0), len(e1)), k + w)
    a = e0[lo:hi] - e0[lo:hi].mean()
    b = e1[lo:hi] - e1[lo:hi].mean()
    if len(a) < 8:
        return 0.0
    n = 1 << (2 * len(a) - 1).bit_length()
    X = np.fft.rfft(a, n); Y = np.fft.rfft(b, n)
    cc = np.fft.irfft(X * np.conj(Y) / (np.abs(X * np.conj(Y)) + 1e-9), n)
    lag = int(np.argmax(cc)) if int(np.argmax(cc)) < len(cc) // 2 \
        else int(np.argmax(cc)) - len(cc)
    return lag / rate * 1e6


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
    tdoa = np.array([gcc_phat(envs[a], envs[b], rate, k) for a, b in PAIRS])

    # range-profile (128 bins over gate) per channel, normalized
    prof = range_profile_from_envs(envs, rate)
    return {
        "rate": rate, "dist_cm": dist_cm, "ref": ref,
        "amps": (amps / max(amps.max(), 1e-9)).astype(np.float32),
        "snr": snr.astype(np.float32),
        "tdoa_us": tdoa.astype(np.float32),
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
