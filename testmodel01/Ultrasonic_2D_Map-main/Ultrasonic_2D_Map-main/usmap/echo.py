"""Echo preprocessing: raw counts -> range profile.

Physics: the echo's time axis IS distance. For a TX->target->RX round trip,
distance_cm = speed_of_sound * t / 2. We baseline-correct, envelope via
Hilbert transform, then resample to a fixed number of range bins so bin i
corresponds to distance RANGE_CM * i / PROFILE_BINS.
"""
import numpy as np
from scipy.signal import hilbert

from .config import PROFILE_BINS, RANGE_CM, SPEED_OF_SOUND_CM_S


def load_echo(npz_path):
    d = np.load(npz_path)
    counts = d["counts"].astype(np.float64)      # (4, N)
    rate = float(d["rate"])                      # samples per second
    return counts, rate


def baseline_subtract(counts, n_fit=100):
    """Remove DC offset per channel using median of first samples (pre-echo)."""
    base = np.median(counts[:, :n_fit], axis=1, keepdims=True)
    return counts - base


def envelope(counts):
    return np.abs(hilbert(counts, axis=1))


def time_to_range_axis(n_samples, rate):
    """Two-way travel: r = c*t/2."""
    t = np.arange(n_samples) / rate
    return SPEED_OF_SOUND_CM_S * t / 2.0          # cm


def range_profile(npz_path):
    """Full chain for one frame -> (4, PROFILE_BINS) normalized profiles."""
    counts, rate = load_echo(npz_path)
    x = envelope(baseline_subtract(counts))
    r = time_to_range_axis(x.shape[1], rate)

    # Bin edges over [0, RANGE_CM]
    edges = np.linspace(0, RANGE_CM, PROFILE_BINS + 1)
    idx = np.clip(np.searchsorted(edges, r) - 1, 0, PROFILE_BINS - 1)

    prof = np.zeros((x.shape[0], PROFILE_BINS))
    cnts = np.zeros(PROFILE_BINS)
    np.add.at(cnts, idx, 1)
    for ch in range(x.shape[0]):
        np.add.at(prof[ch], idx, x[ch])
    prof /= np.maximum(cnts, 1)[None, :]

    # Per-channel normalization (max-abs, guarded)
    mx = prof.max(axis=1, keepdims=True)
    prof = prof / np.maximum(mx, 1e-9)
    return prof.astype(np.float32)


def tof_cm(npz_path, thresh_frac=0.3, ignore_cm=25.0):
    """Baseline physics estimate: first envelope crossing above threshold
    after an ignore zone (excludes burst ringdown/crosstalk)."""
    counts, rate = load_echo(npz_path)
    x = envelope(baseline_subtract(counts))
    peak = x.max(axis=1, keepdims=True)
    r = time_to_range_axis(x.shape[1], rate)
    above = (x > np.maximum(thresh_frac * peak, 30.0)) & (r[None, :] > ignore_cm)
    first = np.argmax(above, axis=1)
    first[~above.any(axis=1)] = len(r) - 1
    return r[first].astype(np.float32)           # (4,)
