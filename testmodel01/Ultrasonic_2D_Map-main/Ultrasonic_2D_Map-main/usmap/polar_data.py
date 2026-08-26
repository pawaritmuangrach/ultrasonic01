"""PolarScan dataset: physics features + polar labels per frame."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATASET, TRAIN_SECTIONS, VAL_SECTIONS, TEST_SECTIONS
from .physics import physics_features, PAIRS, GATE_MAX_CM, GATE_MIN_CM
from .polar_gt import polar_label


def section_pairs(section):
    d = DATASET / section
    depth = {p.stem.split("_")[1]: p for p in sorted(d.glob("depth_*.png"))}
    out = []
    for u in sorted(d.glob("us_*.npz")):
        i = u.stem.split("_")[1]
        if i in depth:
            out.append((u, depth[i]))
    return out


def make_sample(us_path, depth_path):
    f = physics_features(str(us_path))
    dist, valid = polar_label(str(depth_path))
    # nearest-object targets (repo-comparable metrics)
    if valid.any():
        edges = np.linspace(-29.2, 29.2, 14)
        angs = (edges[:-1] + edges[1:]) / 2
        i = int(np.argmin(np.where(valid, dist, 1e9)))
        near_d, near_a = float(dist[i]), float(angs[i])
    else:
        near_d, near_a = np.nan, np.nan
    return {
        "prof": f["prof"],                       # (4,128)
        "tdoa": f["tdoa_us"],                    # (6,)
        "amps": f["amps"], "snr": f["snr"],
        "dist_cm": f["dist_cm"],
        "label": dist, "valid": valid,
        "near_d": near_d, "near_a": near_a,
    }


class PolarDataset(Dataset):
    def __init__(self, sections, cache=True):
        self.pairs = []
        for s in sections:
            self.pairs += section_pairs(s)
        self.cache = cache
        self._c = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if i in self._c:
            s = self._c[i]
        else:
            u, d = self.pairs[i]
            s = make_sample(u, d)
            if self.cache:
                self._c[i] = s
        t = lambda x: torch.from_numpy(np.asarray(x, np.float32))
        return (t(s["prof"]), t(s["tdoa"]), t(s["amps"]), t(s["snr"]),
                t(s["label"]), torch.from_numpy(s["valid"]),
                t([s["dist_cm"]]), t([s["near_d"]]), t([s["near_a"]]))


def splits():
    return (PolarDataset(TRAIN_SECTIONS), PolarDataset(VAL_SECTIONS),
            PolarDataset(TEST_SECTIONS))
