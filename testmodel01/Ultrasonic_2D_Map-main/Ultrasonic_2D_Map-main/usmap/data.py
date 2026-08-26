"""Dataset: pairs each US npz with its nearest depth PNG (they are saved at the
same tick, index-matched by number)."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATASET, TEST_SECTIONS, TRAIN_SECTIONS, VAL_SECTIONS
from .echo import range_profile, tof_cm
from .groundtruth import make_target


def section_files(section):
    d = DATASET / section
    us = sorted(d.glob("us_*.npz"))
    depth = {p.stem.split("_")[1]: p for p in sorted(d.glob("depth_*.png"))}
    pairs = []
    for u in us:
        idx = u.stem.split("_")[1]
        if idx in depth:
            pairs.append((u, depth[idx]))
    return pairs


class UsMapDataset(Dataset):
    def __init__(self, sections, cache_profiles=True):
        self.pairs = []
        for s in sections:
            self.pairs += section_files(s)
        self.cache = cache_profiles

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        u, d = self.pairs[i]
        prof = range_profile(u)                     # (4, BINS)
        tgt, vis_any, vis_per = make_target(d)
        # replace NaN with 0 and provide valid mask
        valid = ~torch.isnan(tgt)
        tgt = torch.nan_to_num(tgt, nan=0.0)
        return (torch.from_numpy(prof), tgt, valid.float(), vis_any,
                vis_per, tof_cm(u))


def splits():
    return (UsMapDataset(TRAIN_SECTIONS), UsMapDataset(VAL_SECTIONS),
            UsMapDataset(TEST_SECTIONS))
