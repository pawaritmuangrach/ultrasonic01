"""PolarScan model: CNN on range profiles + GCC-PHAT TDOA branch.
Outputs: per-bin distance (masked), nearest-object distance & angle."""
import numpy as np
import torch
import torch.nn as nn

from .config import MODELS
from .polar_gt import N_ANG


class PolarNet(nn.Module):
    def __init__(self, bins=128, n_pairs=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),   # 64
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 32
            nn.Flatten(),
        )
        feat = 64 * 32 + n_pairs + 4 + 4 + 1     # + tdoa + amps + snr + phys dist
        self.head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.bin_dist = nn.Linear(128, N_ANG)
        self.bin_logit = nn.Linear(128, N_ANG)   # validity/occupancy
        self.near_d = nn.Linear(128, 1)
        self.near_a = nn.Linear(128, 1)

    def forward(self, prof, tdoa, amps, snr, dist_cm):
        z = self.enc(prof)
        x = torch.cat([z, tdoa, amps, snr, dist_cm / 200.0], dim=1)
        h = self.head(x)
        return {
            "bin_dist": self.bin_dist(h),          # cm
            "bin_valid": self.bin_logit(h),        # logits
            "near_d": self.near_d(h).squeeze(1),
            "near_a": self.near_a(h).squeeze(1),
        }


def masked_bin_loss(out, label, valid):
    m = valid
    if m.sum() == 0:
        return out["bin_dist"].sum() * 0.0
    l1 = nn.functional.l1_loss(out["bin_dist"][m], label[m])
    bce = nn.functional.binary_cross_entropy_with_logits(out["bin_valid"], m.float())
    return l1 + 0.5 * bce


@torch.no_grad()
def eval_metrics(model, dl, device):
    """Returns dict incl. repo-comparable metrics: near-distance MAE,
    nearest-angle MAE, and physics baseline for distance."""
    model.eval()
    md, ma, pd = [], [], []
    with torch.no_grad():
        for prof, tdoa, amps, snr, label, valid, dist_cm, near_d, near_a in dl:
            prof, tdoa, amps, snr = prof.to(device), tdoa.to(device), amps.to(device), snr.to(device)
            dist_cm = dist_cm.to(device)
            out = model(prof, tdoa, amps, snr, dist_cm)
            nd, na = near_d.numpy().squeeze(-1), near_a.numpy().squeeze(-1)
            ok = ~np.isnan(nd)
            if ok.any():
                md.append(np.abs(out["near_d"].cpu().numpy()[ok] - nd[ok]))
                ma.append(np.abs(out["near_a"].cpu().numpy()[ok] - na[ok]))
                pd.append(np.abs(dist_cm.cpu().numpy()[:, 0][ok] - nd[ok]))
    return {
        "model_near_mae": float(np.concatenate(md).mean()) if md else float("nan"),
        "model_angle_mae": float(np.concatenate(ma).mean()) if ma else float("nan"),
        "physics_near_mae": float(np.concatenate(pd).mean()) if pd else float("nan"),
    }
