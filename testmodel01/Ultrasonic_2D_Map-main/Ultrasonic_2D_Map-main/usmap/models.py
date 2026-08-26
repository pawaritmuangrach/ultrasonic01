"""Models: small 1D-CNN echo encoder -> Cartesian map decoder.
Plus physics baseline (paint ToF along receiver rays)."""
import numpy as np
import torch
import torch.nn as nn

from .config import GRID_H, GRID_W, PROFILE_BINS, RECEIVERS


class EchoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = len(RECEIVERS)
        self.net = nn.Sequential(
            nn.Conv1d(ch, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),   # -> 64
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),   # -> 32
            nn.Conv1d(64, 96, 3, stride=2, padding=1), nn.ReLU(),   # -> 16
            nn.Flatten(),
        )
        self.out_dim = 96 * 16
    def forward(self, x):
        return self.net(x)


class UsMapNet(nn.Module):
    """Echo profiles (4,BINS) -> distance grid (1,H,W), cm."""
    def __init__(self):
        super().__init__()
        self.enc = EchoEncoder()
        d = self.enc.out_dim
        self.dec = nn.Sequential(
            nn.Linear(96 * 16 + 8, 504), nn.ReLU(),    # encoder flatten + ToF(4)+amp(4) hints
            nn.Unflatten(1, (8, 3, 21)),               # -> (B,8,3,21) for ConvTranspose2d
            nn.ConvTranspose2d(8, 16, 4, stride=(2, 2), padding=1), nn.ReLU(),   # 6x42
            nn.ConvTranspose2d(16, 16, 4, stride=(8, 3), padding=(1, 0)),        # 48x128
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        # final resize to exact grid happens in forward

    def forward(self, prof, extra):
        z = self.enc(prof)
        z = torch.cat([z, extra], dim=1)
        out = self.dec(z)
        return nn.functional.interpolate(
            out, size=(GRID_H, GRID_W), mode="bilinear", align_corners=False)


def physics_baseline(tof_cm_batch, vis_per):
    """Paint each receiver ray into the grid at its ToF distance.
    tof: (B,4); vis_per: (B,4,H,W). Returns (B,1,H,W) cm grid (0 elsewhere).
    """
    from .config import X_MAX_CM, X_MIN_CM, Y_MAX_CM, Y_MIN_CM
    B = tof_cm_batch.shape[0]
    out = torch.zeros(B, 1, GRID_H, GRID_W)
    xs = torch.linspace(X_MIN_CM, X_MAX_CM, GRID_W)
    ys = torch.linspace(Y_MIN_CM, Y_MAX_CM, GRID_H)
    YY, XX = torch.meshgrid(ys, xs, indexing="ij")
    for b in range(B):
        for i, name in enumerate(RECEIVERS):
            r = float(tof_cm_batch[b, i])
            ang = np.deg2rad(RECEIVERS[name]["angle_deg"])
            rx, ry = RECEIVERS[name]["xy"]
            cxp, cyp = rx + r * np.sin(ang), ry + r * np.cos(ang)
            j = int((cxp - X_MIN_CM) / (X_MAX_CM - X_MIN_CM) * GRID_W)
            k = int((cyp - Y_MIN_CM) / (Y_MAX_CM - Y_MIN_CM) * GRID_H)
            if 0 <= k < GRID_H and 0 <= j < GRID_W:
                out[b, 0, k, j] = r
                # small blob for visibility
                if k + 1 < GRID_H:
                    out[b, 0, k + 1, j] = r
    return out
