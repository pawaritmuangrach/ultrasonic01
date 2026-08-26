"""Inference: one or more us npz files -> predicted rviz-style map (PNG + npy).

Usage:
    .venv/bin/python infer.py dataset/walk_s5/us_000123.npz [more.npz ...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from usmap.config import MODELS, X_MAX_CM, X_MIN_CM, Y_MAX_CM, Y_MIN_CM, ensure_dirs
from usmap.echo import range_profile, tof_cm
from usmap.models import UsMapNet


def predict(model, npz_path):
    prof = torch.from_numpy(range_profile(npz_path))[None]          # (1,4,BINS)
    t = torch.from_numpy(tof_cm(npz_path))[None]
    extra = torch.cat([t / 200.0, prof.max(dim=2).values], dim=1)
    with torch.no_grad():
        return model(prof, extra)[0, 0].numpy()                     # (48,64)


def main(paths):
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UsMapNet().to(device)
    ckpt = torch.load(MODELS / "usmapnet.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in paths:
        grid = predict(model, p)
        stem = p.replace("/", "_").replace(".npz", "")
        np.save(f"{stem}_map.npy", grid)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, origin="lower", cmap="turbo", vmin=0, vmax=250,
                       extent=[X_MIN_CM, X_MAX_CM, Y_MIN_CM, Y_MAX_CM])
        ax.set_title(f"US-predicted map: {p}")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y forward (cm)")
        fig.colorbar(im, label="distance (cm)")
        fig.tight_layout(); fig.savefig(f"{stem}_map.png", dpi=110)
        plt.close(fig)
        print("wrote", f"{stem}_map.png/.npy")


if __name__ == "__main__":
    main(sys.argv[1:])
