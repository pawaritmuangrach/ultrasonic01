"""Evaluate on test walk_s5: model vs physics baseline, plus visual overlays."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .config import EVAL, MODELS, X_MAX_CM, X_MIN_CM, Y_MAX_CM, Y_MIN_CM, ensure_dirs
from .data import splits
from .models import UsMapNet, physics_baseline


def grid_extent():
    return [X_MIN_CM, X_MAX_CM, Y_MIN_CM, Y_MAX_CM]


def save_triplet(tgt, pred, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, img, name in ((axes[0], tgt[0], "Depth GT"), (axes[1], pred[0], "Predicted")):
        im = ax.imshow(np.nan_to_num(img, nan=0), origin="lower",
                       extent=grid_extent(), cmap="turbo", vmin=0, vmax=250)
        ax.set_title(f"{name} ({title})"); fig.colorbar(im, ax=ax, label="cm")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_ds = splits()
    dl = DataLoader(test_ds, batch_size=16)
    model = UsMapNet().to(device)
    ckpt = torch.load(MODELS / "usmapnet.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded model (val MAE {ckpt.get('val_mae', float('nan')):.2f} cm)")

    errs_m, errs_b = [], []
    shown = 0
    with torch.no_grad():
        for prof, tgt, valid, vis_any, vis_per, tof in dl:
            extra = torch.cat([tof / 200.0, prof.max(dim=2).values], dim=1).to(device)
            pred = model(prof.to(device), extra).cpu()
            base = physics_baseline(tof, vis_per)
            m = (valid > 0.5) & (vis_any > 0.5)
            errs_m.append((pred[m] - tgt[m]).abs().numpy())
            errs_b.append((base[m] - tgt[m]).abs().numpy())
            for i in range(min(3, len(pred))):
                if shown < 6:
                    save_triplet(tgt[i].numpy(), pred[i].numpy(),
                                 EVAL / f"sample_{shown:02d}.png", f"test#{shown}")
                    shown += 1

    mae_m = np.concatenate(errs_m).mean()
    mae_b = np.concatenate(errs_b).mean()
    report = (f"Test set: walk_s5\n"
              f"Model  MAE: {mae_m:.2f} cm\n"
              f"Physics baseline MAE: {mae_b:.2f} cm\n")
    print(report)
    (EVAL / "report.txt").write_text(report)


if __name__ == "__main__":
    main()
