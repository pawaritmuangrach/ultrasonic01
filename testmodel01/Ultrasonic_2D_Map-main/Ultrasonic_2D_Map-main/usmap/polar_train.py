"""Train PolarScan v2. Usage: .venv/bin/python -m usmap.polar_train"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import BATCH, EPOCHS, LR, MODELS, ensure_dirs
from .polar_data import splits
from .polar_model import PolarNet, masked_bin_loss, eval_metrics


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, _ = splits()
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    model = PolarNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def loss_fn(out, batch):
        _, _, _, _, label, valid, dist_cm, near_d, near_a = batch
        l_bin = masked_bin_loss(out, label.to(device), valid.to(device))
        nd = near_d.to(device).squeeze(-1)
        na = near_a.to(device).squeeze(-1)
        l_nd = nn.functional.l1_loss(out["near_d"], nd)
        ok = ~torch.isnan(na)
        l_na = (out["near_a"][ok] - na[ok]).abs().mean() / 30.0 if ok.any() \
            else out["near_a"].sum() * 0.0
        return l_bin + 0.5 * l_nd + 2.0 * l_na

    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for batch in train_dl:
            prof, tdoa, amps, snr = (batch[0].to(device), batch[1].to(device),
                                     batch[2].to(device), batch[3].to(device))
            dist_cm = batch[6].to(device)
            out = model(prof, tdoa, amps, snr, dist_cm)
            loss = loss_fn(out, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        m = eval_metrics(model, val_dl, device)
        sched.step()
        print(f"ep {ep+1:3d} loss {tot/max(n,1):7.3f} | "
              f"val near {m['model_near_mae']:6.2f}cm "
              f"ang {m['model_angle_mae']:5.2f}deg "
              f"(physics {m['physics_near_mae']:6.2f}cm)", flush=True)
        score = m["model_near_mae"]
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "metrics": m},
                       MODELS / "polarscan.pt")
    print(f"best val near MAE: {best:.2f} cm")


if __name__ == "__main__":
    main()
