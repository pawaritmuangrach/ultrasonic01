"""Train UsMapNet with masked loss (only cells visible by >=1 receiver AND
having valid depth ground truth contribute)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import BATCH, EPOCHS, LR, MODELS, VAL_SECTIONS, ensure_dirs
from .data import splits
from .models import UsMapNet


def masked_loss(pred, tgt, valid, vis_any):
    m = ((valid.squeeze(1) > 0.5) & (vis_any.squeeze(1) > 0.5))
    if m.sum() == 0:
        return pred.sum() * 0.0
    return nn.functional.l1_loss(pred.squeeze(1)[m], tgt.squeeze(1)[m])


def evaluate(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for prof, tgt, valid, vis_any, vis_per, tof in loader:
            prof, tgt = prof.to(device), tgt.to(device)
            valid, vis_any = valid.to(device), vis_any.to(device)
            extra = torch.cat([tof / 200.0, prof.max(dim=2).values], dim=1).to(device)
            pred = model(prof, extra)
            m = ((valid.squeeze(1) > 0.5) & (vis_any.squeeze(1) > 0.5))
            if m.sum():
                errs.append((pred[:, 0][m] - tgt[:, 0][m]).abs().cpu().numpy())
    return float(np.mean(np.concatenate(errs))) if errs else float("nan")


def main():
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, _ = splits()
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=2, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    model = UsMapNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for prof, tgt, valid, vis_any, vis_per, tof in train_dl:
            prof, tgt = prof.to(device), tgt.to(device)
            valid, vis_any = valid.to(device), vis_any.to(device)
            extra = torch.cat([tof / 200.0, prof.max(dim=2).values], dim=1).to(device)
            loss = masked_loss(model(prof, extra), tgt, valid, vis_any)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        vmae = evaluate(model, val_dl, device)
        sched.step()
        print(f"epoch {ep+1:3d}  train {tot/max(n,1):8.3f} cm  val MAE {vmae:7.3f} cm", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"model": model.state_dict(), "val_mae": best},
                       MODELS / "usmapnet.pt")
    print(f"best val MAE: {best:.3f} cm")


if __name__ == "__main__":
    main()
