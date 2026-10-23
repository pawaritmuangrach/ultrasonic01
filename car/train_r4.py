"""เทรนรอบสี่ — หนึ่งการทดลองต่อหนึ่งคำสั่ง ผลลง car/data/r4_runs/<ชื่อ>.json

    python car/train_r4.py --name base_s0                      ค่าตั้งต้น (สูตรรอบ 3)
    python car/train_r4.py --name act_gelu --act gelu          เปลี่ยน activation
    python car/train_r4.py --name loss_mse --loss mse          เปลี่ยน loss
    python car/train_r4.py --name slice --slice-ms 5           ตัดช่วงเวลา 5 ms เลื่อนทีละ 1 ms

**การแบ่งข้อมูล (เหมือนกันทุกการทดลอง)**
  เทรน s1-s4 · เลือกเช็คพอยต์และจุดตัด s5 ทั้งช่วง · รายงาน s6 ทั้งช่วง
  - กันทั้งช่วง ไม่ใช่ท้ายช่วง เพราะเฟรมในช่วงเดียวกันคล้ายกันนาน 1-2 วินาที (วัดแล้ว)
    ชุดเลือกที่อยู่ช่วงเดียวกับชุดเทรนจะให้คะแนนดีเกินจริง
  - **จุดตัดความน่าจะเป็นเลือกจาก s5 แล้วใช้กับ s6 ตรง ๆ** รอบ 3 เลือกจุดตัดบนชุดรายงานเอง
    ซึ่งทำให้คะแนนดูดีเกินจริงเล็กน้อย รอบนี้แก้แล้ว
  - s6 ถูกแตะครั้งเดียวตอนจบของแต่ละการทดลอง และ **การเลือกว่าสูตรไหนดีที่สุดใช้ s5 เท่านั้น**

ตัวชี้วัด
  IoU      ภาพที่ทายกับภาพจริงทับกันแค่ไหน (0 = ไม่ทับเลย · 1 = ตรงเป๊ะ)
  ระยะพลาด  ค่าเฉลี่ยของ |ระยะที่ทาย - ระยะจริง| เฉพาะช่องที่มีวัตถุจริง (ซม.)
  สูงเงา R2 ทายความสูงของเงา (ตัวแทนท่าทาง) ดีกว่าเดาค่าเฉลี่ยแค่ไหน (0 = เท่าเดา)
  คู่เทียบ  ภาพเฉลี่ย (ไม่ใช้เสียงเลย) และ "สลับคลื่น" (จับคู่คลื่นกับเฉลยผิดคู่)
"""
import os
os.environ.setdefault("MAP2_ROUND", "r4")

import argparse     # noqa: E402
import json         # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402

import mapdata2 as MD                                  # noqa: E402
import model4 as M4                                    # noqa: E402
from mapdata2 import GW, GH, NEAR_CM, FAR_CM           # noqa: E402
from model3 import stack_index, gather                 # noqa: E402
from train_map2 import iou_at, r2                      # noqa: E402
from train_map3 import soft_height                     # noqa: E402

OUT = MD.DATA / "r4_runs"
LOSSES = ("r3", "bdh", "mse", "bce_l1", "dice_huber", "focal")


# ------------------------------------------------------------------ loss
def loss_fn(name, ol, dl, occ, z, torch, F, huber=0.1):
    """ทุก loss คืนสเกลาร์ · ระยะคิดเฉพาะช่องที่มีวัตถุจริง (ระยะในที่ว่างไม่มีความหมาย)

    r3          BCE + Dice + 2 Huber + 0.5 ความสูงเงา   (สูตรรอบสาม = ค่าตั้งต้น)
    bdh         BCE + Dice + 2 Huber                    (ตัดเทอมความสูงออก)
    mse         MSE ของความน่าจะเป็น + 2 MSE ของระยะ    (สิ่งที่ผู้ใช้เสนอตอนแรก)
    bce_l1      BCE + 2 L1                              (ไม่มี Dice)
    dice_huber  Dice + 2 Huber                          (ไม่มี BCE)
    focal       Focal(gamma 2) + Dice + 2 Huber         (เน้นช่องที่ทายยาก)
    """
    p = torch.sigmoid(ol)
    m = occ
    den = m.sum().clamp(min=1)
    err = (torch.sigmoid(dl) - z).abs()
    hub = torch.where(err <= huber, 0.5 * err ** 2, huber * (err - 0.5 * huber))
    hub = (hub * m).sum() / den
    dice = (1 - (2 * (p * occ).sum((1, 2)) + 1) / (p.sum((1, 2)) + occ.sum((1, 2)) + 1)).mean()
    if name in ("r3", "bdh", "bce_l1"):
        bce = F.binary_cross_entropy_with_logits(ol, occ)
    if name == "r3":
        hgt = (soft_height(p, torch) - soft_height(occ, torch)).abs().mean() / GH
        return bce + dice + 2 * hub + 0.5 * hgt
    if name == "bdh":
        return bce + dice + 2 * hub
    if name == "mse":
        return F.mse_loss(p, occ) + 2 * ((err ** 2) * m).sum() / den
    if name == "bce_l1":
        return bce + 2 * (err * m).sum() / den
    if name == "dice_huber":
        return dice + 2 * hub
    if name == "focal":
        ce = F.binary_cross_entropy_with_logits(ol, occ, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** 2 * ce).mean() + dice + 2 * hub
    raise ValueError(name)


# ------------------------------------------------------------------ ตัวชี้วัด
def best_thr(P, occ):
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.96, 0.05):
        v = iou_at(P, occ, t)
        if v > best[0]:
            best = (v, float(t))
    return best[1]


def heights(mask):
    return np.array([MD.shadow_height(m) for m in mask], np.float32)


def metrics(P, D, occ, dep_cm, thr):
    return {"iou": iou_at(P, occ, thr),
            "mae_cm": float(np.abs(D - dep_cm)[occ].mean()) if occ.any() else float("nan"),
            "h_r2": r2(heights(P >= thr), heights(occ)), "thr": thr}


# ------------------------------------------------------------------ ป้อนข้อมูล
class Feeder:
    """เตรียมคลื่นและเป้าเป็นชุด ๆ · โหมดเต็ม หรือ โหมดตัดช่วงเวลา"""

    def __init__(self, counts, dmm, rate, a, dc, scale):
        self.counts, self.dmm, self.a = counts, dmm, a
        self.dc, self.scale = dc, scale
        self.dep = dmm.astype(np.float32) / 10.0                  # ซม. · 0 = ไม่มีวัตถุ
        self.occ = dmm > 0
        if a.slice_ms > 0:
            self.plan = M4.slice_plan(rate, counts.shape[-1], a.slice_ms, a.slice_step_ms)
            self.nsamp = self.plan[0][1] - self.plan[0][0]
        else:
            self.plan = None
            self.nsamp = counts.shape[-1]
        self.K = len(self.plan) if self.plan else 0

    def wave(self, ends):
        return prep_np(gather(self.counts, ends, self.a.stack), self.dc, self.scale, self.a.norm)

    def band(self, ends, k):
        """เป้าของแถบระยะ k: มีวัตถุในแถบไหม และระยะภายในแถบ (0-1)"""
        _, _, lo, hi = self.plan[k]
        d = self.dep[ends]
        occ = self.occ[ends] & (d >= lo) & (d < hi)
        z = np.clip((d - lo) / (hi - lo), 0, 1)
        return occ, z

    def train_batch(self, ends, rng):
        w = self.wave(ends)
        if not self.plan:
            occ = self.occ[ends].astype(np.float32)
            z = np.clip((self.dep[ends] - NEAR_CM) / (FAR_CM - NEAR_CM), 0, 1).astype(np.float32)
            return w, None, occ, z
        # ต่อเฟรมสุ่มมา per_frame ช่วง (ใช้ทุกช่วงทุกรอบจะแพงกว่าโหมดเต็มหลายเท่า)
        ks = rng.integers(0, self.K, (len(ends), self.a.slices_per_frame)).ravel()
        idx = np.repeat(np.arange(len(ends)), self.a.slices_per_frame)
        ws, os_, zs = [], [], []
        for j, k in zip(idx, ks):
            i0, i1, _, _ = self.plan[k]
            ws.append(w[j, :, :, i0:i1])
            o, zz = self.band(ends[j:j + 1], k)
            os_.append(o[0])
            zs.append(zz[0])
        cond = np.eye(self.K, dtype=np.float32)[ks]
        return (np.stack(ws), cond, np.stack(os_).astype(np.float32),
                np.stack(zs).astype(np.float32))


def prep_np(x, dc, scale, mode):
    return M4.prep(x, dc, scale, mode)


def predict(torch, net, fd, ends, bs=64):
    """คืน (ความน่าจะเป็นมีวัตถุ, ระยะ ซม.) ของภาพเต็ม · โหมดตัดช่วง = ต่อทุกแถบเข้าด้วยกัน"""
    net.eval()
    P = np.zeros((len(ends), GH, GW), np.float32)
    D = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, len(ends), bs):
            e = ends[i:i + bs]
            w = fd.wave(e)
            if not fd.plan:
                ol, dl = net(torch.from_numpy(w).float())
                P[i:i + bs] = torch.sigmoid(ol).numpy()
                D[i:i + bs] = (NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(dl)).numpy()
                continue
            best = np.full((len(e), GH, GW), -1.0, np.float32)
            dd = np.zeros_like(best)
            for k, (i0, i1, lo, hi) in enumerate(fd.plan):
                cond = torch.from_numpy(np.eye(fd.K, dtype=np.float32)[[k] * len(e)])
                ol, dl = net(torch.from_numpy(w[:, :, :, i0:i1]).float(), cond)
                p = torch.sigmoid(ol).numpy()
                d = lo + (hi - lo) * torch.sigmoid(dl).numpy()
                take = p > best
                best = np.where(take, p, best)
                dd = np.where(take, d, dd)
            P[i:i + bs] = best
            D[i:i + bs] = dd
    return P, D


# ------------------------------------------------------------------ หลัก
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--act", default="relu", choices=M4.ACTS)
    ap.add_argument("--loss", default="r3", choices=LOSSES)
    ap.add_argument("--norm", default="p999", choices=M4.NORMS)
    ap.add_argument("--stack", type=int, default=5)
    ap.add_argument("--slice-ms", type=float, default=0.0, help="0 = ใช้ทั้งหน้าต่าง")
    ap.add_argument("--slice-step-ms", type=float, default=1.0)
    ap.add_argument("--slices-per-frame", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default="1,2,3,4")
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    import torch
    from torch import nn
    F = nn.functional
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    counts_mm, dmm, sec, src, meta = MD.load()
    counts = np.asarray(counts_mm)                    # ~150 MB อยู่ในหน่วยความจำ อ่านเร็วกว่า
    rate = 66313.0
    ends = stack_index(sec, src, a.stack)
    trs = [int(s) for s in a.train.split(",")]
    tr = ends[np.isin(sec[ends], trs)]
    va = ends[sec[ends] == a.val]
    te = ends[sec[ends] == a.test]
    dc, scale = M4.norm_params(counts, tr, a.norm)
    fd = Feeder(counts, dmm, rate, a, dc, scale)
    net = make_net(torch, nn, a, fd)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[{a.name}] act={a.act} loss={a.loss} norm={a.norm} stack={a.stack} "
          f"slice={a.slice_ms}ms x{fd.K} · {npar:,} พารามิเตอร์")
    print(f"  เทรน s{trs} {len(tr):,} · เลือก s{a.val} {len(va):,} · รายงาน s{a.test} {len(te):,}"
          f" · ปรับค่า dc {dc:.0f} scale {scale:.1f}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = (len(tr) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=nb)
    occ_v, dep_v = dmm[va] > 0, dmm[va].astype(np.float32) / 10.0
    best, hist, ck = -1.0, [], OUT / f"{a.name}.pt"
    t_all = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        t0 = time.time()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), a.batch):
            e = tr[np.sort(perm[i:i + a.batch])]
            w, cond, occ, z = fd.train_batch(e, rng)
            ol, dl = net(torch.from_numpy(w).float(),
                         None if cond is None else torch.from_numpy(cond))
            loss = loss_fn(a.loss, ol, dl, torch.from_numpy(occ), torch.from_numpy(z), torch, F)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        P, D = predict(torch, net, fd, va)
        thr = best_thr(P, occ_v)
        mv = metrics(P, D, occ_v, dep_v, thr)
        mv.update(epoch=ep, loss=tot / nb, sec=time.time() - t0)
        hist.append(mv)
        mark = ""
        if mv["iou"] > best:
            best = mv["iou"]
            torch.save({"model": net.state_dict(), "args": vars(a), "norm": (dc, scale),
                        "thr": thr, "epoch": ep}, ck)
            mark = "  <-"
        print(f"  ep {ep:2d} loss {tot/nb:.4f} | s{a.val} IoU {mv['iou']:.3f} ระยะ {mv['mae_cm']:5.1f}"
              f" ซม. R2 {mv['h_r2']:+.3f} ({mv['sec']:.0f}s){mark}", flush=True)

    # ---- ตอนจบ: โหลดเช็คพอยต์ที่ s5 เลือก แล้วแตะ s6 ครั้งเดียว ด้วยจุดตัดจาก s5
    c = torch.load(ck, map_location="cpu", weights_only=False)
    net.load_state_dict(c["model"])
    occ_t, dep_t = dmm[te] > 0, dmm[te].astype(np.float32) / 10.0
    Pv, Dv = predict(torch, net, fd, va)
    val = metrics(Pv, Dv, occ_v, dep_v, c["thr"])
    P, D = predict(torch, net, fd, te)
    test = metrics(P, D, occ_t, dep_t, c["thr"])
    sh = np.random.default_rng(123).permutation(len(te))
    shuf = metrics(P[sh], D[sh], occ_t, dep_t, c["thr"])
    res = {"name": a.name, "args": vars(a), "params": npar, "norm": [dc, scale],
           "best_epoch": c["epoch"], "val": val, "test": test, "test_shuffled": shuf,
           "hist": hist, "minutes": (time.time() - t_all) / 60}
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  จบ ({res['minutes']:.0f} นาที) · รอบที่เลือก {c['epoch']} · "
          f"s{a.val}: IoU {val['iou']:.3f} ระยะ {val['mae_cm']:.1f} · "
          f"s{a.test}: IoU {test['iou']:.3f} ระยะ {test['mae_cm']:.1f} R2 {test['h_r2']:+.3f} · "
          f"สลับคลื่น IoU {shuf['iou']:.3f}", flush=True)


def make_net(torch, nn, a, fd):
    return M4.make_model(torch, nn, GH, GW, a.stack, fd.nsamp, n_cond=fd.K, act=a.act)


if __name__ == "__main__":
    main()
