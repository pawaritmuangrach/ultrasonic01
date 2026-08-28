#!/usr/bin/env python3
"""โมเดล ML ล้วน — ป้อนคลื่นดิบเข้าไปตรง ๆ ไม่ผ่าน DSP ที่เขียนมือเลย

    python car/train_nn.py                  เทรน (กันช่วงที่ 1 ไว้ทดสอบ)
    python car/train_nn.py --test 5         กันช่วงอื่น
    python car/train_nn.py --rebuild        อ่านไฟล์ดิบใหม่

**ต่างจาก rules.py และ PolarScan ตรงไหน**

| | rules.py | PolarScan | ตัวนี้ |
|---|---|---|---|
| กรองแบนด์ / เปลือกคลื่น / หายอด | เขียนมือ | เขียนมือ | **เน็ตเรียนเอง** |
| อินพุต | ยอด 4 ค่า | โปรไฟล์ 4x128 + ฟีเจอร์ | **คลื่นดิบ 4x871** |
| พารามิเตอร์ | 2 | 593,628 | ดูตอนรัน |

ตัวนี้คือ "ML ทั้งหมด" จริง ๆ — ไม่มีฟิสิกส์ฝังอยู่เลย เน็ตต้องเรียนรู้เองว่า
สัญญาณ 40 kHz ที่พับลงมาเป็น 26 kHz หน้าตาเป็นยังไง ต้องหาเปลือกคลื่นยังไง
และเสียงที่ดังไม่เท่ากันระหว่างช่องแปลว่าเป้าอยู่ทางไหน

**สถาปัตยกรรมออกแบบตามฟิสิกส์ของปัญหา ไม่ใช่ CNN ทั่วไป**

1. **ลำตัวร่วม (shared trunk)** — คอนโวลูชันชุดเดียวใช้กับทั้ง 4 ช่อง
   เพราะหัวรับทั้งสี่เป็นของชนิดเดียวกัน สิ่งที่ต้องตรวจจับ (เอคโค่) หน้าตาเหมือนกัน
   ถ้าให้แต่ละช่องมีน้ำหนักของตัวเอง เน็ตจะจำ "ช่องนี้ดังกว่าเสมอ" แทนที่จะเรียนรูปคลื่น
   และพารามิเตอร์เพิ่ม 4 เท่าโดยไม่ได้อะไร
2. **เคอร์เนลชั้นแรกยาว 15** ครอบคลุมราว 4 คาบของคลื่น 26 kHz ที่ 66 kHz
   สั้นกว่านี้เน็ตมองไม่เห็นรอบคลื่นเต็ม ๆ จะเรียนกรองแบนด์ไม่ได้
3. **นอร์มัลไลซ์ต่อช่อง ด้วยค่ากลางและ MAD** ไม่ใช่ค่าเฉลี่ย/sd
   เพราะเอคโค่คือค่าผิดปกติที่เราต้องการ ถ้าใช้ sd เอคโค่แรงจะไปกดสเกลตัวเอง
   **แต่เก็บสเกลเดิมไว้ป้อนแยก** เพราะความแรงสัมพัทธ์ระหว่างช่องคือข้อมูลบอกทิศ
   (ถ้านอร์มัลไลซ์ทิ้งไปเฉย ๆ จะทิ้งสัญญาณที่ rules.py ใช้ทำนายทั้งหมด)

**เฉลย** = มุมจากกล้อง (`labels.target_angle` จุดศูนย์กลางก้อนเป้า) ตัวเดียวกับที่ rules.py ใช้
จึงเทียบกันได้ตรง ๆ

**แบ่งข้อมูลตามช่วง ไม่สุ่มรายเฟรม** — เฟรมติดกันเหมือนกันแทบทุกอย่าง
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = Path(HERE) / "data"
RAW = DATA / "_raw_nn.npz"
MIN_COV = 0.10          # เป้าต้องกินภาพอย่างน้อยเท่านี้ จึงเชื่อมุมจากกล้อง


# ------------------------------------------------------------------ ข้อมูล
def build_raw():
    """อ่าน npz + png ทุกเฟรมครั้งเดียว เก็บคลื่นดิบทั้งชุดเป็นไฟล์เดียว (30 MB)"""
    import glob
    import cv2
    from labels import target_angle
    pins = [34, 33, 32, 35]
    X, Y, CAM, COV, S, I = [], [], [], [], [], []
    for si, sec in enumerate(sorted(p.name for p in DATA.glob("walk_s*") if p.is_dir())):
        d = DATA / sec
        fs = sorted(glob.glob(str(d / "us_*.npz")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
        for f in fs:
            tag = Path(f).stem.split("_")[1]
            dp = d / f"depth_{tag}.png"
            if not dp.exists():
                continue
            lab = target_angle(cv2.imread(str(dp), cv2.IMREAD_UNCHANGED))
            if lab is None:
                continue
            z = np.load(f)
            idx = {int(p): j for j, p in enumerate(z["pins"])}
            c = z["counts"]
            X.append(np.stack([c[idx[p]] for p in pins]))
            Y.append(lab[0]); CAM.append(lab[1]); COV.append(lab[2])
            S.append(si); I.append(int(tag))
        print(f"  อ่าน {sec}: {len(X)} เฟรม", flush=True)
    np.savez_compressed(RAW, x=np.stack(X).astype(np.uint16),
                        y=np.array(Y, np.float32), cam=np.array(CAM, np.float32),
                        cov=np.array(COV, np.float32), sec=np.array(S, np.int16),
                        idx=np.array(I, np.int32), pins=np.array(pins))


def load(rebuild=False):
    if rebuild or not RAW.exists():
        print("อ่านไฟล์ดิบทั้งชุด (ครั้งเดียว ~50 วินาที) ...", flush=True)
        build_raw()
    z = np.load(RAW)
    keep = z["cov"] >= MIN_COV
    return (z["x"][keep], z["y"][keep], z["sec"][keep], z["idx"][keep],
            z["cam"][keep])


def prep(x):
    """นอร์มัลไลซ์ต่อช่อง + คืนสเกลเดิมแยกออกมา

    คืน (คลื่นที่นอร์มัลแล้ว, log ของสเกลแต่ละช่อง)
    สเกลคือข้อมูลบอกทิศ จึงห้ามทิ้ง — ป้อนเข้าเน็ตแยกต่างหาก
    """
    v = x.astype(np.float32)
    med = np.median(v, axis=2, keepdims=True)
    v = v - med
    mad = np.median(np.abs(v), axis=2, keepdims=True) * 1.4826
    scale = np.maximum(mad, 1e-3)
    return v / scale, np.log(scale[:, :, 0] + 1.0)


# ------------------------------------------------------------------ โมเดล
def make_model(torch, nn, n_ch=4):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # ลำตัวร่วม: ใช้กับทีละช่อง น้ำหนักชุดเดียวกันทั้ง 4 ช่อง
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            )
            # หลังรวมช่อง: ตรงนี้แหละที่เน็ตเทียบระหว่างช่องได้ = ที่มาของทิศ
            self.mix = nn.Sequential(
                nn.Conv1d(32 * n_ch, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96),
                nn.ReLU(),
                nn.Conv1d(96, 96, 3, stride=2, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(96 * 2 + n_ch, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x, sc):
            b, c, n = x.shape
            z = self.trunk(x.reshape(b * c, 1, n))          # น้ำหนักร่วมทุกช่อง
            z = z.reshape(b, c * z.shape[1], z.shape[2])
            z = self.mix(z)
            f = torch.cat([z.mean(-1), z.amax(-1), sc], dim=1)
            return self.head(f).squeeze(1)
    return Net()


# ------------------------------------------------------------------ เทรน
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, default=1, help="ช่วงที่กันไว้ทดสอบ (นับจาก 1)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    x, y, sec, idx, cam = load(a.rebuild)
    xs, sc = prep(x)
    ti = a.test - 1
    tr, te = sec != ti, sec == ti
    # แบ่งช่วงหนึ่งจากชุดเทรนไว้เป็น validation เพื่อเลือกเอพอคที่ดีที่สุด
    vi = [s for s in np.unique(sec) if s != ti][-1]
    va = sec == vi
    tr = tr & ~va
    print(f"เฟรมทั้งหมด {len(y)} · เทรน {tr.sum()} · val {va.sum()} (ช่วง {vi+1}) · "
          f"ทดสอบ {te.sum()} (ช่วง {ti+1})")
    print(f"มุมจากกล้อง {y.min():+.0f}..{y.max():+.0f} องศา\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_model(torch, nn).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"พารามิเตอร์ {npar:,} (rules.py ใช้ 2 · PolarScan ใช้ 593,628)")

    T = lambda v: torch.from_numpy(np.ascontiguousarray(v)).float()
    Xtr, Str, Ytr = T(xs[tr]).to(dev), T(sc[tr]).to(dev), T(y[tr]).to(dev)
    Xva, Sva, Yva = T(xs[va]).to(dev), T(sc[va]).to(dev), T(y[va]).to(dev)
    Xte, Ste, Yte = T(xs[te]).to(dev), T(sc[te]).to(dev), T(y[te]).to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * max(1, len(Ytr) // a.batch + 1))

    def evaluate(X, S, Y):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Y), 256):
                out.append(net(X[i:i+256], S[i:i+256]))
        p = torch.cat(out)
        return float((p - Y).abs().mean()), p.cpu().numpy()

    best, best_state, t0 = float("inf"), None, time.time()
    n = len(Ytr)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, a.batch):
            j = perm[i:i + a.batch]
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(net(Xtr[j], Str[j]), Ytr[j], beta=3.0)
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item() * len(j)
        vmae, _ = evaluate(Xva, Sva, Yva)
        if vmae < best:
            best = vmae
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  loss {tot/n:6.3f}  val MAE {vmae:5.2f} องศา"
                  f"   (ดีสุด {best:5.2f})", flush=True)

    net.load_state_dict(best_state)
    tmae, pred = evaluate(Xte, Ste, Yte)
    base = float(np.abs(y[te] - y[tr].mean()).mean())
    torch.save({"model": best_state, "test_mae": tmae, "holdout": int(ti),
                "params": npar}, DATA / "_nn_model.pt")

    print(f"\nเทรนเสร็จใน {time.time()-t0:.0f} วินาที")
    print(f"\n{'วิธี':<42}{'MAE (องศา)':>12}")
    print(f"{'เดาค่าเฉลี่ยชุดเทรน (ไม่ดูข้อมูล)':<42}{base:>11.2f}")
    print(f"{'rules.py — ฟิต 2 ตัวจากความแรง':<42}{'4.80':>11}")
    print(f"{'ตัวนี้ — ML ล้วนจากคลื่นดิบ':<42}{tmae:>11.2f}")
    e = np.abs(pred - y[te])
    print(f"\nรายละเอียดชุดทดสอบ ({te.sum()} เฟรม)")
    print(f"  มัธยฐานความผิด {np.median(e):5.2f}°  ·  "
          f"ผิดเกิน 10° {np.mean(e>10)*100:4.1f}%  ·  ผิดเกิน 20° {np.mean(e>20)*100:4.1f}%")
    zt = np.searchsorted([-12.0, 4.0], y[te])
    zp = np.searchsorted([-12.0, 4.0], pred)
    print(f"  ทายโซนถูก {np.mean(zt==zp)*100:.0f}%  "
          f"(rules.py ได้ 82% · เดาโซนที่เจอบ่อยสุด "
          f"{max((zt==v).mean() for v in np.unique(zt))*100:.0f}%)")
    print(f"\nเซฟโมเดลที่ {DATA/'_nn_model.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ------------------------------------------------- ใช้งานจริง (สด/เล่นย้อน)
class NNPredictor:
    """ห่อโมเดลให้มีหน้าตาเหมือน predict.Predictor เป๊ะ

    มี push(ping)->(องศา, สด) · zone(องศา) · amps · log4 · rng · stale
    เพื่อให้ predict.py กับ replay.py สลับใช้ได้โดยไม่ต้องแก้ลูปแสดงผล

    **เกลี่ยแบบย้อนหลังล้วน** เหมือน rules.py — วัดแล้วช่วยจริง:
    ทีละเฟรม 4.42 องศา (โซน 87%) -> เกลี่ย 9 เฟรม 3.82 องศา (โซน 93%)
    ทั้งที่โมเดลเทรนแบบทีละเฟรม เพราะความผิดพลาดรายเฟรมไม่สัมพันธ์กัน มัธยฐานจึงกลบได้
    """

    PINS = [34, 33, 32, 35]      # ลำดับที่โมเดลถูกเทรนมา ห้ามสลับ

    def __init__(self, path=None, smooth=9, zones=(-12.0, 4.0), min_pp_mv=60.0):
        import torch
        from collections import deque
        self.torch = torch
        p = Path(path) if path else (DATA / "_nn_model.pt")
        if not p.exists():
            sys.exit(f"ยังไม่มีโมเดลที่ {p} — ต้องเทรนก่อน: "
                     f"python car/train_nn.py --test 1")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.net = make_model(torch, torch.nn)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.holdout = int(ck.get("holdout", -1))
        self.params = int(ck.get("params", 0))
        self.buf = deque(maxlen=max(int(smooth), 1))
        self.zones = list(zones)
        self.min_pp = min_pp_mv
        self.amps = [0.0] * 4
        self.log4 = 0.0          # ไม่มีความหมายกับโมเดลนี้ แต่หน้าจอเดิมขอมา
        self.rng = None      # โมเดลนี้ไม่ได้คำนวณระยะ หน้าจอจะโชว์ 'no range'
        self.stale = 0

    def push(self, ping):
        c = ping["counts"]
        idx = {int(v): i for i, v in enumerate(ping["pins"])}
        x = np.stack([c[idx[p]] for p in self.PINS])[None]      # (1,4,N)
        xs, sc = prep(x)
        # ความแรงดิบ (ไม่ผ่าน DSP) — ไว้โชว์แท่งและจับเฟรมที่เสียงกลับอ่อน
        v = x[0].astype(np.float32)
        self.amps = [float((v[i].max() - v[i].min()) / 4095 * 3.3 * 1000)
                     for i in range(v.shape[0])]
        fresh = max(self.amps) >= self.min_pp
        T = self.torch.from_numpy
        with self.torch.no_grad():
            deg = float(self.net(T(xs).float(), T(sc).float())[0])
        if fresh:
            self.buf.append(deg)
            self.stale = 0
        else:
            self.stale += 1
        if not self.buf:
            return None, False
        return float(np.median(self.buf)), fresh

    def zone(self, deg):
        return int(np.searchsorted(self.zones, deg))
