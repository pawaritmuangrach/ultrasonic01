"""โมเดลที่แปลงคลื่นเสียง 4 ช่อง เป็นภาพ depth 40x30

ไม่มี DSP ที่เขียนมือเลย ป้อนคลื่นดิบเข้าไปตรง ๆ แล้วให้โมเดลหาเอง
ต่างจาก rules.py ที่คนคิดสูตรให้ และต่างจาก train_nn.py ที่ทำนายแค่มุมเดียว

หัวออกมีสองอัน ต่อหนึ่งช่องภาพ:
  occ   - มั่นใจแค่ไหนว่ามีวัตถุอยู่ในระยะที่เซ็นเซอร์เห็น (40-200 ซม.)
  depth - ถ้ามี อยู่ไกลเท่าไร
แยกกันเพราะสองคำถามนี้ไม่เหมือนกัน 'ไม่มีอะไรตรงนี้' ไม่ใช่ 'อยู่ไกลมาก'
ถ้ายัดรวมเป็นเลขเดียว ที่ว่างจะถูกบังคับให้มีค่าระยะ แล้วลากค่าเฉลี่ยเพี้ยนทั้งภาพ
"""
from pathlib import Path
import sys
import numpy as np

from mapdata import GW, GH, NEAR_CM, FAR_CM, PINS, DATA, read_counts


def prep(x):
    """คลื่นดิบ -> คลื่นที่ตัดไฟตรงออกและหารด้วยความแรงของตัวเอง + ความแรงเดิม

    หารด้วยความแรงตัวเองเพื่อให้โมเดลดู **รูปร่างคลื่น** ไม่ใช่ดังเบา
    (คนสะท้อนเสียงแรงต่างกัน 40 เท่าที่ระยะเดียวกัน แล้วแต่ว่าหันตัวยังไง)
    แต่ความแรงก็มีข้อมูลของมัน จึงส่งไปแยกอีกทางเป็น log
    ใช้ MAD ไม่ใช่ส่วนเบี่ยงเบนมาตรฐาน เพราะยอดสะท้อนกินพื้นที่นิดเดียวของคลื่น
    ส่วนเบี่ยงเบนมาตรฐานจะโดนยอดลากจนวัดพื้นเสียงรบกวนไม่ได้
    """
    v = x.astype(np.float32)
    v = v - np.median(v, axis=2, keepdims=True)
    scale = np.maximum(np.median(np.abs(v), axis=2, keepdims=True) * 1.4826, 1e-3)
    return v / scale, np.log(scale[:, :, 0] + 1.0)


def make_model(torch, nn):
    class Up(nn.Module):
        """ขยายภาพไปขนาดที่กำหนดตรง ๆ

        ใช้แทน ConvTranspose2d เพราะ 30x40 หารสองลงตัวไม่สวย (30 -> 15 -> 7.5)
        กำหนดขนาดปลายทางเองชัดเจนกว่าไล่คำนวณ padding ให้ลงตัว
        """
        def __init__(self, hw):
            super().__init__()
            self.hw = hw

        def forward(self, x):
            return nn.functional.interpolate(x, size=self.hw, mode="bilinear",
                                             align_corners=False)

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # อ่านทีละช่องด้วยน้ำหนักชุดเดียวกัน — ฟิสิกส์ของทั้ง 4 ช่องเหมือนกัน
            # ต่างแค่ตำแหน่ง การใช้น้ำหนักร่วมกันจึงได้ข้อมูลต่อพารามิเตอร์มากกว่า
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU())
            # เอา 4 ช่องมาต่อกันแล้วค่อยผสม — ความต่างเวลาระหว่างช่องคือที่มาของมุม
            self.mix = nn.Sequential(
                nn.Conv1d(32 * 4, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU())
            # **ไม่ยุบแกนเวลา** เพราะตำแหน่งบนแกนเวลาคือระยะ ยุบทิ้งคือทิ้งระยะ
            self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                    nn.Linear(96 * 7 + 4, 768), nn.ReLU())
            self.dec = nn.Sequential(
                Up((7, 10)), blk(64, 64),
                Up((15, 20)), blk(64, 32),
                Up((GH, GW)), blk(32, 16),
                nn.Conv2d(16, 2, 3, padding=1))

        def forward(self, w, s):
            b, c, n = w.shape
            f = self.trunk(w.reshape(b * c, 1, n)).reshape(b, c * 32, -1)
            f = self.mix(f)
            f = self.fc(torch.cat([f.flatten(1), s], 1))
            o = self.dec(f.reshape(b, 64, 3, 4))
            return o[:, 0], o[:, 1]      # occ logit, depth logit

    return Net()


def to_cm(depth_logit, torch):
    """บีบผลลัพธ์ให้อยู่ในช่วงที่เซ็นเซอร์เห็นได้จริง โมเดลจะได้ไม่ทายนอกโลก"""
    return NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(depth_logit)


class MapPredictor:
    """ห่อโมเดลให้เรียกง่ายจากหน้าจอสดและหน้าจอเล่นย้อน

    push(ping) -> (occ, depth_cm) ทั้งคู่ขนาด GH x GW
    เกลี่ยย้อนหลังแบบมัธยฐานเหมือนโมเดลก่อนหน้า — คนละเฟรมพลาดคนละที่ มัธยฐานจึงกลบได้
    """

    def __init__(self, path=None, smooth=5):
        import torch
        from collections import deque
        torch.set_num_threads(1)   # เธรดพูล torch ชนกับเธรดกล้อง ดู warmup()
        self.torch = torch
        p = Path(path) if path else (DATA / "_map_model.pt")
        if not p.exists():
            sys.exit(f"ยังไม่มีโมเดลที่ {p} — เทรนก่อนด้วย:\n"
                     f"  python car/train_map.py --test 1")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.net = make_model(torch, torch.nn)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.holdout = int(ck.get("holdout", -1))
        self.params = int(ck.get("params", 0))
        self.score = ck.get("score", {})
        self.occ = deque(maxlen=max(int(smooth), 1))
        self.dep = deque(maxlen=max(int(smooth), 1))
        self.amps = [0.0] * 4
        self.warmup()

    def warmup(self, nsamp=871):
        """ซ้อม forward ก่อนเปิดกล้อง — torch จัด kernel ตอน conv ครั้งแรก
        ถ้าไปจัดตอนเธรด OpenNI ทำงานอยู่ โปรเซสตายเงียบ ๆ บน Windows"""
        r = np.random.default_rng(0)
        self.push({"counts": (2048 + r.normal(0, 30, (4, int(nsamp)))).astype(np.uint16),
                   "pins": list(PINS)})
        self.occ.clear(); self.dep.clear()
        self.amps = [0.0] * 4

    def push(self, ping):
        x = read_counts(ping)[None]
        xs, sc = prep(x)
        v = x[0].astype(np.float32)
        self.amps = [float((v[i].max() - v[i].min()) / 4095 * 3.3 * 1000) for i in range(4)]
        T = self.torch.from_numpy
        with self.torch.no_grad():
            ol, dl = self.net(T(xs).float(), T(sc).float())
            o = self.torch.sigmoid(ol)[0].numpy()
            d = to_cm(dl, self.torch)[0].numpy()
        self.occ.append(o); self.dep.append(d)
        return np.median(self.occ, 0), np.median(self.dep, 0)
