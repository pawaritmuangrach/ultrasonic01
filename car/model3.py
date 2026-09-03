"""โมเดลรอบสาม — 5 การยิงต่อภาพ · CNN แล้ว LSTM · ตาราง 40x30

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ไฟล์นี้ทั้งไฟล์)
═══════════════════════════════════════════════════════════════════════

ไฟล์นี้ **ไม่มีโค้ดหน้าจอเลย** มีแต่การเตรียมข้อมูลกับตัวโครงข่าย

ต่างจากรอบสองสี่อย่าง ทุกอย่างมีเหตุผลจากตัวเลขที่วัดได้ ไม่ใช่เดา

1. **ป้อน 5 การยิงต่อหนึ่งภาพ** แทนที่จะป้อนครั้งเดียว
   รอบสองใช้ครั้งเดียวแล้วเอาผลลัพธ์ 5 ครั้งมาหามัธยฐานทีหลัง
   ซึ่งเป็นการเกลี่ยแบบตายตัวที่โมเดลไม่ได้เรียนรู้อะไรเลย
   ป้อนเข้าไปพร้อมกันแล้วโมเดลเรียนเองว่าจะรวมยังไง
   และได้ของแถมสำคัญ: **การย่อตัวคือการเคลื่อนไหว ไม่ใช่ภาพนิ่ง**

2. **LSTM ข้ามการยิง** วางไว้หลัง mix ที่เดียว
   วัดแล้วที่ 5 การยิง (275 ms) คนเลื่อนไป 6.2 องศา ซึ่งมากกว่าความเบลอ
   ของอาเรย์ทั้งก้อน (4.5 องศา) ถ้ายัดรวมเป็นช่องเพิ่มเฉย ๆ โมเดลจะมอง
   ทั้งห้าเท่ากันหมด การเลื่อนกลายเป็นสัญญาณรบกวน
   LSTM ทำให้ **ลำดับมีความหมาย** เรียนได้เองว่าอันสุดท้ายคือปัจจุบัน

3. **ปรับค่าด้วยช่วงคงที่** ไม่ใช่หารด้วยความแรงของแต่ละเฟรม
   ช่วงคำนวณจาก **section ที่ใช้เทรนเท่านั้น** ถ้าเอา section ทดสอบมาคิดด้วย
   เท่ากับแอบดูข้อสอบ ผลจะดูดีเกินจริง

4. **ตาราง 40x30** ไม่ใช่ 80x60 — ดูเหตุผลที่ mapdata2.GRIDS
"""
import sys
from pathlib import Path

import numpy as np

import mapdata2 as MD
from mapdata2 import GW, GH, NEAR_CM, FAR_CM
from rig2 import PINS

DATA = Path(__file__).resolve().parent / "data"
STACK = 5          # กี่การยิงต่อหนึ่งภาพ
CLIP_PCT = 99.9    # เปอร์เซ็นไทล์ที่ใช้เป็นสเกล · ส่วนที่เกินถูกตัด


# ------------------------------------------------------------------ เตรียมข้อมูล
def norm_params(counts, idx, sample=40):
    """หาจุดกลางกับสเกล จาก **เฟรมที่ใช้เทรนเท่านั้น**

    ทำไมไม่ใช้ค่าต่ำสุด-สูงสุด ทั้งที่ตรงไปตรงมากว่า:
      วัดจากข้อมูลจริง ค่าสุดขั้วห่างจากจุดกลางถึง 646 นับ
      แต่สัญญาณจริงระดับ 99% แค่ 67 นับ
      ถ้าหารด้วย 646 สัญญาณทั่วไปจะเหลือแค่ 0.10 ของช่วง คือแบนติดศูนย์
      เพราะค่าสุดขั้วที่เจอนาน ๆ ครั้งไปกำหนดสเกลให้ทั้งชุด

      ใช้เปอร์เซ็นไทล์ 99.9 แทน ได้ 117 นับ สัญญาณทั่วไปอยู่ที่ 0.57
      แลกกับถูกตัดทิ้งไป 0.1% ของจุด ซึ่งเป็นยอดที่ล้นอยู่แล้ว
    """
    x = counts[idx[::sample]].astype(np.float32)
    dc = float(np.median(x))
    scale = float(np.percentile(np.abs(x - dc), CLIP_PCT))
    return dc, max(scale, 1.0)


def prep(x, dc, scale):
    """คลื่นดิบ -> คลื่นที่พร้อมป้อน · รูปทรงเข้าออกเหมือนกัน

    ต่างจากรอบสองตรงที่ **ไม่หารด้วยความแรงของแต่ละเฟรม**
    รอบสองหารด้วย MAD ของเฟรมนั้น ๆ ทำให้ความดังหายไปจากคลื่น
    ต้องส่งความดังไปอีกทางเป็นอินพุตแยก

    รอบนี้ใช้สเกลคงที่ทั้งชุด **ความดังจึงยังอยู่ในตัวคลื่นเอง**
    ไม่ต้องมีอินพุตแยกอีกต่อไป ซึ่งทำให้โมเดลอ่านง่ายขึ้นด้วย
    """
    v = x.astype(np.float32) - dc
    return np.clip(v / scale, -1.0, 1.0)


def to_cm(t01, torch):
    """0-1 -> เซนติเมตร · sigmoid บังคับให้อยู่ในช่วงที่เซ็นเซอร์เห็นได้เสมอ"""
    return NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(t01)


def stack_index(sec, src, stack=STACK):
    """หาเฟรมที่ย้อนหลังได้ครบ stack ครั้ง **โดยไม่ข้ามรอยต่อช่วง**

    ถ้าปล่อยให้ข้าม ชุดหนึ่งจะมีคลื่นจากคนละช่วงเวลาปนกัน ซึ่งไม่มีอยู่จริง
    ตอนใช้งานสด และจะทำให้คะแนนที่วัดได้ไม่ตรงกับของจริง
    """
    ok = []
    for i in range(len(sec)):
        if sec[i] <= 0:
            continue
        j = i - stack + 1
        if j < 0:
            continue
        if sec[j] == sec[i] and src[j] == src[i]:
            ok.append(i)
    return np.array(ok, np.int64)


def gather(counts, ends, stack=STACK):
    """ดึงคลื่น stack ครั้งที่ลงท้ายด้วยเฟรม ends · ได้ (n, stack, 4, จุดเวลา)"""
    return np.stack([counts[e - stack + 1:e + 1] for e in ends])


# ------------------------------------------------------------------ โครงข่าย
def make_model(torch, nn, gh=GH, gw=GW, stack=STACK):
    """สร้างโครงข่าย · เรียกด้วย (คลื่น) แล้วได้ภาพสองชั้น

        คลื่น (5 การยิง x 4 ไมค์ x 871 จุด)
            |
        [trunk]  CNN อ่านทีละเส้น น้ำหนักชุดเดียวกันทั้ง 20 เส้น
            |    ฟิสิกส์ของทุกเส้นเหมือนกัน ใช้น้ำหนักร่วมกันจึงได้ข้อมูล
            |    ต่อพารามิเตอร์มากกว่า และบังคับให้เรียน 'เสียงสะท้อนหน้าตายังไง'
            |
        [mix]    รวม 4 ไมค์ของแต่ละการยิง -> ได้ 5 เวกเตอร์
            |    **ตรงนี้คือจุดที่มุมเกิดขึ้น** ผลต่างเวลาระหว่างไมค์
            |    ปรากฏก็ต่อเมื่อเอาสี่เส้นมาวางเทียบกัน
            |
        [lstm]   เดินผ่าน 5 เวกเตอร์ตามลำดับเวลา  <- ที่เดียวเท่านั้น
            |    ตรงนี้คือจุดที่ **การเคลื่อนไหว** เกิดขึ้น
            |    เอาสถานะสุดท้ายไปใช้ เพราะการยิงล่าสุดคือปัจจุบัน
            |
        [dec]    ขยายเป็นภาพ 4x5 -> 8x10 -> 15x20 -> 30x40
                 ออกสองชั้น: มีวัตถุไหม / ถ้ามีอยู่ไกลเท่าไร
    """
    class Up(nn.Module):
        """ขยายภาพไปขนาดที่กำหนดตรง ๆ · ชัดกว่าไล่คำนวณ padding ให้ลงตัว"""

        def __init__(self, hw):
            super().__init__()
            self.hw = hw

        def forward(self, x):
            return nn.functional.interpolate(x, size=self.hw, mode="bilinear",
                                             align_corners=False)

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                             nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = stack
            # kernel ใหญ่ (15) ที่ชั้นแรก เพราะคาบของคลื่น 40 kHz ที่สุ่ม 66 kHz
            # กินราว 2.5 จุด ต้องมองกว้างพอเห็นรูปคลื่น ไม่ใช่เห็นทีละจุด
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU())
            self.mix = nn.Sequential(
                nn.Conv1d(32 * 4, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU())
            # **ไม่ยุบแกนเวลาทิ้ง** ตำแหน่งบนแกนเวลาคือระยะ ยุบแล้วระยะหายหมด
            self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                       nn.Linear(96 * 7, 384), nn.ReLU())
            self.lstm = nn.LSTM(384, 320, batch_first=True)
            self.fc = nn.Sequential(nn.Dropout(0.3),
                                    nn.Linear(320, 1280), nn.ReLU())
            self.dec = nn.Sequential(
                Up((8, 10)), blk(64, 64),
                Up((15, 20)), blk(64, 48),
                Up((gh, gw)), blk(48, 24),
                nn.Conv2d(24, 2, 3, padding=1))

        def forward(self, w):
            b, t, c, n = w.shape                      # (batch, 5, 4, 871)
            f = self.trunk(w.reshape(b * t * c, 1, n))
            f = f.reshape(b * t, c * 32, -1)
            f = self.mix(f)
            f = self.embed(f).reshape(b, t, -1)       # (batch, 5, 384)
            # เอาสถานะสุดท้าย เพราะการยิงล่าสุดคือปัจจุบัน ที่เหลือคือบริบท
            f = self.lstm(f)[0][:, -1]
            o = self.dec(self.fc(f).reshape(b, 64, 4, 5))
            return o[:, 0], o[:, 1]

    return Net()


# ------------------------------------------------------------------ ใช้งานจริง
class MapPredictor:
    """โหลดโมเดลที่เทรนไว้ แล้วทำนายจากคลื่นสด

    เก็บการยิงล่าสุด 5 ครั้งไว้เอง ผู้เรียกแค่ push ทีละครั้งเหมือนเดิม
    ระหว่างที่ยังไม่ครบ 5 จะเอาครั้งแรกมาเติมให้ครบ เพื่อให้ทำนายได้ทันที
    ไม่ต้องรอ 5 รอบ (275 ms) ก่อนจะเห็นอะไรบนจอ
    """

    def __init__(self, path=None, smooth=1):
        import torch
        from collections import deque
        torch.set_num_threads(1)   # เธรดพูล torch ชนกับเธรดกล้อง ดู warmup()
        self.torch = torch
        p = Path(path) if path else MD.MODEL
        if not p.exists():
            sys.exit(f"ยังไม่มีโมเดลที่ {p} — เทรนก่อนด้วย:\n"
                     f"  python car/train_map3.py")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.gw, self.gh = ck.get("grid", (GW, GH))
        self.stack = int(ck.get("stack", STACK))
        self.dc, self.scale = ck["norm"]
        self.net = make_model(torch, torch.nn, self.gh, self.gw, self.stack)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.params = int(ck.get("params", 0))
        self.score = ck.get("score", {})
        self.test_section = int(ck.get("test", -1))
        self.buf = deque(maxlen=self.stack)
        self.occ = deque(maxlen=max(int(smooth), 1))
        self.dep = deque(maxlen=max(int(smooth), 1))
        self.amps = [0.0] * 4
        self.warmup()

    def warmup(self, nsamp=871):
        """ซ้อม forward ก่อนเปิดกล้อง — torch จัด kernel ตอน conv ครั้งแรก
        ถ้าไปจัดตอนเธรด OpenNI ทำงานอยู่ โปรเซสตายเงียบ ๆ บน Windows"""
        r = np.random.default_rng(0)
        fake = (self.dc + r.normal(0, 30, (4, int(nsamp)))).astype(np.uint16)
        for _ in range(self.stack):
            self.push({"counts": fake, "pins": list(PINS)})
        self.buf.clear()
        self.occ.clear()
        self.dep.clear()
        self.amps = [0.0] * 4

    def push(self, ping):
        x = MD.read_counts(ping)
        v = x.astype(np.float32)
        self.amps = [float((v[i].max() - v[i].min()) / 4095 * 3.3 * 1000)
                     for i in range(4)]
        if not self.buf:                      # ครั้งแรก เติมให้ครบทันที
            for _ in range(self.stack):
                self.buf.append(x)
        else:
            self.buf.append(x)
        w = prep(np.stack(self.buf)[None], self.dc, self.scale)
        with self.torch.no_grad():
            ol, dl = self.net(self.torch.from_numpy(w).float())
            o = self.torch.sigmoid(ol)[0].numpy()
            d = to_cm(dl, self.torch)[0].numpy()
        self.occ.append(o)
        self.dep.append(d)
        return np.median(self.occ, 0), np.median(self.dep, 0)
