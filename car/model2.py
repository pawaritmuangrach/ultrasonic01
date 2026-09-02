"""โมเดลรอบสอง — คลื่นเสียง 4 ช่อง เป็นภาพระยะ 80x60

ไฟล์นี้มีแต่ **ส่วน ML** ล้วน ไม่มีโค้ดหน้าจอเลย (หน้าจออยู่ที่ live_map2.py)
เขียนแยกกันเพื่อให้อ่านทีละเรื่องได้

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ไฟล์นี้ทั้งไฟล์)
═══════════════════════════════════════════════════════════════════════

ภาพรวมของสิ่งที่โมเดลทำ

    คลื่นดิบ 4 ช่อง x 871 จุด  ──►  ภาพระยะ 80 x 60 (สองค่าต่อช่อง)

ไม่มี DSP ที่คนเขียนเลย ไม่มีการหายอดคลื่น ไม่มีการคำนวณ TDOA ด้วยมือ
ป้อนตัวเลขดิบจาก ADC เข้าไปตรง ๆ แล้วให้โมเดลหาเอาเองว่าจะแปลงยังไง

ทำไมถึงเป็นไปได้ที่จะรู้ทั้งมุมและความสูง
    หัวรับสี่ตัวอยู่มุมทั้งสี่ของแผ่น 110x110 มม.
    เสียงจากวัตถุที่อยู่เยื้องซ้ายจะถึงหัวรับซ้ายก่อนขวา  -> บอกมุมแนวนอน
    เสียงจากวัตถุที่อยู่สูงจะถึงหัวรับบนก่อนล่าง          -> บอกมุมแนวตั้ง
    ความต่างของเวลาที่ว่านี้ซ่อนอยู่ในรูปร่างคลื่น 4 เส้น ซึ่งคือสิ่งที่ป้อนเข้าไป
"""
import sys
from pathlib import Path

import numpy as np

import mapdata2 as MD
from mapdata2 import GW, GH, NEAR_CM, FAR_CM, DATA, read_counts
from rig2 import PINS


# ---------------------------------------------------------------------
#  ขั้นที่ 1 — เตรียมคลื่นก่อนป้อนเข้าโมเดล
# ---------------------------------------------------------------------
def prep(x):
    """คลื่นดิบ -> คลื่นที่พร้อมป้อน + ความแรงเดิมที่แยกออกมา

    ทำสามอย่าง ทีละอย่างมีเหตุผลของมัน

    1. ลบมัธยฐานออก
       ADC วัดแรงดันเทียบกราวด์ สัญญาณของเราลอยอยู่บนไฟตรงราว 2048 นับ
       ไฟตรงนั้นไม่มีข้อมูล มีแต่ทำให้ตัวเลขใหญ่โดยเปล่าประโยชน์
       ใช้มัธยฐานไม่ใช่ค่าเฉลี่ย เพราะยอดสะท้อนดึงค่าเฉลี่ยแต่ไม่ดึงมัธยฐาน

    2. หารด้วยความแรงของตัวเอง (MAD)
       คนสะท้อนเสียงแรงต่างกันได้ถึง 40 เท่าที่ระยะเดียวกัน แล้วแต่ว่าหันตัวยังไง
       ถ้าไม่หาร โมเดลจะไปสนใจว่า 'ดังหรือเบา' ซึ่งแกว่งจนใช้ไม่ได้
       หารแล้วโมเดลได้ดู 'รูปร่างคลื่น' แทน ซึ่งนิ่งกว่ามาก

       ใช้ MAD (มัธยฐานของค่าสัมบูรณ์) ไม่ใช่ส่วนเบี่ยงเบนมาตรฐาน
       เพราะยอดสะท้อนกินพื้นที่นิดเดียวของคลื่นทั้งเส้น ส่วนเบี่ยงเบนมาตรฐาน
       จะโดนยอดลากจนวัดพื้นเสียงรบกวนไม่ได้ MAD ไม่สนใจค่าสุดขั้ว

    3. ส่งความแรงเดิมไปอีกทางเป็น log
       ความแรงไม่ได้ไร้ประโยชน์ ยิ่งไกลยิ่งเบา จึงบอกระยะได้คร่าว ๆ
       แต่มันคนละเรื่องกับรูปร่างคลื่น จึงแยกเป็นอินพุตต่างหาก
       ใช้ log เพราะช่วงมันกว้างมาก (40 เท่า) log ทำให้อยู่ในช่วงที่โมเดลรับไหว
    """
    v = x.astype(np.float32)
    v = v - np.median(v, axis=2, keepdims=True)
    scale = np.maximum(np.median(np.abs(v), axis=2, keepdims=True) * 1.4826, 1e-3)
    return v / scale, np.log(scale[:, :, 0] + 1.0)


# ---------------------------------------------------------------------
#  ขั้นที่ 2 — โครงข่าย
# ---------------------------------------------------------------------
def make_model(torch, nn):
    """สร้างโครงข่าย · คืนวัตถุที่เรียกด้วย (คลื่น, ความแรง) แล้วได้ภาพสองชั้น

    ทางเดินของข้อมูล

        คลื่น (4 ช่อง x 871 จุด)
            |
        [trunk]  อ่านทีละช่องด้วยน้ำหนักชุดเดียวกัน      -> 4 x (32 x 55)
            |    ฟิสิกส์ของทุกช่องเหมือนกัน ต่างแค่ตำแหน่ง
            |    ใช้น้ำหนักร่วมกันจึงได้ข้อมูลต่อพารามิเตอร์มากกว่า
            |    และบังคับให้โมเดลเรียน 'เสียงสะท้อนหน้าตายังไง' ก่อน
            |    แล้วค่อยเอาไปเทียบกันทีหลัง
            |
        [mix]    เอาสี่ช่องมาต่อกันแล้วผสม               -> 96 x 7
            |    **ตรงนี้คือจุดที่มุมเกิดขึ้น** ความต่างของเวลาระหว่างช่อง
            |    จะปรากฏก็ต่อเมื่อเอาสี่ช่องมาวางเทียบกัน
            |
        [fc]     แผ่ทั้งก้อน + ความแรง 4 ค่า             -> 1280
            |    **ไม่ยุบแกนเวลาทิ้ง** เพราะตำแหน่งบนแกนเวลาคือระยะ
            |    ถ้ายุบ (เช่นใช้ average pooling) ระยะจะหายไปหมด
            |
        [dec]    ขยายเป็นภาพ 4x5 -> 8x10 -> 15x20 -> 30x40 -> 60x80
                 ออกสองชั้น: มีวัตถุไหม / ถ้ามีอยู่ไกลเท่าไร
    """
    class Up(nn.Module):
        """ขยายภาพไปขนาดที่กำหนดตรง ๆ

        ใช้แทน ConvTranspose2d เพราะขนาดปลายทางของเรา (60x80) หารสองลงตัวไม่สวย
        กำหนดขนาดเองชัดเจนกว่าไล่คำนวณ padding ให้ลงตัว
        """
        def __init__(self, hw):
            super().__init__()
            self.hw = hw

        def forward(self, x):
            return nn.functional.interpolate(x, size=self.hw, mode="bilinear",
                                             align_corners=False)

    def blk(i, o):
        # conv + batchnorm + relu หนึ่งชุด — batchnorm ช่วยให้เทรนนิ่ง
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # อ่านคลื่นทีละช่อง · stride 2 ทุกชั้น = ย่อแกนเวลาลงครึ่งทุกชั้น
            # kernel ใหญ่ (15) ที่ชั้นแรกเพราะคาบของคลื่น 40 kHz ที่สุ่ม 66 kHz
            # กินราว 2.5 จุด ต้องมองกว้างพอจะเห็นรูปคลื่น ไม่ใช่เห็นทีละจุด
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU())
            self.mix = nn.Sequential(
                nn.Conv1d(32 * 4, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.ReLU())
            self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                    nn.Linear(96 * 7 + 4, 1280), nn.ReLU())
            self.dec = nn.Sequential(
                Up((8, 10)), blk(64, 64),
                Up((15, 20)), blk(64, 48),
                Up((30, 40)), blk(48, 32),
                Up((GH, GW)), blk(32, 16),
                nn.Conv2d(16, 2, 3, padding=1))

        def forward(self, w, s):
            b, c, n = w.shape
            # รวมมิติ batch กับ channel เข้าด้วยกัน เพื่อให้ trunk ตัวเดียว
            # ประมวลผลทั้งสี่ช่องพร้อมกัน แล้วค่อยแยกกลับ
            f = self.trunk(w.reshape(b * c, 1, n)).reshape(b, c * 32, -1)
            f = self.mix(f)
            f = self.fc(torch.cat([f.flatten(1), s], 1))
            o = self.dec(f.reshape(b, 64, 4, 5))
            return o[:, 0], o[:, 1]      # ชั้นแรก = มีวัตถุไหม · ชั้นสอง = ระยะ

    return Net()


def to_cm(depth_logit, torch):
    """บีบผลลัพธ์ให้อยู่ในช่วงที่เซ็นเซอร์เห็นได้จริง

    sigmoid ให้ค่า 0..1 แล้วยืดเป็น 40..200 ซม. โมเดลจึงทายนอกช่วงไม่ได้เลย
    ดีกว่าปล่อยอิสระแล้วหวังว่ามันจะเรียนเอง เพราะค่าที่เป็นไปไม่ได้
    ไม่ควรอยู่ในพื้นที่คำตอบตั้งแต่ต้น
    """
    return NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(depth_logit)


# ---------------------------------------------------------------------
#  ขั้นที่ 3 — ตัวห่อสำหรับใช้งานจริง
# ---------------------------------------------------------------------
class MapPredictor:
    """โหลดโมเดลที่เทรนแล้ว มาใช้ทายทีละเฟรม

    push(ping) -> (occ, depth_cm) ทั้งคู่ขนาด GH x GW

    เกลี่ยย้อนหลังด้วยมัธยฐาน เพราะความผิดพลาดรายเฟรมไม่สัมพันธ์กัน
    วัดในรอบแรกแล้วช่วยจริง (ผิด 4.42 -> 3.82 องศา) ทั้งที่โมเดลเทรนทีละเฟรม
    """

    def __init__(self, path=None, smooth=5):
        import torch
        from collections import deque
        torch.set_num_threads(1)   # เธรดพูล torch ชนกับเธรดกล้อง ดู warmup()
        self.torch = torch
        p = Path(path) if path else MD.MODEL
        if not p.exists():
            sys.exit(f"ยังไม่มีโมเดลที่ {p} — เทรนก่อนด้วย:\n"
                     f"  python car/train_map2.py --test 1")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.net = make_model(torch, torch.nn)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.holdout = int(ck.get("holdout", -1))
        # วิธีแบ่ง train/test ที่ใช้จริง · หน้าจอต้องใช้บอกว่าเฟรมไหนเคยเห็นแล้ว
        # ถ้าไม่รู้ จะไปเดาว่า "ทั้งช่วงนี้ไม่เคยเห็น" ซึ่งผิดสำหรับการแบ่งตามเวลา
        self.split = str(ck.get("split", "section"))
        self.tail = float(ck.get("tail", 0.25))
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
        self.occ.clear()
        self.dep.clear()
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
        self.occ.append(o)
        self.dep.append(d)
        return np.median(self.occ, 0), np.median(self.dep, 0)
