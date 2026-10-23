"""โมเดลรอบสี่ — โครงเดียวกับรอบสาม แต่ตั้งค่าได้ เพื่อใช้ทดลองเทียบกันอย่างยุติธรรม

═══════════════════════════════════════════════════════════════════════
  ส่วนที่ 2 : ML   (ไฟล์นี้ทั้งไฟล์ · การเทรนอยู่ที่ train_r4.py)
═══════════════════════════════════════════════════════════════════════

สิ่งที่ปรับได้ (ทุกอย่างอื่นตรึงไว้เหมือนกันทุกการทดลอง)
  act      ฟังก์ชันกระตุ้นของชั้น conv/linear  relu leaky elu gelu silu mish tanh
           (LSTM มี tanh/sigmoid ในตัวของมันเองเสมอ ไม่ได้เปลี่ยนตาม)
  nsamp    ความยาวคลื่นที่ป้อน  871 จุด = ทั้งหน้าต่าง 13 ms · ~332 จุด = ช่วงละ 5 ms
  n_cond   ขนาดเวกเตอร์บอกตำแหน่งช่วงเวลา (ใช้เฉพาะโหมดตัดช่วง ดูด้านล่าง)
  norm     วิธีปรับขนาดคลื่นก่อนป้อน  p999 (เหมือนรอบ 3) · log (บีบช่วงกว้าง)

โหมดตัดช่วงเวลา (ไอเดียของผู้ใช้: ดูทีละช่วง 0-5 ms, 1-6 ms, 2-7 ms ...)
  ช่วงเวลาหนึ่ง = ช่วงระยะหนึ่ง เพราะเสียงจากของที่ระยะ d มาถึงที่เวลา T0 + 2d/c
  ช่วง 5 ms จึงเห็นได้แค่ของในแถบระยะกว้าง ~86 ซม. ที่ตรงกับช่วงนั้น
  เป้าของแต่ละช่วงจึงเป็น **ภาพเฉพาะแถบระยะนั้น** ไม่ใช่ภาพทั้งภาพ
  (ถ้าใช้ภาพทั้งภาพ ช่วง 0-5 ms จะถูกบังคับให้เดาของที่ไกล 1.8 ม. ซึ่งเสียงยังไม่กลับมา)
  ตอนใช้งานจริงรันทุกช่วงแล้วต่อกันเป็นภาพเดียว
"""
import numpy as np

ACTS = ("relu", "leaky", "elu", "gelu", "silu", "mish", "tanh")
NORMS = ("p999", "log")
CLIP_PCT = 99.9
C_MS, T0_US = 343.0, 1220.0


def act_layer(nn, name):
    table = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.1), "elu": nn.ELU,
             "gelu": nn.GELU, "silu": nn.SiLU, "mish": nn.Mish, "tanh": nn.Tanh}
    if name not in table:
        raise ValueError(f"ไม่รู้จัก activation {name} · มี {list(table)}")
    return table[name]()


# ------------------------------------------------------------------ เตรียมคลื่น
def norm_params(counts, idx, mode, sample=40):
    """จุดกลางและสเกล จาก **เฟรมเทรนเท่านั้น** (เอาชุดทดสอบมาคิดด้วย = แอบดูข้อสอบ)

    p999 : สเกล = เปอร์เซ็นไทล์ 99.9 ของ |คลื่น - จุดกลาง| แล้วตัดที่ ±1 (เหมือนรอบ 3)
    log  : สเกล = ค่ากลางของ |คลื่น - จุดกลาง| (ราวระดับนอยส์) แล้วบีบแบบลอการิทึม
           ฮาร์ดแวร์ใหม่มีช่วงกว้างมาก พัลส์ตรงหลักพัน เอคโค่ไกลหลักสิบ ถ้าหารเชิงเส้น
           เอคโค่ไกลจะเหลือเล็กมาก ลอการิทึมเก็บทั้งสองอย่างไว้ในช่วงที่ใช้ได้
    """
    x = np.asarray(counts[idx[::sample]]).astype(np.float32)
    dc = float(np.median(x))
    a = np.abs(x - dc)
    scale = float(np.percentile(a, CLIP_PCT)) if mode == "p999" else float(np.median(a))
    return dc, max(scale, 1.0)


def prep(x, dc, scale, mode):
    v = x.astype(np.float32) - dc
    if mode == "p999":
        return np.clip(v / scale, -1.0, 1.0)
    return np.sign(v) * np.log1p(np.abs(v) / scale) / np.log1p(4096.0 / scale)


# ------------------------------------------------------------------ ตัดช่วงเวลา
def slice_plan(rate, nsamp, win_ms, step_ms):
    """[(เริ่ม, จบ, ระยะต้นแถบ ซม., ระยะปลายแถบ ซม.)] ของทุกช่วงที่อยู่ในหน้าต่างครบ"""
    w = int(round(win_ms * 1e-3 * rate))
    s = int(round(step_ms * 1e-3 * rate))
    out = []
    i = 0
    while i + w <= nsamp:
        lo = (i / rate - T0_US * 1e-6) * C_MS / 2 * 100
        hi = ((i + w) / rate - T0_US * 1e-6) * C_MS / 2 * 100
        out.append((i, i + w, lo, hi))
        i += s
    return out


# ------------------------------------------------------------------ โครงข่าย
def make_model(torch, nn, gh, gw, stack, nsamp, n_cond=0, act="relu"):
    """CNN -> รวม 4 ไมค์ -> LSTM ข้ามการยิง -> ถอดเป็นภาพ 2 ชั้น (เหมือน model3)"""
    def A():
        return act_layer(nn, act)

    class Up(nn.Module):
        def __init__(self, hw):
            super().__init__()
            self.hw = hw

        def forward(self, x):
            return nn.functional.interpolate(x, size=self.hw, mode="bilinear",
                                             align_corners=False)

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), A())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_cond = n_cond
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), A(),
                nn.Conv1d(16, 32, 9, 2, 4), nn.BatchNorm1d(32), A(),
                nn.Conv1d(32, 32, 9, 2, 4), nn.BatchNorm1d(32), A(),
                nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), A())
            self.mix = nn.Sequential(
                nn.Conv1d(32 * 4, 96, 5, 2, 2), nn.BatchNorm1d(96), A(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), A(),
                nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), A())
            # ความยาวหลังย่อขึ้นกับความยาวคลื่นที่ป้อน คำนวณจากของจริงแทนการฝังตัวเลข
            with torch.no_grad():
                self.trunk.eval()
                self.mix.eval()
                f = self.trunk(torch.zeros(4, 1, nsamp)).reshape(1, 128, -1)
                flat = self.mix(f).numel()
                self.trunk.train()
                self.mix.train()
            self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                       nn.Linear(flat, 384), A())
            self.lstm = nn.LSTM(384, 320, batch_first=True)
            self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(320 + n_cond, 1280), A())
            self.dec = nn.Sequential(
                Up((8, 10)), blk(64, 64),
                Up((15, 20)), blk(64, 48),
                Up((gh, gw)), blk(48, 24),
                nn.Conv2d(24, 2, 3, padding=1))

        def forward(self, w, cond=None):
            b, t, c, n = w.shape
            f = self.trunk(w.reshape(b * t * c, 1, n)).reshape(b * t, c * 32, -1)
            f = self.embed(self.mix(f)).reshape(b, t, -1)
            f = self.lstm(f)[0][:, -1]
            if self.n_cond:
                f = torch.cat([f, cond], 1)
            o = self.dec(self.fc(f).reshape(b, 64, 4, 5))
            return o[:, 0], o[:, 1]

    return Net()
