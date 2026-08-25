#!/usr/bin/env python3
"""อ่าน 4 ช่อง **พร้อมกันในการยิงครั้งเดียว** ด้วยจังหวะคงที่ — โหมดเก็บข้อมูลแบบใหม่

ต่างจาก `ultrasonic.Ultrasonic` (ที่ยิง 2 รอบ อ่านทีละคู่) ตรงที่โหมดนี้ตั้ง ADC ให้
สแกนครบทั้ง 4 ช่องในการยิงครั้งเดียว จึงได้ **ฐานเวลาเดียวกันทั้ง 4 ช่อง**

ทำไมถึงคุ้มแม้อัตราสุ่มจะต่ำลง:
  * ยิง 2 รอบทำให้คู่ (34,35) กับ (32,33) ห่างกัน ~95 ms -> เทียบเวลาข้ามคู่ไม่ได้เลย
    ใช้ได้แค่ 2 คู่เบสไลน์ และเป้าที่เคลื่อนที่จะทำให้คู่ที่สองเพี้ยน
  * ยิงรอบเดียวได้ **6 คู่เบสไลน์** (ทุกคู่ที่จับได้จาก 4 ช่อง) และไม่มีปัญหาเป้าเคลื่อนที่

ข้อแลกเปลี่ยนที่ต้องรู้ (วัดจริง ส.ค. 2026):
  ADC ของ ESP32 มีตัวแปลงตัวเดียว + มัลติเพล็กเซอร์ อัตรารวมตันที่ ~266 kHz
  4 ช่องจึงได้ **~66 kHz/ช่อง** ซึ่งต่ำกว่า Nyquist ของ 40 kHz (ต้อง >80 kHz)
  -> ใช้ **bandpass sampling**: 40 kHz พับลงมาที่ 26.4 kHz ซึ่งอยู่ใต้ Nyquist 33.2 kHz
     เปลือกคลื่นและ TDOA ยังอยู่ครบ (features.signal_band() เลือกย่านกรองให้เอง)
  ความละเอียดเวลาลดจาก 3.76 -> 15 us (~2.1 องศา ซึ่งยังต่ำกว่าความผิดพลาดจริง 11.8 องศา)

จังหวะยิง: บังคับคาบคงที่เสมอ ไม่ยิงเร็วบ้างช้าบ้าง เพื่อรับประกันว่าเอคโค่ของรอบก่อน
ตายสนิทก่อนรอบใหม่ (ถ้ายิงทับกัน เอคโค่เก่าจะโผล่เป็น "วัตถุใกล้ปลอม")

    python car/sync4.py --port COM5          วัดความเร็วและคุณภาพของโหมดนี้
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import serial                                                    # noqa: E402
from scope_view import send_command, read_frame                  # noqa: E402
from ultrasonic import BAUD                                      # noqa: E402

C = 343.0
T0_US = 1220.0        # เวลาหน่วงของระบบ (ยาวชุดยิง + ทริกเกอร์) — ดู features.T0_US

# ค่าตั้งต้นของระบบเก็บข้อมูล — เลือกไว้ให้คาบลงตัวที่ 20 ครั้ง/วินาที
# ที่ 200 cm งานจริงใช้ 47.9 ms (เก็บ 13.1 + ส่ง 34.8) จึงพอดีกับคาบ 50 ms
MAX_CM, PERIOD_MS = 200.0, 50.0


class Sync4:
    """เปิดครั้งเดียว ยิงซ้ำได้ด้วยคาบคงที่ · ทุกปิงได้ครบ 4 ช่องจากการยิงครั้งเดียว"""

    def __init__(self, port="COM5", pins="34,35,32,33", max_cm=MAX_CM,
                 rate_req=800000, cycles=16, period_ms=PERIOD_MS, baud=BAUD,
                 verbose=False):
        self.pins = [int(p) for p in pins.split(",")]
        if len(self.pins) != 4:
            raise ValueError("โหมดนี้ต้องใช้ 4 ช่องเท่านั้น")
        self.max_cm = float(max_cm)
        self.bad_frames = 0
        self.ser = serial.Serial(port, baud, timeout=10)
        self.ser.setDTR(False)
        self.ser.setRTS(False)
        time.sleep(2.0)                      # รอ ESP32 บูต
        self.ser.reset_input_buffer()

        send_command(self.ser, f"b {cycles}", settle=0.3)
        send_command(self.ser, "c " + ",".join(str(p) for p in self.pins), settle=0.6)
        send_command(self.ser, f"f {rate_req}", settle=0.5)

        # อัตราที่ ADC ทำได้จริงรู้ไม่ได้จนกว่าจะอ่านเฟรมแรก จึงตั้งชั่วคราวแล้ววัดเอง
        send_command(self.ser, "n 900", settle=0.4)
        probe = self._read(retries=4)
        if probe is None:
            self.close()
            raise RuntimeError("อ่านเฟรมแรกไม่ได้ — เช็คสายและพอร์ต")
        if probe["data"].shape[0] != 4:
            self.close()
            raise RuntimeError(
                f"ตั้ง 4 ช่องไม่สำเร็จ (ได้ {probe['data'].shape[0]} ช่อง) — "
                f"โหมดนี้ไม่ยอมถอยไปใช้การยิงหลายรอบ")
        self.rate = float(probe["rate"])
        self.skew_us = float(probe.get("skew_us", 0.0))

        # จำนวนตัวอย่างที่พอดีกับระยะที่ต้องการ ไม่เก็บเกิน = ไม่เสียเวลาส่งเปล่า
        need_us = T0_US + 2 * self.max_cm / 100.0 / C * 1e6
        self.samples = int(np.ceil(need_us * 1e-6 * self.rate)) + 16
        send_command(self.ser, f"n {self.samples}", settle=0.4)

        # คาบขั้นต่ำ: ต้องยาวกว่าทั้ง (เวลาเสียงไปกลับ) และ (เวลาเก็บ+ส่ง)
        self.acq_ms = self.samples / self.rate * 1e3
        self.tx_ms = self.samples * 4 * 2 * 10 / baud * 1e3
        self.acoustic_ms = 2 * self.max_cm / 100.0 / C * 1e3
        floor_ms = max(self.acoustic_ms * 1.3, self.acq_ms + self.tx_ms)
        self.period = (period_ms if period_ms else floor_ms + 3.0) / 1e3
        if period_ms and period_ms < floor_ms:
            # เตือนแต่ไม่ห้าม — ถ้าคาบสั้นกว่างานจริง จังหวะจะรวนและอาจเกิดเอคโค่ทับรอบ
            print(f"  !! คาบที่ตั้ง {period_ms:.1f} ms สั้นกว่าที่งานต้องการ "
                  f"{floor_ms:.1f} ms (เสียง {self.acoustic_ms*1.3:.1f} · "
                  f"เก็บ+ส่ง {self.acq_ms+self.tx_ms:.1f}) — จังหวะจะไม่คงที่")
        self._next = self._t_prev = None
        self.pings = 0           # จำนวนปิงทั้งหมด (ใช้เป็นตัวหาร)
        self.late = 0            # จำนวนครั้งที่ยิงช้ากว่าคาบที่ตั้งไว้
        self.late_ms = 0.0
        from collections import deque
        self.periods = deque(maxlen=200)   # คาบจริงที่วัดได้
        if verbose:
            print(f"  4 ช่องพร้อมกัน · {self.rate:.0f} Hz/ช่อง · {self.samples} ตัวอย่าง "
                  f"({self.samples/self.rate*1e3:.1f} ms = {self.max_cm:.0f} cm)")
            print(f"  คาบ {self.period*1e3:.1f} ms  "
                  f"(เสียงต้องการ {self.acoustic_ms:.1f} · เก็บ {self.acq_ms:.1f} · "
                  f"ส่ง {self.tx_ms:.1f})")

    def _read(self, retries=3):
        for _ in range(retries):
            self.ser.reset_input_buffer()
            self.ser.write(b"r\n")
            fr = read_frame(self.ser, verbose=False)
            if fr is not None:
                return fr
            self.bad_frames += 1
        return None

    def ping(self, retries=3):
        """ยิงหนึ่งครั้งตามจังหวะ คืน dict แบบเดียวกับ Ultrasonic.ping() หรือ None

        รอให้ครบคาบก่อนเสมอ แม้รอบก่อนจะเสร็จเร็ว — จังหวะที่สม่ำเสมอสำคัญกว่า
        ความเร็วสูงสุด เพราะรับประกันว่าเอคโค่รอบเก่าตายสนิทแล้วทุกครั้ง
        """
        now = time.time()
        self.pings += 1
        if self._next is not None:
            if now < self._next:
                time.sleep(self._next - now)
            else:
                # มาช้ากว่ากำหนด = งานในลูป (รวมการวาดภาพของผู้เรียก) ยาวกว่าคาบ
                # จังหวะจะไม่คงที่อีกต่อไป ต้องนับไว้ให้เห็น ไม่ใช่ปล่อยหลุดเงียบ ๆ
                self.late += 1
                self.late_ms += (now - self._next) * 1e3
        self._next = max(time.time(), (self._next or time.time())) + self.period
        if self._t_prev is not None:
            self.periods.append((time.time() - self._t_prev) * 1e3)
        self._t_prev = time.time()
        fr = self._read(retries)
        if fr is None or fr["data"].shape[0] != 4:
            return None
        return {"counts": fr["data"].astype(np.uint16), "rate": float(fr["rate"]),
                "skew_us": float(fr.get("skew_us", 0.0)), "pins": self.pins,
                "npairs": 1, "simultaneous": True, "t": time.time()}

    def cadence(self):
        """คืน (คาบจริงกลาง ms, สัดส่วนที่ยิงช้ากว่ากำหนด) — ใช้ตรวจว่าจังหวะคงที่จริงไหม"""
        if not self.periods:
            return float("nan"), 0.0
        # ต้องหารด้วยจำนวนปิงทั้งหมด ไม่ใช่ความยาว deque (ซึ่งจำกัดที่ 200)
        return float(np.median(self.periods)), self.late / max(self.pings, 1)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _selftest(a):
    """วัดความเร็วจริงและคุณภาพเอคโค่ — ใช้ตรวจก่อนเอาไปเก็บ dataset"""
    import features as F
    us = Sync4(port=a.port, pins=a.pins, max_cm=a.max_cm,
               period_ms=a.period_ms, verbose=True)
    lo, hi = F.signal_band(us.rate)
    print(f"  ย่านกรองที่เลือกให้: {lo/1e3:.1f}-{hi/1e3:.1f} kHz "
          f"(40 kHz พับมาที่ {abs(F.F0 - round(F.F0/us.rate)*us.rate)/1e3:.1f} kHz)\n")
    rows, t0, got = [], time.time(), 0
    for _ in range(a.pings):
        p = us.ping()
        if p is None:
            continue
        got += 1
        envs = [F.envelope_of(p["counts"][c], p["rate"]) for c in range(4)]
        k, rng, ref = F.common_peak(envs, p["rate"])
        i0 = max(1, int((T0_US + 2 * F.GATE_MIN_CM / 100 / C * 1e6) * 1e-6 * p["rate"]))
        i1 = min(len(envs[ref]),
                 int((T0_US + 2 * F.GATE_MAX_CM / 100 / C * 1e6) * 1e-6 * p["rate"]))
        nf = max(float(np.median(envs[ref][i0:i1])), 1e-12)
        per_ch = [F.echo_peak(e, p["rate"])[1] for e in envs]
        rows.append([rng, float(envs[ref][k]) / nf, float(np.std(per_ch))]
                    + [float(e[k]) * 1e3 for e in envs])
    dt = (time.time() - t0) / max(got, 1)
    us.close()
    if not rows:
        sys.exit("อ่านไม่ได้เลย")
    r = np.array(rows)
    print(f"  ได้ {got}/{a.pings} ปิง · **{1/dt:.1f} fps** ({dt*1e3:.0f} ms/ping) · "
          f"เฟรมเสีย {us.bad_frames}")
    print(f"  ระยะที่วัดได้ : กลาง {np.median(r[:,0]):6.1f} cm · "
          f"แกว่ง (ส่วนเบี่ยงเบน) {r[:,0].std():5.1f} cm")
    print(f"  SNR ของยอด   : กลาง {np.median(r[:,1]):6.1f} เท่า")
    print(f"  4 ช่องเห็นตรงกันไหม (ส่วนเบี่ยงเบนของระยะข้ามช่อง): "
          f"{np.median(r[:,2]):.1f} cm")
    print(f"  ความแรงที่ยอดรายช่อง (mV): " +
          "  ".join(f"g{p}={np.median(r[:,3+i]):.0f}" for i, p in enumerate(us.pins)))
    print("\n  อ่านผล: 'แกว่ง' ต่ำ + 'เห็นตรงกัน' ต่ำ + SNR สูง = เอคโค่จริง เชื่อได้")
    print("          ถ้าแกว่งหลายสิบ cm แปลว่ากำลังจับเสียงขยะ ไม่ใช่เป้า")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--max-cm", type=float, default=MAX_CM)
    ap.add_argument("--period-ms", type=float, default=PERIOD_MS,
                    help="บังคับคาบยิง (ไม่ระบุ = คำนวณให้สั้นที่สุดที่ยังปลอดภัย)")
    ap.add_argument("--pings", type=int, default=40)
    _selftest(ap.parse_args())


if __name__ == "__main__":
    main()
