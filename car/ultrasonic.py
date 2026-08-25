"""อ่านอัลตราซาวด์จาก ESP32 (esp32_scope) — แยกให้ตัวเก็บ dataset เรียกใช้

ห่อไปป์ไลน์ใน tools/ ที่พิสูจน์แล้วใน Stage 1-2 ให้เปิดพอร์ตครั้งเดียว ยิงซ้ำได้

รองรับ 2 หรือ 4 ช่อง:
  2 ช่อง (34,35)        = ยิงครั้งเดียว อ่านคู่เดียว
  4 ช่อง (34,35,32,33)  = ยิงสองครั้ง อ่านทีละคู่ (ADC บน ESP32 อ่านพร้อมกันได้แค่ 2)
                          ได้ผลเพราะฉากนิ่งตอนเก็บ dataset — คู่ (34,35)=A140-C140 และ
                          (32,33)=A80-C80 คือเบสไลน์แนวนอนสองขนาด วัด TDOA ภายในแต่ละคู่
                          (ข้ามคู่ไม่ซิงก์กัน แต่เราไม่ใช้ — master/slave อ่านพร้อมกันไว้ Stage C)
"""
import os
import sys
import time

import numpy as np
import serial

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from scope_view import send_command, read_frame, counts_to_volts   # noqa: E402,F401


# ต้องตรงกับ Serial.begin() ในเฟิร์มแวร์ esp32_scope.ino เสมอ
# 2 Mbaud แทน 921600 เพราะเวลาส่งเฟรมดิบคือคอขวดของทั้งระบบ (ดู car/README.md)
BAUD = 2000000

SWITCH_SETTLE = 0.0      # วินาทีที่รอหลังสั่ง ADC สลับคู่ช่อง (ดู _capture_pair)


class Ultrasonic:
    def __init__(self, port="COM5", baud=BAUD, pins="34,35",
                 rate=800000, samples=4000, cycles=16):
        # 4000 ตัวอย่าง/ช่อง ที่ 266 kS/s = 15 ms = ระยะ 2.6 m ครอบคลุมช่วงที่กล้องเห็น
        # และลดข้อมูลที่ส่ง = เจอเฟรมเสียน้อยลงเมื่อกล้องแย่งแบนด์วิดท์ USB
        self.ser = serial.Serial(port, baud, timeout=10)
        self.ser.setDTR(False)
        self.ser.setRTS(False)
        time.sleep(2.0)                       # รอ ESP32 บูต
        self.ser.reset_input_buffer()

        self.pins = [int(p) for p in pins.split(",")]
        self.nch = len(self.pins)
        if self.nch not in (2, 4):
            raise ValueError("รองรับ 2 หรือ 4 ช่องเท่านั้น")
        # จัดเป็นคู่ ADC อ่านพร้อมกันได้ทีละ 2
        self.pairs = [self.pins[i:i + 2] for i in range(0, self.nch, 2)]

        send_command(self.ser, f"f {rate}", settle=0.5)
        send_command(self.ser, f"n {samples}", settle=0.4)
        # 16 รอบให้ SNR ดีกว่า 8 รอบ 4.6 dB ที่ 1 m แลกกับโซนบอดลึกขึ้นเป็น 7 cm
        send_command(self.ser, f"b {cycles}", settle=0.3)
        # ถ้าคู่เดียว ตั้งช่องครั้งเดียวไว้เลย ไม่ต้องสลับทุกปิง
        self._single_pair = len(self.pairs) == 1
        if self._single_pair:
            send_command(self.ser, f"c {self.pairs[0][0]},{self.pairs[0][1]}", settle=0.6)
        self.bad_frames = 0      # เฟรมที่เสียแล้วต้องยิงซ้ำ — ใช้ดูสุขภาพสาย USB

    def _capture_pair(self, pair, fire, retries):
        """ยิงและอ่านหนึ่งคู่ คืน (data[2,n], rate, skew) หรือ None"""
        if not self._single_pair:
            # ไม่ต้องหน่วงหลังสั่งสลับช่อง ADC เลย — วัดจริง ส.ค. 2026 ที่ settle
            # 0.20/0.08/0.03/0.00 วิ ได้ 1.3/1.8/2.2/**2.6** fps โดย**เฟรมเสีย 0 ทุกค่า**
            # ดีเลย์นี้จึงเป็นการรอเปล่า ๆ ที่กินเวลาไปครึ่งหนึ่งของรอบเก็บข้อมูล
            # (ถ้าวันหนึ่ง bad_frames ไต่ขึ้น ค่อยตั้ง SWITCH_SETTLE กลับเป็น 0.03)
            send_command(self.ser, f"c {pair[0]},{pair[1]}", settle=SWITCH_SETTLE)
        for _ in range(retries):
            self.ser.reset_input_buffer()
            self.ser.write(b"r\n" if fire else b"q\n")
            fr = read_frame(self.ser, verbose=False)
            if fr is not None and fr["data"].shape[0] == 2:
                return fr["data"], float(fr["rate"]), float(fr.get("skew_us", 0.0))
            self.bad_frames += 1
        return None

    def ping(self, fire=True, retries=3):
        """ยิงแล้วเก็บทุกช่อง คืน dict หรือ None

        4 ช่อง = ยิงสองครั้ง ทีละคู่ (ฉากต้องนิ่งช่วง ~1 วินาทีนี้)
        counts เรียงตาม self.pins: [ch0,ch1] = คู่แรก, [ch2,ch3] = คู่สอง
        """
        datas, rate, skew = [], None, None
        for pair in self.pairs:
            got = self._capture_pair(pair, fire, retries)
            if got is None:
                return None
            d, rate, skew = got
            datas.append(d)
        n = min(d.shape[1] for d in datas)      # ตัดให้ยาวเท่ากันเผื่อคลาดกันเฟรมละแซมเปิล
        counts = np.vstack([d[:, :n] for d in datas]).astype(np.uint16)
        return {
            "counts": counts,                   # [ch][sample] ดิบ 0..4095 เรียงตาม pins
            "rate": rate,                       # ตัวอย่าง/วินาที ต่อช่อง
            "skew_us": skew,                    # เหลื่อมภายในคู่ (ค่าคงที่ หักได้)
            "pins": self.pins,
            "npairs": len(self.pairs),
            "t": time.time(),
        }

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
