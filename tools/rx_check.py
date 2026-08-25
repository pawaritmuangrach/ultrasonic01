#!/usr/bin/env python3
"""rx_check.py - ตรวจภาครับอย่างเดียว ไม่ยิงคลื่น

ใช้ตอนเพิ่งต่อวงจรเสร็จ ก่อนจะไปวัดระยะ ตอบสามคำถาม:
  1. ไบแอสอยู่ที่ 1.65 V จริงไหม (ถ้าไม่ใช่ = VREF หรือ R ไบแอสผิด)
  2. มีฮัมไฟบ้าน 50 Hz เข้ามาเท่าไหร่ (ถ้าเป็นร้อย mV = ต่อผิดที่ไหนสักแห่ง)
  3. ที่ย่าน 40 kHz เงียบดีไหม ตอนไม่มีเสียง

วัดสองรอบเพราะอัตราสุ่มเดียวมองไม่เห็นทั้งสองปัญหา รอบช้า (20 kS/s) หน้าต่างยาวพอ
จะแยก 50 Hz ออกจาก 100 Hz ได้ รอบเร็ว (500 kS/s) ถึงจะเห็นย่าน 40 kHz

    python tools/rx_check.py --port COM5
    python tools/rx_check.py --port COM5 --pin 35      # เช็คช่องที่ 2
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import serial
from scope_view import send_command, read_frame, counts_to_volts

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--port", default="COM5", help="พอร์ตอนุกรม เช่น COM5")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--pin", type=int, default=34, help="ขา ADC ที่จะอ่าน (34 = ช่อง 1, 35 = ช่อง 2)")
a = ap.parse_args()

ser = serial.Serial(a.port, a.baud, timeout=4)
ser.setDTR(False)
ser.setRTS(False)
time.sleep(2.0)
ser.reset_input_buffer()
send_command(ser, f"c {a.pin}", settle=0.4)
send_command(ser, "n 24000", settle=0.3)


def grab(rate):
    send_command(ser, f"f {rate}", settle=0.4)
    ser.reset_input_buffer()
    ser.write(b"q\n")                    # q = เก็บแบบเงียบ ไม่ยิงเบิร์สต์
    return read_frame(ser, verbose=False)


def band(fq, sp, lo, hi):
    """พลังงานรวมในแถบความถี่ lo..hi หน่วย mV"""
    m = (fq >= lo) & (fq < hi)
    return float(np.sqrt((sp[m] ** 2).sum())) * 1e3 if m.any() else 0.0


for rate, tag in ((20000, "ช้า"), (500000, "เร็ว")):
    fr = grab(rate)
    if fr is None:
        print(f"[{tag}] ไม่ได้เฟรม")
        continue
    fs = fr["rate"]
    c = fr["data"][0]
    v = counts_to_volts(c).astype(float)
    n = len(v)
    ac = v - v.mean()
    sp = np.abs(np.fft.rfft(ac * np.hanning(n))) * 4.0 / n
    fq = np.fft.rfftfreq(n, 1.0 / fs)

    print(f"\n=== {tag}  fs={fs/1e3:.1f} kS/s  n={n}  หน้าต่าง {n/fs*1e3:.1f} ms ===")
    print(f"  ไฟตรง (ไบแอส) {v.mean()*1e3:8.1f} mV     (ต้องได้ ~1520-1650 mV)")
    print(f"  ต่ำสุด/สูงสุด  {v.min()*1e3:8.1f} / {v.max()*1e3:.1f} mV")
    print(f"  Vpp           {(v.max()-v.min())*1e3:8.1f} mV")
    print(f"  AC rms        {ac.std()*1e3:8.2f} mV")
    print(f"  ชนราง ล่าง/บน {int((c==0).sum()):6d} / {int((c>=4095).sum())}   (ต้องเป็น 0/0)")
    if rate == 20000:
        print(f"  50 Hz         {band(fq, sp, 45, 55):8.2f} mV   (ดีคือ < 10 mV)")
        print(f"  100 Hz        {band(fq, sp, 95, 105):8.2f} mV")
        print(f"  150 Hz        {band(fq, sp, 145, 155):8.2f} mV")
        print(f"  1-9 kHz       {band(fq, sp, 1000, 9000):8.2f} mV")
    else:
        print(f"  30-50 kHz     {band(fq, sp, 30000, 50000):8.2f} mV   (ย่านที่เราใช้)")
        print(f"  50-200 kHz    {band(fq, sp, 50000, 200000):8.2f} mV")
    k = np.argsort(sp)[::-1]
    seen, top = [], []
    for i in k:
        if fq[i] < 20:
            continue
        if any(abs(fq[i] - f) < fs / n * 4 for f in seen):
            continue
        seen.append(fq[i])
        top.append((fq[i], sp[i] * 1e3))
        if len(top) == 5:
            break
    print("  ยอดเด่น       " + " · ".join(f"{f:.0f} Hz {m:.2f} mV" for f, m in top))

ser.close()
