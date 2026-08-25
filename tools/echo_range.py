#!/usr/bin/env python3
"""echo_range.py - วัดระยะให้เป็นตัวเลขเดียว พร้อมค่ากระจาย

ต่างจาก echo_profile.py ตรงที่ตัวนี้ไม่ได้หา "ยอดของเอนเวโลป" แต่ใช้ matched filter

เหตุผล: พัลส์ตรงกับเอคโค่คือคลื่นลูกเดียวกัน — เบิร์สต์เดียวกัน ทรานสดิวเซอร์คู่เดิม
เวลาไต่ขึ้นเท่ากัน — พัลส์ตรงจึงเป็นแม่แบบที่ตรงเป๊ะของเอคโค่ พอเอาสัญญาณทั้งเส้นมา
หาสหสัมพันธ์กับแม่แบบ พลังงานของพัลส์ที่เคยกระจายอยู่ทั่วช่วง ~1 ms จะยุบมารวมกัน
เป็นยอดแคบยอดเดียว ยอดเอนเวโลปเดิมมันแบน ตำแหน่งยอดเลยเดินไปเดินมาทุกช็อต

แล้วยังทำ parabolic interpolation รอบยอดอีกชั้น จึงได้ความละเอียดต่ำกว่าหนึ่งแซมเปิล

    python tools/echo_range.py --port COM5 --expect-cm 20
    python tools/echo_range.py --port COM5 --shots 20 --min-cm 40
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import serial
from scope_view import send_command, read_frame, counts_to_volts, bandpass, envelope

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--port", default="COM5", help="พอร์ตอนุกรม เช่น COM5")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--pin", type=int, default=34, help="ขา ADC (34 = ช่อง 1, 35 = ช่อง 2)")
ap.add_argument("--shots", type=int, default=10, help="ยิงกี่ครั้ง (แต่ละครั้งได้ระยะหนึ่งค่า)")
ap.add_argument("--min-cm", type=float, default=30.0,
                help="เริ่มค้นหาเอคโค่ตั้งแต่ระยะนี้ขึ้นไป กันไปเจอหางของพัลส์ตรง")
ap.add_argument("--expect-cm", type=float, help="ระยะที่วัดด้วยไม้บรรทัด เพื่อให้เทียบให้")
a = ap.parse_args()

C = 343.0
ser = serial.Serial(a.port, a.baud, timeout=5)
ser.setDTR(False)
ser.setRTS(False)
time.sleep(2.0)
ser.reset_input_buffer()
send_command(ser, f"c {a.pin}", settle=0.4)

res, fs = [], None
for _ in range(a.shots):
    ser.reset_input_buffer()
    ser.write(b"r\n")
    fr = read_frame(ser, verbose=False)
    if fr is None:
        continue
    fs = fr["rate"]
    v = counts_to_volts(fr["data"][0]).astype(float)
    x = bandpass(v - v.mean(), fs, 25e3, 60e3)
    env = envelope(x)
    floor = np.median(env[int(0.9 * len(env)):])
    lead = int(np.flatnonzero(env > floor * 8)[0])
    dpk = lead + int(np.argmax(env[lead:lead + int(1e-3 * fs)]))

    # แม่แบบ = พัลส์ตรง ตั้งแต่ขอบขึ้นไปจนสุดหาง
    t0 = max(0, lead - int(50e-6 * fs))
    t1 = min(len(x), dpk + int(500e-6 * fs))
    tmpl = x[t0:t1] * np.hanning(t1 - t0)
    tmpl -= tmpl.mean()

    corr = np.correlate(x, tmpl, mode="valid")
    cenv = envelope(corr)
    lag0 = t0                            # ตำแหน่งที่แม่แบบตรงกับตัวเอง = ระยะ 0
    lo = lag0 + int(2 * (a.min_cm / 100) / C * fs)
    k = lo + int(np.argmax(cenv[lo:]))
    y0, y1, y2 = cenv[k-1], cenv[k], cenv[k+1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    lag = (k + d) - lag0
    res.append((lag / fs * 1e6, cenv[k], np.median(cenv)))
ser.close()

if not res:
    sys.exit("ไม่ได้เฟรมเลยสักช็อต - เช็คว่า esp32_scope แฟลชอยู่ และไม่มีโปรแกรมอื่นจับพอร์ตค้างไว้")

r = np.array(res)
us, amp, base = r[:, 0], r[:, 1], r[:, 2]
cm = us * 1e-6 * C / 2 * 100
print(f"เฉลี่ย {len(r)} ช็อต · fs {fs/1e3:.1f} kS/s · ค้นหาตั้งแต่ {a.min_cm:.0f} cm ขึ้นไป\n")
for i, (u, c_) in enumerate(zip(us, cm), 1):
    print(f"  ยิงที่ {i:2d}: หน่วงเวลา {u:8.1f} µs = ระยะ {c_:7.2f} cm")
print(f"\nระยะเฉลี่ย  {cm.mean():.2f} cm")
print(f"กระจาย     {cm.std()*10:.2f} mm  (เวลา {us.std():.2f} µs)")
print(f"ช่วง       {cm.min():.2f} .. {cm.max():.2f} cm")
print(f"SNR ของยอดสหสัมพันธ์ {20*np.log10(amp.mean()/base.mean()):.1f} dB")
if a.expect_cm:
    print(f"\nวางไว้จริง {a.expect_cm:.0f} cm  →  ต่าง {cm.mean()-a.expect_cm:+.2f} cm")
