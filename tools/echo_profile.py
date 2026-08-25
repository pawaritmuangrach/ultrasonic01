#!/usr/bin/env python3
"""echo_profile.py - ยิงคลื่นแล้ววาดกราฟแท่งว่าเสียงสะท้อนกลับมาตอนไหนบ้าง

นี่คือตัวที่ให้ "ภาพรวม" ใช้ตอนยังไม่รู้ว่าเอคโค่อยู่ตรงไหน หรืออยากเห็นว่ามี
เอคโค่กี่ก้อน (สะท้อนสองเด้ง สะท้อนจากผนัง ฯลฯ)

วิธีอ่าน t = 0: ไม่ได้นับจากตอนสั่งยิง แต่นับจาก **ยอดของพัลส์ตรง** ที่วิ่งลัด
จากหัวส่งไปหัวรับบนแผ่นเดียวกัน เพราะทั้งพัลส์ตรงและเอคโค่ต่างก็ผ่านหัวรับ
ตัวเดียวกัน เวลาที่ทรานสดิวเซอร์ใช้ไต่ขึ้นจึงเท่ากันและหักล้างกันไปในผลต่าง
ถ้าไปจัดแนวขอบขึ้นกับยอด จะได้ระยะเพี้ยนไปหนึ่งเวลาไต่ขึ้นทุกครั้ง

    python tools/echo_profile.py --port COM5 --expect-cm 20
    python tools/echo_profile.py --port COM5 --shots 10 --span-ms 6
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
ap.add_argument("--shots", type=int, default=5, help="ยิงกี่ครั้งแล้วเฉลี่ย")
ap.add_argument("--expect-cm", type=float, help="ระยะที่วางไว้จริง เพื่อให้สคริปต์เทียบให้")
ap.add_argument("--span-ms", type=float, default=4.0, help="มองไปข้างหน้ากี่มิลลิวินาที")
ap.add_argument("--bin-us", type=float, default=100.0, help="ความกว้างของแต่ละแท่ง")
a = ap.parse_args()

C = 343.0                                # ความเร็วเสียงในอากาศ 20 องศา
ser = serial.Serial(a.port, a.baud, timeout=5)
ser.setDTR(False)
ser.setRTS(False)
time.sleep(2.0)
ser.reset_input_buffer()
send_command(ser, f"c {a.pin}", settle=0.4)

stack, fs = [], None
for _ in range(a.shots):
    ser.reset_input_buffer()
    ser.write(b"r\n")                    # r = ยิงเบิร์สต์แล้วเก็บ
    fr = read_frame(ser, verbose=False)
    if fr is None:
        continue
    fs = fr["rate"]
    v = counts_to_volts(fr["data"][0]).astype(float)
    env = envelope(bandpass(v - v.mean(), fs, 25e3, 60e3))
    floor = np.median(env[int(0.9 * len(env)):])
    lead = int(np.flatnonzero(env > floor * 8)[0])
    # ยอดของพัลส์ตรง: ค้นเฉพาะในช่วงหางของมันเอง กว้าง 1 ms
    dpk = lead + int(np.argmax(env[lead:lead + int(1e-3 * fs)]))
    stack.append((env, dpk, floor, dpk - lead))
ser.close()

if not stack:
    sys.exit("ไม่ได้เฟรมเลยสักช็อต - เช็คว่า esp32_scope แฟลชอยู่ และไม่มีโปรแกรมอื่นจับพอร์ตค้างไว้")

nspan = int(a.span_ms * 1e-3 * fs)
al = np.stack([e[p:p + nspan] for e, p, _, _ in stack])
avg = al.mean(axis=0)
floor = float(np.mean([f for _, _, f, _ in stack]))
ring = float(np.mean([r for _, _, _, r in stack])) / fs * 1e6
t_us = np.arange(nspan) / fs * 1e6
rng_cm = t_us * 1e-6 * C / 2 * 100

print(f"เฉลี่ย {len(stack)} ช็อต · fs {fs/1e3:.1f} kS/s · พื้นหลัง {floor*1000:.2f} mV")
print(f"เวลาไต่ขึ้นของทรานสดิวเซอร์ (ขอบ→ยอด) {ring:.0f} µs · จัดแนวที่ยอดพัลส์ตรง = 0 cm\n")
step = int(a.bin_us * 1e-6 * fs)
peak = avg.max()
print(f"{'ระยะ (cm)':>10} {'t (µs)':>8} {'mV':>9}  โปรไฟล์")
for i in range(0, nspan - step, step):
    m = float(avg[i:i + step].max())
    print(f"{rng_cm[i]:10.1f} {t_us[i]:8.0f} {m*1000:9.2f}  {'#' * int(round(46 * m / peak))}")

if a.expect_cm:
    want_us = 2 * (a.expect_cm / 100) / C * 1e6
    print(f"\nคาดว่าเอคโค่ที่ {a.expect_cm:.0f} cm ตกที่ {want_us:.0f} µs หลังยอดพัลส์ตรง")
    # หายอดเฉพาะที่ ข้ามหางของพัลส์ตรงไป
    d = np.diff(avg)
    loc = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    loc = [i for i in loc if t_us[i] > 300 and avg[i] > floor * 6]
    loc.sort(key=lambda i: -avg[i])
    print("ยอดเฉพาะที่แรงสุด 5 อันดับ:")
    for i in loc[:5]:
        print(f"  {rng_cm[i]:7.1f} cm  ({t_us[i]:6.0f} µs)  {avg[i]*1000:8.2f} mV  "
              f"SNR {20*np.log10(avg[i]/floor):5.1f} dB")
