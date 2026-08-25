#!/usr/bin/env python3
"""aim.py - หมุนแผ่นสะท้อนไปพลาง ดูความแรงไปพลาง จนกว่าจะได้จุดสูงสุด

แผ่นเรียบสะท้อนเสียงเหมือนกระจก กลีบสะท้อนแคบเท่ากับ lambda/ความกว้างของแผ่น
แผ่นกว้าง 100 mm ที่ 40 kHz ได้กลีบกว้างแค่ 4.9 องศา - เล็งด้วยตาไม่มีทางเข้า
วิธีเดียวที่ใช้ได้คือหมุนไปเรื่อยๆ แล้วดูว่าตรงไหนเสียงกลับมาแรงที่สุด

    python tools/aim.py --port COM5 --gate-cm 42

หมุนแผ่นช้าๆ ทีละนิด ดูแถบให้ยาวที่สุด แล้วหยุด · Ctrl-C เพื่อออก
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import serial
from scope_view import send_command, read_frame, counts_to_volts, bandpass, envelope

C = 343.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--gate-cm", type=float, required=True, help="ระยะโดยประมาณของเป้า")
    ap.add_argument("--gate-width-cm", type=float, default=20.0)
    ap.add_argument("--pin", type=int, default=34)
    ap.add_argument("--rate", type=int, default=500000)
    ap.add_argument("--samples", type=int, default=8000)
    a = ap.parse_args()

    ser = serial.Serial(a.port, a.baud, timeout=8)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(2.0)
    ser.reset_input_buffer()
    send_command(ser, f"f {a.rate}", settle=0.4)
    send_command(ser, f"c {a.pin}", settle=0.4)
    send_command(ser, f"n {a.samples}", settle=0.4)

    print(f"\nประตูที่ {a.gate_cm:.0f} +-{a.gate_width_cm/2:.0f} cm · หมุนแผ่นช้าๆ ให้แถบยาวที่สุด")
    print("Ctrl-C เพื่อหยุด\n")
    best = 0.0
    try:
        while True:
            ser.reset_input_buffer()
            ser.write(b"r\n")
            fr = read_frame(ser, verbose=False)
            if fr is None:
                continue
            fs = fr["rate"]
            burst_us = float(fr.get("burst_us", 200.0))
            v = counts_to_volts(fr["data"][0]).astype(float)
            e = envelope(bandpass(v - v.mean(), fs, 25e3, 60e3))
            floor = float(np.median(e[int(0.85 * len(e)):]))
            t0 = burst_us + 235.0
            lo = max(0, int((t0 + (a.gate_cm - a.gate_width_cm / 2) / 100 * 2 / C * 1e6)
                            * 1e-6 * fs))
            hi = min(len(e), int((t0 + (a.gate_cm + a.gate_width_cm / 2) / 100 * 2 / C * 1e6)
                                 * 1e-6 * fs))
            if hi - lo < 8:
                print("ประตูอยู่นอกบันทึก - ปรับ --gate-cm")
                break
            k = lo + int(np.argmax(e[lo:hi]))
            amp = float(e[k])
            best = max(best, amp)
            rng = (k / fs * 1e6 - t0) * 1e-6 * C / 2 * 100
            snr = 20 * np.log10(amp / floor) if floor else 0.0
            bar = "#" * int(round(56 * amp / best)) if best else ""
            mark = "  <-- ดีที่สุด" if amp >= best * 0.995 else ""
            print(f"{amp*1e3:7.1f} mV  SNR {snr:5.1f} dB  ที่ {rng:5.1f} cm  |{bar:<56}|{mark}")
    except KeyboardInterrupt:
        print(f"\nแรงที่สุดที่เจอ {best*1e3:.1f} mV")
        print("ถ้าตัวเลขตอนหยุดใกล้ค่านี้ = เล็งเข้าแล้ว ถ้าห่างมาก ให้หมุนกลับไปหาจุดนั้น")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
