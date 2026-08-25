#!/usr/bin/env python3
"""match_transducers.py - วัดความถี่เรโซแนนซ์ของหัวรับทีละตัว แล้วคัดชุดที่เข้ากัน

ทำไมต้องคัด: หัวรับ 40 kHz เป็นตัวสั่นพ้อง Q สูง เฟสจึงกวาดเร็วมากรอบเรโซแนนซ์
สองตัวที่ f0 ต่างกันจะให้เฟสต่างกันที่ความถี่ใช้งาน และเฟสที่ต่างกันคือ Δt ที่ผิดโดยตรง
วัดจริงบนรีก 2 ช่องแรกได้ f0 ต่างกัน 874 Hz = Δt 0.90 µs ซึ่งกินงบ Stage 2 ไป 89%

วิธีใช้ - เสียบหัวรับทีละตัวเข้าที่อินพุตของช่องที่ 1 (บอร์ด RX รู 5B กับ ราง −)
แล้วรันทีละตัว ใช้ช่องเดิม บอร์ดเดิม หัวส่งเดิม ตลอด ความต่างที่วัดได้จึงเป็นของหัวรับล้วนๆ

    python tools/match_transducers.py --port COM5 R01
    python tools/match_transducers.py --port COM5 R02
    ...
    python tools/match_transducers.py --report          # สรุปและจัดชุดที่ดีที่สุด

ผลเก็บสะสมใน hardware/transducer_match.json
"""
import os
import sys
import json
import time
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(os.path.dirname(HERE), "hardware", "transducer_match.json")
Q_ASSUMED = 20.0
F_DRIVE = 40_000.0


def load():
    return json.load(open(STORE, encoding="utf-8")) if os.path.exists(STORE) else {}


def phase_at(f, f0, Q=Q_ASSUMED):
    """เฟสของตัวสั่นพ้องอันดับสองที่ความถี่ f หน่วยองศา"""
    return -np.degrees(np.arctan(2 * Q * (f - f0) / f0))


def report():
    d = load()
    if len(d) < 2:
        sys.exit(f"ยังมีข้อมูลแค่ {len(d)} ตัว ต้องวัดอย่างน้อย 2 ตัวก่อน")
    items = sorted(d.items(), key=lambda kv: kv[1]["f0"])
    print(f"\nวัดไว้ {len(items)} ตัว · เรียงตามความถี่เรโซแนนซ์\n")
    print(f"{'ชื่อ':>8} {'f0 (kHz)':>10} {'ยอด (mV)':>11} {'เทียบตัวต่ำสุด':>14}")
    lo = items[0][1]["f0"]
    for k, v in items:
        print(f"{k:>8} {v['f0']/1e3:10.3f} {v['amp']*1e3:10.1f} {v['f0']-lo:+12.0f} Hz")

    f0s = np.array([v["f0"] for _, v in items])
    names = [k for k, _ in items]
    print(f"\nช่วงทั้งกอง {f0s.max()-f0s.min():.0f} Hz")

    for n in (2, 6, 9):
        if len(items) < n:
            continue
        # เรียงแล้ว ชุดที่แคบที่สุดคือช่วงติดกันเสมอ
        i = int(np.argmin([f0s[j+n-1] - f0s[j] for j in range(len(f0s)-n+1)]))
        span = f0s[i+n-1] - f0s[i]
        worst = max(abs(phase_at(F_DRIVE, a) - phase_at(F_DRIVE, b))
                    for a, b in itertools.combinations(f0s[i:i+n], 2))
        dt = worst / 360 * (1e6 / F_DRIVE)
        ok = "ผ่าน" if dt < 1.0 else "เกินงบ"
        print(f"\nชุดที่ดีที่สุด {n} ตัว: {', '.join(names[i:i+n])}")
        print(f"   ช่วง {span:.0f} Hz · เฟสต่างสูงสุด {worst:.1f}° · Δt {dt:.2f} µs  → {ok}")


def measure(a):
    import serial
    from scope_view import send_command, read_frame, counts_to_volts, bandpass, envelope

    ser = serial.Serial(a.port, a.baud, timeout=8)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(2.0)
    ser.reset_input_buffer()
    send_command(ser, "f 500000", settle=0.4)
    send_command(ser, "n 8000", settle=0.4)
    send_command(ser, f"c {a.pin}", settle=0.4)

    def sweep(freqs, shots):
        out = []
        for hz in freqs:
            send_command(ser, f"s {int(hz)}", settle=0.25)
            amps = []
            for _ in range(shots):
                ser.reset_input_buffer()
                ser.write(b"r\n")
                fr = read_frame(ser, verbose=False)
                if fr is None:
                    continue
                fs = fr["rate"]
                v = counts_to_volts(fr["data"][0]).astype(float)
                env = envelope(bandpass(v - v.mean(), fs, 20e3, 70e3))
                amps.append(env[int(1.2e-3 * fs):].max())   # ข้ามพัลส์ตรง
            out.append(np.mean(amps) if amps else 0.0)
        return np.array(out)

    def peak(fr, ys):
        k = int(np.argmax(ys))
        if k in (0, len(ys) - 1):
            return fr[k], ys[k]
        y0, y1, y2 = ys[k-1], ys[k], ys[k+1]
        den = y0 - 2*y1 + y2
        d = 0.5 * (y0 - y2) / den if den else 0.0
        return fr[k] + d * (fr[1] - fr[0]), y1

    coarse = np.arange(36000, 45001, 1000.0)
    print(f"กวาดหยาบ {coarse[0]/1e3:.0f}-{coarse[-1]/1e3:.0f} kHz ...")
    yc = sweep(coarse, 3)
    fc, _ = peak(coarse, yc)
    fine = np.arange(max(35000, fc - 1000), min(46000, fc + 1001), 250.0)
    print(f"กวาดละเอียดรอบ {fc/1e3:.1f} kHz ...")
    yf = sweep(fine, 4)
    f0, amp = peak(fine, yf)
    send_command(ser, "s 40000", settle=0.3)
    ser.close()

    d = load()
    d[a.label] = {"f0": float(f0), "amp": float(amp), "when": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    json.dump(d, open(STORE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n  {a.label}:  f0 = {f0/1e3:.3f} kHz   ยอด {amp*1e3:.1f} mV")
    others = {k: v for k, v in d.items() if k != a.label}
    if others:
        near = min(others.items(), key=lambda kv: abs(kv[1]["f0"] - f0))
        df = f0 - near[1]["f0"]
        ph = abs(phase_at(F_DRIVE, f0) - phase_at(F_DRIVE, near[1]["f0"]))
        print(f"  ใกล้ที่สุดคือ {near[0]} ต่างกัน {df:+.0f} Hz "
              f"→ เฟส {ph:.1f}° → Δt {ph/360*25:.2f} µs")
    print(f"  เก็บแล้ว {len(d)} ตัว · ดูสรุปด้วย --report")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?", help="ชื่อหัวรับตัวนี้ เช่น R01")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--pin", type=int, default=34, help="ช่องที่ใช้เป็นเครื่องวัด")
    ap.add_argument("--report", action="store_true", help="สรุปผลที่เก็บไว้ ไม่ต้องต่อบอร์ด")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.label:
        measure(a)
    else:
        ap.error("ใส่ชื่อหัวรับ เช่น R01 หรือใช้ --report")


if __name__ == "__main__":
    main()
