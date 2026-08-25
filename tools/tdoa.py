#!/usr/bin/env python3
"""tdoa.py - วัดความต่างของเวลามาถึงระหว่างหัวรับสองตัว (Stage 2)

สองอย่างที่ต้องทำให้ถูก ไม่งั้นตัวเลขสวยแต่ผิด

1. ความกำกวมของเฟส
   สัญญาณ 40 kHz ซ้ำตัวเองทุก 25 µs สหสัมพันธ์จึงมียอดสูงพอกันหลายยอด
   ห่างกันยอดละ 25 µs การหยิบยอดที่สูงที่สุดจะกระโดดข้ามคาบเป็นครั้งคราว
   (ต้นแบบ Nano เดิมเจอปัญหานี้ทั้งโปรเจกต์) แก้ด้วยการหาสองชั้น -
   เอนเวโลปให้ค่าหยาบที่ไม่กำกวม แล้วเฟสให้ค่าละเอียดในคาบที่เอนเวโลปชี้

2. ค่าเหลื่อมของ ADC
   ADC ตัวเดียวเดินวนอ่านทีละช่อง ช่องที่สองจึงถูกอ่านช้ากว่าช่องแรกเสมอ
   หนึ่งช่องการแปลง เฟิร์มแวร์วัดค่านี้จากเวลาจริงแล้วส่งมาใน skew_us
   เป็นค่าคงที่ที่รู้แน่นอน จึงลบทิ้งได้ ต่างจาก jitter ที่ลบไม่ได้

ตรวจสอบตัวเองได้ในตัว: TX ที่รู T1 ห่างจาก A140 กับ C140 เท่ากันเป๊ะ
**Δt ของพัลส์ตรงจึงต้องเป็น 0** ค่าที่อ่านได้จริงคือออฟเซ็ตของสองช่องล้วนๆ

    python tools/tdoa.py --port COM5 --shots 30
    python tools/tdoa.py --port COM5 --shots 30 --offset-us 1.23   # หักออฟเซ็ตที่เคยวัดไว้
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
BASELINE_MM = 242.5          # A140 -> C140


def lag_parabolic(y, k):
    """ยอดแบบต่ำกว่าหนึ่งตัวอย่าง จากพาราโบลาสามจุดรอบดัชนี k"""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    den = y0 - 2 * y1 + y2
    return k + (0.5 * (y0 - y2) / den if den else 0.0)


def delta_samples(a, b, half):
    """หน่วงของ b เทียบ a หน่วยเป็นตัวอย่าง (บวก = b มาถึงทีหลัง)

    ทำสองชั้น แต่ทั้งสองชั้นมาจาก**ตัวสหสัมพันธ์ตัวเดียวกัน**

      หยาบ = ยอดของเอนเวโลปของสหสัมพันธ์  -> ไม่กำกวม บอกว่าอยู่คาบไหน
      ละเอียด = ยอดของสหสัมพันธ์ดิบที่ใกล้ค่าหยาบที่สุด -> บอกตำแหน่งในคาบนั้น

    เดิมค่าหยาบมาจากการเอาเอนเวโลปของ *สัญญาณ* สองเส้นมาสหสัมพันธ์กัน ซึ่งแกว่ง
    σ 4.5 µs เพราะหัวรับสองตัวเรโซแนนซ์คนละความถี่ หางจึงลากไม่เท่ากันและรูปร่าง
    เอนเวโลปไม่เหมือนกัน พอค่าหยาบพลาดเกินครึ่งคาบ (12 µs) ชั้นละเอียดก็ไปเกาะ
    คาบข้างเคียง = cycle slip วัดได้ 4 ครั้งใน 30 ช็อต

    เอนเวโลปของสหสัมพันธ์ต่างออกไป มันคือผลรวมพลังงานของทั้งพัลส์ยุบมาเป็นยอด
    เดียว ความไม่เหมือนกันของหางจึงเฉลี่ยหายไปแทนที่จะสะสม
    """
    n = len(a)
    lags = np.arange(-half, half + 1)
    cf = np.array([np.dot(b[half:n - half], a[half - L:n - half - L]) for L in lags])
    ce = envelope(cf - cf.mean())
    coarse = float(int(np.argmax(ce)) - half)
    peaks = np.flatnonzero((cf[1:-1] > cf[:-2]) & (cf[1:-1] >= cf[2:])) + 1
    if not len(peaks):
        return None, None
    # คืนยอดทั้งหมด ไม่เลือกให้ - การเลือกคาบทำทีหลังโดยดูทั้งชุดพร้อมกัน
    cands = np.array([lag_parabolic(cf, int(k)) - half for k in peaks])
    return cands, coarse


def resolve(cands_us, coarse_us, period_us):
    """เลือกคาบให้ทุกช็อต โดยใช้ค่ากลางของทั้งชุดเป็นหลัก ไม่ใช่ค่าหยาบทีละช็อต

    ทำไมเชื่อค่าหยาบทีละช็อตไม่ได้: หัวรับสองตัวเรโซแนนซ์ห่างกัน 874 Hz ทำให้
    **หน่วงเอนเวโลป**ของสองช่องต่างกันหลายสิบ µs ขณะที่ **หน่วงเฟส** ต่างกันไม่ถึง 1 µs
    ค่าหยาบวัดอย่างแรก ค่าละเอียดวัดอย่างหลัง ทั้งสองจึงห่างกันเป็นระบบเกินครึ่งคาบ
    พอค่าหยาบไปตกกลางระหว่างสองคาบ การเลือก "ยอดที่ใกล้ค่าหยาบที่สุด" ก็กลายเป็น
    โยนเหรียญ - วัดจริงได้ slip 11 จาก 30

    สิ่งที่ใช้แทนคือข้อเท็จจริงที่ว่าเป้าไม่ขยับระหว่างชุดวัด ทุกช็อตจึงต้องตอบ
    ค่าเดียวกัน เริ่มจากค่าหยาบเป็นจุดตั้งต้น แล้ววนให้ทุกช็อตเกาะค่ากลางร่วมกัน

    ข้อจำกัดที่ต้องรู้: ใช้ได้เมื่อเป้าอยู่นิ่งกว่าครึ่งคาบ (12.5 µs = 7 mm ที่ 42 cm)
    ตลอดชุด ถ้าจะกวาดมุมต้องแยกชุดวัดต่อหนึ่งตำแหน่ง ห้ามขยับกลางชุด
    """
    pick = np.array([c[np.argmin(np.abs(c - k))] for c, k in zip(cands_us, coarse_us)])
    for _ in range(4):
        m = float(np.median(pick))
        new = np.array([c[np.argmin(np.abs(c - m))] for c in cands_us])
        if np.allclose(new, pick):
            break
        pick = new
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--rate", type=int, default=800000,
                    help="อัตรารวมทั้งสองช่อง 800k วัดแล้วได้จริง 266k ต่อช่อง")
    ap.add_argument("--samples", type=int, default=8000, help="ตัวอย่างต่อช่อง")
    ap.add_argument("--offset-us", type=float, default=None,
                    help="ออฟเซ็ตของสองช่องที่เคยวัดไว้ หักออกจากผล")
    ap.add_argument("--win-us", type=float, default=1600.0, help="ความกว้างหน้าต่างสหสัมพันธ์")
    ap.add_argument("--list", action="store_true", help="พิมพ์ค่าของทุกช็อต")
    ap.add_argument("--gate-cm", type=float,
                    help="ระยะโดยประมาณของเป้า ค้นหาเฉพาะแถวนั้น "
                         "จำเป็นเมื่อมีผนังหรือของอื่นสะท้อนแรงกว่าเป้า")
    ap.add_argument("--gate-width-cm", type=float, default=20.0,
                    help="ความกว้างของหน้าต่างค้นหา รอบ --gate-cm")
    ap.add_argument("--direct", action="store_true",
                    help="วัด Δt ของพัลส์ตรงแทนเอคโค่ = ออฟเซ็ตวงจรล้วนๆ "
                         "ไม่ต้องมีเป้า ไม่มีความคลาดจากการวาง (TX ที่ T1 ห่าง "
                         "A140 กับ C140 เท่ากัน Δt เชิงเรขาคณิต = 0)")
    a = ap.parse_args()

    # ออฟเซ็ตของคู่ช่องนี้ วัดไว้ครั้งเดียวแล้วใช้ตลอด จนกว่าจะเปลี่ยนฮาร์ดแวร์
    cal = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "hardware", "tdoa_calibration.json")
    if a.offset_us is None and os.path.exists(cal):
        import json
        a.offset_us = float(json.load(open(cal, encoding="utf-8"))["offset_us"])
        print(f"# ใช้ออฟเซ็ต {a.offset_us:+.2f} us จาก hardware/tdoa_calibration.json")

    if a.offset_us is None:
        a.offset_us = 0.0

    ser = serial.Serial(a.port, a.baud, timeout=10)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(2.0)
    ser.reset_input_buffer()
    send_command(ser, f"f {a.rate}", settle=0.5)
    send_command(ser, "c 34,35", settle=0.6)
    send_command(ser, f"n {a.samples}", settle=0.4)

    rows = []
    fs = skew = None
    for _ in range(a.shots):
        ser.reset_input_buffer()
        ser.write(b"r\n")
        fr = read_frame(ser, verbose=False)
        if fr is None or fr["data"].shape[0] != 2:
            continue
        fs = fr["rate"]
        skew = float(fr.get("skew_us", 0.0))
        burst_us = float(fr.get("burst_us", 200.0))
        half = int(a.win_us * 1e-6 * fs / 2)
        x = [bandpass(counts_to_volts(c).astype(float) - counts_to_volts(c).mean(),
                      fs, 25e3, 60e3) for c in fr["data"]]
        # เกาะยอดที่แรงที่สุดของช่องที่ 1 ไปเลย
        #
        # เดิมโค้ดนี้พยายามหา "พัลส์ตรง" ก่อนแล้วค่อยหาเอคโค่ถัดจากนั้น ใช้ไม่ได้
        # หลังย้าย TX ไปรู T1 - หัวส่งหลบมุมหัวรับมากขึ้นจนพัลส์ตรงเหลือไม่กี่ mV
        # ตัวตรวจจับเลยไปจับเอคโค่มาเรียกว่าพัลส์ตรง แล้ววัดหางของเอคโค่เองต่อ
        # ได้ตัวเลขที่ดูเหมือนใช้ได้แต่ไม่มีความหมายเลย
        e0, e1 = envelope(x[0]), envelope(x[1])
        floor = np.median(e0[int(0.85 * len(e0)):])
        if a.direct:
            # พัลส์ตรง = การมาถึงครั้งแรกหลังยิง ก่อนเอคโค่ใดๆ
            # ค้นในหน้าต่างแคบตั้งแต่จบเบิร์สต์ไปอีก ~40 cm ของระยะเดินทาง
            t0 = burst_us + 235.0
            lo_i = max(0, int((t0 - 60.0) * 1e-6 * fs))
            hi_i = min(len(e0), int((t0 + 700.0) * 1e-6 * fs))
            if hi_i - lo_i < 8:
                continue
            centre = lo_i + int(np.argmax(e0[lo_i:hi_i]))
        elif a.gate_cm:
            # ยอดที่แรงที่สุดไม่ใช่เป้าเสมอไป - ผนังคือแผ่นเรียบขนาดยักษ์และ
            # สะท้อนแรงกว่าเป้าเล็กๆ ได้หลายเท่า วัดจริงเจอผนังที่ 69 cm แรงกว่า
            # กระป๋องที่ 40 cm ถึง 3 เท่า จึงต้องบอกไปว่าให้มองแถวไหน
            #
            # t0 เอาจากดัชนีที่เฟิร์มแวร์ยิง บวกเวลาไต่ขึ้นของทรานสดิวเซอร์คร่าวๆ
            # หยาบระดับ +-5 cm ซึ่งพอสำหรับใช้เป็นประตู ไม่ได้ใช้เป็นค่าที่วัด
            t0 = burst_us + 235.0
            lo_i = max(0, int((t0 + (a.gate_cm - a.gate_width_cm / 2) / 100 * 2 / C * 1e6)
                              * 1e-6 * fs))
            hi_i = min(len(e0), int((t0 + (a.gate_cm + a.gate_width_cm / 2) / 100 * 2 / C * 1e6)
                                    * 1e-6 * fs))
            if hi_i - lo_i < 8:
                continue
            centre = lo_i + int(np.argmax(e0[lo_i:hi_i]))
        else:
            centre = int(np.argmax(e0))
        snr = 20 * np.log10(e0[centre] / floor)
        if snr < (12 if a.direct else 20):
            continue
        lo, hi = max(0, centre - half * 2), min(len(x[0]), centre + half * 2)
        if hi - lo < 4 * half:
            continue
        cands, coarse = delta_samples(x[0][lo:hi], x[1][lo:hi], half)
        if cands is None:
            continue
        rng_cm = (centre / fs * 1e6 - burst_us - 235.0) * 1e-6 * C / 2 * 100
        rows.append({
            "range_cm": rng_cm,
            # ช่อง 1 ถูกอ่านช้ากว่าช่อง 0 อยู่ skew - เอาออกจากผลที่วัดได้
            "cands": cands / fs * 1e6 - skew,
            "coarse": coarse / fs * 1e6 - skew,
            "peak_us": centre / fs * 1e6,
            "snr": snr,
            "amp0": e0[centre] * 1e3,
            "amp1": e1[lo:hi].max() * 1e3,   # ในหน้าต่างเดียวกัน ไม่ใช่ทั้งบันทึก
        })
    ser.close()

    if len(rows) < 3:
        sys.exit(f"ได้แค่ {len(rows)} ช็อตที่ใช้ได้ - เช็คว่ามีเป้าสะท้อนอยู่ตรงหน้า")

    dc = np.array([r["coarse"] for r in rows])
    de = resolve([r["cands"] for r in rows], dc, 1e6 / 40e3) - a.offset_us
    snr = np.array([r["snr"] for r in rows])
    print()
    print(f"{len(rows)} shots | {fs/1e3:.1f} kS/s per channel | "
          f"ADC skew {skew:.2f} us (subtracted)")
    print(f"target at roughly {np.mean([r['range_cm'] for r in rows]):.0f} cm"
          + (f"  (gate {a.gate_cm:.0f} +-{a.gate_width_cm/2:.0f} cm)" if a.gate_cm else "  (strongest peak)"))
    print(f"peak SNR {snr.mean():.1f} dB | amplitude "
          f"{np.mean([r['amp0'] for r in rows]):.0f} and "
          f"{np.mean([r['amp1'] for r in rows]):.0f} mV")
    print()
    print(f"{'':22}{'median':>11}{'sigma':>10}{'range':>20}")
    print(f"{'envelope (coarse)':22}{np.median(dc):9.2f} us{dc.std():8.2f} us"
          f"{dc.min():10.1f} ..{dc.max():7.1f}")
    print(f"{'phase (fine)':22}{np.median(de):9.2f} us{de.std():8.2f} us"
          f"{de.min():10.1f} ..{de.max():7.1f}")
    if a.list:
        print()
        print("  shot   phase(us)  coarse(us)   peak(us)   snr(dB)")
        for i, r in enumerate(rows, 1):
            print(f"  {i:4d} {de[i-1]:11.2f} {r['coarse']:11.2f} "
                  f"{r['peak_us']:10.0f} {r['snr']:9.1f}")
    sigma = float(de.std())
    print()
    print(f"Stage 2 needs sigma(dt) < 1.00 us  ->  measured {sigma:.2f} us  "
          f"{'PASS' if sigma < 1.0 else 'not yet'}")
    T = 1e6 / 40e3
    slips = int(np.sum(np.abs(de - np.median(de)) > T / 2))
    print(f"envelope-vs-phase bias {np.median(dc)-np.median(de):+.1f} us "
          f"(from the 874 Hz resonance mismatch; half a period is {1e6/40e3/2:.1f} us)")
    print(f"cycle slips (further than {T/2:.1f} us from the median): {slips} of {len(de)}")
    ang = np.degrees(np.arcsin(np.clip(np.median(de) * 1e-6 * C / (BASELINE_MM / 1000), -1, 1)))
    print(f"angle {ang:+.2f} deg | sensitivity "
          f"{np.degrees(np.arcsin(1e-6*C/(BASELINE_MM/1000))):.3f} deg per us")
    print("(channel offset not removed yet - calibrate with a target dead ahead first)")


if __name__ == "__main__":
    main()
