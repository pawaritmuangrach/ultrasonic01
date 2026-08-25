#!/usr/bin/env python3
"""วัด "พื้นเสียงรบกวน" ของภาครับแต่ละช่อง — ใช้ตรวจคุณภาพสาย/กราวด์ ไม่ต้องใช้กล้อง

ทำไมต้องวัดตัวนี้ ไม่ใช่แค่ amp ของเอคโค่:
  ช่องที่รับสัญญาณรบกวนสูงจะมี "ยอดปลอม" โผล่สุ่มๆ ทั่วช่วงระยะ ซึ่งไปชนะเอคโค่จริง
  ตอนหายอด (argmax) ทำให้อ่านระยะผิดโดยที่ amp ดูสูงปกติ — วัดจริง ส.ค. 2026:
  ช่องบอร์ด 2 (g32/g33) พื้นเสียงสูงกว่าบอร์ด 1 (g34) ~3 เท่า จับเป้าถูกแค่ 27-51%
  ที่ระยะ >90 cm ขณะที่ g34 ได้ 89-100%

สิ่งที่ตัดออกไปแล้วด้วยการวัด (อย่าเสียเวลาไล่ซ้ำ):
  * ไม่ใช่สายจั๊มยาว — เปลี่ยนสาย GPIO32/33 ให้สั้นแล้ว พื้นเสียงเท่าเดิม (9.0 -> 9.0)
    เหตุผลเชิงวงจร: ขาออกออปแอมป์อิมพีแดนซ์ต่ำ (2.2k) รับ noise ยาก และ noise ที่เข้า
    หลังภาคขยายไม่ถูกขยายต่อ — จุดที่ไวคือ *ขาเข้า* (สายหัวรับ) ไม่ใช่ขาออก
  * ไม่ใช่คลื่นจากตัวส่ง — ปิดการยิง TX แล้วพื้นเสียงแทบเท่าเดิม (10.2 -> 9.5)
  * ไม่ใช่ EMI ที่ความถี่ใดความถี่หนึ่ง — สเปกตรัมแบนเรียบ สูงกว่า ~3-4 เท่าทุกย่าน
    ตั้งแต่ 0-133 kHz (ถ้าเป็น EMI จะเห็นเป็นยอดที่ความถี่เฉพาะ)
  * ไม่ใช่หัวรับ/สายหัวรับ — ถอดหัวรับ A80 ออกแล้ว noise "สูงขึ้น" (9.0 -> 12.4)
    เพราะอินพุตลอยกลายเป็นเสาอากาศ (อาการเดียวกับตอน Rx2 หลุด)
  * ไม่ใช่ไฟเลี้ยง/VREF ที่ยืมข้ามบอร์ด — ถ้าเป็นแหล่งร่วม noise สองช่องบนบอร์ดเดียวกัน
    ต้องสัมพันธ์กัน แต่วัดได้ corr(ch2,ch3) = -0.009 (ไม่สัมพันธ์เลย) = เกิดแยกรายช่อง
  * ไม่ใช่ขา ADC ของ ESP32 — สลับสายที่ ESP32 แล้ว noise "ตามวงจรไป" ไม่ได้อยู่กับขา
    (A80 ไปอยู่ g34 ก็ยัง 9.0 · C140 มาอยู่ g32 ได้ 5.2) ขามีผลบ้างแต่เล็ก (~1.3x)
  * ไม่ใช่ชิปออปแอมป์ — เปลี่ยน MCP6004 ตัวใหม่แล้วเท่าเดิม (9.0 -> 9.2)
  * ไม่ใช่ Cg ผิดค่า — ถ้า Cg เป็น 100nF แทน 1nF อัตราส่วนที่ย่าน 0-5 kHz ต้องสูง ~18 เท่า
    แต่วัดได้ 2.0 เท่า (ผังทั้งสองบอร์ดระบุ 1nF ตรงกัน)
  * **เส้นทางกราวด์ขากลับสำคัญที่สุดเท่าที่วัดมา** (ส.ค. 2026 บนแผ่นไข่ปลา):
    อ่านสัญญาณจากแผ่นหนึ่งด้วย ESP32 อีกตัวโดยให้กราวด์เชื่อมกันผ่านสาย USB อ้อมคอม
    -> พื้นเสียง 14.9 · ต่อสายกราวด์ตรงจาก H3 ขา GND ไป GND ของ ESP32 นั้น -> **4.5**
    (ดีขึ้น 3.3 เท่าจากสายเส้นเดียว) => สายสัญญาณต้องมีสายกราวด์คู่เดินเคียงเสมอ
  * เครื่องวัดเองก็เติม noise: ช่องเดียวกันอ่านด้วย ESP32 ตัวใหม่ได้ 24.0 ตัวเก่าได้ 14.9
    => เวลาเทียบสองวงจร ต้องอ่านด้วย ADC ตัวเดียวกัน ไม่งั้นเทียบเครื่องวัดแทนวงจร
  => เหลือผู้ต้องสงสัยเดียว: **ค่าตัวต้านทานบนบอร์ด 2 ผิด** โดยเฉพาะ R ไบแอส (10k)
     ซึ่งเป็นตัวเดียวที่เพิ่ม noise ได้มากโดยแทบไม่เปลี่ยนเกน (noise ความร้อน ~ sqrt(R))
     -> วัดด้วยมัลติมิเตอร์เทียบกับบอร์ด 1 ทีละตัว

ค่าที่ดูสามตัว (ยิ่งต่ำยิ่งดีทั้งหมด):
  พื้นเสียง (median)  ระดับสัญญาณในช่วงที่ไม่ควรมีอะไรสะท้อนกลับ
  crest (max/median)  ความ "แหลม" ของสัญญาณรบกวน — สูง = มียอดปลอมโดด ๆ (มัก EMI)
  ยอดปลอมสูงสุด       ตัวเลขที่เอคโค่จริงต้องเอาชนะให้ได้

    python tools/noise_check.py --port COM5 --pins 34,35,32,33
    python tools/noise_check.py --port COM5 --pins 34,35,32,33 --pings 20
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
sys.path.insert(0, HERE)

C = 343.0

# ค่าอ้างอิงที่วัดไว้ก่อนเปลี่ยนสาย (ส.ค. 2026) ไว้เทียบว่าดีขึ้นไหม
# วัดใหม่ ส.ค. 2026 ด้วย --pins 34,36 --samples 12000 (บอร์ดทดลอง g34 · ไข่ปลา g36)
# ค่าเก่าที่วัดคนละการตั้งค่ากันเทียบตรง ๆ ไม่ได้ จึงเลิกใช้
BEFORE = {34: (3.0, 12.8, 167.0), 36: (4.5, 5.3, 61.1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--pins", default="34,35,32,33")
    ap.add_argument("--pings", type=int, default=12)
    # 4000 ตัวอย่าง @400 kHz = 10 ms = แค่ 0.86 m — สั้นกว่าช่วงเงียบที่จะวัด
    # ต้องเก็บให้ยาวพอครอบ 250 cm ไม่งั้นดัชนีช่วงเงียบหลุดออกนอกข้อมูล
    ap.add_argument("--samples", type=int, default=12000,
                    help="ตัวอย่างต่อช่อง (12000 @400kHz = 30 ms = 5.1 m)")
    ap.add_argument("--quiet-from-cm", type=float, default=180.0,
                    help="เริ่มนับพื้นเสียงจากระยะนี้ (ควรไกลกว่าวัตถุใดๆ ในห้อง)")
    ap.add_argument("--quiet-to-cm", type=float, default=250.0)
    a = ap.parse_args()

    from ultrasonic import Ultrasonic
    from features import envelope_of, T0_US, GATE_MIN_CM, GATE_MAX_CM

    def idx(cm, rate):
        return int((T0_US + 2 * cm / 100 / C * 1e6) * 1e-6 * rate)

    print(f"เปิด {a.port} · ช่อง {a.pins} · ยิง {a.pings} ครั้ง ...")
    us = Ultrasonic(port=a.port, pins=a.pins, samples=a.samples)
    pins = us.pins
    acc = {p: {"nf": [], "crest": [], "mx": [], "pk": []} for p in pins}
    quiet = {p: [] for p in pins}
    skipped = {}      # พื้นเสียงตอน "ไม่ยิง TX" = สัญญาณรบกวนล้วนๆ
    try:
        # รอบแรก: ไม่ยิง TX เลย -> วัดว่าสัญญาณรบกวนมาจากวงจร/สิ่งแวดล้อม ไม่ใช่จาก TX
        for _ in range(max(4, a.pings // 3)):
            ping = us.ping(fire=False)
            if ping is None:
                continue
            rate = ping["rate"]
            for ci, p in enumerate(pins):
                e = envelope_of(ping["counts"][ci], rate)
                q0, q1 = idx(a.quiet_from_cm, rate), min(len(e), idx(a.quiet_to_cm, rate))
                if q1 > q0 + 20:
                    quiet[p].append(float(np.median(e[q0:q1])) * 1e3)
        got = 0
        for _ in range(a.pings * 2):
            if got >= a.pings:
                break
            ping = us.ping()
            if ping is None:
                continue
            got += 1
            rate = ping["rate"]
            for ci, p in enumerate(pins):
                e = envelope_of(ping["counts"][ci], rate)
                q0, q1 = idx(a.quiet_from_cm, rate), min(len(e), idx(a.quiet_to_cm, rate))
                g0, g1 = max(1, idx(GATE_MIN_CM, rate)), min(len(e), idx(GATE_MAX_CM, rate))
                if q1 <= q0 + 20 or g1 <= g0:
                    skipped[p] = (q0, q1, len(e))
                    continue
                seg = e[q0:q1]
                nf = max(float(np.median(seg)), 1e-12)
                acc[p]["nf"].append(nf * 1e3)
                acc[p]["crest"].append(float(seg.max()) / nf)
                acc[p]["mx"].append(float(seg.max()) * 1e3)
                acc[p]["pk"].append(float(e[g0:g1].max()) * 1e3)
    finally:
        us.close()

    for p, (q0, q1, n) in skipped.items():
        print(f"!! g{p}: ช่วงเงียบ (ดัชนี {q0}-{q1}) หลุดนอกข้อมูลที่มี {n} ตัวอย่าง "
              f"— เพิ่ม --samples หรือลด --quiet-to-cm")
    if not any(acc[p]["nf"] for p in pins):
        sys.exit("อ่านข้อมูลไม่ได้เลย — เช็คสายและพอร์ต")

    print(f"\nวัดจาก {got} ปิง · ช่วงเงียบ {a.quiet_from_cm:.0f}-{a.quiet_to_cm:.0f} cm\n")
    print(f"{'ช่อง':<8}{'พื้นเสียง':>10}{'crest':>8}{'ยอดปลอม':>10}{'ยอดสูงสุด':>11}"
          f"   {'เทียบก่อนเปลี่ยนสาย':<22}")
    for p in pins:
        d = acc[p]
        if not d["nf"]:
            print(f"g{p:<7}  (ไม่มีข้อมูล)")
            continue
        nf, cr, mx = np.mean(d["nf"]), np.mean(d["crest"]), np.mean(d["mx"])
        pk = np.mean(d["pk"])
        qv = np.mean(quiet[p]) if quiet[p] else float("nan")
        cmp_txt = f"ไม่ยิง TX {qv:5.1f}"
        if p in BEFORE:
            b_nf = BEFORE[p][0]
            chg = nf / b_nf
            mark = "ดีขึ้น" if chg < 0.8 else ("แย่ลง" if chg > 1.25 else "เท่าเดิม")
            cmp_txt += f" | ก่อนแก้ {b_nf:.1f} ({mark} {chg:.2f}x)"
        print(f"g{p:<7}{nf:>10.1f}{cr:>8.1f}{mx:>10.1f}{pk:>11.1f}   {cmp_txt}")

    nfs = {p: np.mean(acc[p]["nf"]) for p in pins if acc[p]["nf"]}
    best, worst = min(nfs, key=nfs.get), max(nfs, key=nfs.get)
    print(f"\nช่องเงียบสุด g{best} ({nfs[best]:.1f}) · ดังสุด g{worst} ({nfs[worst]:.1f}) "
          f"· ต่างกัน {nfs[worst]/nfs[best]:.1f} เท่า")
    print("เป้าหมาย: ทุกช่องพื้นเสียงใกล้เคียงกัน (ต่างกัน < 1.5 เท่า) จึงจะเชื่อยอดของทุกช่องได้")


if __name__ == "__main__":
    main()
