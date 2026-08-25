"""อ่านแรงดันไฟตรง (DC) ที่ขาหนึ่งของ ESP32 — ไว้ใช้เป็นมัลติมิเตอร์ผ่านซีเรียล

ใช้เทสออปแอมป์: ต่อเอาต์พุต (ขา 8) -> GPIO36 (SP) แล้วรันตัวนี้
ไม่ยิง TX (fire=False) จึงเป็นการอ่านแรงดันนิ่ง ๆ ล้วน

    python tools/read_dc.py --port COM6 --pin 36
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
sys.path.insert(0, HERE)


def _repeat(us, pins, names, a):
    """วัดซ้ำหลายรอบโดยเปิดพอร์ตครั้งเดียว — ใช้จับจังหวะที่มีเสียงเข้าหัวรับ"""
    hdr = "  ".join(f"{(names[i] if i < len(names) else f'GPIO{p}'):>18}"
                    for i, p in enumerate(pins))
    print(f"\n  รอบ  {hdr}")
    for k in range(a.repeat):
        cols = []
        for i in range(len(pins)):
            dc, pp = [], []
            for _ in range(a.pings):
                fr = us.ping(fire=False)
                if fr is None:
                    continue
                v = fr["counts"][i].astype(np.float64) / 4095.0 * 3.3
                dc.append(v.mean())
                pp.append(float(np.percentile(v, 99.5) - np.percentile(v, 0.5)) * 1000)
            if not dc:
                cols.append(f"{'--':>18}")
                continue
            cols.append(f"{np.mean(dc):8.3f}V {np.mean(pp):7.1f}mVpp")
        print(f"  {k+1:>3}  " + "  ".join(cols), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--pins", default="36,39",
                    help="GPIO สองขาที่จะอ่านพร้อมกัน (36=SP · 39=SN)")
    ap.add_argument("--names", default="",
                    help="ชื่อจุดวัด คั่นด้วยจุลภาค เช่น 'ขาบน Rv1,จุดต่อ Rv1-Rv2'")
    ap.add_argument("--pings", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1,
                    help="วัดซ้ำกี่รอบ (เปิดพอร์ตครั้งเดียว) — ใช้ดูค่าตอนมีเสียง/ไม่มีเสียง")
    a = ap.parse_args()

    from ultrasonic import Ultrasonic
    us = Ultrasonic(port=a.port, pins=a.pins)
    pins = us.pins
    names = [s.strip() for s in a.names.split(",")] if a.names else []
    if a.repeat > 1:
        _repeat(us, pins, names, a)
        us.close()
        return
    acc = {i: {"dc": [], "pp": []} for i in range(len(pins))}
    try:
        for _ in range(a.pings):
            p = us.ping(fire=False)
            if p is None:
                continue
            for i in range(len(pins)):
                v = p["counts"][i].astype(np.float64) / 4095.0 * 3.3
                acc[i]["dc"].append(v.mean())
                # Vpp ตัดหาง 0.5% กันสไปก์เดี่ยว ๆ ทำให้ตัวเลขเว่อร์เกินจริง
                acc[i]["pp"].append(
                    float(np.percentile(v, 99.5) - np.percentile(v, 0.5)) * 1000)
    finally:
        us.close()

    if not acc[0]["dc"]:
        sys.exit("อ่านไม่ได้ — เช็คพอร์ต/สาย")
    print()
    for i, pin in enumerate(pins):
        v = float(np.mean(acc[i]["dc"]))
        pp = float(np.mean(acc[i]["pp"]))
        nm = names[i] if i < len(names) else f"GPIO{pin}"
        if v < 0.25:
            tag = "≈0V (ลง GND หรือขาลอย/ไม่มีไฟมา)"
        elif v > 2.9:
            tag = "≈3.3V (ติดไฟบวก หรือชนเพดาน ADC)"
        elif 1.4 <= v <= 1.9:
            tag = "≈1.65V (ครึ่งไฟเลี้ยง — ค่าที่ต้องการ)"
        else:
            tag = "ค่ากลาง ๆ"
        print(f"  GPIO{pin:<3} {nm:<22} DC {v:5.3f} V · AC {pp:6.1f} mVpp   {tag}")
    print(f"\n(เฉลี่ยจาก {len(acc[0]['dc'])} ครั้ง · ไม่ยิง TX)")


if __name__ == "__main__":
    main()
