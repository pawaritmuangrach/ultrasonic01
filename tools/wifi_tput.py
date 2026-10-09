"""วัด WiFi throughput ของ ESP32 — สลับเน็ตไปหามัน วัด แล้วต่อกลับให้เอง

ทำไมต้องสลับเน็ต
  ESP32 ปล่อยสัญญาณเป็น AP ของตัวเอง (192.168.4.1) เครื่องนี้มี WiFi ใบเดียว
  จึงอยู่สองที่พร้อมกันไม่ได้ ระหว่างวัดเน็ตจะหลุดราว 40 วินาที
  **สคริปต์ต่อเน็ตเดิมกลับให้เองเสมอ** แม้จะพังกลางทาง (อยู่ใน finally)
  ถ้าพังจริง ๆ Windows ก็ยังต่อเน็ตเดิมกลับอัตโนมัติอยู่ดี เพราะเป็นเครือข่ายที่จำไว้แล้ว

ทำไมต้องวัดฝั่งนี้ ไม่เชื่อเลขจาก ESP32
  ฝั่ง ESP32 นับตอน write() คืนค่า ซึ่งแค่แปลว่าข้อมูลเข้าคิว TCP แล้ว
  ไม่ได้แปลว่าถึงปลายทาง เลขที่จริงคือเลขที่นับตอนรับได้ที่นี่

    python tools/wifi_tput.py

ผลออกที่จอ และเขียนลง tools/_wifi_tput.txt เผื่อ session หลุดตอนสลับเน็ต
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

AP_SSID = "esp32-tput"
AP_PASS = "tput12345"          # รหัสใช้ครั้งเดียวที่ตั้งไว้ในเฟิร์มแวร์ ไม่ใช่ของใคร
ESP_IP = "192.168.4.1"
PORT = 5001
OUT = Path(__file__).resolve().parent / "_wifi_tput.txt"

PROFILE = """<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{s}</name>
  <SSIDConfig><SSID><name>{s}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication>
      <encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType>
      <protected>false</protected><keyMaterial>{p}</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>"""


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def current_ssid():
    out = sh("netsh", "wlan", "show", "interfaces").stdout
    for ln in out.splitlines():
        p = ln.split(":", 1)
        if len(p) == 2 and p[0].strip().lower() == "ssid":
            return p[1].strip()
    return None


def connect(ssid, tries=20):
    sh("netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}")
    for _ in range(tries):
        time.sleep(1.0)
        if current_ssid() == ssid:
            return True
    return False


def reachable(ip, port, secs=20):
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            with socket.create_connection((ip, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.7)
    return False


def measure():
    """รับข้อมูลจนสายปิด · แบ่งช่วงตามที่เฟิร์มแวร์ส่ง (stream ก่อน แล้ว frame)

    ฝั่งนี้ไม่รู้ขอบเฟรม จึงวัดเป็นอัตรารวมกับอัตราต่อวินาที แล้วรายงานทั้งคู่
    อัตราต่อวินาทีสำคัญกว่าค่าเฉลี่ย เพราะ fps ถูกกำหนดโดยช่วงที่ช้าที่สุด
    """
    s = socket.create_connection((ESP_IP, PORT), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(10)
    total, per_sec, t0, mark, chunk = 0, [], time.time(), time.time(), 0
    while True:
        try:
            b = s.recv(65536)
        except socket.timeout:
            break
        if not b:
            break
        total += len(b)
        chunk += len(b)
        if time.time() - mark >= 1.0:
            per_sec.append(chunk / (time.time() - mark) / 1e6)
            chunk, mark = 0, time.time()
    s.close()
    el = time.time() - t0
    return {"bytes": total, "sec": round(el, 2),
            "MBps_avg": round(total / 1e6 / el, 3) if el else 0,
            "MBps_per_sec": [round(v, 3) for v in per_sec]}


def main():
    home = current_ssid()
    print(f"เน็ตตอนนี้: {home}")
    if home is None:
        sys.exit("อ่าน SSID ปัจจุบันไม่ได้ — ต่อ WiFi อยู่หรือเปล่า")
    xml = Path(OUT.parent / "_ap.xml")
    xml.write_text(PROFILE.format(s=AP_SSID, p=AP_PASS), encoding="utf-8")
    res = {"home": home}
    try:
        sh("netsh", "wlan", "add", "profile", f"filename={xml}", "user=current")
        print(f"สลับไป {AP_SSID} ...")
        if not connect(AP_SSID):
            sys.exit(f"ต่อ {AP_SSID} ไม่ได้ — ESP32 เปิด AP อยู่ไหม (สั่ง ap ทางซีเรียล)")
        print(f"ต่อแล้ว · รอ {ESP_IP}:{PORT} ...")
        if not reachable(ESP_IP, PORT):
            sys.exit("ต่อ AP ได้ แต่เรียก ESP32 ไม่ติด")
        print("กำลังวัด ~20 วินาที อย่าเพิ่งปิด ...")
        res.update(measure())
    finally:
        print(f"ต่อ {home} กลับ ...")
        ok = connect(home)
        sh("netsh", "wlan", "delete", "profile", f"name={AP_SSID}")
        xml.unlink(missing_ok=True)
        res["restored"] = ok
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"เน็ตกลับมา: {ok}  ·  ผลอยู่ที่ {OUT}")

    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res.get("MBps_avg"):
        f = 41488 / (res["MBps_avg"] * 1e6)
        print(f"\nเฟรม AD7606 8 ช่อง 41,488 ไบต์ -> ส่ง {f*1e3:.1f} ms -> "
              f"{1/(f+0.0129):.1f} fps (รวมเวลาเก็บ 12.9 ms)")


if __name__ == "__main__":
    main()
