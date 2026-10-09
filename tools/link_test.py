"""วัดสายซีเรียลของ ESP32-S3 — ความเร็วจริง และไบต์เพี้ยนหรือไม่

    python tools/link_test.py --port COM4                  ไล่ baud ผ่านชิปแปลง
    python tools/link_test.py --native COM9                วัด USB ของตัวชิป (ไม่มี baud)

ทุก baud เริ่มใหม่หมด (รีเซ็ตบอร์ด เปิดที่ 115200 สั่งเปลี่ยน baud) เพื่อไม่ให้ baud ที่พัง
ลากรอบถัดไปพังตาม · เลขที่รายงานคือเลขฝั่ง PC เท่านั้น ฝั่งบอร์ดนับตอนเขียนเข้าคิว
ซึ่งยังไม่ได้แปลว่าถึงปลายทาง

ผลต่อ baud: เร็วจริงกี่ MB/s · ไบต์เพี้ยน · แพ็กเก็ตหาย · เทียบเพดานทฤษฎี (baud/10)
"""
import argparse
import struct
import sys
import time

import numpy as np
import serial

PKT, HDR = 1024, 8
MAGIC = b"\xA5\x5A\xC3\x3C"
TRAIL = b"\xDE\xAD\xBE\xEF"
FRAME_AD7606 = 41488          # 8 ช่อง x 2593 ตัวอย่าง x 2 ไบต์ ที่ 200 ซม.
WINDOW_MS = 12.9              # เวลาเก็บของระยะ 200 ซม.
# เรียงจากน้อยไปมาก และหยุดที่ 4M: วัดแล้วที่ 5M ชิปแปลง CH343 ไม่ส่งข้อมูลกลับมาเลย
# แล้วไดรเวอร์ Windows ค้างตอนปิดพอร์ต โปรเซสฆ่าไม่ตาย ต้องถอดสาย USB ถึงปล่อยพอร์ต
# (ใส่ --bauds 5000000 เองได้ถ้าอยากลองซ้ำ)
BAUDS = [921600, 2000000, 3000000, 4000000]


def reset_open(port, baud=115200):
    s = serial.Serial(port, baud, timeout=0.2, write_timeout=2)
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.1)
    s.setRTS(False)
    time.sleep(1.6)
    s.reset_input_buffer()
    return s


def parse(buf):
    """นับแพ็กเก็ตถูก/เพี้ยน/ไบต์ขยะ · คืน (good, bad, junk, sent)"""
    good = bad = junk = 0
    sent = None
    pos = buf.find(MAGIC)
    ar = np.arange(HDR, PKT, dtype=np.int64)
    while 0 <= pos < len(buf):
        if buf[pos:pos + 4] == TRAIL:
            if pos + 8 <= len(buf):
                sent = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
            break
        if pos + PKT > len(buf):
            break
        seq = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        want = ((seq * 7 + ar) & 0xFF).astype(np.uint8).tobytes()
        if buf[pos + HDR:pos + PKT] == want:
            good += 1
            nxt = pos + PKT
        else:
            bad += 1
            nxt = pos + 4
        cands = [i for i in (buf.find(MAGIC, nxt), buf.find(TRAIL, nxt)) if i >= 0]
        nxt_hit = min(cands) if cands else len(buf)
        if good and nxt_hit > nxt and bad == 0:
            junk += nxt_hit - nxt
        pos = nxt_hit if cands else -1
    return good, bad, junk, sent


def stream(s, ms):
    s.reset_input_buffer()
    s.write(f"g {ms}\n".encode())
    buf = bytearray()
    t_first = t_last = None
    t_end = time.time() + ms / 1000 + 6
    while time.time() < t_end:
        d = s.read(max(1, s.in_waiting))
        if not d:
            if t_first and time.time() - t_last > 1.5:
                break
            continue
        now = time.time()
        if t_first is None:
            t_first = now
        t_last = now
        buf += d
        if len(buf) >= 8 and buf[-8:-4] == TRAIL:
            break
    return bytes(buf), (t_last - t_first) if t_first else 0.0


def report(label, rate_cap, buf, el):
    if not buf or el <= 0:
        print(f"{label:>10}  ไม่ได้ข้อมูลเลย", flush=True)
        return None
    good, bad, junk, sent = parse(buf)
    mbps = good * PKT / el / 1e6
    loss = (sent - good - bad) if sent is not None else None
    cap = f"{rate_cap / 1e6:5.2f}" if rate_cap else "    -"
    eff = f"{mbps / (rate_cap / 1e6) * 100:5.0f}%" if rate_cap else "     -"
    print(f"{label:>10}  {mbps:6.3f} MB/s  เพดานทฤษฎี {cap}  ได้ {eff}  "
          f"ถูก {good:5d}  เพี้ยน {bad:4d}  ขยะ {junk:6d}  หาย "
          f"{'?' if loss is None else loss}", flush=True)
    return mbps, bad, junk, loss


def fps_line(mbps):
    tx = FRAME_AD7606 / (mbps * 1e6)
    return f"    -> เฟรม AD7606 41,488 ไบต์ ส่ง {tx * 1e3:5.1f} ms -> {1 / (tx + WINDOW_MS / 1e3):5.1f} fps"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port")
    ap.add_argument("--native")
    ap.add_argument("--ms", type=int, default=3000)
    ap.add_argument("--bauds", default=",".join(str(b) for b in BAUDS))
    a = ap.parse_args()
    if not a.port and not a.native:
        sys.exit("ต้องระบุ --port หรือ --native")

    if a.port:
        print(f"UART ผ่านชิปแปลง {a.port} · ส่งรัว {a.ms} ms ต่อ baud\n")
        for baud in (int(b) for b in a.bauds.split(",")):
            s = None
            print(f"{baud:>10}  เริ่ม ...", flush=True)   # ถ้าค้าง จะรู้ว่าค้างที่ baud ไหน
            try:
                s = reset_open(a.port)
                s.write(f"b {baud}\n".encode())
                t0, got = time.time(), b""
                while time.time() - t0 < 2 and b"#BAUD" not in got:
                    got += s.read(64)
                if b"#BAUD" not in got:
                    print(f"{baud:>10}  บอร์ดไม่ตอบคำสั่งเปลี่ยน baud")
                    continue
                time.sleep(0.15)
                try:
                    s.baudrate = baud
                except (serial.SerialException, ValueError, OSError) as e:
                    print(f"{baud:>10}  Windows ไม่ยอมตั้ง baud นี้ ({e})")
                    continue
                time.sleep(0.2)
                buf, el = stream(s, a.ms)
                r = report(str(baud), baud / 10, buf, el)
                if r and r[1] == 0 and r[2] == 0 and not r[3]:
                    print(fps_line(r[0]))
            finally:
                if s:
                    s.close()
    if a.native:
        print(f"\nUSB ของตัวชิป {a.native} · ส่งรัว {a.ms} ms\n")
        s = serial.Serial(a.native, 115200, timeout=0.2)
        time.sleep(0.5)
        buf, el = stream(s, a.ms)
        r = report("native", None, buf, el)
        if r and r[1] == 0 and r[2] == 0 and not r[3]:
            print(fps_line(r[0]))
        s.close()


if __name__ == "__main__":
    main()
