"""ตรวจสถานะฮาร์ดแวร์ระหว่างเก็บข้อมูล — ให้รู้ทันทีถ้าระบบเปลี่ยนกลางทาง

ทำไมต้องมี
  วัดจริงเมื่อ ต.ค. 2026: เอคโค่ของชุดเดียวกันต่างกันราวเท่าตัวระหว่างสองวัน และนอยส์เคยพุ่ง
  3-6 เท่าครั้งเดียวโดยไม่รู้สาเหตุ (สงสัยสายชาร์จโน้ตบุ๊ก) ถ้าชุดข้อมูลหนึ่งผสมสภาพฮาร์ดแวร์
  สองแบบเข้าด้วยกัน โมเดลจะเรียนความต่างของฮาร์ดแวร์แทนความต่างของฉาก และเราจะไม่รู้เลย

สิ่งที่ตรวจ (ใช้ตัวเลขที่ **ไม่ขึ้นกับฉาก** เป็นหลัก)
  PULSE   พัลส์ตรงจาก TX ถึงตัวรับ (-15..13 ซม. บนแกนระยะ) เป็นตัวบอกกำลังส่ง+เกนรับ
          ฉากเปลี่ยน คนขยับ ก็ไม่ทำให้มันเปลี่ยน ถ้ามันเปลี่ยนแปลว่าฮาร์ดแวร์เปลี่ยน
  NOISE   นอยส์ของวงจรรับตอนไม่ยิง (คำสั่ง q) รายช่อง
  CLIP    สัดส่วนตัวอย่างที่ชนเพดาน ADC ในช่วงเอคโค่ที่เราสนใจ (40-200 ซม.)
          ตัดแบนแล้วแอมพลิจูดจริงหาย ขยายภาพก็ไม่เห็น
  DC      ระดับพักรายช่อง

ใช้ 3 จุด: ก่อนอัด · หลังแต่ละช่วง · หลังอัดครบ แล้วเทียบกัน
"""
import json
from pathlib import Path

import numpy as np

C, T0_US = 343.0, 1220.0
PULSE_CM, ECHO_CM = (-15.0, 13.0), (40.0, 200.0)

# นอยส์ตอนไม่ยิงรายช่อง (RMS นับ ADC): ปกติวัดได้ 16-45 (16 ต.ค. 2026)
# ผิดปกติที่เคยเจอ 122-127 ในวันที่ 15 ต.ค. จึงตั้งเตือนที่ 80 (ครึ่งทางในสเกลลอการิทึม)
NOISE_WARN = 80.0
# เปลี่ยนเกิน 25% ถือว่าฮาร์ดแวร์เปลี่ยน · เกณฑ์นี้มาจากความต่างของการวัดซ้ำที่ 6 นาที (~8%)
# บวกเผื่อ ไม่ใช่ค่าที่พิสูจน์แล้ว ปรับได้
DRIFT_WARN = 0.25


def _mask(n, rate, lo, hi):
    dist = (np.arange(n) / rate - T0_US * 1e-6) * C / 2 * 100
    return (dist >= lo) & (dist < hi)


def stats_fired(frames, rate):
    """frames (m, 4, n) ของปิงที่ยิง -> พัลส์ตรง · การชนเพดานช่วงเอคโค่ · DC"""
    f = np.asarray(frames)
    x = f.astype(np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    mp = _mask(f.shape[-1], rate, *PULSE_CM)
    me = _mask(f.shape[-1], rate, *ECHO_CM)
    clip = (f[..., me] <= 2) | (f[..., me] >= 4093)
    return {"n": int(f.shape[0]),
            "pulse_rms": float(np.sqrt((x[..., mp] ** 2).mean())),
            "clip_echo_pct": 100.0 * float(clip.mean()),
            "dc": [float(v) for v in np.median(f, axis=(0, 2))]}


def stats_quiet(frames):
    """frames (m, 4, n) ของปิงที่ไม่ยิง -> นอยส์ RMS รายช่อง"""
    x = np.asarray(frames).astype(np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    return {"n": int(x.shape[0]),
            "noise_rms": [float(v) for v in np.sqrt((x ** 2).mean(axis=(0, 2)))]}


def preflight(us, n=20):
    """ยิง n ปิง แล้วไม่ยิง n ปิง ใช้แค่เซ็นเซอร์ (ไม่ต้องมีกล้อง) ราว 2-5 วินาที"""
    fired, quiet = [], []
    for fire, bucket in ((True, fired), (False, quiet)):
        guard = 0
        while len(bucket) < n and guard < n * 4:
            guard += 1
            p = us.ping(fire=fire, retries=1)
            if p is not None:
                bucket.append(p["counts"])
    if len(fired) < 3 or len(quiet) < 3:
        return {"error": f"อ่านได้ยิง {len(fired)} / ไม่ยิง {len(quiet)} ปิง น้อยเกินไป"}
    return {"fired": stats_fired(np.stack(fired), us.rate),
            "quiet": stats_quiet(np.stack(quiet)), "rate": float(us.rate)}


def report(label, res, ref=None, ref_name=""):
    """พิมพ์สรุปและเตือน · ref = ผลก่อนหน้า (preflight ของครั้งก่อน หรือก่อนอัดครั้งนี้)"""
    if "error" in res:
        print(f"  [{label}] ตรวจไม่ได้: {res['error']}")
        return []
    warns = []
    nz, p = res["quiet"]["noise_rms"], res["fired"]["pulse_rms"]
    print(f"  [{label}] PULSE {p:6.0f} · นอยส์ TL/TR/BL/BR "
          f"{' '.join('%4.0f' % v for v in nz)} · ชนเพดานช่วงเอคโค่ "
          f"{res['fired']['clip_echo_pct']:.1f}% · DC {' '.join('%4.0f' % v for v in res['fired']['dc'])}")
    if max(nz) > NOISE_WARN:
        warns.append(f"นอยส์สูงผิดปกติ (สูงสุด {max(nz):.0f} > {NOISE_WARN:.0f}) "
                     f"— เช็คสายชาร์จ สายที่หลวม และของไฟฟ้าใกล้แผ่น ก่อนอัด")
    if ref and "fired" in ref:
        r = p / max(ref["fired"]["pulse_rms"], 1e-9)
        print(f"            เทียบ{ref_name or 'ครั้งก่อน'}: PULSE x{r:.2f}")
        if abs(r - 1.0) > DRIFT_WARN:
            warns.append(f"PULSE เปลี่ยน x{r:.2f} จาก{ref_name or 'ครั้งก่อน'} "
                         f"— กำลังส่งหรือเกนเปลี่ยน ข้อมูลสองชุดนี้ไม่ควรรวมกัน")
    for w in warns:
        print(f"  !! {w}")
    return warns


def latest_session(root, exclude_name):
    """ผล preflight ของครั้งล่าสุดที่ไม่ใช่ชุดนี้ · (ชื่อ, ผล) หรือ (None, None)"""
    cands = [p for p in Path(root).glob("*_session.json") if p.name != f"{exclude_name}_session.json"]
    for p in sorted(cands, key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            res = d.get("pre") or d.get("post")
            if res and "fired" in res:
                return d.get("name", p.stem), res
        except (OSError, ValueError):
            continue
    return None, None


def section_stats(out_dir, k=150):
    """สถิติจากไฟล์ที่บันทึกไว้จริงของช่วงหนึ่ง (สุ่มเว้นระยะ k เฟรม)"""
    files = sorted(Path(out_dir).glob("us_*.npz"))
    if not files:
        return {"error": "ไม่มีไฟล์"}
    step = max(1, len(files) // k)
    frames, rate = [], None
    for f in files[::step][:k]:
        z = np.load(f)
        frames.append(z["counts"])
        rate = rate or float(z["rate"])
    st = stats_fired(np.stack(frames), rate)
    st["frames_total"] = len(files)
    return st


def report_section(st, ref):
    if "error" in st:
        print(f"  QA: {st['error']}")
        return
    r = st["pulse_rms"] / max(ref["pulse_rms"], 1e-9)
    print(f"  QA: PULSE {st['pulse_rms']:.0f} (x{r:.2f} ของช่วงแรก) · ชนเพดานช่วงเอคโค่ "
          f"{st['clip_echo_pct']:.1f}% · DC {' '.join('%.0f' % v for v in st['dc'])}")
    if abs(r - 1.0) > DRIFT_WARN:
        print(f"  !! PULSE เปลี่ยน x{r:.2f} ภายในรอบนี้ — ฮาร์ดแวร์เปลี่ยนระหว่างอัด "
              f"(สายหลวม? แผ่นถูกขยับ?) ช่วงนี้อย่าปนกับช่วงอื่นโดยไม่ตรวจ")
