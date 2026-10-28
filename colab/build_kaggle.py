"""โน้ตบุ๊กสำหรับ Kaggle — ใช้ zip ชุดเดียวกับ Colab (สร้างด้วย build_colab.py ก่อน)

    python colab/build_kaggle.py      ได้ colab/ultrasonic_<รอบ>_kaggle.ipynb

ต่างจาก Colab สามอย่าง
  - ข้อมูลเข้าทาง Kaggle Dataset (/kaggle/input/...) ไม่ใช่ Google Drive
    Kaggle มักแตก zip ให้เองตอนอัปโหลด โน้ตบุ๊กจึงรับได้ทั้งแบบ zip และแบบแตกแล้ว
  - ผลเขียนที่ /kaggle/working/runs แล้วรวมเป็น runs.zip ตอนจบ (ดาวน์โหลดจากแท็บ Output)
  - ควรรันแบบ "Save Version → Save & Run All" รันเบื้องหลังได้ถึง 12 ชั่วโมง ปิดเบราว์เซอร์ได้
"""
import json

from build_colab import OUTDIR, QUEUE, REPORT_GLOB, ROUND, SPLIT, STAGE, code, md

BASE = "r5dev"          # แคชที่อยู่บน Kaggle แล้ว (ขั้น final เติมห้อง D ทีหลัง)


def notebook():
    q = "[\n" + "".join(f"    ({n!r}, {e!r}),\n" for n, e in QUEUE) + "]"
    cells = [
        md(f"""
# ultrasonic รอบห้า ({ROUND}) บน Kaggle

**ตั้งค่าก่อนรัน (แผงขวา → Session options)**
- Accelerator: **GPU T4 x2** (ใช้จริงแค่ตัวเดียว)
- Input: เพิ่ม Dataset ที่อัปโหลด `ultrasonic_{ROUND}_data.zip` กับ `ultrasonic_code.zip`

การแบ่ง: เทรนห้องเดิม (hw2) + ห้อง B · เลือกด้วยห้อง C ทั้งห้อง · ห้อง D ไม่อยู่ในชุดนี้
"""),
        code("""
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "ไม่มี — Session options > Accelerator > GPU T4 x2")
"""),
        md("## หาไฟล์จาก Dataset แล้วเตรียมโฟลเดอร์ทำงาน"),
        code(f"""
import os, glob, shutil, zipfile
WORK = '/kaggle/working/r5'
RUNS = '/kaggle/working/runs'
os.makedirs(RUNS, exist_ok=True)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
found = []
# แบบ zip (ถ้า Kaggle ไม่ได้แตกให้)
for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
    zipfile.ZipFile(z).extractall(WORK); found.append(z)
# แบบที่ Kaggle แตกให้แล้ว: หาโฟลเดอร์ car/ ที่มี train_r4.py และแคช
for p in glob.glob('/kaggle/input/**/car', recursive=True):
    shutil.copytree(p, f'{{WORK}}/car', dirs_exist_ok=True); found.append(p)
car = f'{{WORK}}/car'
cache = f'{{car}}/data/_map2_cache_{BASE}_40x30'
assert os.path.exists(f'{{car}}/train_r4.py'), f'ไม่พบโค้ด · เจอ {{found}}'
assert os.path.exists(f'{{cache}}/meta.json'), f'ไม่พบแคชข้อมูล · เจอ {{found}}'
os.makedirs(f'{{car}}/data', exist_ok=True)
if os.path.lexists(f'{{car}}/data/r4_runs'):
    os.remove(f'{{car}}/data/r4_runs')
os.symlink(RUNS, f'{{car}}/data/r4_runs')
os.chdir(car)
os.environ['MAP2_ROUND'] = '{BASE}'
os.environ['PYTHONIOENCODING'] = 'utf-8'
print('พร้อม · ข้อมูลจาก', found)
"""),
    ] + ([] if STAGE == "dev" else [
        md("## ประกอบข้อมูลห้อง D จากชิ้นที่อัปโหลด แล้วรวมเป็นแคช r5"),
        code(f"""
import hashlib, subprocess, sys
parts = sorted(glob.glob('/kaggle/input/**/roomD.zip.part*', recursive=True))
want = open(glob.glob('/kaggle/input/**/roomD.sha256', recursive=True)[0]).read().strip()
assert len(parts) == 6, f'ต้องมี 6 ชิ้น เจอ {{len(parts)}}: {{parts}}'
blob = b''.join(open(p, 'rb').read() for p in parts)
got = hashlib.sha256(blob).hexdigest()
assert got == want, f'ไฟล์ไม่ครบหรือเพี้ยน sha256 {{got[:16]}} != {{want[:16]}}'
open('/kaggle/working/roomD.zip', 'wb').write(blob)
zipfile.ZipFile('/kaggle/working/roomD.zip').extractall(WORK)
r = subprocess.run([sys.executable, 'merge_cache.py', 'r5dev', 'r5d'], env=dict(os.environ, MAP2_ROUND='r5'),
                   capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0
os.environ['MAP2_ROUND'] = 'r5'
print('sha256 ตรง · แคช r5 พร้อม')
"""),
    ]) + [
        md("## ทดสอบสั้น ๆ 1 รอบ (~1 นาที) — ดูเวลาต่อรอบ"),
        code(f"""
!python train_r4.py --name smoke --epochs 1 --threads 2 --train roomB_s1 --val roomC_s3 --test ""
for ext in ('json', 'pt', 'log'):
    p = f'{{RUNS}}/smoke.{{ext}}'
    if os.path.exists(p): os.remove(p)
"""),
        md("## รันคิว (~1.5 ชั่วโมงบน T4)"),
        code(f"""
import subprocess, sys, time
SPLIT = {SPLIT!r}
QUEUE = {q}
for name, extra in QUEUE:
    if os.path.exists(f'{{RUNS}}/{{name}}.json'):
        print('ข้าม (เสร็จแล้ว):', name); continue
    cmd = [sys.executable, 'train_r4.py', '--name', name, '--epochs', '12', '--threads', '2'] + SPLIT + extra
    print('=' * 70, '\\nเริ่ม', name, ' '.join(extra), flush=True)
    t = time.time()
    with open(f'{{RUNS}}/{{name}}.log', 'w', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='', flush=True); log.write(line)
        p.wait()
    print(f"{{'เสร็จ' if p.returncode == 0 else 'ล้ม'}} {{name}} · {{(time.time() - t) / 60:.0f}} นาที")
"""),
        md("## สรุปผล และรวมไฟล์ผลเป็น runs_json.zip (ดาวน์โหลดจากแท็บ Output)"),
        code(f"!python r4_report.py '{REPORT_GLOB}'\n" + """
with zipfile.ZipFile('/kaggle/working/runs_json.zip', 'w') as z:
    for f in sorted(glob.glob(f'{RUNS}/*.json')) + sorted(glob.glob(f'{RUNS}/*.log')):
        z.write(f, os.path.basename(f))
print('รวมแล้ว', len(glob.glob(f'{RUNS}/*.json')), 'ผล -> /kaggle/working/runs_json.zip')
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    nb = OUTDIR / f"ultrasonic_{ROUND}_{STAGE}_kaggle.ipynb"
    nb.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(nb)


if __name__ == "__main__":
    main()
