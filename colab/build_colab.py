"""สร้างชุดสำหรับเทรนบน Google Colab (GPU T4) — โค้ด + ข้อมูล r4 + โน้ตบุ๊ก

    python colab/build_colab.py

ได้สองไฟล์ในโฟลเดอร์ colab/
  ultrasonic_r4_colab.zip    โค้ดเทรน + แคชข้อมูล r4 (อัปโหลดไป Google Drive)
  ultrasonic_r4_colab.ipynb  โน้ตบุ๊ก (เปิดใน Colab: File > Upload notebook)

**ไม่แก้ไฟล์ใน car/ เลย** ตัว train_r4.py ในชุดนี้คือสำเนาที่เติมการส่งข้อมูลขึ้น GPU
ตอนแพ็ก เพื่อไม่กระทบคิวที่กำลังรันบนเครื่องอยู่ ค่าอื่นทุกอย่างเหมือนต้นฉบับ
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAR = ROOT / "car"
OUTDIR = ROOT / "colab"
ZIP = OUTDIR / "ultrasonic_r4_colab.zip"
NB = OUTDIR / "ultrasonic_r4_colab.ipynb"
CODE = ["mapdata2.py", "rig2.py", "model2.py", "model3.py", "model4.py", "train_map2.py",
        "train_map3.py", "baselines_r4.py", "r4_report.py"]
CACHE = CAR / "data" / "_map2_cache_r4_40x30"

# (ข้อความเดิม, ข้อความใหม่) · ต้องเจอข้อความเดิมครั้งเดียวพอดี ไม่งั้นหยุด
GPU_PATCH = [
    ('    ap.add_argument("--threads", type=int, default=6)\n',
     '    ap.add_argument("--threads", type=int, default=6)\n'
     '    ap.add_argument("--device", default="auto", help="auto = ใช้ GPU ถ้ามี")\n'),
    ('    torch.set_num_threads(a.threads)\n',
     '    torch.set_num_threads(a.threads)\n'
     '    dev = torch.device(a.device if a.device != "auto" else\n'
     '                       ("cuda" if torch.cuda.is_available() else "cpu"))\n'
     '    torch.backends.cudnn.benchmark = True\n'
     '    print(f"  อุปกรณ์: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))\n'),
    ('    net = make_net(torch, nn, a, fd)\n', '    net = make_net(torch, nn, a, fd).to(dev)\n'),
    ('            ol, dl = net(torch.from_numpy(w).float(),\n'
     '                         None if cond is None else torch.from_numpy(cond))\n'
     '            loss = loss_fn(a.loss, ol, dl, torch.from_numpy(occ), torch.from_numpy(z), torch, F)\n',
     '            ol, dl = net(torch.from_numpy(w).float().to(dev),\n'
     '                         None if cond is None else torch.from_numpy(cond).to(dev))\n'
     '            loss = loss_fn(a.loss, ol, dl, torch.from_numpy(occ).to(dev),\n'
     '                           torch.from_numpy(z).to(dev), torch, F)\n'),
    ('    net.eval()\n    P = np.zeros((len(ends), GH, GW), np.float32)\n',
     '    net.eval()\n    dev = next(net.parameters()).device\n'
     '    P = np.zeros((len(ends), GH, GW), np.float32)\n'),
    ('                ol, dl = net(torch.from_numpy(w).float())\n'
     '                P[i:i + bs] = torch.sigmoid(ol).numpy()\n'
     '                D[i:i + bs] = (NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(dl)).numpy()\n',
     '                ol, dl = net(torch.from_numpy(w).float().to(dev))\n'
     '                P[i:i + bs] = torch.sigmoid(ol).cpu().numpy()\n'
     '                D[i:i + bs] = (NEAR_CM + (FAR_CM - NEAR_CM) * torch.sigmoid(dl)).cpu().numpy()\n'),
    ('                cond = torch.from_numpy(np.eye(fd.K, dtype=np.float32)[[k] * len(e)])\n'
     '                ol, dl = net(torch.from_numpy(w[:, :, :, i0:i1]).float(), cond)\n'
     '                p = torch.sigmoid(ol).numpy()\n'
     '                d = lo + (hi - lo) * torch.sigmoid(dl).numpy()\n',
     '                cond = torch.from_numpy(np.eye(fd.K, dtype=np.float32)[[k] * len(e)]).to(dev)\n'
     '                ol, dl = net(torch.from_numpy(w[:, :, :, i0:i1]).float().to(dev), cond)\n'
     '                p = torch.sigmoid(ol).cpu().numpy()\n'
     '                d = lo + (hi - lo) * torch.sigmoid(dl).cpu().numpy()\n'),
]


def patched_train():
    s = (CAR / "train_r4.py").read_text(encoding="utf-8")
    for old, new in GPU_PATCH:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"แพตช์ GPU หาข้อความเดิมเจอ {n} ครั้ง (ต้อง 1): {old[:60]!r}")
        s = s.replace(old, new)
    return s


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook():
    cells = [
        md("""
# ultrasonic รอบสี่ — เทรนบน Colab (GPU)

**ก่อนเริ่ม**
1. เมนู **Runtime → Change runtime type → T4 GPU** แล้วกด Save
2. อัปโหลด `ultrasonic_r4_colab.zip` ไปไว้ใน Google Drive ที่โฟลเดอร์ **`MyDrive/ultrasonic_r4/`**
3. รันทีละเซลล์จากบนลงล่าง (Shift+Enter)

ผลทุกการทดลองเก็บลง Drive (`MyDrive/ultrasonic_r4/runs/`) ถ้าเซสชันหลุด รันเซลล์ใหม่ได้เลย
การทดลองที่เสร็จแล้วจะถูกข้าม · **ปล่อยแท็บนี้เปิดไว้** Colab ฟรีตัดเซสชันถ้าไม่มีความเคลื่อนไหวนาน
"""),
        md("## 1. ตรวจว่าได้ GPU"),
        code("""
import torch
print("torch", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("ไม่มี GPU — ไปที่ Runtime > Change runtime type > T4 GPU แล้วรันเซลล์นี้ใหม่")
"""),
        md("## 2. เชื่อม Google Drive แล้วแตกไฟล์"),
        code("""
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, zipfile
DRIVE_DIR = '/content/drive/MyDrive/ultrasonic_r4'
ZIP = f'{DRIVE_DIR}/ultrasonic_r4_colab.zip'
RUNS = f'{DRIVE_DIR}/runs'
WORK = '/content/r4'
assert os.path.exists(ZIP), f'ไม่พบ {ZIP} — อัปโหลด zip ไปที่โฟลเดอร์นี้ใน Drive ก่อน'
os.makedirs(RUNS, exist_ok=True)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
zipfile.ZipFile(ZIP).extractall(WORK)
# ผลเขียนตรงลง Drive ผ่านลิงก์ จะได้ไม่หายตอนเซสชันหลุด
os.symlink(RUNS, f'{WORK}/car/data/r4_runs')
os.chdir(f'{WORK}/car')
os.environ['MAP2_ROUND'] = 'r4'
os.environ['PYTHONIOENCODING'] = 'utf-8'
print('พร้อม:', sorted(f for f in os.listdir('.') if f.endswith('.py')))
"""),
        md("""
## 3. ทดสอบสั้น ๆ (~1-3 นาที)

เทรน 1 รอบด้วยข้อมูลช่วงเดียว ดูว่าทุกอย่างทำงานและ **หนึ่งรอบใช้กี่วินาที**
(บนเครื่องที่บ้าน รอบเต็มใช้ ~110 วินาที · ส่งตัวเลขนี้ให้ Claude ประเมินเวลาได้)
"""),
        code("""
!python train_r4.py --name colab_smoke --train 1 --epochs 1 --threads 2
for ext in ('json', 'pt', 'log'):
    p = f'{RUNS}/colab_smoke.{ext}'
    if os.path.exists(p):
        os.remove(p)
"""),
        md("""
## 4. รันคิวการทดลอง (ทั้งหมด ~1.5-2 ชั่วโมงบน T4)

ชื่อขึ้นต้นด้วย `gpu_` เพื่อไม่ชนกับผลที่รันบนเครื่อง · ค่าตั้งต้น = สูตรรอบ 3
เปลี่ยน **ทีละอย่าง** จากค่าตั้งต้น (ดูเหตุผลใน `car/ROUND4.md`)

- `gpu_base_s0/1/2` ค่าตั้งต้น 3 ค่าสุ่ม = ความต่างที่เกิดจากความบังเอิญ · s0 เทียบกับบนเครื่องด้วย
- `gpu_loss_*` วิธีวัดความผิด 5 แบบ · `gpu_act_*` ฟังก์ชันกระตุ้น 6 แบบ
- `gpu_norm_log` บีบคลื่นแบบลอการิทึม · `gpu_slice5` ไอเดียตัดช่วงเวลา 5 ms เลื่อนทีละ 1 ms
- `gpu_lc_*` เส้นโค้งการเรียนรู้ (เทรน 1, 2, 3 ช่วง) · `gpu_long40_base` เทรน 40 รอบ
"""),
        code("""
import subprocess, sys, time
QUEUE = [
    ("gpu_base_s0", []),
    ("gpu_base_s1", ["--seed", "1"]),
    ("gpu_base_s2", ["--seed", "2"]),
    ("gpu_loss_bdh", ["--loss", "bdh"]),
    ("gpu_loss_mse", ["--loss", "mse"]),
    ("gpu_loss_bce_l1", ["--loss", "bce_l1"]),
    ("gpu_loss_dice_huber", ["--loss", "dice_huber"]),
    ("gpu_loss_focal", ["--loss", "focal"]),
    ("gpu_act_leaky", ["--act", "leaky"]),
    ("gpu_act_elu", ["--act", "elu"]),
    ("gpu_act_gelu", ["--act", "gelu"]),
    ("gpu_act_silu", ["--act", "silu"]),
    ("gpu_act_mish", ["--act", "mish"]),
    ("gpu_act_tanh", ["--act", "tanh"]),
    ("gpu_norm_log", ["--norm", "log"]),
    ("gpu_slice5", ["--slice-ms", "5", "--slice-step-ms", "1"]),
    ("gpu_lc_s1", ["--train", "1"]),
    ("gpu_lc_s12", ["--train", "1,2"]),
    ("gpu_lc_s123", ["--train", "1,2,3"]),
    ("gpu_long40_base", ["--epochs", "40"]),
]
EPOCHS = "12"

for name, extra in QUEUE:
    if os.path.exists(f'{RUNS}/{name}.json'):
        print('ข้าม (เสร็จแล้ว):', name)
        continue
    cmd = [sys.executable, 'train_r4.py', '--name', name, '--epochs', EPOCHS, '--threads', '2'] + extra
    print('=' * 70, '\\nเริ่ม', name, ' '.join(extra), flush=True)
    t = time.time()
    with open(f'{RUNS}/{name}.log', 'w', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='', flush=True)
            log.write(line)
        p.wait()
    print(f"{'เสร็จ' if p.returncode == 0 else 'ล้ม'} {name} · {(time.time() - t) / 60:.0f} นาที")
"""),
        md("## 5. สรุปผลทั้งหมด"),
        code("!python r4_report.py"),
        md("""
## 6. ส่งผลกลับ

ไฟล์ผลอยู่ใน Google Drive ที่ `MyDrive/ultrasonic_r4/runs/` (`*.json` คือผล · `*.pt` คือโมเดล)
ดาวน์โหลด `*.json` ทั้งหมดไปวางที่ `car/data/r4_runs/` ในเครื่อง แล้วบอก Claude
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    OUTDIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in CODE:
            z.write(CAR / f, f"car/{f}")
        z.writestr("car/train_r4.py", patched_train())
        for f in sorted(CACHE.iterdir()):
            z.write(f, f"car/data/{CACHE.name}/{f.name}")
    NB.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{ZIP.name} {ZIP.stat().st_size / 1e6:.0f} MB · {NB.name}")


if __name__ == "__main__":
    main()
