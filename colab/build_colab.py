"""สร้างชุดสำหรับเทรนบน Google Colab (GPU T4)

    python colab/build_colab.py                 รอบ r5dev (เลือกสูตร · ไม่มีห้องสอบ D)

ได้สามไฟล์ในโฟลเดอร์ colab/
  ultrasonic_<รอบ>_data.zip   แคชข้อมูล (ใหญ่ · อัปโหลดไป Drive ครั้งเดียวต่อรอบข้อมูล)
  ultrasonic_code.zip         โค้ดเทรน (เล็ก · อัปโหลดใหม่ทุกครั้งที่โค้ดเปลี่ยน)
  ultrasonic_<รอบ>.ipynb      โน้ตบุ๊ก (เปิดใน Colab: File > Upload notebook)

**r5dev ไม่มีข้อมูลห้อง D อยู่ในชุดเลย** ห้อง D เป็นข้อสอบสุดท้ายที่ประกาศไว้ก่อนเก็บ
การไม่ใส่ไว้เลยทำให้เผลอเลือกสูตรจากห้อง D ไม่ได้แม้ตั้งใจ
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAR = ROOT / "car"
OUTDIR = ROOT / "colab"
CODE = ["mapdata2.py", "rig2.py", "model2.py", "model3.py", "model4.py", "train_map2.py",
        "train_map3.py", "train_r4.py", "baselines_r4.py", "r4_report.py", "merge_cache.py", "person_target.py"]
import os
STAGE = os.environ.get("STAGE", "dev")      # dev = เลือกสูตร (ไม่มีห้อง D) · final = ข้อสอบห้อง D
ROUND = "r5dev" if STAGE == "dev" else "r5"
CACHE = CAR / "data" / f"_map2_cache_{ROUND}_40x30"

# รอบเลือกสูตร: เทรนห้องเดิม (hw2) + ห้อง B · เลือกด้วยห้อง C ทั้งห้อง (ห้องที่ไม่เคยเห็น)
SPLIT = (["--train", "hw2_*,roomB_*", "--val", "roomC_*", "--test", ""] if STAGE == "dev"
         else [])          # ขั้น final: แต่ละรายการกำหนดการแบ่งเอง
F5 = ["--loss", "bce_l1", "--slice-ms", "5"]
SP_C = ["--train", "hw2_*,roomB_*", "--val", "roomC_*", "--test", "roomD_*"]
# final_all: ห้อง C อยู่ในชุดเทรนแล้ว ชุดเลือก (roomC_s3) มีไว้แค่พิมพ์ความคืบหน้า ไม่ได้ใช้เลือกอะไร
SP_ALL = ["--train", "hw2_*,roomB_*,roomC_*", "--val", "roomC_s3", "--test", "roomD_*",
          "--select", "last", "--thr", "0.35"]
QUEUE_FINAL = [
    ("final_s0", SP_C + F5),
    ("final_s1", SP_C + F5 + ["--seed", "1"]),
    ("final_s2", SP_C + F5 + ["--seed", "2"]),
    ("final_base_s0", SP_C),
    ("final_base_s1", SP_C + ["--seed", "1"]),
    ("final_all_s0", SP_ALL + F5),
    ("final_all_s1", SP_ALL + F5 + ["--seed", "1"]),
]
QUEUE_DEV = [
    ("r5_base_s0", []),
    ("r5_base_s1", ["--seed", "1"]),
    ("r5_l1_s0", ["--loss", "bce_l1"]),
    ("r5_l1_s1", ["--loss", "bce_l1", "--seed", "1"]),
    ("r5_mse_s0", ["--loss", "mse"]),
    ("r5_bdh_w2", ["--loss", "bdh"]),
    ("r5_bdh_w4", ["--loss", "bdh", "--w-depth", "4"]),
    ("r5_l1_slice5_s0", ["--loss", "bce_l1", "--slice-ms", "5"]),
    ("r5_l1_slice5_s1", ["--loss", "bce_l1", "--slice-ms", "5", "--seed", "1"]),
    ("r5_l1_slice3", ["--loss", "bce_l1", "--slice-ms", "3"]),
    ("r5_l1_slice8", ["--loss", "bce_l1", "--slice-ms", "8"]),
    ("r5_l1_leaky", ["--loss", "bce_l1", "--act", "leaky"]),
]
# ขั้น person: เฉลยเฉพาะสิ่งที่เคลื่อนที่ (person_target.py) · สูตรและการแบ่งเหมือนขั้น final ทุกอย่าง
PT = ["--target", "person"]
SP_IN = ["--train", "hw2_s1,hw2_s2,hw2_s3,hw2_s4", "--val", "hw2_s5", "--test", "hw2_s6"]
QUEUE_PERSON = [
    ("person_s0", SP_C + F5 + PT),
    ("person_s1", SP_C + F5 + PT + ["--seed", "1"]),
    ("person_s2", SP_C + F5 + PT + ["--seed", "2"]),
    ("person_base_s0", SP_C + PT),
    ("person_inroom_s0", SP_IN + F5 + PT),
    ("person_all_s0", SP_ALL + F5 + PT),
]
QUEUE = {"dev": QUEUE_DEV, "final": QUEUE_FINAL, "person": QUEUE_PERSON}[STAGE]
REPORT_GLOB = {"dev": "r5_*.json", "final": "final_*.json", "person": "person_*.json"}[STAGE]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook():
    q = "[\n" + "".join(f"    ({n!r}, {e!r}),\n" for n, e in QUEUE) + "]"
    cells = [
        md(f"""
# ultrasonic รอบห้า ({ROUND}) — เลือกสูตรด้วยห้องที่ไม่เคยเห็น

**ก่อนเริ่ม**
1. **Runtime → Change runtime type → T4 GPU** → Save
2. อัปโหลดสองไฟล์ไปที่ Google Drive โฟลเดอร์ **`MyDrive/ultrasonic_r4/`** (โฟลเดอร์เดิม)
   - `ultrasonic_{ROUND}_data.zip` (ข้อมูล)
   - `ultrasonic_code.zip` (โค้ด · ทับของเดิมได้)
3. รันทีละเซลล์จากบนลงล่าง

การแบ่ง: เทรน **ห้องเดิม (hw2) + ห้อง B** · เลือกรอบ/จุดตัด/สูตร ด้วย **ห้อง C ทั้งห้อง**
· **ห้อง D ไม่อยู่ในชุดนี้** (ข้อสอบสุดท้าย ใช้รอบหน้าครั้งเดียว)
"""),
        md("## 1. ตรวจ GPU"),
        code("""
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "ไม่มี — Runtime > Change runtime type > T4 GPU")
"""),
        md("## 2. เชื่อม Drive แล้วแตกไฟล์"),
        code(f"""
from google.colab import drive
drive.mount('/content/drive')
import os, shutil, zipfile
D = '/content/drive/MyDrive/ultrasonic_r4'
RUNS = f'{{D}}/runs'
WORK = '/content/r5'
for z in ('ultrasonic_{ROUND}_data.zip', 'ultrasonic_code.zip'):
    assert os.path.exists(f'{{D}}/{{z}}'), f'ไม่พบ {{D}}/{{z}} — อัปโหลดก่อน'
os.makedirs(RUNS, exist_ok=True)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
for z in ('ultrasonic_{ROUND}_data.zip', 'ultrasonic_code.zip'):
    zipfile.ZipFile(f'{{D}}/{{z}}').extractall(WORK)
os.symlink(RUNS, f'{{WORK}}/car/data/r4_runs')
os.chdir(f'{{WORK}}/car')
os.environ['MAP2_ROUND'] = '{ROUND}'
os.environ['PYTHONIOENCODING'] = 'utf-8'
print('พร้อม')
"""),
        md("""
## 3. รันคิว (~1-1.5 ชั่วโมงบน T4)

ทุกการทดลองใช้การแบ่งเดียวกัน (`SPLIT`) เปลี่ยนทีละอย่าง ดูเหตุผลใน `car/ROUND4.md` หัวข้อ 7
"""),
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
        md("## 4. สรุปผล"),
        code(f"!python r4_report.py '{REPORT_GLOB}'"),
        md("ส่งตารางนี้ให้ Claude หรือบอกว่าเสร็จแล้ว (Claude อ่านจากลิงก์โฟลเดอร์ Drive ที่แชร์ได้)"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    OUTDIR.mkdir(exist_ok=True)
    if not (CACHE / "meta.json").exists():
        raise SystemExit(f"ยังไม่มีแคช {CACHE} — สร้างด้วย MAP2_ROUND={ROUND} python car/mapdata2.py")
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    if STAGE == "dev" and any(s.startswith("roomD") for s in meta["sets"]):
        raise SystemExit("แคชนี้มีห้อง D (ข้อสอบ) อยู่ — หยุด")
    dz = OUTDIR / f"ultrasonic_{ROUND}_data.zip"
    with zipfile.ZipFile(dz, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(CACHE.iterdir()):
            z.write(f, f"car/data/{CACHE.name}/{f.name}")
    cz = OUTDIR / "ultrasonic_code.zip"
    with zipfile.ZipFile(cz, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(CAR / f, f"car/{f}")
    nb = OUTDIR / f"ultrasonic_{ROUND}.ipynb"
    nb.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dz.name} {dz.stat().st_size / 1e6:.0f} MB · {cz.name} {cz.stat().st_size / 1e3:.0f} KB · {nb.name}")


if __name__ == "__main__":
    main()
