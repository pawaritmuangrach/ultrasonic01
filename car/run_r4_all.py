"""รันการทดลองรอบสี่ทั้งหมดตามคิว — รายการนี้คือบันทึกว่ารันอะไรไปบ้าง

    python car/run_r4_all.py              รันทุกอย่างที่ยังไม่มีผล (ข้ามที่เสร็จแล้ว รันต่อได้)
    python car/run_r4_all.py --list       ดูรายการ

แบ่งเป็นสองคิวรันขนานกัน คิวละ 3 เธรด (เครื่องมี 6 คอร์จริง) log ของแต่ละการทดลอง
อยู่ที่ car/data/r4_runs/<ชื่อ>.log
"""
import argparse
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "r4_runs")
EP = "12"

# (ชื่อ, อาร์กิวเมนต์เพิ่มจากค่าตั้งต้น) · ค่าตั้งต้น = สูตรรอบ 3 (relu · r3 · p999 · 5 การยิง)
Q1 = [
    ("base_s0", []),
    ("loss_bdh", ["--loss", "bdh"]),
    ("loss_mse", ["--loss", "mse"]),
    ("loss_bce_l1", ["--loss", "bce_l1"]),
    ("loss_dice_huber", ["--loss", "dice_huber"]),
    ("loss_focal", ["--loss", "focal"]),
    ("norm_log", ["--norm", "log"]),
    ("slice5", ["--slice-ms", "5", "--slice-step-ms", "1"]),
]
Q2 = [
    ("__baselines__", None),
    ("base_s1", ["--seed", "1"]),
    ("base_s2", ["--seed", "2"]),
    ("act_leaky", ["--act", "leaky"]),
    ("act_elu", ["--act", "elu"]),
    ("act_gelu", ["--act", "gelu"]),
    ("act_silu", ["--act", "silu"]),
    ("act_mish", ["--act", "mish"]),
    ("act_tanh", ["--act", "tanh"]),
]


def done(name):
    if name == "__baselines__":
        return os.path.exists(os.path.join(OUT, "baseline_knn.json"))
    return os.path.exists(os.path.join(OUT, f"{name}.json"))


def run_queue(q, tag):
    env = dict(os.environ, MAP2_ROUND="r4", PYTHONIOENCODING="utf-8")
    for name, extra in q:
        if done(name):
            print(f"[{tag}] ข้าม {name} (มีผลแล้ว)", flush=True)
            continue
        if name == "__baselines__":
            cmd = [sys.executable, os.path.join(HERE, "baselines_r4.py")]
            log = os.path.join(OUT, "baselines.log")
        else:
            cmd = [sys.executable, os.path.join(HERE, "train_r4.py"), "--name", name,
                   "--epochs", EP, "--threads", "3"] + extra
            log = os.path.join(OUT, f"{name}.log")
        print(f"[{tag}] เริ่ม {name}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        print(f"[{tag}] {'เสร็จ' if r.returncode == 0 else 'ล้ม (ดู ' + log + ')'} {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.list:
        for q in (Q1, Q2):
            for n, e in q:
                print(f"{'เสร็จ' if done(n) else 'รอ  '}  {n:18} {' '.join(e or [])}")
        return
    ts = [threading.Thread(target=run_queue, args=(q, t)) for q, t in ((Q1, "คิว1"), (Q2, "คิว2"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("ครบทุกการทดลอง", flush=True)


if __name__ == "__main__":
    main()
