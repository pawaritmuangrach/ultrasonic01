"""รันไขว้ทุกช่วง — กันทีละช่วงไว้ทดสอบ ดูว่าผลนิ่งจริงไหม"""
import subprocess, sys, re, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for t in range(1, 6):
    out = subprocess.run([sys.executable, "-u", os.path.join(HERE, "train_nn.py"),
                          "--test", str(t), "--epochs", "80", "--seed", "0"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    nn = re.search(r"ML ล้วนจากคลื่นดิบ\s+([\d.]+)", out)
    bs = re.search(r"ไม่ดูข้อมูล\)\s+([\d.]+)", out)
    zn = re.search(r"ทายโซนถูก (\d+)%", out)
    res[t] = (float(nn.group(1)) if nn else float("nan"),
              float(bs.group(1)) if bs else float("nan"),
              int(zn.group(1)) if zn else -1)
    print(f"กันช่วง s{t}: ML {res[t][0]:5.2f}°  เดามั่ว {res[t][1]:5.2f}°  "
          f"โซนถูก {res[t][2]}%", flush=True)
a = np.array([v[0] for v in res.values()]); b = np.array([v[1] for v in res.values()])
z = np.array([v[2] for v in res.values()])
print(f"\n{'':12}{'ML ล้วน':>10}{'เดามั่ว':>10}{'rules.py':>10}")
print(f"{'เฉลี่ย':12}{a.mean():>9.2f}°{b.mean():>9.2f}°{'4.80°':>10}")
print(f"{'ช่วง':12}{f'{a.min():.2f}-{a.max():.2f}':>10}{'':>10}{'4.5-6.2':>10}")
print(f"{'โซนถูกเฉลี่ย':12}{z.mean():>9.0f}%{'':>10}{'82%':>10}")
