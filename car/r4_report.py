"""รวมผลทุกการทดลองรอบสี่เป็นตาราง Markdown (พิมพ์ออกจอ เอาไปวางใน ROUND4.md)

    python car/r4_report.py
"""
import json
import os
import sys

os.environ.setdefault("MAP2_ROUND", "r4")
import mapdata2 as MD     # noqa: E402

OUT = MD.DATA / "r4_runs"


def row(r):
    a = r.get("args", {})
    v, t = r["val"], r["test"]
    sh = r.get("test_shuffled", {})
    cfg = (f"{a.get('act', '-')} · {a.get('loss', '-')} · {a.get('norm', '-')}"
           + (f" · ตัด {a['slice_ms']:g} ms" if isinstance(a, dict) and a.get("slice_ms") else "")
           if isinstance(a, dict) else str(a))
    return (f"| {r['name']} | {cfg} | {v['iou']:.3f} | {v['mae_cm']:.1f} | {t['iou']:.3f} | "
            f"{t['mae_cm']:.1f} | {t['h_r2']:+.3f} | "
            f"{sh.get('iou', float('nan')):.3f} | {r.get('best_epoch', '-')} | {r.get('minutes', 0):.0f} |")


def main():
    files = sorted(OUT.glob("*.json"))
    if not files:
        sys.exit("ยังไม่มีผล")
    rs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rs.sort(key=lambda r: (not r["name"].startswith("baseline"), r["name"]))
    print("| การทดลอง | ตั้งค่า | s5 IoU | s5 ระยะ ซม. | **s6 IoU** | **s6 ระยะ ซม.** | s6 สูงเงา R2 "
          "| s6 สลับคลื่น IoU | รอบที่เลือก | นาที |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(row(r))


if __name__ == "__main__":
    main()
