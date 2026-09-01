"""ตัวบอกท่าให้ทำตามระหว่างอัด dataset — วาดหุ่นไม้ขีดบอกท่า พร้อมนับถอยหลัง

    python car/pose_cue.py --section 1        เปิดหน้าต่างบอกท่าของช่วงที่ 1

**ทำไมต้องมี** โมเดลชุดเดิมเรียนรูปร่างไม่ได้เลย เพราะในข้อมูล
**ตำแหน่งอธิบายทุกอย่างได้อยู่แล้ว** ท่าทางแทบไม่เคยเปลี่ยนขณะที่ตำแหน่งคงที่
โมเดลจึงเรียนแค่ 'อยู่ตรงไหน ไกลเท่าไร' แล้วแปะรูปร่างเฉลี่ยลงไป

รอบนี้จึงบังคับให้ **ท่าเปลี่ยนขณะตำแหน่งคงที่** และ **ตำแหน่งเปลี่ยนขณะท่าเดิม**
สองอย่างนี้แยกออกจากกันในข้อมูล โมเดลถึงจะมีอะไรให้เรียนเรื่องรูปร่าง

ไม่ยุ่งกับการอัดเลย รันคู่ขนานกับ record.py ได้ — กล้อง depth เป็นคนบอกเฉลย
อยู่แล้วว่าท่าไหน ตัวนี้แค่ทำให้ **คนหน้ากล้อง** เปลี่ยนท่าอย่างเป็นระบบ

วาดเป็นรูปไม่ใช่ตัวหนังสือ เพราะ OpenCV เขียนภาษาไทยไม่ได้ และรูปหุ่นไม้ขีด
บอกท่าได้ชัดกว่าคำอธิบายอยู่แล้ว
"""
import argparse
import time

import numpy as np

W, H = 900, 760
HOLD_S = 6.0            # ถือท่าละกี่วินาที
BG = (18, 20, 24)
INK = (235, 240, 248)
DIM = (120, 126, 138)
HOT = (60, 210, 250)

# ตำแหน่งที่ให้ยืนในแต่ละช่วง — ท่าชุดเดียวกันทุกช่วง จึงแยกท่ากับตำแหน่งออกจากกัน
SECTIONS = [
    ("center, 100 cm", "ยืนกลาง ห่าง 100 ซม."),
    ("center,  70 cm", "ยืนกลาง ห่าง 70 ซม."),
    ("center, 140 cm", "ยืนกลาง ห่าง 140 ซม."),
    ("LEFT,   100 cm", "ยืนเยื้องซ้าย ห่าง 100 ซม."),
    ("RIGHT,  100 cm", "ยืนเยื้องขวา ห่าง 100 ซม."),
]

# หุ่นไม้ขีด: (หัว, ลำตัว, แขนซ้าย, แขนขวา, ขาซ้าย, ขาขวา) พิกัด 0..1 y ขึ้นบน
POSES = [
    ("ARMS DOWN", dict(head=(.5, .86), hip=(.5, .46), sh=(.5, .74),
                       la=[(.42, .60), (.40, .46)], ra=[(.58, .60), (.60, .46)],
                       ll=[(.44, .24), (.43, .04)], rl=[(.56, .24), (.57, .04)])),
    ("ARMS OUT (T)", dict(head=(.5, .86), hip=(.5, .46), sh=(.5, .74),
                          la=[(.34, .74), (.18, .74)], ra=[(.66, .74), (.82, .74)],
                          ll=[(.44, .24), (.43, .04)], rl=[(.56, .24), (.57, .04)])),
    ("ARMS UP", dict(head=(.5, .86), hip=(.5, .46), sh=(.5, .74),
                     la=[(.38, .86), (.34, .99)], ra=[(.62, .86), (.66, .99)],
                     ll=[(.44, .24), (.43, .04)], rl=[(.56, .24), (.57, .04)])),
    ("ONE ARM UP", dict(head=(.5, .86), hip=(.5, .46), sh=(.5, .74),
                        la=[(.42, .60), (.40, .46)], ra=[(.62, .86), (.66, .99)],
                        ll=[(.44, .24), (.43, .04)], rl=[(.56, .24), (.57, .04)])),
    ("HALF CROUCH", dict(head=(.5, .68), hip=(.5, .34), sh=(.5, .58),
                         la=[(.38, .48), (.36, .36)], ra=[(.62, .48), (.64, .36)],
                         ll=[(.40, .18), (.44, .04)], rl=[(.60, .18), (.56, .04)])),
    ("FULL CROUCH", dict(head=(.5, .52), hip=(.5, .24), sh=(.5, .44),
                         la=[(.36, .34), (.40, .22)], ra=[(.64, .34), (.60, .22)],
                         ll=[(.34, .12), (.44, .04)], rl=[(.66, .12), (.56, .04)])),
    ("BEND FORWARD", dict(head=(.66, .70), hip=(.5, .46), sh=(.60, .64),
                          la=[(.60, .48), (.60, .32)], ra=[(.62, .48), (.62, .32)],
                          ll=[(.46, .24), (.45, .04)], rl=[(.54, .24), (.53, .04)])),
    ("TURN SIDEWAYS", dict(head=(.5, .86), hip=(.5, .46), sh=(.5, .74),
                           la=[(.52, .60), (.53, .46)], ra=[(.52, .60), (.53, .46)],
                           ll=[(.50, .24), (.52, .04)], rl=[(.50, .24), (.48, .04)])),
]


def _p(pt, x0, y0, w, h):
    """พิกัด 0..1 (y ขึ้น) -> พิกเซล (y ลง)"""
    return int(x0 + pt[0] * w), int(y0 + (1.0 - pt[1]) * h)


def draw_figure(img, pose, x0, y0, w, h, col=INK, th=7):
    import cv2
    P = pose
    hd = _p(P["head"], x0, y0, w, h)
    sh = _p(P["sh"], x0, y0, w, h)
    hp = _p(P["hip"], x0, y0, w, h)
    cv2.circle(img, (hd[0], hd[1] - int(h * .05)), int(h * .055), col, th, cv2.LINE_AA)
    cv2.line(img, sh, hp, col, th, cv2.LINE_AA)
    for key, root in (("la", sh), ("ra", sh), ("ll", hp), ("rl", hp)):
        pts = [root] + [_p(q, x0, y0, w, h) for q in P[key]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, th, cv2.LINE_AA)


def frame(sec, idx, left, cycle):
    import cv2
    img = np.full((H, W, 3), BG, np.uint8)
    name, pose = POSES[idx]
    nxt = POSES[(idx + 1) % len(POSES)]
    where = SECTIONS[sec - 1][0]

    cv2.putText(img, f"SECTION {sec}   stand: {where}", (26, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, HOT, 2, cv2.LINE_AA)
    cv2.putText(img, f"cycle {cycle}   pose {idx + 1}/{len(POSES)}", (26, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)

    draw_figure(img, pose, 60, 110, 420, 500)
    cv2.putText(img, name, (60, 660), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
                INK, 3, cv2.LINE_AA)

    cv2.putText(img, "NEXT", (560, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                DIM, 1, cv2.LINE_AA)
    draw_figure(img, nxt[1], 560, 170, 290, 340, DIM, 4)
    cv2.putText(img, nxt[0], (560, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                DIM, 2, cv2.LINE_AA)

    # แถบนับถอยหลัง
    f = max(0.0, left) / HOLD_S
    cv2.rectangle(img, (26, H - 54), (W - 26, H - 26), (44, 48, 56), -1)
    cv2.rectangle(img, (26, H - 54), (26 + int((W - 52) * f), H - 26), HOT, -1)
    cv2.putText(img, f"{left:4.1f}s", (W - 130, H - 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 2, cv2.LINE_AA)
    cv2.putText(img, "q quit   space pause", (26, H - 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, DIM, 1, cv2.LINE_AA)
    return img


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", type=int, default=1, help="ช่วงที่เท่าไร (1-5)")
    ap.add_argument("--hold", type=float, default=HOLD_S, help="ถือท่าละกี่วินาที")
    ap.add_argument("--cycles", type=int, default=4, help="วนกี่รอบ")
    a = ap.parse_args()
    globals()["HOLD_S"] = a.hold

    total = a.cycles * len(POSES) * a.hold
    print(f"ช่วง {a.section}: {SECTIONS[a.section - 1][1]}")
    print(f"{len(POSES)} ท่า x {a.cycles} รอบ x {a.hold:.0f} วินาที = {total/60:.1f} นาที")
    print("เริ่มอัดด้วย record.py ก่อน แล้วค่อยกดหน้าต่างนี้")

    cv2.namedWindow("pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("pose", W, H)
    t0, paused, pt = time.time(), False, 0.0
    while True:
        now = (pt if paused else time.time() - t0)
        step = int(now // a.hold)
        if step >= a.cycles * len(POSES):
            print("ครบแล้ว")
            break
        idx = step % len(POSES)
        cyc = step // len(POSES) + 1
        left = a.hold - (now - step * a.hold)
        cv2.imshow("pose", frame(a.section, idx, left, cyc))
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            if paused:
                t0 = time.time() - pt
            else:
                pt = now
            paused = not paused
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
