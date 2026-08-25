"""Generate the 3 TX + 9 RX array plate as STEP and STL, straight from the
coordinate table the analysis code uses.

Run:  python hardware/plate_model.py

Why the model lives here as code rather than in a CAD file: the transducer
positions are also the numbers the TDOA maths uses. If the plate is drawn by
hand in a GUI those two copies drift apart, and every range comes out subtly
wrong with nothing to point at. Here there is one source of truth - this file
writes plate_coords.csv itself - and changing a radius is a one-line edit plus
a re-run.

Written in build123d's algebra mode rather than its builder mode: builder mode
finds its context by walking the call stack, so any operation moved into a
helper function silently loses the builder it belongs to. Algebra mode has no
hidden context, which makes the repeated boss/bore/label code shareable.

The design splits into a hub and three identical arms because the assembled
plate spans ~340 mm, wider than most print beds. The split sits between the
r=40 and r=80 receivers - the only gap on an arm wide enough for a lap joint
that does not cut into a boss.
"""

import csv
import math
import os
import struct

import numpy as np
from build123d import *

# ----------------------------------------------------------------- config --

T = 4.0            # plate thickness [mm]
BORE = 19.5        # transducer hole: 16.0 mm can + 1.75 mm of O-ring each side
BOSS_OD = 23.0     # 1.75 mm wall around the bore
BOSS_H = 10.0      # boss height above the plate; two O-rings stack inside it

LEG_W = 28.0       # width of the hub legs and of the arms
HUB_DISC_R = 34.0  # central disc, sized to contain the three TX bosses
HUB_LEG_R = 68.0   # how far the hub legs reach

JOINT_R0, JOINT_R1 = 53.0, 68.0   # lap joint: hub top face / arm underside
JOINT_T = 2.0                     # each half is half the plate thickness
JOINT_FIT = 0.2                   # pocket is this much wider than the tongue

# Two identical M3 clearance holes rather than a dowel plus a screw. A 3 mm
# locating pin would hold the arm straighter, but sourcing dowel pins is a
# detour and the pocket walls already do most of the work. If you do have
# pins, set BOLT_A_D to 3.0 and press one in.
BOLT_A_R, BOLT_A_D = 56.0, 3.4
BOLT_B_R, BOLT_B_D = 65.0, 3.4

ARM_R1 = 170.0                    # arm tip
MOUNT_R, MOUNT_D = 160.0, 4.5     # mounting hole, doubles as a camera fiducial
CENTER_D = 6.5                    # 1/4"-20 tripod clearance

SLOT_R, SLOT_L, SLOT_W = 110.0, 20.0, 3.0   # acoustic break across each arm

TX_R, TX_ANGLES = 20.0, (30.0, 150.0, 270.0)
ARM_ANGLES = (90.0, 210.0, 330.0)
RX_HUB_R = 40.0                   # near ring, carried by the hub
RX_ARM_R = (80.0, 140.0)          # mid and far, carried by the arms

LABEL_H, LABEL_DEPTH = 4.5, 0.6

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cad")

# --------------------------------------------------------------- helpers --

def polar(r, deg):
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def at(r, deg, spin=0.0):
    """Location at polar (r, deg) with local +Y pointing outward."""
    x, y = polar(r, deg)
    return Location((x, y, 0), (0, 0, deg - 90 + spin))


def boss(pt):
    """Raised collar that holds the two isolating O-rings."""
    return Pos(pt[0], pt[1], T) * extrude(Circle(BOSS_OD / 2), BOSS_H)


def bore(pt):
    return Pos(pt[0], pt[1], -0.5) * extrude(Circle(BORE / 2), T + BOSS_H + 1)


def hole(pt, d, z0=-0.5, h=None):
    return Pos(pt[0], pt[1], z0) * extrude(Circle(d / 2), h or (T + 1))


def label(txt, loc):
    """Engraved into the top face - raised text would be crushed by the bed."""
    return loc * Pos(0, 0, T) * extrude(Text(txt, font_size=LABEL_H), -LABEL_DEPTH)


# ------------------------------------------------------------------- hub --

def build_hub():
    part = extrude(Circle(HUB_DISC_R), T)
    for a in ARM_ANGLES:
        part += at(HUB_LEG_R / 2, a) * extrude(Rectangle(LEG_W, HUB_LEG_R), T)

    pts = [polar(TX_R, a) for a in TX_ANGLES] + [polar(RX_HUB_R, a) for a in ARM_ANGLES]
    for p in pts:
        part += boss(p)

    # lap-joint pockets, cut down from the top face
    for a in ARM_ANGLES:
        part -= (at((JOINT_R0 + JOINT_R1) / 2, a) * Pos(0, 0, T) *
                 extrude(Rectangle(LEG_W + JOINT_FIT,
                                   JOINT_R1 - JOINT_R0 + JOINT_FIT), -JOINT_T))

    for p in pts:
        part -= bore(p)
    part -= hole((0, 0), CENTER_D)
    for a in ARM_ANGLES:
        part -= hole(polar(BOLT_A_R, a), BOLT_A_D)
        part -= hole(polar(BOLT_B_R, a), BOLT_B_D)

    part -= label("+Y", Location((0, 11, 0)))
    for name, a in zip(("T1", "T2", "T3"), TX_ANGLES):
        part -= label(name, at(30.0, a))
    for name, a in zip(("A40", "B40", "C40"), ARM_ANGLES):
        part -= label(name, at(22.0, a))
    return part


# ------------------------------------------------------------------- arm --

def build_arm():
    """Modelled along +Y so every radius reads the same as on the assembly."""
    mid = (JOINT_R0 + ARM_R1) / 2
    part = Pos(0, mid) * extrude(
        RectangleRounded(LEG_W, ARM_R1 - JOINT_R0, LEG_W / 2 - 0.01), T)

    # tongue: thin the inner end from below so it drops into the hub pocket
    part -= (Pos(0, (JOINT_R0 + JOINT_R1) / 2) *
             extrude(Rectangle(LEG_W + 1, JOINT_R1 - JOINT_R0), JOINT_T))

    for r in RX_ARM_R:
        part += boss((0, r))
    for r in RX_ARM_R:
        part -= bore((0, r))

    part -= hole((0, BOLT_A_R), BOLT_A_D)
    part -= hole((0, BOLT_B_R), BOLT_B_D)
    part -= hole((0, MOUNT_R), MOUNT_D)
    # slot runs across the arm, so structure-borne sound has to detour round it
    part -= Pos(0, SLOT_R, -0.5) * extrude(SlotOverall(SLOT_L, SLOT_W), T + 1)

    part -= label("80", Location((0, 95, 0)))
    part -= label("140", Location((0, 154, 0)))
    return part


# ------------------------------------------------------------------ main --

def write_coords():
    rows = []
    for name, a in zip(("T1", "T2", "T3"), TX_ANGLES):
        rows.append((name, *polar(TX_R, a), TX_R, a, "TX"))
    for arm, a in zip("ABC", ARM_ANGLES):
        for r in (RX_HUB_R,) + RX_ARM_R:
            rows.append((f"{arm}{int(r)}", *polar(r, a), r, a, "RX"))
        rows.append((f"M{arm}", *polar(MOUNT_R, a), MOUNT_R, a, "MOUNT"))
    with open(os.path.join(HERE, "plate_coords.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_mm", "y_mm", "radius_mm", "angle_deg", "role"])
        for name, x, y, r, a, role in rows:
            w.writerow([name, f"{x:.2f}", f"{y:.2f}", f"{r:g}", f"{a:g}", role])
    return len(rows)


def render_preview(asm, path):
    """Top view drawn from the exported mesh, so the picture cannot disagree
    with the file that actually gets printed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    with open(os.path.join(OUT, "plate_assembly.stl"), "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tris = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    z = tris[:, :, 2].mean(axis=1)
    order = np.argsort(z)               # painter's algorithm: low faces first
    tris, z = tris[order], z[order]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=110)
    shade = plt.cm.Blues(0.25 + 0.7 * (z - z.min()) / (np.ptp(z) + 1e-9))
    ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=shade, edgecolors="none"))
    colour = {"TX": "#e03131", "RX": "#1971c2", "MOUNT": "#2f9e44"}
    with open(os.path.join(HERE, "plate_coords.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x, y = float(row["x_mm"]), float(row["y_mm"])
            c = colour[row["role"]]
            ax.plot(x, y, "+", color=c, ms=9, mew=1.6)
            ax.annotate(row["id"], (x, y), textcoords="offset points",
                        xytext=(10, 6), fontsize=8, color=c, weight="bold")
    bb = asm.bounding_box()
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"plate assembly - top view   {bb.size.X:.0f} x {bb.size.Y:.0f} mm")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    hub, arm = build_hub(), build_arm()
    for name, part in (("plate_hub", hub), ("plate_arm", arm)):
        export_step(part, os.path.join(OUT, name + ".step"))
        export_stl(part, os.path.join(OUT, name + ".stl"))
        bb = part.bounding_box()
        print(f"{name:10s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:5.1f} mm"
              f"   ปริมาตร {part.volume/1000:6.1f} cm3")

    asm = Compound(children=[hub] + [Rot(0, 0, a - 90) * Pos(0, 0, T - JOINT_T) * arm
                                     for a in ARM_ANGLES])
    export_step(asm, os.path.join(OUT, "plate_assembly.step"))
    export_stl(asm, os.path.join(OUT, "plate_assembly.stl"))
    bb = asm.bounding_box()
    print(f"{'assembly':10s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:5.1f} mm"
          f"   พิมพ์ดุม 1 + แขน 3")

    print(f"\nเขียน plate_coords.csv จากโมเดลเดียวกัน ({write_coords()} จุด)")
    render_preview(asm, os.path.join(OUT, "plate_preview.png"))
    print("วาดภาพตัวอย่าง cad/plate_preview.png")


if __name__ == "__main__":
    main()
