"""PolarScan v2: predict a full polar scan (nearest distance per angular bin)
from ultrasound alone — beating ultrasonic01's single-object (dist, angle) model.

Label: depth PNG -> N_ANG angular bins covering camera H-FOV; per bin the
nearest reliable surface distance (percentile-5, middle vertical band,
400-2500 mm trust window) + validity mask. Same philosophy as the repo's
labels.py but projected through the pinhole so bins are TRUE angles, not
pixel columns.
"""
import numpy as np
import torch
from PIL import Image

FOV_H_DEG = 58.4          # Astra Pro horizontal FOV (matches repo)
N_ANG = 13                # ~4.5 deg/bin
V_BAND = 0.40             # middle 40% vertical band (cut floor/ceiling)
MIN_MM, MAX_MM = 400, 2500
MIN_PIX = 10


def load_depth_mm(png_path):
    return np.array(Image.open(png_path)).astype(np.float32)


def polar_label(png_path):
    """-> (dist_cm[N_ANG], valid[N_ANG]). Nearest surface per true-angle bin."""
    depth = load_depth_mm(png_path)
    h, w = depth.shape
    fx = w / (2 * np.tan(np.deg2rad(FOV_H_DEG) / 2))
    r0, r1 = int(h * (0.5 - V_BAND / 2)), int(h * (0.5 + V_BAND / 2))
    band = depth[r0:r1, :]

    yy, xx = np.mgrid[0:band.shape[0], 0:w]
    Z = band                                   # forward mm
    X = (xx - w / 2) * Z / fx                  # lateral mm

    ok = (Z >= MIN_MM) & (Z <= MAX_MM)
    ang = np.degrees(np.arctan2(X[ok], Z[ok]))          # -29..+29 deg
    z = Z[ok]

    edges = np.linspace(-FOV_H_DEG / 2, FOV_H_DEG / 2, N_ANG + 1)
    bi = np.clip(np.searchsorted(edges, ang) - 1, 0, N_ANG - 1)

    dist = np.zeros(N_ANG, np.float32)
    valid = np.zeros(N_ANG, bool)
    order = np.argsort(bi)
    bi_s, z_s = bi[order], z[order]
    bounds = np.searchsorted(bi_s, np.arange(N_ANG + 1))
    for b in range(N_ANG):
        seg = z_s[bounds[b]:bounds[b + 1]]
        if seg.size >= MIN_PIX:
            k = max(0, int((seg.size - 1) * 0.05))      # p5 = nearest-ish robust
            dist[b] = float(np.partition(seg, k)[k]) / 10.0
            valid[b] = True
    return dist, valid


def nearest_object(dist, valid):
    """Repo-style metric targets: distance & angle of nearest valid bin."""
    if not valid.any():
        return None
    edges = np.linspace(-FOV_H_DEG / 2, FOV_H_DEG / 2, N_ANG + 1)
    angs = (edges[:-1] + edges[1:]) / 2
    i = int(np.argmin(np.where(valid, dist, 1e9)))
    return float(dist[i]), float(angs[i])
