"""Ground truth: depth PNG (mm, 0=invalid) -> rviz-like Cartesian target grid.

For each grid cell we take the robust median of valid depth pixels whose
position falls in that cell. Cells are additionally masked by which sensor
cones can physically observe them (visibility mask used for loss masking).
"""
import numpy as np
import torch
from PIL import Image

from .config import (GRID_H, GRID_W, RECEIVERS, X_MAX_CM, X_MIN_CM, Y_MAX_CM,
                     Y_MIN_CM, BEAM_HALF_ANGLE_DEG)


def load_depth_cm(png_path):
    d = np.array(Image.open(png_path)).astype(np.float32)  # mm
    return d / 10.0                                        # cm


# Assumed intrinsics: standard 58 deg H-FOV depth camera (user: model unknown).
HFOV_DEG = 58.0


def pinhole_fx(width):
    return width / (2 * np.tan(np.deg2rad(HFOV_DEG) / 2))


def cell_centers():
    xs = np.linspace(X_MIN_CM, X_MAX_CM, GRID_W + 1)
    ys = np.linspace(Y_MIN_CM, Y_MAX_CM, GRID_H + 1)
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    return cx, cy


def visibility_mask():
    """(4, GRID_H, GRID_W) bool: can receiver ch see this cell?"""
    cx, cy = cell_centers()
    XX, YY = np.meshgrid(cx, cy)                            # (H, W)
    masks = np.zeros((len(RECEIVERS), GRID_H, GRID_W), dtype=bool)
    for i, name in enumerate(RECEIVERS):
        ang = np.deg2rad(RECEIVERS[name]["angle_deg"])
        rx, ry = RECEIVERS[name]["xy"]
        dx = XX - rx
        dy = YY - ry
        bearing = np.arctan2(dx, dy)                        # 0 = +Y forward
        masks[i] = np.abs(bearing - ang) < np.deg2rad(BEAM_HALF_ANGLE_DEG * 2)
    return masks


def depth_to_grid(depth_cm, min_valid_px=2, sensor_height_frac=0.5):
    """Back-project depth pixels (pinhole) and rasterize a horizontal slice
    at the sensor's height band into the grid. -> (1, GRID_H, GRID_W), NaN=empty."""
    H, W = depth_cm.shape
    fx = pinhole_fx(W)
    yy, xx = np.mgrid[0:H, 0:W]
    Z = depth_cm                                   # forward distance (cm)
    X = (xx - W / 2) * Z / fx                      # right (cm)

    valid = Z > 0

    xw = X_MAX_CM - X_MIN_CM
    yw = Y_MAX_CM - Y_MIN_CM
    gx = ((X - X_MIN_CM) / xw * GRID_W).astype(int)
    gy = ((Z - Y_MIN_CM) / yw * GRID_H).astype(int)
    ok = valid & (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)

    grid = np.full(GRID_W * GRID_H, np.nan, dtype=np.float32)
    lin = gy[ok] * GRID_W + gx[ok]
    z_s = Z[ok]
    order = np.argsort(lin)
    lin_s, z_s = lin[order], z_s[order]
    bounds = np.searchsorted(lin_s, np.arange(GRID_W * GRID_H + 1))
    for c in range(GRID_W * GRID_H):
        seg = z_s[bounds[c]:bounds[c + 1]]
        if len(seg) >= min_valid_px:
            grid[c] = float(np.median(seg))
    return grid.reshape(1, GRID_H, GRID_W)


def make_target(png_path):
    """Target + visibility mask tensors for training."""
    tgt = depth_to_grid(load_depth_cm(png_path))
    vis = visibility_mask()                                  # (4,H,W)
    vis_any = vis.any(axis=0)                                # (H,W)
    return torch.from_numpy(tgt), torch.from_numpy(vis_any[None].astype(np.float32)), \
           torch.from_numpy(vis.astype(np.float32))
