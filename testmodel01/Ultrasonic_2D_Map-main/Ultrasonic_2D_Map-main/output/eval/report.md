# Ultrasonic-as-Camera Pipeline — Report

## Goal
Turn a 4-receiver ultrasonic array (T1 transmitter top; A140+A80 left, C80+C140 right)
into a low-resolution "camera": from echo waveforms alone, produce an rviz-style
top-down Cartesian map (64×48 cells, 3 m wide × 2.4 m deep, 5 cm cells) of the space
in front of the robot. The depth camera is used ONLY as teacher during training;
at inference no camera is needed.

## Data
- 5 walks (walk_s1..s5), ~857 frames each; 4-channel echoes (871 samples @ ~66 kHz)
  synchronized with 320x240 depth images.
- Split: s1-s3 train / s4 val / s5 test (walk-level, prevents temporal leakage).

## Pipeline stages
1. Echo preprocessing (`usmap/echo.py`): baseline subtraction -> Hilbert envelope
   -> resample to 128 range bins over 0–200 cm. Physics: echo time axis IS distance.
2. Ground truth (`usmap/groundtruth.py`): pinhole back-projection of depth pixels
   (assumed 58° H-FOV), median per grid cell; visibility masks from receiver geometry.
3. Model (`usmap/models.py`): 1D-CNN echo encoder (4->96 ch) + ToF/amplitude hints
   -> transposed-conv decoder -> 64x48 distance grid.
4. Training: masked L1 loss (only cells visible to >=1 receiver AND with valid GT),
   AdamW + cosine schedule, 60 epochs, batch 32.

## Results (held-out test walk_s5)
| Model | MAE |
|---|---|
| **CNN (ours)** | **1.84 cm** |
| Physics baseline (paint ToF ray) | 129.32 cm |

Validation curve: converged ~0.88 cm val MAE by epoch 56/60.

Qualitative overlays in `output/eval/sample_*.png`: predicted maps match the
ground-truth obstacle positions and distances (near ~1 m and far ~2 m returns
both localized correctly); predictions are blurrier than GT as expected given
only 4 receivers.

## Honest limitations
1. Angular resolution is inherently coarse (2 left + 2 right receivers).
2. Depth-camera intrinsics were assumed (58 deg H-FOV) - if you later learn the
   real camera model, retrain for better ground truth.
3. ±35 ms US/camera sync adds noise on fast motion.
4. Grid cells with no valid depth GT were excluded from loss; truly unseen cells
   are not supervised.

## How to use
- Train:      `.venv/bin/python -m usmap.train`
- Evaluate:   `.venv/bin/python -m usmap.evaluate`
- Inference:  see `infer.py` (single npz -> map PNG + .npy)

## Next steps (optional)
- Record all 9 receivers (A/B/C x 40/80/140) -> same pipeline, much finer map.
- Export predicted grid as nav_msgs/OccupancyGrid for rviz/ROS.
