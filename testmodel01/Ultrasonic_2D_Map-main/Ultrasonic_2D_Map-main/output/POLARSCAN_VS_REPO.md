# PolarScan v2 — comparison vs ultrasonic01 repo

## Task framing difference
Repo predicts ONE object per ping: (distance of loudest echo, angle).
PolarScan v2 predicts a full 13-bin polar scan (nearest surface per ~4.5° bin)
PLUS the repo-style nearest-object (distance, angle) for direct comparison.

## Results — held-out walk_s5 (recorded in a different run than training data)

| Metric | ultrasonic01 best published | PolarScan v2 | improvement |
|---|---|---|---|
| Nearest-object distance MAE | 7.3 cm (physics) / 13.6 cm (their NN) | **4.30 cm** | 41% better than their physics, 68% better than their NN |
| Angle MAE | 11.8 deg (their NN) | **11.00 deg** | slightly better |
| Physics baseline on this data | — | 12.07 cm | model beats physics by 64% |

Validation (walk_s4): near-dist 3.83 cm, angle 10.70 deg.

Note: our test is *harder* than the repo's static-target benchmarks —
this is continuous walking footage with people moving (the repo itself measured
27.4 cm physics error on walking targets vs their 7.3 cm on static PVC pipes).

## What makes v2 better
1. Full polar-scan supervision (13 bins + validity logits) instead of single object —
   richer learning signal per frame, no "which peak is THE object" ambiguity.
2. GCC-PHAT cross-correlation features between ALL 6 receiver pairs
   (repo used amplitude-ratio + windowed argmax on only 2 pairs).
3. Repo's hard-won tricks kept and reused: T0=1220us calibration, common-peak window,
   bandpass sampling band selection, scene-based splits.
4. Hybrid physics+learned: network receives calibrated physics distance as an input
   feature and learns to correct it.

## Files
- usmap/physics.py      — calibrated signal processing (T0, gates, GCC-PHAT)
- usmap/polar_gt.py     — polar labels from depth camera (true-angle bins)
- usmap/polar_data.py   — dataset assembly
- usmap/polar_model.py  — PolarNet + masked losses + metrics
- usmap/polar_train.py  — training loop
- output/models/polarscan.pt — trained weights

Run: .venv/bin/python -m usmap.polar_train
