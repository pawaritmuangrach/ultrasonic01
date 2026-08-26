# Ultrasonic 2D Map — Teaching Ultrasound to "See" Like a Camera

**Give a robot eyes made of sound.**

This project turns a cheap 4-receiver ultrasonic array (the kind of sensor that
normally only tells you "something is 80 cm ahead") into a low-resolution
**camera**: from raw echo waveforms alone, it produces an rviz-style top-down
map of the space in front of the robot — where obstacles are, how far away,
and at what angle.

The trick: during training, a depth camera acts as a **teacher**, labeling every
ultrasonic recording with the true shape of the room. After training, the model
runs on **ultrasound alone** — no camera needed. This matters because depth
cameras fail exactly where ultrasound is strongest:

| Condition | Depth camera (Astra/RealSense) | This ultrasonic array |
|---|---|---|
| Glass / mirrors / glossy surfaces | ❌ blind | ✅ works |
| Transparent plastic | ❌ blind | ✅ works |
| Closer than 0.6 m | ❌ blind (min range) | ✅ works |
| Direct sunlight | ❌ washed out | ✅ works |
| Cost | $$$ | $ |

---

## 1. The hardware this was built for

Built as the next stage of [ultrasonic01](https://github.com/pawaritmuangrach/ultrasonic01) — same hardware, same recorded datasets, better model.

```
                 ▲ T1 (transmitter, fires 40 kHz burst)
    A140 ●───────────● C140      ← outer pair, baseline 242 mm
    A80  ●───────────● C80       ← inner pair, baseline 139 mm

   ESP32 samples all 4 receivers simultaneously @ ~66 kHz/channel
   (bandpass sampling: the 40 kHz echo aliases down to ~26 kHz, intact)
```

- **T1** fires a 16-cycle burst (~400 µs)
- Echoes bounce off obstacles and return; each receiver hears them at slightly
  different times and loudnesses
- **Distance** lives in *when* the echo arrives (time-of-flight, sound = 343 m/s,
  round trip = 58.3 µs per cm)
- **Direction** lives in *differences between channels* (TDOA between receiver
  pairs + amplitude ratios)

## 2. The dataset (not in git — record with ultrasonic01's `car/record.py`)

5 continuous walks (`walk_s1`–`walk_s5`) of ~60 s each, walking through a real
room with people moving. Every ~70 ms, two things were saved simultaneously:

| File | Content |
|---|---|
| `us_XXXXXX.npz` | `counts[4][871]` uint16 raw ADC samples @ ~66 kHz, plus `rate`, `pins`, timestamps, `sync_ms`, quality flags |
| `depth_XXXXXX.png` | 320×240 uint16 depth image in **millimeters** (0 = invalid), from the Astra camera looking the same direction |

Total ≈ 4,300 synchronized (echo, truth) pairs. Split for all experiments:
**s1–s3 train / s4 validation / s5 test** — split by walk session, never by
random frames, because consecutive frames are nearly identical and random
splitting leaks answers into training.

## 3. How the model works (PolarScan v2)

```
raw echoes ──► calibrated DSP ──► features ──► neural net ──► polar map
(4×871)        physics.py         (4,128)+     PolarNet       13 bins × dist
               T0 calibration     6× TDOA                     + nearest obj
               bandpass           amplitudes                  distance + angle
               common peak        SNR                         + validity mask
               GCC-PHAT
```

Step by step:

1. **Envelope extraction** — the raw wave oscillates at 40 kHz; we don't care
   about the wiggles, we care *when it gets loud*. A Hilbert transform extracts
   the loudness-over-time curve ("envelope") per channel.
2. **T0 calibration** — the system has a fixed delay (burst length + filter
   lag). Calibrated against the camera: wrong T0 shifted every reading by
   +13 cm; fixing it cut error from 20 → 9.7 cm.
3. **Range gate** — only look for echoes in the 40–200 cm window, otherwise
   argmax latches onto burst ring-down and reads garbage.
4. **Common peak** — the obstacle is at ONE distance, so all channels should
   peak at the same time. Use the quietest channel (least noise) to define the
   shared window. Worth 20.6 → 7.3 cm on its own.
5. **GCC-PHAT cross-correlation** — for all **6 receiver pairs**, compute the
   precise sub-sample time difference between their envelopes. This is the
   physically correct way to measure angle, replacing hand-tuned features.
6. **Range profiles** — each channel's envelope resampled to 128 distance bins
   (bin i ≈ distance i·1.25 cm). The network sees the full echo SHAPE, not just
   the loudest peak — multipath and echo width carry information.
7. **PolarNet** — small CNN over the profiles, concatenated with TDOA/amplitude/
   SNR/calibrated-physics-distance features, decoded into:
   - **13 angular-bin distances** (a mini radar sweep, ~4.5° per bin)
   - **validity flag** per bin (is anything actually there?)
   - **nearest-object distance & angle** (the number a collision-avoider needs)
8. **Training** — masked loss: only bins where the camera actually saw surface
   AND the bin is inside the sensors' physical coverage contribute. The camera
   labels are built by back-projecting depth pixels through the pinhole model.

### Why it beats ultrasonic01

| | ultrasonic01 | PolarScan v2 (this repo) |
|---|---|---|
| Output | ONE object per ping (single dist + angle) | Full 13-bin scan + object |
| Angle features | amplitude ratio + windowed argmax on 2 pairs | GCC-PHAT on all 6 pairs |
| Range approach | pure physics (argmax), throws away echo shape | physics as input feature, NN learns corrections |
| Nearest-object MAE* | 7.3 cm static / ~27 cm walking | **4.30 cm walking** |
| Angle MAE | 11.8° | **11.0°** |

\* Their 7.3 cm was measured on a static PVC pipe target. On walking-target
footage their own docs report ~27 cm error. Our 4.30 cm test set is continuous
walking footage — a harder setting.

Full write-up: [`output/POLARSCAN_VS_REPO.md`](output/POLARSCAN_VS_REPO.md)

## 4. Second model included: Grid CNN (generation 1)

First attempt: map echoes directly to a 64×48 Cartesian grid (3 m × 2.4 m,
~5 cm cells) via CNN encoder-decoder. Test result: **1.84 cm cell-MAE vs
129 cm for naive ToF painting**. Kept for reference and comparison;
PolarScan supersedes it because polar bins match the array's actual geometry.

## 5. Installation & usage

```bash
# environment (CPU torch is enough — model trains in ~15 min)
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy scipy matplotlib pillow

# put dataset in ./dataset/walk_s1..s5 (see §2)

# train PolarScan v2
.venv/bin/python -m usmap.polar_train
# → output/models/polarscan.pt, prints val metrics each epoch

# train grid CNN (gen 1)
.venv/bin/python -m usmap.train
.venv/bin/python -m usmap.evaluate

# single-frame inference — ultrasound ONLY, no camera
.venv/bin/python infer.py dataset/walk_s5/us_000123.npz
```

## 6. Live web demo (mini-rviz)

```bash
.venv/bin/python webapp_polar.py    # then open http://localhost:8766
```

- Radar-style display: blue wedges = the 13-bin ultrasonic scan, orange dots =
  camera ground truth (toggleable), range rings every 50 cm, ▲ = sensor array
- Pick any walk/frame, hit **▶ Play** to animate like a live radar sweep
- Live stats: nearest object distance+angle, physics-only estimate, MAE vs GT
- Per-bin table comparing model vs camera

Older grid demo: `.venv/bin/python webapp.py` → http://localhost:8765

## 7. Project layout

```
usmap/                    python package
├── config.py             sensor geometry, grid size, splits, hyperparameters
├── physics.py            ★ calibrated DSP: T0=1220µs, gates, common peak,
│                           GCC-PHAT TDOA, range profiles
├── polar_gt.py           depth PNG → 13 true-angle polar labels
├── polar_data.py         dataset assembly for PolarScan
├── polar_model.py        PolarNet architecture + losses + metrics
├── polar_train.py        training loop
├── echo.py               envelope/range-profile utilities (grid model)
├── groundtruth.py        depth PNG → Cartesian GT grid
├── data.py, models.py    grid CNN model
├── train.py, evaluate.py grid CNN train/eval
webapp_polar.py           live radar server (:8766)
webapp.py                 grid demo server (:8765)
webapp/                   frontend HTML
infer.py                  CLI inference on one npz
output/POLARSCAN_VS_REPO.md   benchmark write-up
output/models/            trained weights (gitignored — retrain or ask)
```

## 8. Honest limitations

- **Angular resolution**: 4 receivers fundamentally limits direction accuracy
  (~11°). Recording all 9 RX positions (A/B/C × 40/80/140) drops into the same
  pipeline unchanged — just more input channels.
- **Camera intrinsics assumed**: 58.4° H-FOV (Astra Pro spec). If your exact
  camera differs, labels shift slightly; retraining fixes it.
- **Sync noise**: US↔camera offset ±35 ms smears labels during fast motion.
- **Single-room generalization**: test split is a different recording session
  but the same room. Cross-room transfer is untested.
- **Specular surfaces can still fool everyone**: smooth walls angled away
  reflect sound elsewhere — MIMO (3 TX, planned in ultrasonic01 Stage B+)
  addresses this.

## 9. ROS / rviz (planned)

The 13-bin scan maps directly onto `sensor_msgs/LaserScan`. An export node is
the next step — the web app's radar view is already a stand-in for rviz.

## 10. Credits

Hardware, recording pipeline, calibration lessons (T0, common peak, bandpass
sampling), and datasets: [pawaritmuangrach/ultrasonic01](https://github.com/pawaritmuangrach/ultrasonic01).
Modeling, PolarScan v2, evaluation: this repo.
