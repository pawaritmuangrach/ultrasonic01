"""Central configuration for the ultrasonic-as-camera pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = ROOT / "output"
FEAT = OUT / "features"
MODELS = OUT / "models"
EVAL = OUT / "eval"

# --- Echo preprocessing ---
RANGE_CM = 200.0            # matches meta.json max_cm
PROFILE_BINS = 128          # resampled range-profile length per channel
SPEED_OF_SOUND_CM_S = 34300.0

# --- Sensor geometry (array frame: +Y forward, X right, origin at array center) ---
# Receivers in npz channel order (pin 34,35,32,33)
CHANNEL_NAMES = ["A140", "A80", "C80", "C140"]
# (x_cm, y_cm) receiver positions; angles deg from +Y (negative = left)
RECEIVERS = {
    "A140": {"xy": (-6.0, 0.0), "angle_deg": -35.0},
    "A80":  {"xy": (-3.5, 0.0), "angle_deg": -17.5},
    "C80":  {"xy": (3.5, 0.0),  "angle_deg": +17.5},
    "C140": {"xy": (6.0, 0.0),  "angle_deg": +35.0},
}
TX = {"name": "T1", "xy": (0.0, 1.5)}   # top of array
BEAM_HALF_ANGLE_DEG = 20.0              # per-receiver sensitivity cone

# --- Target grid (rviz-like Cartesian map) ---
GRID_W = 64                 # cells across X
GRID_H = 48                 # cells across Y
X_MIN_CM, X_MAX_CM = -150.0, 150.0   # 3 m wide
Y_MIN_CM, Y_MAX_CM = 10.0, 250.0     # 2.4 m deep, starts 10 cm ahead of array

# --- Training ---
TRAIN_SECTIONS = ["walk_s1", "walk_s2", "walk_s3"]
VAL_SECTIONS = ["walk_s4"]
TEST_SECTIONS = ["walk_s5"]
BATCH = 32
EPOCHS = 60
LR = 1e-3


def ensure_dirs():
    for d in (OUT, FEAT, MODELS, EVAL):
        d.mkdir(parents=True, exist_ok=True)
