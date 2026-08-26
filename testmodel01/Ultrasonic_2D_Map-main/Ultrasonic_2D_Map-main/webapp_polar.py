"""Live PolarScan web app: 13-bin radar-style map from the v2 model.

Run:  .venv/bin/python webapp_polar.py   -> http://localhost:8766
"""
import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch

from usmap.config import DATASET
from usmap.physics import physics_features
from usmap.polar_gt import N_ANG, FOV_H_DEG, polar_label

ROOT = Path(__file__).resolve().parent
DEVICE = "cpu"
_model = None


def get_model():
    global _model
    if _model is None:
        from usmap.polar_model import PolarNet
        _model = PolarNet()
        ckpt = torch.load(ROOT / "output" / "models" / "polarscan.pt",
                          map_location=DEVICE)
        _model.load_state_dict(ckpt["model"])
        _model.eval()
    return _model


class PolarNetWrap(torch.nn.Module):
    """Lazy import wrapper so module-level model load stays simple."""
    def __init__(self):
        super().__init__()
        from usmap.polar_model import PolarNet
        self.net = PolarNet()

    def forward(self, *a):
        return self.net(*a)


def predict(npz_path):
    f = physics_features(str(npz_path))
    prof = torch.from_numpy(f["prof"])[None]
    tdoa = torch.from_numpy(f["tdoa_us"])[None]
    amps = torch.from_numpy(f["amps"])[None]
    snr = torch.from_numpy(f["snr"])[None]
    dist_cm = torch.tensor([[f["dist_cm"]]], dtype=torch.float32)
    with torch.no_grad():
        out = get_model()(prof, tdoa, amps, snr, dist_cm)
    bins = out["bin_dist"][0].numpy()
    valid = (torch.sigmoid(out["bin_valid"][0]).numpy() > 0.5)
    bins = np.where(valid, np.clip(bins, 0, 250), None)
    return {
        "bins": [None if b is None else round(float(b), 1) for b in bins],
        "near_d": round(float(out["near_d"][0]), 1),
        "near_a": round(float(out["near_a"][0]), 1),
        "physics": round(float(f["dist_cm"]), 1),
        "tof": None,
    }


def gt_bins(depth_path):
    d, v = polar_label(str(depth_path))
    return [round(float(x), 1) if vv else None for x, vv in zip(d, v)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "webapp" / "polar.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/api/walks":
            walks = {}
            for d in sorted(DATASET.glob("walk_s*")):
                walks[d.name] = [f.name for f in sorted(d.glob("us_*.npz"))]
            self._send(200, json.dumps(walks).encode())
        elif path.startswith("/api/frame/"):
            m = re.match(r"/api/frame/(\w+)/([\w.]+)$", path)
            if not m:
                self._send(404, b'{"error":"bad"}'); return
            walk, fname = m.groups()
            u = DATASET / walk / fname
            if not u.exists():
                self._send(404, b'{"error":"not found"}'); return
            r = predict(u)
            idx = fname.split("_")[1].split(".")[0]
            dp = DATASET / walk / f"depth_{idx}.png"
            r["gt"] = gt_bins(dp) if dp.exists() else None
            self._send(200, json.dumps(r).encode())
        else:
            self._send(404, b'{"error":"not found"}')


if __name__ == "__main__":
    port = 8766
    print(f"PolarScan live on http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
