"""Local web app: explains the pipeline + live inference on us_*.npz frames.

Run:  .venv/bin/python webapp.py
Then open http://localhost:8765
"""
import base64
import io
import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch

from usmap.config import DATASET, GRID_H, GRID_W, X_MAX_CM, X_MIN_CM, Y_MAX_CM, Y_MIN_CM
from usmap.echo import range_profile, tof_cm
from usmap.models import UsMapNet

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = UsMapNet().to(DEVICE)
        ckpt = torch.load(ROOT / "output" / "models" / "usmapnet.pt",
                          map_location=DEVICE)
        _model.load_state_dict(ckpt["model"])
        _model.eval()
    return _model


def predict_grid(npz_bytes):
    """npz file bytes -> (48,64) float grid."""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.write(npz_bytes)
    tmp.close()
    try:
        prof = range_profile(tmp.name)                    # (4,BINS)
        t = tof_cm(tmp.name)
    finally:
        os.unlink(tmp.name)
    pt = torch.from_numpy(prof)[None]
    tt = torch.from_numpy(t)[None]
    extra = torch.cat([tt / 200.0, pt.max(dim=2).values], dim=1)
    with torch.no_grad():
        return get_model()(pt.to(DEVICE), extra.to(DEVICE))[0, 0].cpu().numpy()


def list_walks():
    walks = {}
    for d in sorted(DATASET.glob("walk_s*")):
        files = sorted(d.glob("us_*.npz"))
        walks[d.name] = [f.name for f in files]
    return walks


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
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
            html = (ROOT / "webapp" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/walks":
            self._send(200, json.dumps(list_walks()).encode())
        elif path == "/api/config":
            cfg = {
                "grid": [GRID_W, GRID_H],
                "extent": [X_MIN_CM, X_MAX_CM, Y_MIN_CM, Y_MAX_CM],
                "channels": ["A140 (left outer)", "A80 (left inner)",
                             "C80 (right inner)", "C140 (right outer)"],
                "range_cm": 200,
            }
            self._send(200, json.dumps(cfg).encode())
        elif path.startswith("/api/frame/"):
            m = re.match(r"/api/frame/(\w+)/([\w.]+)$", path)
            if not m:
                self._send(404, b'{"error":"bad path"}'); return
            walk, fname = m.groups()
            f = DATASET / walk / fname
            if not f.exists():
                self._send(404, b'{"error":"not found"}'); return
            grid = predict_grid(f.read_bytes())
            # also load paired depth for comparison
            idx = fname.split("_")[1].split(".")[0]
            depth_png = DATASET / walk / f"depth_{idx}.png"
            gt = None
            if depth_png.exists():
                from PIL import Image
                from usmap.groundtruth import depth_to_grid
                gt = depth_to_grid(np.array(Image.open(depth_png)).astype(np.float32) / 10.0)[0]
            gt_json = [[None if np.isnan(v) else round(float(v), 1) for v in row]
                       for row in gt] if gt is not None else None
            resp = {
                "grid": [[round(float(v), 1) for v in row] for row in grid],
                "gt": gt_json,
                "tof": [round(float(x), 1) for x in tof_cm(str(f))],
            }
            self._send(200, json.dumps(resp).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            try:
                grid = predict_grid(data)
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}).encode()); return
            resp = {"grid": [[round(float(v), 1) for v in row] for row in grid]}
            self._send(200, json.dumps(resp).encode())
        else:
            self._send(404, b'{"error":"not found"}')


if __name__ == "__main__":
    port = 8765
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
