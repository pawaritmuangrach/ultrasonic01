import shutil, torch, numpy as np
from usmap.config import MODELS
import usmap.polar_train as pt
for s in (1, 2):
    torch.manual_seed(s); np.random.seed(s)
    print(f"===== seed {s} =====", flush=True)
    pt.main()
    shutil.copy(MODELS / "polarscan.pt", MODELS / f"polarscan_seed{s}.pt")
