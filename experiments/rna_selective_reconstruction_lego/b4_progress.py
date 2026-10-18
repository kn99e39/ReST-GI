"""Count rendered views in b4 h5 files (a view is written once its beauty render finishes)."""
import sys, h5py, numpy as np
for p in sys.argv[1:]:
    try:
        with h5py.File(p, "r") as f:
            c = f["color"]; n = c.shape[0]
            done = sum(1 for i in range(n) if np.abs(np.asarray(c[i][::4099], np.float32)).max() > 0)
            print(p.split("/")[-1], f"{done}/{n}")
    except Exception as e:
        print(p.split("/")[-1], "busy/unreadable", type(e).__name__)
