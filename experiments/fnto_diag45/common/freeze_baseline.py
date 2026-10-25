"""WL45 - freeze record of the immutable diagnostic baseline (read-only; sha256 of every input both tracks use).

    python freeze_baseline.py <repo root> <out.json>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(root, out):
    root = Path(root); r1 = root / "results/fnto_lego_r1"
    files = {
        "K_k44P/kernels.pt": r1 / "K_k44P/kernels.pt", "K_r1P/kernels.pt": r1 / "K_r1P/kernels.pt",
        "anchors.npz (a1)": r1 / "anchors.npz", "rel_T0.npz": r1 / "rel_T0.npz", "rel_T1.npz": r1 / "rel_T1.npz", "rel_T1_inc.npz": r1 / "rel_T1_inc.npz",
        "eval_px/heldout_T0.npz": r1 / "eval_px/heldout_T0.npz", "eval_px/heldout_T1.npz": r1 / "eval_px/heldout_T1.npz",
        "eval_vis_k44.npz": r1 / "eval_vis_k44.npz", "eval_vis.npz": r1 / "eval_vis.npz",
        "refs/heldout_T0_first.npz": r1 / "refs/heldout_T0_first.npz", "refs/heldout_T1_first.npz": r1 / "refs/heldout_T1_first.npz",
        "refs/heldout_T0_full.npz": r1 / "refs/heldout_T0_full.npz", "refs/heldout_T1_full.npz": r1 / "refs/heldout_T1_full.npz",
        "eval_K_k44P/views.npz": r1 / "eval_K_k44P/views.npz", "eval_K_r1P/views.npz": r1 / "eval_K_r1P/views.npz",
        "prep2_holdout/meta.json": r1 / "stagea/prep2_holdout/meta.json", "prep2_train/meta.json": r1 / "stagea/prep2_train/meta.json",
    }
    rec = {"files": {}, "missing": []}
    for k, p in files.items():
        if p.exists():
            rec["files"][k] = {"sha256": sha(p), "bytes": p.stat().st_size}
        else:
            rec["missing"].append(k)
    try:
        rec["git_head"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # server copy is not a git checkout
        rec["git_head"] = f"n/a ({type(e).__name__})"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v["sha256"][:16] for k, v in rec["files"].items()}, indent=1), rec["missing"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
