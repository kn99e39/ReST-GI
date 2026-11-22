"""WL104K-A shared helpers: authenticated WL104K inputs and re-used WL104K modules (nothing in WL104K is modified)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104k"))
import common104k as CM  # noqa: E402
import fit104k as FT  # noqa: E402
import eval104k as EV  # noqa: E402

IO, GF, CL = CM.IO, CM.GF, CM.CL
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
REPO = CM.REPO
K_ROOT = REPO / "results/ecro_wl84_104k/primary"
OUT = REPO / "results/ecro_wl84_104ka"
SCENES = tuple(P["stage_C"]["scenes"])
Stop = IO.Stop


def authenticate_k():
    """WL104K raw files against evaluation/104K/manifest.json raw_sha256."""
    man = IO.read(REPO / "evaluation/104K/manifest.json")["raw_sha256"]
    seen = {}
    for rel in ("primary/fit_primary/states.npz", "primary/fit_primary/fit.json", "primary/fit_extended/states.npz",
                "primary/eval/eval.json", "primary/eval/lanes.npz", "primary/decision.json"):
        p = REPO / "results/ecro_wl84_104k" / rel
        if CM.sha(p) != man[rel]:
            raise Stop(f"FROZEN_HASH_MISMATCH: {rel}")
        seen["results/ecro_wl84_104k/" + rel] = man[rel]
    return seen


def states(fit_dir, n):
    """{arm: [state per start]} with arrays per material, from a WL104K-format fit dir."""
    fj = CM.read(fit_dir / "fit.json")
    R = len(fj["start_keys"])
    st, _ = EV.load_states(fit_dir / "states.npz", R, n)
    return st, R, fj


def write(p, o):
    CM.write(p, o)


def read(p):
    return CM.read(p)
