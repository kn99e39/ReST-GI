"""WL104K-C shared helpers. Re-uses the WL104K-B modules (common104kb, fit104kb, eval104kb, data104kb) unchanged.

Owner instruction 2026-11-24: this batch uses at most 50 % of the GPU. GPU time is limited externally by gpu_duty.ps1 (duty-cycle
suspension of every python.exe whose command line contains 'ecro_wl84_104kc'); device memory is limited here with
torch.cuda.set_per_process_memory_fraction(0.5). Neither changes any numerical operation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KB_DIR = HERE.parent / "ecro_wl84_104kb"
sys.path.insert(0, str(KB_DIR))
import common104kb as KB  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5)

CM, IO, GF, FT, EV, KA = KB.CM, KB.IO, KB.GF, KB.FT, KB.EV, KB.KA
DEV = KB.DEV
REPO = KB.REPO
OUT = REPO / "results/ecro_wl84_104kc"
PROTOCOL = HERE / "protocol.json"
PROTOCOL_A = HERE / "protocol_stageA.json"
P = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else {}
PA = json.loads(PROTOCOL_A.read_text(encoding="utf-8")) if PROTOCOL_A.exists() else {}
KB_ROOT = KB.OUT / "primary"                     # WL104K-B run root (read-only)
K_FIT = REPO / "results/ecro_wl84_104k/primary/fit_primary"   # WL104K archived A / P fits (read-only)
NAMES = KB.NAMES
LUMA = KB.LUMA
Stop = IO.Stop
N_MAT = 9


def sha(p):
    return KB.sha(p)


def write(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    KB.write(p, o)


def read(p):
    return KB.read(p)


def commit():
    return KB.commit()


def kb_manifest_raw():
    """WL104K-B evaluation manifest raw_sha256 (paths relative to results/ecro_wl84_104kb)."""
    return IO.read(REPO / "evaluation/104K-B/manifest.json").get("raw_sha256", {})


def authenticate_kb(rels):
    """Authenticate WL104K-B raw files (relative to results/ecro_wl84_104kb) against evaluation/104K-B/manifest.json."""
    raw = kb_manifest_raw()
    seen = {}
    for rel in rels:
        if rel not in raw:
            raise Stop(f"NO_FROZEN_HASH: WL104K-B {rel}")
        h = sha(KB.OUT / rel)
        if h != raw[rel]:
            raise Stop(f"FROZEN_HASH_MISMATCH: WL104K-B {rel}")
        seen["results/ecro_wl84_104kb/" + rel] = h
    return seen


def load_states(path, R, n=N_MAT):
    """states.npz (start-major [R * n, ...]) -> list over starts of {'y': [n, 58], 'g': [n, 8, 3, 5]}."""
    st, _ = EV.load_states(Path(path), R, n)
    return st["P"]


def flat_x(states):
    """list over starts of {'y','g'} -> x [R * n, 178] float64 (start-major)."""
    return np.concatenate([np.concatenate([s["y"], np.asarray(s["g"]).reshape(len(s["y"]), -1)], 1) for s in states])


def archived_P(n=N_MAT):
    """WL104K archived P 12k states (4 starts), via common104ka.states (WL104K manifest authentication)."""
    seen = KA.authenticate_k()
    hist, R, fj = KA.states(K_FIT, n)
    return hist["P"], fj, seen
