"""Immutable experiment paths and numerical helpers."""
from importlib import import_module
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for directory in ("ecro_denoise105", "ecro_denoise107"):
    sys.path.insert(0, str(HERE.parent / directory))

from common107 import read, sha, write, SPPS, VIEWS  # noqa: E402,F401

PROTOCOL = HERE / "protocol108.json"
P = read(PROTOCOL)
ARMS = tuple(P["arms"])
SEEDS = P["training"]["train_seeds"]


def assessment107():
    """LabServer63 retains the literal approved Amendment 1 evaluator as a sibling."""
    module = HERE.parent / "ecro_denoise107" / "real107_amend1.py"
    return import_module(module.stem if module.is_file() else "real107")
