"""Run an UNCHANGED WL104K-B script (experiments/ecro_wl84_104kb/<script>) under the WL104K-C process name, so that the
WL104K-C GPU duty-cycle supervisor (gpu_duty.ps1, matching 'ecro_wl84_104kc') covers it. The torch caching allocator is capped
at 50 % of the device memory (owner instruction 2026-11-24); this changes no numerical operation.

  python run_kb.py <script.py> [args ...]
"""
import runpy
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "ecro_wl84_104kb"
script = KB_DIR / sys.argv[1]
sys.path.insert(0, str(KB_DIR))
sys.argv = [str(script)] + sys.argv[2:]
import torch  # noqa: E402
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5)
runpy.run_path(str(script), run_name="__main__")
