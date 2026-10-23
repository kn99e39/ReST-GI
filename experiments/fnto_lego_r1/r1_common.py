"""FNTO r1 (worklog 42, validity-restoration candidate) shared definitions. Extends fnto_common (Batch 6, unchanged).

Corrections relative to Batch 6 (audit findings in brackets):
  ports      on-surface origins, clusters split by connected component and a 3 mm plane offset along the dominant
             axis, S_PORT surface samples per port for visibility and gathers                      [F1 / audit #1]
  support    stratified representatives over the FULL declared radius with represented-area weights [F3 / #4]
  Stage A    intrinsic-all paths (internal reflection / TIR kept, free-space segments absorbed), point-sampled pixels,
             no sample clamping, spot / point / multi-light / sun probe families                    [F4, F5, F2 / #2 #5 #8 #9]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego"))
from fnto_common import *  # noqa: E402,F401,F403
import fnto_common as C0  # noqa: E402

OUT1 = C0.ROOT / "results/fnto_lego_r1"
H_PLANE = 0.003          # plane-offset quantisation along the dominant normal axis [m]
S_PORT = 4               # surface samples per port (visibility average, gather origins)
EPS1 = 1e-4              # ray offset from an ON-SURFACE sample along its own triangle normal [m]
H_MID1 = 0.08            # coarse source ports of the mid-range kernel (r1) [m]
K_BAND = 32              # band representatives over the full R_LOCAL ball
K_MID = 64               # mid representatives over the full R_MID ball
