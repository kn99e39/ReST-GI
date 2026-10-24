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

HERE1 = Path(__file__).resolve().parent   # this directory (HERE is fnto_lego's after the star import)
OUT1 = C0.ROOT / "results/fnto_lego_r1"
H_PLANE = 0.003          # plane-offset quantisation along the dominant normal axis [m]
S_PORT = 4               # surface samples per port (visibility average, gather origins)
EPS1 = 1e-4              # ray offset from an ON-SURFACE sample along its own triangle normal [m]
H_MID1 = 0.08            # coarse source ports of the mid-range kernel (r1) [m]
H_BAND1 = 0.015          # band source ports: 1.5 cm aggregates of fine ports (area-mean incident) [m]
K_BAND_A0 = 48           # a0 band representatives over the full R_LOCAL ball (r1_anchors; failed Gate 2b)
K_MID_A0 = 64            # a0 mid representatives over the full R_MID ball
K_BAND = 96              # amendment 1 (r1_reps_a1): near-exact + orientation-stratified area-proportional picks
K_MID = 128
