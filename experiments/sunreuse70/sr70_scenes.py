"""WL70 states: the frozen WL69 C1 pair (T0 floor, T1 floor_c4) and the prospectively fixed diagnostic invalidation / acceptance controls.
Controls are isolated variants of T1 used only for the sun-state contract and its physical identity test (never benchmark scenes):
  N1_sun      sun zenith 40 -> 45 deg (azimuth 20 unchanged)                    expected: reject (D4)
  N2_occluder black 2 x 2 x 0.2 cm box 3 cm above the slab top on the sun path    expected: reject (D6)
  N3_medium   albedo blue 0.9 -> 0.8 (sigma_t, g unchanged)                      expected: reject (D2)
  N4_rotate   slab rotated 90 deg about the vertical axis through its centre    expected: reject (D3/D4: world boundary identical, material
              (the square slab occupies the same world volume)                  frame differs)
  P1_offpath  the same black box 12 cm downstream of the slab, 3 cm above it     expected: accept (no change of the incident field)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55"))
import k55_common as KCm  # noqa: E402,F401
import k55_fixtures as KF  # noqa: E402
import p53_fixtures as FX  # noqa: E402

CONTROLS = ("N1_sun", "N2_occluder", "N3_medium", "N4_rotate", "P1_offpath")
EXPECT = dict(T1="accept", N1_sun="reject", N2_occluder="reject", N3_medium="reject", N4_rotate="reject", P1_offpath="accept")
MEDIUM = (FX.SIGMA_T, tuple(FX.ALBEDO), FX.G_PHASE)


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def box(sd, centre, scale=(0.2, 0.2, 0.2)):
    """Black box = the slab rest mesh scaled about its centre and placed at `centre` (geometry for the shadow test only)."""
    m = sd["comps"][0][2]; c = 0.5 * (m["V"].min(0) + m["V"].max(0))
    return ("blocker", "black", m, (m["V"] - c) * np.asarray(scale) + np.asarray(centre, float))


def state(name):
    """-> (scene dict, medium tuple)."""
    if name == "T0":
        return KF.scene_def("floor"), MEDIUM
    sd = KF.scene_def("floor_c4"); n, k, m, V = sd["comps"][0]; s, E = sd["lights"][0]
    ctr = 0.5 * (V.min(0) + V.max(0)); top = np.array([ctr[0], ctr[1], V[:, 2].max()])
    if name == "T1":
        return sd, MEDIUM
    if name == "N1_sun":
        sd["lights"] = [(KF.sun_s(45.0, 20.0), E)]; return sd, MEDIUM
    if name == "N2_occluder":
        up = -unit(s); sd["comps"].append(box(sd, top + up * (0.03 / up[2]))); return sd, MEDIUM
    if name == "N3_medium":
        return sd, (FX.SIGMA_T, (FX.ALBEDO[0], FX.ALBEDO[1], 0.8), FX.G_PHASE)
    if name == "N4_rotate":
        c, s9 = 0.0, 1.0; Rz = np.array([[c, -s9, 0.0], [s9, c, 0.0], [0.0, 0.0, 1.0]])
        sd["comps"][0] = (n, k, m, (V - ctr) @ Rz.T + ctr); return sd, MEDIUM
    if name == "P1_offpath":
        h = unit([s[0], s[1], 0.0]); sd["comps"].append(box(sd, top + 0.12 * h + np.array([0.0, 0.0, 0.03]))); return sd, MEDIUM
    raise KeyError(name)
