"""Focused analytic / synthetic tests of the WL101I contract-v2 gates. No renderer.

python tests101i.py   (from experiments/ecro_sss101i; needs experiments/ecro86 for the real geometry)
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ecro86"))
sys.path.insert(0, str(HERE.parent / "ecro_sss101a"))
import gates101i as Q  # noqa: E402

KEYS3 = ("L", "LL", "vL", "gL")
KEYS = ("n", "c", "v", "g", "k_singular", "k_ring", "k_cap", "k_junction", "k_inside_box", "j_L", "j_LL", "j_v", "j_g",
        "sh_n", "sh_L", "sh_LL", "sh_v", "sh_vL", "sh_g", "sh_gL", "sh_gLL",
        *[f"{p}{t}" for t in ("10", "100") for p in ("nf", "dg", "dgL", "dg2_", "dgL2_", "dgdgL", "dv", "dvL", "dvL2_", "nfL")])


def real_geom():
    import scene86 as S
    from common import P as A
    g = S.Geom(A["geometry"])
    return {"psid": g.PSID, "parea": g.PAREA, "po": g.PO, "pe1": g.PE1, "pe2": g.PE2,
            "l1": np.linalg.norm(g.PE1, axis=1), "l2": np.linalg.norm(g.PE2, axis=1), "psph_iz": g.PSPH[:, 1]}, A


def small_geom(n_patch=2):
    return {"psid": np.zeros(n_patch, int), "parea": np.full(n_patch, 0.0625), "l1": np.full(n_patch, .25), "l2": np.full(n_patch, .25)}


def packet(n_patch=2, per_cell=256, L=0.5, cos=0.5):
    slots = n_patch * 64
    pk = {k: np.zeros((slots, 4, 4)) for k in KEYS}
    for k in KEYS3:
        pk[k] = np.zeros((slots, 4, 4, 3))
    pk["n"][:] = per_cell; pk["v"][:] = per_cell; pk["g"][:] = per_cell * cos; pk["c"][:] = per_cell * cos
    pk["L"][:] = per_cell * L; pk["LL"][:] = per_cell * L * L; pk["vL"][:] = per_cell * L; pk["gL"][:] = per_cell * cos * L
    sh = per_cell * 0.15
    pk.update(sh_n=np.full((slots, 4, 4), sh), sh_L=np.full((slots, 4, 4), sh * L), sh_LL=np.full((slots, 4, 4), sh * L * L),
              sh_v=np.full((slots, 4, 4), sh), sh_vL=np.full((slots, 4, 4), sh * L), sh_g=np.full((slots, 4, 4), sh * cos),
              sh_gL=np.full((slots, 4, 4), sh * cos * L), sh_gLL=np.full((slots, 4, 4), sh * cos * L * L))
    pk["slot_tie"] = np.zeros(slots); pk["slot_boundary_miss"] = np.zeros(slots)
    return pk


def test_singular_geometry():
    geom, _ = real_geom()
    h = 2.1241822779460237e-07
    sing = Q.singular_fractions(geom, h)
    r_s2 = 4 * 0.4 * h - 4 * h * h
    assert abs(sing["r_s_m"] ** 2 - r_s2) <= 1e-12 * r_s2
    pid, s = sing["floor_pid"], sing["floor_subcell"]
    assert pid == 155 and s == 3
    assert abs(sing["f_sub"][pid * 64, s] - math.pi * r_s2 / (geom["parea"][pid] / 4)) < 1e-15
    # The cap measure distributed over the sphere's bottom subcells integrates back to its area.
    area = sum(sing["f_sub"][p * 64, k] * geom["parea"][p] / 4 for p in np.flatnonzero(geom["psid"] == 6) for k in range(4))
    assert abs(area - 2 * math.pi * 0.4 * 2 * h) < 1e-15
    return {"r_s_m": sing["r_s_m"], "floor_subcell_fraction": float(sing["f_sub"][pid * 64, s])}


def test_s3(rs):
    geom, _ = real_geom()
    h = 2.1241822779460237e-07
    sing = Q.singular_fractions(geom, h)
    slots = len(geom["psid"]) * 64
    pk = {"n": np.full((slots, 4, 4), 256.0), "k_singular": np.zeros((slots, 4, 4)),
          "slot_tie": np.zeros(slots), "slot_boundary_miss": np.zeros(slots)}
    expected = 1024 * sing["f_sub"]
    draw = rs.poisson(expected)
    pk["k_singular"][:, :, 0] = draw
    ok, _, _ = Q.s3([pk], geom, h, 3.8e-6, 4096)
    bad = {k: v.copy() for k, v in pk.items()}
    bad["k_singular"][10, 0, 0] += 1                 # singular sample in a cell with no registered measure
    fail_unreg, _, _ = Q.s3([bad], geom, h, 3.8e-6, 4096)
    band = {k: v.copy() for k, v in pk.items()}
    band["slot_tie"][777] = 60                         # far above the lattice-band expectation (~0.3)
    fail_band, _, _ = Q.s3([band], geom, h, 3.8e-6, 4096)
    assert ok["pass"] and not fail_unreg["pass"] and not fail_band["pass"]
    return {"consistent": ok["pass"], "unregistered_detected": not fail_unreg["pass"], "band_excess_detected": not fail_band["pass"]}


def sing_zero(n_patch=2):
    return {"f_slot": np.zeros(n_patch * 64), "f_sub": np.zeros((n_patch * 64, 4))}


def test_s4_silhouette():
    geom = small_geom()
    z = 5.0
    proxy = {"L": 0.1745, "model": 0.01}
    pk = packet()
    # Slot 0: 200 flips at m/10 add visible weight (cos 0.5, L = 0.9); a tenth more at m/100 (linear model).
    k = 200
    pk["dg10"][0, 0, 0], pk["dgL10"][0, 0, 0] = k * 0.5, k * 0.5 * 0.9
    pk["dg100"][0, 0, 0], pk["dgL100"][0, 0, 0] = 1.1 * k * 0.5, 1.1 * k * 0.5 * 0.9
    pk["dg2_10"][0, 0, 0], pk["dgL2_10"][0, 0, 0], pk["dgdgL10"][0, 0, 0] = k * 0.25, k * (0.5 * 0.9) ** 2, k * 0.25 * 0.9
    # Slot 1: one tiny flip of weight 1e-4.
    pk["dg10"][1, 0, 0], pk["dgL10"][1, 0, 0] = 1e-4, 1e-4 * 0.5
    pk["dg100"][1, 0, 0], pk["dgL100"][1, 0, 0] = 1.1e-4, 1.1e-4 * 0.5
    res = Q.s4(pk, geom, sing_zero(), np.zeros(128), np.zeros(128), 4096, proxy, z, "slot", "native")["G"]
    g, gL = 256 * 16 * 0.5, 256 * 16 * 0.5 * 0.5
    truth = -((gL + k * 0.45) / (g + k * 0.5) - 0.5) / 0.9
    assert abs(res["b"][0] - truth) < 1e-12
    assert res["state"][0] == -1, res["state"][0]       # material and resolved -> FAIL
    assert res["state"][1] == 1                          # negligible -> PASS
    # UNRESOLVED: a moderate effect with large uncertainty.
    pk2 = packet()
    pk2["dg10"][0, 0, 0], pk2["dgL10"][0, 0, 0] = 10 * 0.5, 10 * 0.5 * 0.9
    pk2["dg100"][0, 0, 0], pk2["dgL100"][0, 0, 0] = 11 * 0.5, 11 * 0.5 * 0.9
    pk2["dg2_10"][0, 0, 0], pk2["dgL2_10"][0, 0, 0], pk2["dgdgL10"][0, 0, 0] = 10 * 0.25, 10 * 0.45 ** 2, 10 * 0.25 * 0.9
    r2 = Q.s4(pk2, geom, sing_zero(), np.zeros(128), np.zeros(128), 4096, proxy, z, "slot", "native")["G"]
    assert r2["state"][0] == 2, (r2["b"][0], r2["se"][0], r2["B"][0], r2["theta"][0])
    return {"silhouette_recovered": float(res["b"][0]), "fail": True, "pass": True, "unresolved": True}


def test_s4_singular_positive_control():
    geom = small_geom()
    pk = packet(L=0.26)
    sing = sing_zero()
    sing["f_sub"][:64, 3] = 0.032
    sing["f_slot"][:64] = 0.008
    res = Q.s4(pk, geom, sing, np.zeros(128), np.zeros(128), 4096, {"L": 0.1745, "model": 0.01}, 5.0, "sub", "candidate")["S_G"]
    assert np.all(res["state"][:64, 3] == -1)
    assert np.all(res["state"][64:, :] == 1)
    small = sing_zero(); small["f_sub"][:64, 3] = 6.8e-5
    ok = Q.s4(pk, geom, small, np.zeros(128), np.zeros(128), 4096, {"L": 0.1745, "model": 0.01}, 5.0, "sub", "candidate")["S_G"]
    assert np.all(ok["state"][:64, 3] == 1)
    return {"expanded_singular_fails": True, "frozen_singular_passes": True}


def test_s5():
    a = packet(); b = {k: v.copy() for k, v in a.items()}
    same = Q.s5([a], [b], 5.0)
    b["n"][0, 0, 0] -= 1; b["n"][0, 1, 0] += 1          # one sample moves across a spatial border
    b["L"][0, 0, 0] -= 0.9; b["L"][0, 1, 0] += 0.9
    b["g"][0, 0, 0] -= 0.5; b["g"][0, 1, 0] += 0.5; b["gL"][0, 0, 0] -= 0.45; b["gL"][0, 1, 0] += 0.45
    moved = Q.s5([a], [b], 5.0)
    assert same["pass"] and moved["rows"][0]["moved_samples"] == 1 and moved["rows"][0]["slot_n_identical"]
    return {"identical_pass": True, "moved_detected": moved["rows"][0]["moved_across_spatial_subcells"] == 1,
            "max_abs_S_G_change": moved["rows"][0]["max_abs_S_G_change"]}


def run_all(seed=101909):
    rs = np.random.default_rng(seed)
    return {"singular_geometry": test_singular_geometry(), "s3": test_s3(rs), "s4_silhouette": test_s4_silhouette(),
            "s4_singular_positive_control": test_s4_singular_positive_control(), "s5": test_s5(), "seed": seed}


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1))
