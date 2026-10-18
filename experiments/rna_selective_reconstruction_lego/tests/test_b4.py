"""Contract tests for Batch 4 (RNA venv; GPU for the model fixtures). Contracts only; never architecture evidence.

    python tests/test_b4.py --out <dir> [--h5-T0 <oracle_T0_s1_val.h5> --h5-T1 <oracle_T1_s1_val.h5>]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership_lego"))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_compose as C  # noqa: E402
import b4_model as E  # noqa: E402
import b4_oracle as O  # noqa: E402
import rso_model as M  # noqa: E402

BASE = {"resolution": [64, 64], "triplane_features_num_channels": 8, "triplane_features_min_sigma": 1.0,
        "initial_sigma": 8.0, "iterations_to_sigma_1": 20000, "mlp_num_hidden_layers": 2, "mlp_hidden_layer_neurons": 32}


def sample_surface(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nrm = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(nrm, axis=1)
    t = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n); fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    return a[t] + (b[t] - a[t]) * u[:, None] + (c[t] - a[t]) * v[:, None], nrm[t] / np.linalg.norm(nrm[t], axis=1, keepdims=True)


def t_identity_rigid_translation():
    """The same physical points (canonical samples on A and on B) rendered in T0, T1 and T1b resolve to the same unit
    and the same canonical position in every state; A units = nearest A centre of the canonical point (never B's);
    B points -> unit 16; on-surface distance within the worklog-36 tolerance; deterministic."""
    rng = np.random.default_rng(11)
    m = np.load(E.ROOT / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz"); bm = np.load(E.ROOT / "results/rna_selective_reconstruction_lego/scene/B_box_T0.npz")
    xa, na = sample_surface(m["V"].astype(np.float64), m["F"], 20000, rng)
    xb, nb = sample_surface(bm["V"].astype(np.float64), bm["F"], 5000, rng)
    x, n = np.concatenate([xa, xb]), np.concatenate([na, nb]); isB = np.r_[np.zeros(len(xa), bool), np.ones(len(xb), bool)]
    cent = np.load(E.ROOT / "results/rna_surface_ownership_lego/partition_L16.npz")["centres"]
    cellA = ((xa[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
    out = {}
    labs = {}
    for s in ("T0", "T1", "T1b"):
        own = E.make_owner(s); aoff, boff = E.state_offsets(s)
        w = x + np.where(isB[:, None], boff[None], aoff[None])
        lab, d = own(w, n); lab2, _ = own(w, n)
        can = own.canonical(w, lab)
        labs[s] = lab
        out[s] = {"dist_q999": float(np.quantile(d, 0.999)), "A_cell_agreement": float((lab[~isB] == cellA).mean()), "B_all_unit16": bool((lab[isB] == 16).all()),
                  "A_never_16": bool((lab[~isB] != 16).all()), "canonical_max_err": float(np.abs(can - x).max()), "deterministic": bool(np.array_equal(lab, lab2))}
    same = bool(np.array_equal(labs["T0"], labs["T1"]) and np.array_equal(labs["T0"], labs["T1b"]))
    ok = same and all(r["dist_q999"] <= 1.125e-3 and r["A_cell_agreement"] > 0.999 and r["B_all_unit16"] and r["A_never_16"]
                      and r["canonical_max_err"] < 1e-9 and r["deterministic"] for r in out.values())
    return {"pass": ok, "same_unit_in_all_states": same, **out}


def t_oracle_estimator_and_noise():
    """Known change fields under independent noise: the cross-product estimator recovers the RMS change (unit 0:
    0.010, unit 1: 0, unit 2: 0.004), the naive estimate is biased by noise, results are deterministic; the 3x3
    visibility-uniformity mask removes shadow-edge pixels."""
    rng = np.random.default_rng(5)
    n = 300000; unit = rng.integers(0, 3, n); cnt = np.ones(n, bool)
    true = np.zeros((n, 3)); true[unit == 0] = 0.010 * rng.choice([-1, 1], ((unit == 0).sum(), 3)); true[unit == 2] = 0.004 * rng.choice([-1, 1], ((unit == 2).sum(), 3))
    sig = 0.02
    res = []
    for _ in range(2):
        r2 = np.random.default_rng(9)
        d1 = true + r2.normal(0, sig, (n, 3)); d2 = true + r2.normal(0, sig, (n, 3))
        S, NO, NA, NP = (np.zeros(3) for _ in range(4))
        O.accumulate(d1, d2, unit, cnt, S, NO, NA, NP)
        res.append((np.sqrt(np.clip(S / NP, 0, None)), np.sqrt(NA / NP), np.sqrt(NO / NP)))
    rms, naive, noise = res[0]
    vis = np.zeros((32, 32), bool); vis[:, 16:] = True
    u3 = O.uniform3(vis.ravel(), 32).reshape(32, 32)
    edge_ok = (not u3[:, 15:17].any()) and u3[5:-5, 2:14].all() and u3[5:-5, 18:30].all()
    ok = abs(rms[0] - 0.010) < 0.0008 and rms[1] < 0.0015 and abs(rms[2] - 0.004) < 0.0012 and naive[1] > 0.01 and \
        all(np.array_equal(a, b) for a, b in zip(res[0], res[1])) and edge_ok
    return {"pass": bool(ok), "rms": rms.tolist(), "naive": naive.tolist(), "noise_of_mean": noise.tolist(), "shadow_edge_removed": bool(edge_ok)}


def fixture(seed=0):
    """4 units on a strip: units 0-2 = object A (one component, x in [-1, 0.5]), unit 3 = object B (own component)
    whose canonical frame is x in [0.6, 1.0]; in 'T1' B is translated by +0.5."""
    import torch

    specs = []
    for k in range(3):
        lo = np.array([-1 + 0.5 * k, -0.1, -0.5]); specs.append({"params": dict(BASE), "lo": lo - 0.02, "hi": lo + np.array([0.5, 0.2, 1.0]) + 0.02})
    specs.append({"params": dict(BASE), "lo": np.array([0.58, -0.12, -0.52]), "hi": np.array([1.02, 0.12, 0.52])})
    torch.manual_seed(seed)
    dmin, dmax = (-1.0, -0.1, -0.5), (1.6, 0.1, 0.5)
    m = M.OwnedTriplaneSet(specs, (dmin, dmax)).cuda().eval(); m.set_iteration_index(10 ** 9)
    centres = np.array([[-0.75, 0, 0], [-0.25, 0, 0], [0.25, 0, 0], [0.8, 0, 0]])
    xs = np.random.default_rng(seed).uniform(-1, 1, (6000, 3)); lab = np.clip(((xs[:, 0] + 1) / 0.5).astype(int), 0, 3)
    comp = np.where(lab == 3, 1, 0)
    return m, C.LegoPUComposer(m, centres, lab, comp, 0.05), np.array(dmin), np.array(dmax)


def queries(n, seed, obj_b=False):
    rng = np.random.default_rng(seed)
    lo, hi = ((0.6, -0.1, -0.5), (1.0, 0.1, 0.5)) if obj_b else ((-1, -0.1, -0.5), (0.5, 0.1, 0.5))
    p = rng.uniform(lo, hi, (n, 3)); p[:, 1] = 0
    d = rng.normal(size=(3, n, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return p, d[0].astype(np.float32), d[1].astype(np.float32), d[2].astype(np.float32)


def t_coordinate_stability_reconstruction_and_influence():
    """(a) a frozen unit's prediction for a B point is identical in T0 and T1 up to float rounding of the canonical
    shift (<= 1e-6; no frame drift), while the uncorrected world position would change it;
    (b) re-initialising and training D = {1, 3} with the others frozen leaves units 0, 2 bit-identical; (c) the composed
    output changes exactly on S_1 u B_1 u S_3 u B_3 (B_3 empty: B has its own component), nowhere else."""
    import torch

    m, comp_, dmin, dmax = fixture(seed=3)
    pa, ca, la, na = queries(20000, 1); pb, cb, lb, nb = queries(4000, 2, obj_b=True)
    p = np.concatenate([pa, pb]); c, l, n = (np.concatenate(x) for x in ((ca, cb), (la, lb), (na, nb)))
    own = np.clip(((p[:, 0] + 1) / 0.5).astype(int), 0, 2); own[len(pa):] = 3
    cp = np.where(own == 3, 1, 0); vis = np.ones(len(p), bool)
    delta = np.array([0.5, 0, 0])
    world_T1 = p + np.where((own == 3)[:, None], delta[None], 0)
    canon_T1 = world_T1 - np.where((own == 3)[:, None], delta[None], 0)
    f0 = comp_(p.astype(np.float32), c, l, n, vis, own, cp, p)
    f1 = comp_(canon_T1.astype(np.float32), c, l, n, vis, own, cp, canon_T1)
    stable_err = float(np.abs(f0 - f1).max()); stable = stable_err <= 1e-6  # canonical = (p + delta) - delta: equal up to float rounding
    shifted_wrong = comp_(world_T1.astype(np.float32), c, l, n, vis, own, cp, world_T1)
    drift_detectable = bool(not np.array_equal(f0[own == 3], shifted_wrong[own == 3]))
    sd0 = copy.deepcopy(m.state_dict()); D = [1, 3]
    for k in D:
        m.units[k].apply(M.rmodels.init_gaussian)
    for k, u in enumerate(m.units):
        u.requires_grad_(k in D)
    opt = torch.optim.Adam([x for x in m.parameters() if x.requires_grad], lr=1e-2); m.train()
    for it in range(8):
        sel = np.random.default_rng(100 + it).choice(len(p), 6000, replace=False)
        pos = (p[sel] - dmin) / (dmax - dmin)
        Tt = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda").unsqueeze(0)
        loss = m(Tt(np.concatenate([pos, own[sel, None]], 1)), Tt(c[sel]), Tt(l[sel]), Tt(n[sel])).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); m.set_iteration_index(10 ** 9)
    frozen = all(torch.equal(m.state_dict()[k], sd0[k]) for k in sd0 if k.startswith("units.") and int(k.split(".")[1]) not in D)
    after = comp_(p.astype(np.float32), c, l, n, vis, own, cp, p)
    changed = (after != f0).any(1)
    infl = np.zeros(len(p), bool)
    for k in D:
        infl |= comp_.influenced_by(k, own, cp, p)
    exact = bool(np.array_equal(changed, infl)); band_nbr = int((infl & ~np.isin(own, D)).sum())
    return {"pass": stable and drift_detectable and frozen and exact and band_nbr > 0, "frozen_B_T0_T1_max_abs_diff": stable_err,
            "uncorrected_world_would_differ": drift_detectable, "valid_units_bit_identical": frozen, "changed_equals_union_of_influence": exact,
            "neighbour_band_points": band_nbr, "influenced": int(infl.sum())}


def t_dataset_adapter(h5_T0, h5_T1):
    """Real oracle data: T0/T1 dataset AABBs are equal (no normalisation drift); in T1 every B pixel's canonical position
    lies on B's T0 box surface and A pixels are not shifted; owner counts include unit 16."""
    import h5py

    with h5py.File(h5_T0, "r") as f0, h5py.File(h5_T1, "r") as f1:
        same_aabb = bool(np.array_equal(f0.attrs["aabb_min"], f1.attrs["aabb_min"]) and np.array_equal(f0.attrs["aabb_max"], f1.attrs["aabb_max"]))
        own = E.make_owner("T1")
        lo, hi = map(np.array, E.SCENE["B_mesh"]["aabb_T0"])
        errs, nB, shiftA = [], 0, 0
        for v in range(4):
            a = f1["alpha"][v][:, 0] >= 0.999
            P = f1["position"][v][a].astype(np.float64); N = f1["normal"][v][a].astype(np.float64)
            lab, _ = own(P, N); can = own.canonical(P, lab); b = lab == 16; nB += int(b.sum())
            q = can[b]; inside = np.clip(q, lo, hi); d_out = np.linalg.norm(q - inside, axis=1)
            d_in = np.min(np.stack([q - lo, hi - q], -1), axis=(-1)).min(-1)
            errs.append(np.where(d_out > 0, d_out, np.abs(d_in))); shiftA += int((np.abs(can[~b] - P[~b]) > 0).any(1).sum())
        e = np.concatenate(errs) if errs else np.zeros(1)
    ok = same_aabb and nB > 0 and float(np.quantile(e, 0.999)) < 2e-3 and shiftA == 0
    return {"pass": ok, "T0_T1_aabb_equal": same_aabb, "B_pixels": nB, "B_canonical_dist_to_T0_box_surface_q999": float(np.quantile(e, 0.999)), "A_pixels_shifted": shiftA}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--h5-T0", default=None); ap.add_argument("--h5-T1", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    tests = {"identity_rigid_translation": t_identity_rigid_translation, "oracle_estimator_and_noise": t_oracle_estimator_and_noise,
             "coordinate_stability_reconstruction_and_influence": t_coordinate_stability_reconstruction_and_influence}
    if args.h5_T0:
        tests["dataset_adapter"] = lambda: t_dataset_adapter(args.h5_T0, args.h5_T1)
    res = {}
    for name, fn in tests.items():
        try:
            r = fn()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        res[name] = r
        print(name, "PASS" if r["pass"] else "FAIL", {k: v for k, v in r.items() if k not in ("pass", "error")}, r.get("error", ""), flush=True)
    (out / "tests.json").write_text(json.dumps(res, indent=1, default=str))
    return 0 if all(r["pass"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
