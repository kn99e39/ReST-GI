"""WL104K-B focused contracts. Each mode writes <out.json> with per-test {passed, values} and a top-level 'passed';
the process exits with status 1 after writing when any test fails (CONTRACT_TESTS_FAILED).

  python tests104kb.py <out.json>                                   (a) pre-fit contracts: response identity vs WL104K, starts,
                                                                        E2 objective (numpy + finite differences), O1 L-BFGS
                                                                        (real subset + synthetic box QP + freezing), O0 Adam path
                                                                        vs fit104k.train, synthetic truth
  python tests104kb.py <out.json> --fitted <fit_dir> [--fitted ...]  (b) fitted states: shape (178 / material), gates in [0, 1],
                                                                        finite, reciprocity, pointwise envelope, MC albedo; fit.json
                                                                        (when present): states hash, budget, 178 parameters
  python tests104kb.py <out.json> --data <root>                     (c) layout (L, P = 20, peak K = 305), qualified non-smoke
                                                                        data.json, hashes, isolation of <root>/data/*.npz
  python tests104kb.py <out.json> --leak <root> [--eval-script F]   (d) eval104kb.py test refuses without the hashed dev decision,
                       [--leak-static-only] [--timeout S]               with a tampered dev hash / O1 states / missing eligible F
                                                                        (LEAK_GUARD); its dev path never opens data/test*.npz,
                                                                        scenes*, test_eval or decision.json and reproduces the
                                                                        dev decision (stage_F, selections, fits_sha256)

Seeds: protocol seeds.tests (random states, directions) and seeds.albedo_mc (MC albedo). In fitted mode every (arm directory,
start r) has its own streams default_rng([seed, crc32(arm_dir_name), r]), independent of the order of --fitted arguments.
Data mode reads geometry members (directions, masks, radii) and file hashes only: response arrays (a, b, peak_a, peak_b) are
never loaded. Leak mode copies an allow-list of the run root (data/, fits/, dev_eval/, scenes_new/) to a temporary directory.
"""
import argparse
import ast
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch

import common104kb as KB
import fit104kb as FB
import tests104k as T          # WL104K contracts (rand_dirs, reciprocity, mc_albedo, flat_eval, random_gates); unmodified

CM, GF, CL, FT, IO = KB.CM, KB.GF, KB.CL, KB.FT, KB.IO
DEV = KB.DEV
SEEDS = KB.P["seeds"]
MUS = (0.01, 0.05, 0.1, 0.3, 0.6, 1.0)
LANE_SETS = ("train_coverage", "dev", "test")
PEAK_SETS = ("dev_peak", "test_peak")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def entry(passed, **values):
    return {"passed": bool(passed), "values": values}


def flat_x(y, g):
    """y [U, 58], g [U, 8, 3, 5] -> x [U, 178] (fit104kb layout)."""
    return np.concatenate([np.asarray(y, np.float64), np.asarray(g, np.float64).reshape(len(y), -1)], 1)


def subset(ds, m):
    return {k: v[m:m + 1] for k, v in ds.items()}


def data64(ds, n, R, objective, target=None):
    """fit104kb.Data with every lane tensor rebuilt in float64 from the float64 arrays (the constructor rounds to float32);
    region masks are the module's own (exact 0 / 1) cast to float64."""
    d = FB.Data(ds, n, R, objective, target)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV, dtype=torch.float64)
    d.wi, d.wo = t(ds["wi"]).reshape(-1, 3).repeat(R, 1), t(ds["wo"]).reshape(-1, 3).repeat(R, 1)
    d.ref = t(ds["mean"] if target is None else target).reshape(-1, 3).repeat(R, 1)
    d.w = t(ds["w"]).reshape(-1).repeat(R)
    for k in ("core", "shoulder"):
        if hasattr(d, k):
            setattr(d, k, getattr(d, k).double())
    return d


def mirror_angle_deg(wi, wo):
    """Angle between wo and the mirror of wi about z, via atan2 (independent of the arccos form in common104j)."""
    m = wi * np.array([-1.0, -1.0, 1.0])
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(m, wo), axis=-1), (m * wo).sum(-1)))


REGIONS = {"e_core": (0.0, FB.CORE_DEG), "e_shoulder": (FB.CORE_DEG, FB.SHOULDER_DEG)}


def e_region_numpy(pred, ds, lo, hi):
    """Protocol region term per material [n] in float64: mean over probes of sum_region w |F - R|_1 / sum_region w R_1."""
    rad = mirror_angle_deg(ds["wi"], ds["wo"])
    reg = ((rad >= lo) & (rad < hi)) * ds["w"]
    num = (np.abs(pred - ds["mean"]).sum(-1) * reg).sum(-1)
    den = (ds["mean"].sum(-1) * reg).sum(-1)
    return (num / np.maximum(den, 1e-12)).mean(-1), int((reg > 0).sum(-1).min())


# ------------------------------------------------------------------ duck-typed objectives for the optimizer contracts
class Recorder:
    """Wraps a Data-like object; records the objective and the point at every gradient evaluation (accepted points) and
    counts calls."""

    def __init__(self, data):
        self.data, self.calls, self.accepted, self.points = data, 0, [], []

    def f(self, x):
        v = self.data.f(x)
        self.calls += 1
        if x.requires_grad:
            self.accepted.append(v.detach().cpu().numpy().copy())
            self.points.append(x.detach().cpu().numpy().astype(np.float64))
        return v


class BoxQuadratic:
    """f_u(x) = 0.5 (y - c_y)' A_u (y - c_y) + 0.5 sum d_u (g - c_g)^2 (dense SPD on the 58 free coordinates, diagonal on the
    120 box coordinates): minimiser under 0 <= g <= 1 is (c_y, clip(c_g, 0, 1)). Units flagged `wrong` return sum(x) with the
    NEGATED gradient, so every line search fails (freezing contract)."""

    def __init__(self, rng, U, wrong=()):
        Q = np.linalg.qr(rng.standard_normal((U, 58, 58)))[0]
        ev = np.exp(rng.uniform(0, np.log(100.0), (U, 58)))                  # condition number <= 100
        A = np.einsum("uij,uj,ukj->uik", Q, ev, Q)
        c = np.concatenate([rng.standard_normal((U, 58)), rng.uniform(-0.6, 1.6, (U, 120))], 1)
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=DEV)
        self.A, self.c, self.d = t(A), t(c), t(rng.uniform(0.5, 5.0, (U, 120)))
        self.wrong = torch.zeros(U, dtype=torch.bool, device=DEV)
        self.wrong[list(wrong)] = True
        self.opt = np.concatenate([c[:, :58], np.clip(c[:, 58:], 0, 1)], 1)

    def f(self, x):
        dy, dg = x[:, :58] - self.c[:, :58], x[:, 58:] - self.c[:, 58:]
        q = 0.5 * torch.einsum("ui,uij,uj->u", dy, self.A, dy) + 0.5 * (self.d * dg * dg).sum(-1)
        lin = (2 * x.detach() - x).sum(-1)                                     # value sum(x), gradient -1
        return torch.where(self.wrong, lin, q)


# ------------------------------------------------------------------ (a) pre-fit contracts
def prefit(res):
    rows, raw, _, _, ref, seen = CM.inputs()
    seen.update(KB.KA.authenticate_k())
    n = len(rows)
    rng = np.random.default_rng(SEEDS["tests"])
    ds = KB.training_set("original", n)
    t = {}

    # 1 response identity: fit104kb.Data objective at the archived WL104K P states == fit104k.FitData('P') score
    st, R, fj = KB.KA.states(KB.KA.K_ROOT / "fit_primary", n)
    x_hist = np.concatenate([flat_x(s["y"], s["g"]) for s in st["P"]])                     # start-major [R n, 178]
    with torch.no_grad():
        ob_new = FB.Data(ds, n, R, "original").f(torch.as_tensor(x_hist, device=DEV, dtype=torch.float32)).cpu().numpy()
        fd = FT.FitData(ds, n, R)
        ob_old = fd.score("P", FT.tensors({"y": x_hist[:, :58], "g": x_hist[:, 58:].reshape(-1, 8, 3, 5)})).cpu().numpy().ravel()
        del fd
    ob_arch = np.asarray(fj["P"]["final_objective"], np.float64).ravel()
    resp = []
    ev, _ = CM.eval_reference(ref, n)
    wi_r, wo_r = ev["wi"][:, :, :256], ev["wo"][:, :, :256]
    for trial in range(3):
        y = GF.from_original(CM.J.CI.generic(n)) + rng.standard_normal((n, 58))
        g = T.random_gates(rng, n, "P")
        Fp = CM.predict("P", {"y": y, "g": g}, wi_r, wo_r)
        for m in range(n):
            a = torch.as_tensor(wi_r[m].reshape(-1, 3), dtype=torch.float64, device=DEV)
            b = torch.as_tensor(wo_r[m].reshape(-1, 3), dtype=torch.float64, device=DEV)
            with torch.no_grad():
                Fk = KB.sdm_eval_t(torch.as_tensor(y[m], dtype=torch.float64, device=DEV),
                                   torch.as_tensor(g[m], dtype=torch.float64, device=DEV), a, b).cpu().numpy()
            resp.append(float(np.abs(Fk - Fp[m].reshape(-1, 3)).max() / Fp[m].max()))
    npar = 58 + 8 * 3 * 5
    v = {"objective_rel_max": rel(ob_new, ob_old), "vs_archived_fit_json_final_rel_max_informational": rel(ob_new, ob_arch),
         "sdm_eval_t_vs_predict_max_rel_peak": max(resp), "parameters_per_material": npar, "fit104kb_NPAR": FB.NPAR,
         "archived_y_cols": int(x_hist.shape[1] - 120), "start_keys": fj["start_keys"]}
    t["response_identity"] = entry(v["objective_rel_max"] <= 1e-6 and max(resp) <= 1e-12 and npar == FB.NPAR == 178
                                   and x_hist.shape[1] == 178, **v)

    # 2 starts
    s4 = KB.starts_y(n, KB.START_KEYS[:4])
    s4_old = FT.starts_y(n, CM.STARTS)
    s8 = KB.starts_y(n).reshape(len(KB.START_KEYS), n, 58)
    dmin = min(float(np.abs(s8[i] - s8[j]).max(-1).min()) for i, j in itertools.combinations(range(len(s8)), 2))
    t["starts"] = entry(np.array_equal(s4, s4_old) and list(CM.STARTS) == KB.START_KEYS[:4] and len(s8) == 8 and dmin > 0,
                        first4_bitwise_equal=bool(np.array_equal(s4, s4_old)), keys=KB.START_KEYS, wl104k_keys=list(CM.STARTS),
                        min_over_pairs_and_materials_of_max_abs_difference=dmin)

    # 3 E2 objective: numpy float64, float32 path, decomposition and finite-difference gradient
    y = GF.from_original(CM.J.CI.generic(n)) + 0.5 * rng.standard_normal((n, 58))
    g = rng.uniform(0.1, 0.9, (n, 8, 3, 5))
    states = {"random_interior": (y, g), "archived_P_generic": (st["P"][0]["y"], st["P"][0]["g"])}
    d64e = data64(ds, n, 1, "e2")
    d32e, d32o = FB.Data(ds, n, 1, "e2"), FB.Data(ds, n, 1, "original")
    rad_a, rad_b = mirror_angle_deg(ds["wi"], ds["wo"]), KB.mirror_radius(ds["wi"], ds["wo"])
    e3 = {"lambda_core": FB.LAMBDA_CORE, "lambda_shoulder": FB.LAMBDA_SHOULDER, "core_deg": FB.CORE_DEG,
          "shoulder_deg": FB.SHOULDER_DEG, "region_mask_disagreements_atan2_vs_arccos": int(sum(
              (((rad_a >= lo) & (rad_a < hi)) != ((rad_b >= lo) & (rad_b < hi))).sum() for lo, hi in REGIONS.values()))}
    ok3 = (FB.LAMBDA_CORE == 1.0 and FB.LAMBDA_SHOULDER == 1.0 and FB.CORE_DEG == 3.0 and FB.SHOULDER_DEG == 12.0
           and e3["region_mask_disagreements_atan2_vs_arccos"] == 0)
    for name, (yy, gg) in states.items():
        pred = CM.predict("P", {"y": yy, "g": gg}, ds["wi"], ds["wo"])
        x = flat_x(yy, gg)
        with torch.no_grad():
            T64 = {k: v.cpu().numpy() for k, v in d64e.terms(torch.as_tensor(x, device=DEV)).items()}
            xf = torch.as_tensor(x, device=DEV, dtype=torch.float32)
            T32 = {k: v.cpu().numpy() for k, v in d32e.terms(xf).items()}
            O32 = d32o.f(xf).cpu().numpy()
        r = {}
        for term, (lo, hi) in REGIONS.items():
            e_np, mincount = e_region_numpy(pred, ds, lo, hi)
            r[term] = {"float64_vs_numpy_rel": rel(T64[term], e_np), "float32_vs_numpy_rel": rel(T32[term], e_np),
                       "min": float(T64[term].min()), "all_finite": bool(np.isfinite(T64[term]).all()),
                       "min_lanes_with_weight_per_probe": mincount, "per_material": e_np.tolist()}
            ok3 &= (r[term]["float64_vs_numpy_rel"] <= 1e-10 and r[term]["float32_vs_numpy_rel"] <= 1e-4 and r[term]["min"] >= 0
                    and r[term]["all_finite"] and mincount > 0)
        reg = T64["e_core"] + T64["e_shoulder"]
        r["decomposition_float64_abs"] = float(np.abs(T64["objective"] - T64["el1"] - T64["ealb"] - reg).max())
        r["e2_minus_original_minus_regions_float32_rel"] = rel(T32["objective"] - O32, T32["e_core"] + T32["e_shoulder"])
        ok3 &= r["decomposition_float64_abs"] <= 1e-12 and r["e2_minus_original_minus_regions_float32_rel"] <= 1e-4
        e3[name] = r
    x64 = torch.as_tensor(flat_x(y, g), device=DEV)
    fd_res = {}
    m = 2                                                                              # hard_1 row
    coords = {"y0_O0_shape": 0, "y1_O0_shape": 1, "y23": 23, "y40": 40, "gate_O0_ch0_mu1": 58 + 0 * 15 + 0 * 5 + 4,
              "gate_O1_ch2_mu0.7": 58 + 1 * 15 + 2 * 5 + 3, "gate_L_ch1_mu0.4": 58 + 3 * 15 + 1 * 5 + 2,
              "gate_I0_ch2_mu0.15": 58 + 4 * 15 + 2 * 5 + 1}
    for term in ("objective", "e_core", "e_shoulder"):
        fun = lambda xx: d64e.terms(xx)[term].sum()
        xg = x64.clone().requires_grad_(True)
        (gr,) = torch.autograd.grad(fun(xg), xg)
        for cname, j in coords.items():
            h = 1e-6
            vals = []
            for sgn in (1, -1):
                xx = x64.clone(); xx[m, j] += sgn * h
                with torch.no_grad():
                    vals.append(fun(xx).item())
            gfd = (vals[0] - vals[1]) / (2 * h)
            ga = gr[m, j].item()
            fd_res[f"{term}:{cname}"] = {"autograd": ga, "fd": gfd, "rel": abs(ga - gfd) / max(abs(gfd), 1e-12)}
    e3["finite_difference"] = fd_res
    ok3 &= all(v["rel"] <= 1e-5 for v in fd_res.values()) and any(abs(v["autograd"]) > 0 for v in fd_res.values())
    t["e2_objective"] = entry(ok3, **e3)
    del d64e, d32e, d32o
    torch.cuda.empty_cache()

    # 4 O1 projected L-BFGS: real subset (1 material, 2 starts, budget 60); synthetic box QP; freezing
    budget = 60
    y2 = KB.starts_y(n, KB.START_KEYS[:2]).reshape(2, n, 58)[:, m]
    x0 = torch.as_tensor(np.concatenate([y2, np.ones((2, 120))], 1), device=DEV, dtype=torch.float32)
    rec = Recorder(FB.Data(subset(ds, m), 1, 2, "original"))
    xl, _, info = FB.lbfgs(rec, x0, budget)
    acc = np.stack(rec.accepted)
    inc = float(np.max((acc[1:] - acc[:-1]) / np.abs(acc[:-1])))
    gx = xl[:, 58:].cpu().numpy()
    # first accepted step per unit (protocol: the first iteration scales alpha so that max |alpha d| = 0.01)
    pts = np.stack(rec.points)                                                     # [gradient calls, U, 178]
    first = []
    for u in range(pts.shape[1]):
        mv = [k for k in range(1, len(pts)) if np.any(pts[k, u] != pts[0, u])]
        first.append(float(np.abs(pts[mv[0], u] - pts[0, u]).max()) if mv else None)
    real = {"evaluations_per_unit": info["evaluations_per_unit"], "gradient_evaluations_per_unit": info["gradient_evaluations_per_unit"],
            "objective_calls": rec.calls, "gradient_calls": len(rec.accepted), "initial": acc[0].tolist(), "final": acc[-1].tolist(),
            "max_relative_increase_between_accepted_points": inc, "gate_min": float(gx.min()), "gate_max": float(gx.max()),
            "terminated_early": info["terminated_early"], "iterations": info["iterations"],
            "first_accepted_step_max_abs_per_unit": first, "first_step_limit": 0.01 + 1e-6}
    ok_real = (max(info["evaluations_per_unit"]) <= budget and gx.min() >= 0 and gx.max() <= 1 and inc <= 1e-6
               and bool((acc[-1] < acc[0]).all()) and max(info["gradient_evaluations_per_unit"]) <= len(rec.accepted)
               and all(v is not None and 0 < v <= 0.01 + 1e-6 for v in first))
    qrng = np.random.default_rng(SEEDS["tests"] + 1)
    qp = BoxQuadratic(qrng, 3)
    q0 = torch.as_tensor(np.concatenate([qrng.standard_normal((3, 58)), qrng.uniform(0, 1, (3, 120))], 1), device=DEV)
    xq, _, iq = FB.lbfgs(qp, q0, 4000)
    errq = np.abs(xq.cpu().numpy() - qp.opt).max(1)
    at_bound = int(((qp.opt[:, 58:] == 0) | (qp.opt[:, 58:] == 1)).sum())
    quad = {"max_abs_error_per_unit": errq.tolist(), "evaluations_per_unit": iq["evaluations_per_unit"],
            "optimum_gate_coordinates_at_bound": at_bound, "terminated_early": iq["terminated_early"]}
    ok_quad = bool(errq.max() <= 1e-4 and at_bound > 0 and max(iq["evaluations_per_unit"]) <= 4000)
    qf = BoxQuadratic(np.random.default_rng(SEEDS["tests"] + 2), 2, wrong=(1,))
    f0 = torch.as_tensor(np.concatenate([np.zeros((2, 58)), np.ones((2, 120))], 1), device=DEV)
    xf_, _, ifz = FB.lbfgs(qf, f0, 200)
    frz = {"terminated_early": ifz["terminated_early"], "evaluations_per_unit": ifz["evaluations_per_unit"],
           "frozen_unit_moved": float((xf_[1] - f0[1]).abs().max()), "expected_frozen_evaluations": 1 + 3 * 13}
    ok_frz = (ifz["terminated_early"] == [False, True] and ifz["evaluations_per_unit"][1] == 1 + 3 * 13
              and frz["frozen_unit_moved"] == 0 and ifz["evaluations_per_unit"][0] > ifz["evaluations_per_unit"][1])
    t["o1_lbfgs"] = entry(ok_real and ok_quad and ok_frz, real_subset=real, real_subset_passed=bool(ok_real),
                          box_quadratic=quad, box_quadratic_passed=ok_quad, freezing=frz, freezing_passed=bool(ok_frz))

    # 5 O0 Adam path == fit104k.train (50 steps, 1 material x 2 starts)
    steps = 50
    xa, _, _ = FB.adam(FB.Data(subset(ds, m), 1, 2, "original"), x0.clone(), steps)
    term, _ = FT.train(FT.FitData(subset(ds, m), 1, 2), "P", {"y": y2, "g": np.ones((2, 8, 3, 5))}, steps)
    xa = xa.cpu().numpy().astype(np.float64)
    ad = {"y_rel": rel(xa[:, :58], term["y"]), "g_rel": rel(xa[:, 58:], term["g"].reshape(2, -1)),
          "bitwise_equal": bool(np.array_equal(xa, flat_x(term["y"], term["g"]))), "steps": steps}
    t["o0_adam_path"] = entry(ad["y_rel"] <= 1e-5 and ad["g_rel"] <= 1e-5, **ad)

    # 6 synthetic truth: exactly representable, O0 alpha = 0.03, other outer lobes / F0 / amplitudes unchanged
    tr = KB.synthetic_truth(raw["R0"])
    y0 = GF.from_original(raw["R0"])
    with torch.no_grad():
        pt = GF.decode(torch.as_tensor(tr["y"], dtype=torch.float64, device=DEV))
        p0 = GF.decode(torch.as_tensor(y0, dtype=torch.float64, device=DEV))
    o0 = CL.OUTER[0]
    sy = {"alpha_u_O0_max_abs_dev": float((pt["au"][:, o0] - 0.03).abs().max()),
          "alpha_v_O0_max_abs_dev": float((pt["av"][:, o0] - 0.03).abs().max()),
          "other_lobe_alpha_change": float(max((pt[k][:, 1:] - p0[k][:, 1:]).abs().max() for k in ("au", "av"))),
          "f0_lam_s_eta_change": float(max((pt[k] - p0[k]).abs().max() for k in ("f0", "lam", "s", "eta"))),
          "gates_equal_SP": bool(all(np.array_equal(tr["g"][:, j], np.broadcast_to(KB.SP_GATES[c], (n, 3, 5)))
                                     for j, c in enumerate(KB.COMP))),
          "gates_in_box": bool((tr["g"] >= 0).all() and (tr["g"] <= 1).all())}
    tgt = CM.predict("P", tr, ds["wi"], ds["wo"])
    dt = data64(ds, n, 1, "e2", target=tgt)
    with torch.no_grad():
        sy["objective_at_truth_float64_max"] = float(dt.f(torch.as_tensor(flat_x(tr["y"], tr["g"]), device=DEV)).max())
    del dt
    sy["truth_differs_from_R0_identity_rel"] = rel(CM.predict("P", tr, wi_r, wo_r),
                                                   CM.predict("P", {"y": y0, "g": np.ones((n, 8, 3, 5))}, wi_r, wo_r))
    t["synthetic_truth"] = entry(sy["alpha_u_O0_max_abs_dev"] <= 1e-12 and sy["alpha_v_O0_max_abs_dev"] <= 1e-12
                                 and sy["other_lobe_alpha_change"] <= 1e-12 and sy["f0_lam_s_eta_change"] <= 1e-12
                                 and sy["gates_equal_SP"] and sy["gates_in_box"] and sy["objective_at_truth_float64_max"] <= 1e-12
                                 and sy["truth_differs_from_R0_identity_rel"] > 1e-2, **sy)
    res["tests"] = t
    res["inputs_sha256"] = seen


# ------------------------------------------------------------------ (b) fitted states
def state_contracts(st, n, rng, rng_mc):
    """Reciprocity (interior cos >= 0.05; grazing-included descriptive), pointwise envelope vs the ungated base, MC albedo."""
    r = T.reciprocity("P", st, rng, n)
    rg = T.reciprocity("P", st, rng, n, L=1024, zmin=1e-3)
    wi, wo = T.rand_dirs(rng, n * 4096, 1e-3).reshape(n, 4096, 3), T.rand_dirs(rng, n * 4096, 1e-3).reshape(n, 4096, 3)
    Fs, Fb = T.flat_eval("P", st, wi, wo), T.flat_eval("A", {"y": st["y"]}, wi, wo)
    peak = Fb.max()
    env = {"min_F": float(Fs.min()), "max_excess_over_base_rel_peak": float((Fs - Fb).max() / peak)}
    mean, se = T.mc_albedo("P", st, n, MUS, rng_mc, N=1 << 20)
    alb = {"max_albedo": float(mean.max()), "max_albedo_minus_4se": float((mean - 4 * se).max()),
           "max_albedo_plus_4se": float((mean + 4 * se).max()), "per_material_max": mean.max((1, 2)).tolist()}
    with torch.no_grad():
        lam = GF.decode(torch.as_tensor(st["y"], device=DEV, dtype=torch.float64))["lam"].sum(-1).max().item()
    ok = {"reciprocity": r["f_rel"] <= 1e-12 and r["cos_identity_rel"] <= 1e-12,
          "envelope": env["min_F"] >= 0 and env["max_excess_over_base_rel_peak"] <= 1e-15,
          "albedo": alb["max_albedo_minus_4se"] <= 1.0, "sum_lambda": lam <= 1 + 1e-6}
    return {"reciprocity_interior": r, "reciprocity_grazing_included_descriptive": rg, "envelope": env, "mc_albedo": alb,
            "max_sum_lambda": lam, "parts": ok, "passed": bool(all(ok.values()))}


def fit_record(d, fj):
    """fit.json contract (when present): states hash, budget respected and not exceeded, 178 parameters per material."""
    if not fj:
        return {"fit_json_present": False, "passed": True}
    ev = fj.get("evaluations_per_unit") or []
    v = {"fit_json_present": True, "states_sha256_recorded": fj.get("states_sha256"), "budget": fj.get("budget"),
         "budget_respected": fj.get("budget_respected"), "max_evaluations_per_unit": max(ev) if ev else None,
         "parameters_per_material": fj.get("parameters_per_material")}
    v["states_sha256_match"] = fj.get("states_sha256") == KB.sha(d / "states.npz")
    v["passed"] = bool(v["states_sha256_match"] and fj.get("budget_respected") is True and ev
                       and isinstance(fj.get("budget"), int) and max(ev) <= fj["budget"]
                       and fj.get("parameters_per_material") == 178)
    return v


def fitted(res, dirs):
    rows = CM.inputs()[0]
    n = len(rows)
    t = {}
    for d in dirs:
        with np.load(d / "states.npz") as z:
            S = {k: z[k] for k in z.files}
        fj = KB.read(d / "fit.json") if (d / "fit.json").exists() else {}
        rec = fit_record(d, fj)
        Py, Pg = S["P_y"], S["P_g"]
        R = len(fj["start_keys"]) if "start_keys" in fj else Py.shape[0] // n
        shape_ok = (Py.ndim == 2 and Py.shape == (R * n, 58) and Pg.shape == (R * n, 8, 3, 5)
                    and Py.shape[1] + int(np.prod(Pg.shape[1:])) == 178)
        fin = bool(np.isfinite(Py).all() and np.isfinite(Pg).all())
        box = {"gate_min": float(Pg.min()), "gate_max": float(Pg.max())}
        per = {}
        key = zlib.crc32(d.name.encode("utf-8"))                 # stable per-arm stream key (independent of argument order)
        if shape_ok and fin:
            for r in range(R):
                st = {"y": Py[r * n:(r + 1) * n], "g": Pg[r * n:(r + 1) * n]}
                per[f"start{r}"] = state_contracts(st, n, np.random.default_rng([SEEDS["tests"], key, r]),
                                                   np.random.default_rng([SEEDS["albedo_mc"], key, r]))
                print(d.name, "start", r, per[f"start{r}"]["passed"], flush=True)
        ok = rec["passed"] and shape_ok and fin and box["gate_min"] >= 0 and box["gate_max"] <= 1 and len(per) == R and all(
            v["passed"] for v in per.values())
        t[str(d)] = entry(ok, starts=R, arm=fj.get("arm"), shape_178_per_material=bool(shape_ok), finite=fin, **box,
                          states_sha256=KB.sha(d / "states.npz"), fit_record=rec,
                          rng_key={"seeds": [SEEDS["tests"], SEEDS["albedo_mc"]], "crc32_arm_dir": key, "arm_dir": d.name},
                          per_start=per)
    res["tests"] = t


# ------------------------------------------------------------------ (c) data independence
def walk_json(o, path=()):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk_json(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_json(v, path + (str(i),))


def recorded_sets(dj):
    """{set: record dict} for every dict in data.json carrying 'sequence_range'; the set is the deepest path key among
    train_coverage / dev / test, '_peak' appended when a later path key contains 'peak'."""
    out = {}
    for path, d in walk_json(dj):
        if "sequence_range" not in d:
            continue
        hit = [i for i, k in enumerate(path) if k in LANE_SETS or k in PEAK_SETS]
        if not hit:
            continue
        name = path[hit[-1]]
        if name in ("dev", "test") and any("peak" in k for k in path[hit[-1] + 1:]):
            name += "_peak"
        out.setdefault(name, []).append({"path": "/".join(path), **d})
    return out


def recorded_hashes(dj):
    out = {}
    for _, d in walk_json(dj):
        for k, v in d.items():
            if isinstance(v, str) and k.endswith(".npz") and re.fullmatch(r"[0-9a-f]{64}", v):
                out.setdefault(Path(k).name, set()).add(v)
    return out


def historical_registry():
    """Sequence ranges and initstates recorded before WL104K-B (protocol stream_registry)."""
    jd = KB.read(KB.REPO / "results/ecro_wl84_104j/primary/data/data.json")["isolation"]["sequence_ranges"]
    rng = {"WL104D": (0, 735870976), "WL104D_amendment1": (12288000000, 12363497472),
           "WL104KA_scenes": (2 ** 42, 4470000000000)}
    for i, (a, b) in enumerate(jd):
        rng[f"WL104J_{i}"] = (int(a), int(b))
    sa = sorted((KB.OUT).glob("**/stageA.json"))
    if sa:
        g = KB.read(sa[0])["grid"]
        rng["stageA"] = (2 ** 31 * g["walks"], (2 ** 31 + g["pairs"]) * g["walks"])          # stageA104kb pair0 = 2^31
    else:
        rng["stageA_conservative"] = (2 ** 47, 2 ** 48)
    jp = IO.read(KB.REPO / "experiments/ecro_wl84_104j/protocol.json")["seeds"]
    seeds = {104460, 10441901, 104470, jp["fit_walks"], jp["transfer_walks"], SEEDS["stageA_walks"]} | set(range(1041210, 1041230))
    return rng, seeds


def expected_streams():
    d = KB.P["data"]
    s = {"train_coverage": d["train_coverage"], "dev": d["dev"], "test": d["test"],
         "dev_peak": d["dev"]["peak_grid"], "test_peak": d["test"]["peak_grid"]}
    return {k: {"seed": v["walk_seed"], "pair0": v["walk_pair0"], "walks": v.get("walks_per_lane", v.get("walks"))}
            for k, v in s.items()}


def fold_rows(x):
    """[..., 3] -> contiguous float64 [N, 3] with -0.0 folded to +0.0 (exact row identity as data104kb.row_keys)."""
    return np.ascontiguousarray(np.asarray(x, np.float64).reshape(-1, 3) + 0.0)


def exact_shared_rows(a, b):
    """Number of rows of a [N, 3] exactly equal (bitwise float64, -0.0 == +0.0) to a row of b [M, 3]."""
    va = fold_rows(a).view(np.dtype((np.void, 24))).ravel()
    vb = fold_rows(b).view(np.dtype((np.void, 24))).ravel()
    return int(np.isin(va, vb).sum())


def duplicate_rows(x):
    """Number of rows of x [N, 3] that exactly repeat an earlier row (-0.0 folded)."""
    r = fold_rows(x)
    return int(len(r) - len({row.tobytes() for row in r}))


# Geometry-only keys read in data mode. The response arrays (a, b, peak_a, peak_b) of ANY set are never read here: the npz is
# opened as a lazy NpzFile and only these members are accessed (protocol pre_freeze_clarifications, data-stage isolation).
LANE_GEOM = ("wi", "wo", "use", "incidence", "polar_deg")
PEAK_GEOM = ("peak_wi", "peak_wo", "use", "radius_deg")
RESPONSE_KEYS = ("a", "b", "peak_a", "peak_b")
LANES_PER_PROBE = {"train_coverage": 1024, "dev": 1024, "test": 2048}
PROBES = 20
PEAK_RADII = np.r_[0.0, np.repeat(np.r_[0.25 * np.arange(1, 17), 5.0, 6.0, 8.0], 16)]      # [0] + 16 x 16 rings + {5, 6, 8} x 16
PEAK_K = 305


def load_geometry(path, keys):
    """{key: array} for the geometry keys present in the npz, plus the member list; no other member is read."""
    with np.load(path) as z:
        members = list(z.files)
        return {k: z[k] for k in keys if k in members}, members


def data_check(res, root):
    rows = CM.inputs()[0]
    n = len(rows)
    dd = root / "data"
    dj = KB.read(dd / "data.json")
    t = {}
    # data.json status: qualified (all sets) and a non-smoke run
    q = dj.get("qualified")
    t["data_json_status"] = entry(isinstance(q, dict) and q.get("all") is True and dj.get("smoke") is False,
                                  qualified=q, smoke=dj.get("smoke"))
    # sha256
    rh = recorded_hashes(dj)
    files = sorted(p.name for p in dd.glob("*.npz"))
    req = [f"{s}.npz" for s in LANE_SETS + PEAK_SETS]
    sh = {f: {"actual": KB.sha(dd / f), "recorded": sorted(rh.get(f, []))} for f in files}
    t["sha256"] = entry(all(f in files for f in req) and all(len(v["recorded"]) == 1 and v["recorded"][0] == v["actual"]
                                                             for v in sh.values()), files=sh, required=req)
    # geometry arrays only (lazy NpzFile; response members never accessed)
    L, Pk, members = {}, {}, {}
    for s in LANE_SETS:
        if (dd / f"{s}.npz").exists():
            L[s], members[s] = load_geometry(dd / f"{s}.npz", LANE_GEOM)
    for s in PEAK_SETS:
        if (dd / f"{s}.npz").exists():
            Pk[s], members[s] = load_geometry(dd / f"{s}.npz", PEAK_GEOM)
    orig, _ = load_geometry(KB.REPO / "results/ecro_wl84_104j/primary/data/fit_set.npz", ("wi", "wo", "use"))
    L_all = {**L, "train_original": orig}
    inc = {s: (v["incidence"] if "incidence" in v else v["wi"][:, 0]) for s, v in L_all.items()}
    lay, lay_ok = {}, True
    for s, v in L.items():
        miss = [k for k in LANE_GEOM if k not in v]
        if miss:
            lay[s] = {"missing_keys": miss, "members": members[s]}
            lay_ok = False
            continue
        u = v["use"].astype(bool)
        lanes_proto = int(KB.P["data"][s]["lanes_per_probe"])
        lay[s] = {"shape": list(v["wo"].shape), "incidence_equals_wi": bool(np.array_equal(
            np.broadcast_to(v["incidence"][:, None], v["wi"].shape)[u], v["wi"][u])), "active": int(u.sum()),
                  "lanes_per_probe": int(v["wo"].shape[1]), "lanes_expected": LANES_PER_PROBE[s], "lanes_protocol": lanes_proto,
                  "probes_per_material": v["wo"].shape[0] / n, "incidences": int(len(v["incidence"])),
                  "polar_entries": int(v["polar_deg"].size), "response_members_present_not_read": [k for k in RESPONSE_KEYS
                                                                                                    if k in members[s]]}
        lay_ok &= (v["wo"].shape == (n * PROBES, LANES_PER_PROBE[s], 3) and lanes_proto == LANES_PER_PROBE[s]
                   and v["wi"].shape == v["wo"].shape and v["use"].shape == v["wo"].shape[:2]
                   and v["incidence"].shape == (n * PROBES, 3) and v["polar_deg"].size == n * PROBES
                   and lay[s]["incidence_equals_wi"])
    for s, v in Pk.items():
        miss = [k for k in PEAK_GEOM if k not in v]
        if miss:
            lay[s] = {"missing_keys": miss, "members": members[s]}
            lay_ok = False
            continue
        base = s.replace("_peak", "")
        u = v["use"].astype(bool)
        rad = np.asarray(v["radius_deg"], np.float64)
        geo = mirror_angle_deg(v["peak_wi"], v["peak_wo"])                       # atan2 radius of every grid point
        rad_ok = rad.shape == (PEAK_K,) and bool(np.abs(rad - PEAK_RADII).max() <= 1e-12)
        lay[s] = {"shape": list(v["peak_wo"].shape), "active": int(u.sum()),
                  "same_incidence_as_lane_set": bool(base in L and "incidence" in L[base]
                                                     and np.array_equal(v["peak_wi"][:, 0], L[base]["incidence"])),
                  "points_per_probe": int(v["peak_wo"].shape[1]), "points_expected": PEAK_K,
                  "radius_layout_equals_protocol": rad_ok,
                  "max_abs_geometric_radius_minus_radius_deg": (float(np.abs(geo - rad[None, :]).max())
                                                                if rad.shape == geo.shape[1:] else None),
                  "response_members_present_not_read": [k for k in RESPONSE_KEYS if k in members[s]]}
        lay_ok &= (v["peak_wo"].shape == (n * PROBES, PEAK_K, 3) and v["peak_wi"].shape == v["peak_wo"].shape
                   and v["use"].shape == v["peak_wo"].shape[:2] and rad_ok and lay[s]["same_incidence_as_lane_set"]
                   and lay[s]["max_abs_geometric_radius_minus_radius_deg"] is not None
                   and lay[s]["max_abs_geometric_radius_minus_radius_deg"] <= 1e-6)
    t["layout"] = entry(bool(lay_ok) and len(L) == 3 and len(Pk) == 2, **lay)
    if not (lay_ok and len(L) == 3 and len(Pk) == 2):
        res["tests"] = t
        return
    # incidences: no exactly-shared incident direction between sets, none repeated within a new set; min angle descriptive
    pairs = {}
    for a, b in itertools.combinations(sorted(inc), 2):
        cos = np.clip(inc[a] @ inc[b].T, -1, 1)
        pairs[f"{a}|{b}"] = {"exact_shared": exact_shared_rows(inc[a], inc[b]), "min_angle_deg": float(np.degrees(np.arccos(cos.max())))}
    t["incidence_disjoint"] = entry(all(v["exact_shared"] == 0 for v in pairs.values()), **pairs)
    within = {s: duplicate_rows(inc[s]) for s in LANE_SETS}
    t["incidence_unique_within_set"] = entry(all(v == 0 for v in within.values()), duplicate_incidences=within,
                                             note="train_original shares the WL104D grid across materials by design (not checked)")
    # active outgoing directions per material across sets (lane sets incl. train_original, plus peak grids)
    act = {}
    for s, v in L_all.items():
        u = v["use"].astype(bool).reshape(n, -1)
        wo = v["wo"].reshape(n, -1, 3)
        act[s] = [wo[m][u[m]] for m in range(n)]
    for s, v in Pk.items():
        u = v["use"].astype(bool).reshape(n, -1)
        wo = v["peak_wo"].reshape(n, -1, 3)
        act[s] = [wo[m][u[m]] for m in range(n)]
    outp = {}
    for a, b in itertools.combinations(sorted(act), 2):
        outp[f"{a}|{b}"] = sum(exact_shared_rows(act[a][m], act[b][m]) for m in range(n))
    t["outgoing_disjoint"] = entry(all(v == 0 for v in outp.values()), exact_shared_active_outgoing=outp)
    # stream registry
    hist, hseeds = historical_registry()
    rec = recorded_sets(dj)
    exp = expected_streams()
    st = {}
    ranges = {}
    ok = True
    for s, e in exp.items():
        r = rec.get(s, [])
        if len(r) != 1:
            st[s] = {"found": len(r), "error": "expected exactly one record with sequence_range"}
            ok = False
            continue
        r = r[0]
        lo, hi = (int(x) for x in r["sequence_range"])
        arr = L.get(s) if s in L else Pk.get(s)
        nact = None if arr is None else int(np.asarray(arr["use"], bool).sum())
        seed = r.get("walk_seed", r.get("initstate", r.get("seed")))
        v = {"path": r["path"], "range": [lo, hi], "expected_lo": e["pair0"] * e["walks"], "active": nact,
             "length_equals_active_x_walks": nact is not None and hi - lo == nact * e["walks"], "recorded_seed": seed,
             "expected_seed": e["seed"], "at_or_above_2^44": lo >= 2 ** 44}
        v["passed"] = bool(lo == v["expected_lo"] and v["length_equals_active_x_walks"] and v["at_or_above_2^44"]
                           and (seed is None or int(seed) == e["seed"]) and e["seed"] not in hseeds)
        ok &= v["passed"]
        st[s] = v
        ranges[s] = (lo, hi)
    allr = {**{f"new:{k}": v for k, v in ranges.items()}, **{f"hist:{k}": v for k, v in hist.items()}}
    overl = [f"{a}|{b}" for (a, x), (b, y) in itertools.combinations(allr.items(), 2)
             if (a.startswith("new:") or b.startswith("new:")) and x[0] < y[1] and y[0] < x[1]]
    seeds_new = [e["seed"] for e in exp.values()]
    t["stream_registry"] = entry(ok and not overl and len(set(seeds_new)) == len(seeds_new), sets=st, overlaps=overl,
                                 historical_ranges=hist, historical_initstates=sorted(hseeds), new_initstates=seeds_new)
    res["tests"] = t
    res["data_json_qualified_informational"] = dj.get("qualified")


# ------------------------------------------------------------------ (d) leak guard
WRAPPER = r'''
import os, re, runpy, sys
script, log, root = sys.argv[1], sys.argv[2], os.path.normcase(os.path.abspath(sys.argv[3]))
pat = re.compile(r"^test.*\.npz$")
def category(p):
    """test data (data/test*.npz anywhere) or, under the run root, scenes* / test_eval / decision.json; None otherwise."""
    a = os.path.normcase(os.path.abspath(p))
    if pat.match(os.path.basename(a)) and os.path.basename(os.path.dirname(a)) == "data":
        return "test_data"
    try:
        rel = os.path.relpath(a, root)
    except ValueError:                                   # other drive
        return None
    head = rel.split(os.sep)[0]
    if head == os.pardir or os.path.isabs(rel):
        return None
    if head.startswith("scenes"):
        return "scenes"
    if head == "test_eval":
        return "test_eval"
    if rel == "decision.json":
        return "decision"
    return None
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        c = category(p)
        if c is not None:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(c + "\t" + p + "\n")
            raise PermissionError("tests104kb: blocked " + c + ": " + p)
sys.addaudithook(hook)
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.argv = [script] + sys.argv[4:]
runpy.run_path(script, run_name="__main__")
'''


def run_guarded(script, args, root, tmp, label, timeout):
    """Run `script args` under the audit hook (root = the copied run root). A timeout is a failed run (timed_out True)."""
    wrap, log = tmp / "audit_wrapper.py", tmp / f"opened_{label}.txt"
    wrap.write_text(WRAPPER, encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1"}
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-B", str(wrap), str(script), str(log), str(root), *args], cwd=str(script.parent),
                           env=env, capture_output=True, text=True, timeout=timeout)
        rc, out, timed_out = p.returncode, p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        dec = lambda s: s.decode("utf-8", "replace") if isinstance(s, bytes) else (s or "")
        rc, out, timed_out = None, dec(e.stdout) + dec(e.stderr), True
    blocked = [ln.split("\t", 1) for ln in log.read_text(encoding="utf-8").splitlines() if ln] if log.exists() else []
    return {"returncode": rc, "timed_out": timed_out, "timeout_seconds": timeout,
            "test_files_opened": [p for c, p in blocked if c == "test_data"],
            "blocked_opens": [f"{c}: {p}" for c, p in blocked], "leak_guard_in_output": "LEAK_GUARD" in out,
            "seconds": time.monotonic() - t0, "output_tail": out[-1500:]}


def refused(r, require_guard=True):
    """Refusal: finished (no timeout) with a non-zero status, no blocked open attempted, LEAK_GUARD reported (if required)."""
    return bool(not r["timed_out"] and r["returncode"] not in (0, None) and not r["blocked_opens"]
                and (r["leak_guard_in_output"] or not require_guard))


LEAK_COPY = ("data", "fits")                     # allow-list; dev_eval/ on request, scenes_new/ when present


def copy_root(src, dst, skip_test_data, dev_eval=False):
    """Allow-list copy of a run root: data/ (test*.npz dropped when skip_test_data), fits/, dev_eval/ (only when dev_eval) and
    scenes_new/ if present. Nothing else (scenes/, test_eval/, decision.json, contracts/, logs/, ...) is copied."""
    dst.mkdir(parents=True)
    ign = (lambda d, names: {x for x in names if re.match(r"^test.*\.npz$", x)}) if skip_test_data else None
    for sub in LEAK_COPY + (("dev_eval",) if dev_eval else ()) + ("scenes_new",):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub, ignore=ign if sub == "data" else None)
    if skip_test_data and not dev_eval:
        # guarded dev rerun: coverage fits record their run-root data_dir; point it at the copy's data/ (same file hash), so the
        # rerun's FIT_DATA_MISMATCH check compares like with like. Only this copy is rewritten (refusal copies stay byte-identical).
        for fj in (dst / "fits").glob("*/fit.json"):
            d = json.loads(fj.read_text(encoding="utf-8"))
            if d.get("data_dir") and Path(d["data_dir"]).resolve() == (src / "data").resolve():
                d["data_dir"] = str((dst / "data").resolve())
                fj.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return sorted(p.name for p in dst.iterdir())


def static_scan(script):
    """Every string literal naming test data, with its enclosing function (descriptive)."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    hits = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                    re.search(r"\btest(_peak)?(\*)?\.npz\b", node.value) or node.value in ("test", "test_peak")):
                hits.append({"line": node.lineno, "literal": node.value, "function": getattr(fn, "name", "<module>")})
    seen, out = set(), []
    for h in sorted(hits, key=lambda h: (h["line"], h["function"] == "<module>")):
        if h["line"] not in seen:
            seen.add(h["line"]); out.append(h)
    return out


DEV_COMPARE = ("stage_F.eligible", "stage_F.run", "stage_F.spec", "selections", "fits_sha256")


def dev_fields(dev):
    return {k: (dev.get("stage_F", {}).get(k.split(".")[1]) if k.startswith("stage_F.") else dev.get(k)) for k in DEV_COMPARE}


def leak(res, root, script, static_only, timeout):
    t = {}
    if not script.exists():
        t["eval_script_present"] = entry(False, script=str(script))
        res["tests"] = t
        return
    t["static_scan_descriptive"] = entry(True, literals=static_scan(script))
    real_dev = root / "dev_eval" / "dev.json"
    tmp = Path(tempfile.mkdtemp(prefix="wl104kb_leak_"))
    try:
        a = tmp / "root_no_dev_decision"
        cp = copy_root(root, a, skip_test_data=False)
        r = run_guarded(script, ["test", str(a)], a, tmp, "test_no_dev", timeout)
        t["test_refused_without_dev_decision"] = entry(refused(r), copied=cp, **r)
        if real_dev.exists():
            dev = KB.read(real_dev)
            b = tmp / "root_bad_dev_hash"
            cp = copy_root(root, b, skip_test_data=False, dev_eval=True)
            (b / "dev_eval" / "dev.json.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
            r = run_guarded(script, ["test", str(b)], b, tmp, "test_bad_hash", timeout)
            t["test_refused_with_tampered_dev_hash"] = entry(refused(r, require_guard=False), copied=cp, **r)
            # one byte appended to fits/O1/states.npz after the dev decision
            o = tmp / "root_tampered_O1"
            cp = copy_root(root, o, skip_test_data=False, dev_eval=True)
            st = o / "fits" / "O1" / "states.npz"
            if st.exists():
                with open(st, "ab") as fh:
                    fh.write(b"\0")
                r = run_guarded(script, ["test", str(o)], o, tmp, "test_tampered_O1", timeout)
                t["test_refused_with_tampered_O1_states"] = entry(refused(r), copied=cp, **r)
            else:
                t["test_refused_with_tampered_O1_states"] = entry(False, copied=cp, error="fits/O1/states.npz missing in the run root")
            # F eligible at dev but fits/F missing
            if dev.get("stage_F", {}).get("run") is True:
                f = tmp / "root_missing_F"
                cp = copy_root(root, f, skip_test_data=False, dev_eval=True)
                shutil.rmtree(f / "fits" / "F", ignore_errors=True)
                r = run_guarded(script, ["test", str(f)], f, tmp, "test_missing_F", timeout)
                t["test_refused_with_missing_F"] = entry(refused(r) and not (f / "fits" / "F").exists(), copied=cp, **r)
        if not static_only:
            c = tmp / "root_without_test_data"
            cp = copy_root(root, c, skip_test_data=True)
            left = sorted(p.name for p in (c / "data").glob("test*"))
            r = run_guarded(script, ["dev", str(c)], c, tmp, "dev", timeout)
            r["dev_json_written"] = (c / "dev_eval" / "dev.json").exists()
            t["dev_path_never_opens_test_data"] = entry(not r["timed_out"] and r["returncode"] == 0 and not r["blocked_opens"]
                                                        and r["dev_json_written"] and not left, copied=cp,
                                                        test_files_in_copy=left, **r)
            if real_dev.exists():
                if r["dev_json_written"]:
                    want, got = dev_fields(KB.read(real_dev)), dev_fields(KB.read(c / "dev_eval" / "dev.json"))
                    diff = [k for k in DEV_COMPARE if want[k] != got[k]]
                    t["dev_rerun_matches_dev_decision"] = entry(not diff and want["fits_sha256"] is not None
                                                                and want["selections"] is not None, compared=list(DEV_COMPARE),
                                                                differing=diff, dev_decision=want, guarded_rerun=got)
                else:
                    t["dev_rerun_matches_dev_decision"] = entry(False, error="guarded dev rerun wrote no dev.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["tests"] = t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fitted", type=Path, action="append")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--leak", type=Path)
    ap.add_argument("--eval-script", type=Path, default=KB.HERE / "eval104kb.py")
    ap.add_argument("--leak-static-only", action="store_true", help="(d) skip the guarded dev run")
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()
    t0 = time.monotonic()
    res = {"protocol_sha256": KB.sha(KB.PROTOCOL), "commit": KB.commit()}
    if args.fitted:
        res["mode"] = "fitted"
        fitted(res, args.fitted)
    elif args.data:
        res["mode"] = "data"
        data_check(res, args.data)
    elif args.leak:
        res["mode"] = "leak"
        leak(res, args.leak, args.eval_script.resolve(), args.leak_static_only, args.timeout)
    else:
        res["mode"] = "prefit"
        prefit(res)
    res["passed"] = bool(res["tests"]) and all(v["passed"] for v in res["tests"].values())
    res["seconds"] = time.monotonic() - t0
    if torch.cuda.is_available():
        res["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    KB.write(args.out, res)
    print(res["mode"], "passed", res["passed"], {k: v["passed"] for k, v in res["tests"].items()}, round(res["seconds"], 1), "s",
          flush=True)
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
