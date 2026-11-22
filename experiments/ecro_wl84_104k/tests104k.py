"""WL104K Stage C contracts (gate before physical fitting): baseline identity, reciprocity, passivity, gradients, trainability,
Dr.Jit parity, synthetic-target representability, A reproduction path. Exits non-zero on failure.

Usage: python tests104k.py <out_json> [--fitted <fit_dir>]   (--fitted: re-run reciprocity / passivity on fitted states)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import common104k as CM
import fit104k as FT

GF, CL, IO = CM.GF, CM.CL, CM.IO
DEV = CM.DEV


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def rand_dirs(rng, n, zmin=1e-3):
    z = zmin + (1 - zmin) * rng.random(n)
    r = np.sqrt(1 - z * z)
    ph = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def flat_eval(arm, st, wi, wo):
    """st with leading n; wi / wo [n, L, 3] -> F [n, L, 3]."""
    return CM.predict(arm, st, wi[:, None], wo[:, None])[:, 0]


def random_gates(rng, n, arm):
    g = rng.random((n, CM.NJ[arm], 3, 5))
    g[rng.random(g.shape) < 0.15] = 0.0
    g[rng.random(g.shape) < 0.15] = 1.0
    return g


def reciprocity(arm, st, rng, n, L=4096, zmin=0.05):
    wi, wo = rand_dirs(rng, n * L, zmin).reshape(n, L, 3), rand_dirs(rng, n * L, zmin).reshape(n, L, 3)
    Ff, Fr = flat_eval(arm, st, wi, wo), flat_eval(arm, st, wo, wi)
    ff, fr = Ff / wo[..., 2:], Fr / wi[..., 2:]
    return {"f_rel": rel(ff, fr), "cos_identity_rel": rel(wi[..., 2:] * Ff, wo[..., 2:] * Fr)}


def mc_albedo(arm, st, n, mus, rng, N=1 << 20):
    """Directional albedo per channel by uniform-hemisphere MC (pdf 1 / 2pi): returns mean [n, M, 3], se [n, M, 3]."""
    M = len(mus)
    mean = np.zeros((n, M, 3)); se = np.zeros((n, M, 3))
    for j, mu in enumerate(mus):
        wi1 = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
        u = rng.random((N, 2))
        z = u[:, 0]; r = np.sqrt(np.maximum(0, 1 - z * z)); ph = 2 * np.pi * u[:, 1]
        wo = np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)
        F = flat_eval(arm, st, np.broadcast_to(wi1, (n, N, 3)).copy(), np.broadcast_to(wo, (n, N, 3)).copy()) * 2 * np.pi
        mean[:, j] = F.mean(1); se[:, j] = F.std(1) / math.sqrt(N)
    return mean, se


def contracts_on(states, rng, n, label):
    """Reciprocity and passivity on a dict {name: (arm, st)}."""
    res = {}
    mus = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
    for name, (arm, st) in states.items():
        r = reciprocity(arm, st, rng, n)
        rg = reciprocity(arm, st, rng, n, L=1024, zmin=1e-3)
        # pointwise envelope vs the ungated base
        wi, wo = rand_dirs(rng, n * 4096, 1e-3).reshape(n, 4096, 3), rand_dirs(rng, n * 4096, 1e-3).reshape(n, 4096, 3)
        Fs, Fb = flat_eval(arm, st, wi, wo), flat_eval("A", {"y": st["y"]}, wi, wo)
        peak = Fb.max()
        env = {"min_F": float(Fs.min()), "max_excess_over_base_rel": float((Fs - Fb).max() / peak)}
        mean, se = mc_albedo(arm, st, n, mus, rng, N=1 << 18 if label == "fitted" else 1 << 20)
        alb = {"max_albedo": float(mean.max()), "max_albedo_plus_4se": float((mean + 4 * se).max()),
               "max_albedo_minus_4se": float((mean - 4 * se).max())}
        with torch.no_grad():
            lam = GF.decode(torch.as_tensor(st["y"], device=DEV, dtype=torch.float64))["lam"].sum(-1).max().item()
        ok = (r["f_rel"] <= 1e-12 and r["cos_identity_rel"] <= 1e-12 and env["min_F"] >= 0 and env["max_excess_over_base_rel"] <= 1e-15
              and alb["max_albedo_minus_4se"] <= 1.0 and lam <= 1 + 1e-6)
        res[name] = {"reciprocity_interior": r, "reciprocity_grazing_included": rg, "envelope": env, "mc_albedo": alb,
                     "max_sum_lambda": lam, "passed": bool(ok)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fitted", type=Path)
    args = ap.parse_args()
    rows, raw, compiled, desc, ref, seen = CM.inputs()
    n = len(rows)
    rng = np.random.default_rng(CM.P["seeds"]["tests"])
    res = {}
    if args.fitted:
        R = len(CM.read(args.fitted / "fit.json")["start_keys"])
        with np.load(args.fitted / "states.npz") as z:
            S = {k: z[k] for k in z.files}
        states = {}
        for arm in ("T", "P"):
            for r in range(R):
                states[f"{arm}_start{r}"] = (arm, {"y": S[f"{arm}_y"].reshape(R, n, 58)[r],
                                                   "g": S[f"{arm}_g"].reshape(R, n, *S[f"{arm}_g"].shape[1:])[r]})
        res["fitted"] = contracts_on(states, rng, n, "fitted")
        res["passed"] = all(v["passed"] for v in res["fitted"].values())
        res["commit"] = CM.commit()
        CM.write(args.out, res)
        print("fitted contracts", res["passed"], flush=True)
        return
    # ---------------- base states
    base = {"R0": GF.from_original(raw["R0"]), "WL104D_A": GF.from_original(raw["A"]), "WL104D_C": GF.from_original(raw["C"]),
            "ER_L": GF.from_original(raw["ER_L"])}
    yr = GF.from_original(CM.J.CI.generic(n)) + rng.standard_normal((n, 58))
    base["random"] = yr
    ye = yr.copy(); ye[:, GF.SL_KNOT] = rng.choice([-8.0, 8.0], (n, 15)); ye[:, :21] = rng.choice([-8.0, 8.0], (n, 21))
    base["extreme"] = ye
    ev, _ = CM.eval_reference(ref, n)
    L = 512
    wi_r, wo_r = ev["wi"][:, :, :L], ev["wo"][:, :, :L]
    # ---------------- 1 baseline identity (float64, reference lanes) and float32 training path
    t1 = {}
    for name, y in base.items():
        FA = CM.predict("A", {"y": y}, wi_r, wo_r)
        for arm in ("T", "P"):
            t1[f"{name}_{arm}"] = rel(CM.predict(arm, {"y": y, "g": CM.identity_gates(n, arm)}, wi_r, wo_r), FA)
    fit = CM.load_set("fit_set", n)
    data = FT.FitData(fit, n, 1)
    oA = FT.objective(data, "A", {"y": base["R0"]})
    f32 = {arm: float(np.max(np.abs(FT.objective(data, arm, {"y": base["R0"], "g": CM.identity_gates(n, arm)}) - oA) / oA))
           for arm in ("T", "P")}
    # float32 response identity on fit lanes
    with torch.no_grad():
        p, _ = CM.decode("A", {"y": base["R0"]}, torch.float32)
        lanes = 4096
        wi = torch.as_tensor(fit["wi"][:, :, :lanes // CM.NP].reshape(-1, 3), device=DEV, dtype=torch.float32)
        wo = torch.as_tensor(fit["wo"][:, :, :lanes // CM.NP].reshape(-1, 3), device=DEV, dtype=torch.float32)
        rep = wi.shape[0] // n
        pp = {k: v[:, None].expand(n, rep, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
        Fa = CL.eval_closure("ER", pp, wi, wo).cpu().numpy()
        f32_resp = {}
        for arm in ("T", "P"):
            g = torch.ones(n * rep, CM.NJ[arm], 3, 5, device=DEV)
            f32_resp[arm] = rel(CM.eval_sdm(pp, g, wi, wo).cpu().numpy(), Fa)
    res["baseline_identity"] = {"float64_reference_lanes": t1, "float32_objective_rel": f32, "float32_response_rel": f32_resp}
    ok_id = max(t1.values()) <= 1e-12 and max(f32_resp.values()) <= 2e-6 and max(f32.values()) <= 2e-6
    # ---------------- 2 + 3 reciprocity and passivity (random gates incl. 0 / 1 knots)
    states = {}
    for name in ("R0", "random", "extreme", "WL104D_A"):
        for arm in ("T", "P"):
            states[f"{name}_{arm}"] = (arm, {"y": base[name], "g": random_gates(rng, n, arm)})
        states[f"{name}_A"] = ("A", {"y": base[name]})
    res["contracts"] = contracts_on(states, rng, n, "random")
    ok_rp = all(v["passed"] for v in res["contracts"].values())
    # ---------------- 4 gradients (float64) at identity and interior gates, base parameters
    data64 = FT.FitData(fit, n, 1)
    for k in ("wi", "wo", "ref", "w"):
        setattr(data64, k, getattr(data64, k).double())
    grads = {}
    for arm, gname, gval in (("P", "identity", 1.0), ("P", "interior", 0.7), ("T", "identity", 1.0)):
        y = torch.as_tensor(base["R0"], dtype=torch.float64, device=DEV)
        g = torch.full((n, CM.NJ[arm], 3, 5), gval, dtype=torch.float64, device=DEV)
        for key, idx in (("g", (2, 0 if arm == "T" else 3, 1, 2)), ("g", (5, 0, 0, 4)), ("y", (1, 23))):
            x = (g if key == "g" else y).clone().requires_grad_(True)
            st = {"y": y, "g": g, key: x}
            obj = data64.score(arm, st).sum()
            ga = torch.autograd.grad(obj, x)[0][idx].item()
            h = 1e-6
            v = []
            for sgn in (1, -1):
                xx = (g if key == "g" else y).clone(); xx[idx] += sgn * h
                with torch.no_grad():
                    v.append(data64.score(arm, {"y": y, "g": g, key: xx}).sum().item())
            gfd = (v[0] - v[1]) / (2 * h)
            grads[f"{arm}_{gname}_{key}_{idx}"] = {"autograd": ga, "fd": gfd, "rel": abs(ga - gfd) / max(abs(gfd), 1e-12)}
    res["gradients"] = grads
    ok_gr = all(v["rel"] <= 1e-5 for v in grads.values()) and all(abs(v["autograd"]) > 0 for k, v in grads.items() if "_g_" in k)
    # ---------------- 5 trainability: 200 steps from identity on the physical data, every material moves a gate
    tr = {}
    for arm in ("T", "P"):
        term, _ = FT.train(data, arm, {"y": base["R0"], "g": CM.identity_gates(n, arm)}, 200)
        mv = np.abs(1 - term["g"]).reshape(n, -1).max(1)
        tr[arm] = {"max_move_per_material": mv.tolist(), "all_materials_moved": bool((mv > 1e-4).all()),
                   "within_box": bool((term["g"] >= 0).all() and (term["g"] <= 1).all())}
    res["trainability"] = tr
    ok_tr = all(v["all_materials_moved"] and v["within_box"] for v in tr.values())
    # ---------------- 6 Dr.Jit parity of the experiment-local evaluator
    yR = base["R0"]
    comp = GF.compile_states(yR)
    comp = comp.cpu().numpy() if torch.is_tensor(comp) else np.asarray(comp)
    par = {}
    wi_f, wo_f = rand_dirs(rng, 1 << 14, 1e-3), rand_dirs(rng, 1 << 14, 1e-3)
    base_par, id_par = [], []
    for m in (0, 2, 7):
        Ft = CM.predict("A", {"y": yR[m:m + 1]}, wi_f[None, None], wo_f[None, None])[0, 0]
        Fa = CM.J.CI.drjit_eval(comp[m], wi_f, wo_f)                          # unmodified compiled runtime
        base_par.append(rel(Fa, Ft))
        id_par.append(rel(CM.dr_eval_numpy(CM.DrSDM(np.asarray(comp[m]), np.ones((8, 3, 5))), wi_f, wo_f), Fa))
    for arm in ("T", "P"):
        g = random_gates(rng, n, arm)
        errs = []
        for m in (0, 2, 7):
            Fd = CM.dr_eval_numpy(CM.DrSDM(np.asarray(comp[m]), g[m]), wi_f, wo_f)
            Ft = CM.predict(arm, {"y": yR[m:m + 1], "g": g[m:m + 1]}, wi_f[None, None], wo_f[None, None])[0, 0]
            errs.append(rel(Fd, Ft))
        par[arm] = max(errs)
    par["base_runtime_vs_torch"] = max(base_par)
    par["sdm_identity_vs_base_runtime"] = max(id_par)
    res["drjit_parity"] = par
    ok_dr = (max(par["T"], par["P"]) <= max(2e-6, 1.5 * par["base_runtime_vs_torch"]) and par["sdm_identity_vs_base_runtime"] <= 2e-6)
    # ---------------- 7 synthetic representability (truth arm at truth state reproduces itself; ST / SP differ from A)
    syn = {}
    for ctl in ("ST", "SP"):
        tarm, truth = FT.synthetic_truth(ctl, yR)
        Ft = CM.predict(tarm, truth, wi_r, wo_r)
        syn[f"{ctl}_differs_from_shared_base"] = rel(Ft, CM.predict("A", {"y": yR}, wi_r, wo_r))
        if ctl == "SP":
            syn["SP_as_P_equals_truth"] = rel(CM.predict("P", truth, wi_r, wo_r), Ft)
        else:
            gP = np.repeat(truth["g"], 8, 1)
            syn["ST_representable_by_P"] = rel(CM.predict("P", {"y": yR, "g": gP}, wi_r, wo_r), Ft)
    res["synthetic_representability"] = syn
    ok_syn = syn["ST_representable_by_P"] <= 1e-12 and syn["SP_as_P_equals_truth"] == 0 and all(
        v > 1e-2 for k, v in syn.items() if "differs" in k)
    res["passed_parts"] = {"baseline_identity": bool(ok_id), "reciprocity_passivity": bool(ok_rp), "gradients": bool(ok_gr),
                           "trainability": bool(ok_tr), "drjit_parity": bool(ok_dr), "synthetic_representability": bool(ok_syn)}
    res["passed"] = all(res["passed_parts"].values())
    res["commit"] = CM.commit()
    res["inputs_sha256"] = seen
    CM.write(args.out, res)
    print(json.dumps(res["passed_parts"]), json.dumps(res["baseline_identity"]["float32_response_rel"]), json.dumps(par), flush=True)
    if not res["passed"]:
        raise IO.Stop("CONTRACT_TESTS_FAILED")


if __name__ == "__main__":
    main()
