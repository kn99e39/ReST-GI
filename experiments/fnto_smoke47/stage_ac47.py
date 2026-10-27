"""WL47 Stage A (synthetic oracle evidence) and Stage C (A0 / A1 / B0 / B1 closed-form linear operators).

    python stage_ac47.py a <out dir>                       # Stage A: S0 / S1 oracle, paired equal-irradiance fields
    python stage_ac47.py c <out dir> [<mir T npz>]         # Stage C on S0, S1 (and MIR when the renderer basis exists)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke47_core as K  # noqa: E402


def stage_a(out):
    rng = np.random.default_rng(4791)
    LA = K.sample_fields(64, K.P, K.J, K.OMP, rng); keep = (K.EMAP_S @ LA.T).sum(0) > 0; LA = LA[keep][:32]
    LB = K.pair_fields(LA, K.P, K.J, K.OMP, rng)
    rec = {"bins": {"theta_deg": np.degrees(K.TH).round(3).tolist(), "phi_deg": np.degrees(K.PH).round(1).tolist(),
                    "omega_perp": K.OMP.round(6).tolist(), "sum_omega_perp": float(K.OMP.sum())}}
    for name, T in (("S0", K.oracle_s0()), ("S1", K.oracle_s1())):
        EA, EB = K.EMAP_S @ LA.T, K.EMAP_S @ LB.T
        YA, YB = T @ LA.T, T @ LB.T
        nA, nB = np.linalg.norm(YA, axis=0), np.linalg.norm(YB, axis=0)
        D = np.linalg.norm(YA - YB, axis=0) / np.maximum(np.maximum(nA, nB), 1e-30)
        nl = np.ones_like(T, bool)
        for q in range(K.P):
            nl[q * K.J:(q + 1) * K.J, q * K.J:(q + 1) * K.J] = False
        Dnl = np.linalg.norm((T * nl) @ (LA - LB).T, axis=0) / np.maximum(np.linalg.norm((T * nl) @ LA.T, axis=0), 1e-30)
        ang_in = np.linalg.norm(LA - LB, axis=1) / np.linalg.norm(LA, axis=1)
        alb = K.flux_albedo(T, np.tile(K.OMP, K.P), np.tile(K.OMP, K.P))
        rec[name] = {
            "E_match_max_rel": float(np.max(np.abs(EA - EB) / np.maximum(EA, 1e-30) * (EA > 0))),
            "angular_input_rel_diff_mean": float(ang_in.mean()),
            "output_rel_diff_mean": float(D.mean()), "output_rel_diff_median": float(np.median(D)), "output_rel_diff_max": float(D.max()),
            "nonlocal_output_rel_diff_mean": float(np.nanmean(Dnl)),
            "per_pair_output_rel_diff": D.round(6).tolist(),
            "max_column_flux_albedo": float(alb.max()), "min_entry": float(T.min()),
            "nonlocal_share": float(np.linalg.norm((T * nl) @ LA.T) / np.linalg.norm(T @ LA.T)),
            "rank": int(np.linalg.matrix_rank(T)),
        }
        np.savez(out / f"stage_a_{name}.npz", T=T, LA=LA, LB=LB, YA=YA.T, YB=YB.T)
    c = PA_crit(rec)
    rec["criteria"] = c
    (out / "stage_a.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: rec[k] for k in ("criteria",)}, indent=1))


def PA_crit(rec):
    return {"S1_sensitive": bool(rec["S1"]["output_rel_diff_mean"] >= 0.10 and rec["S1"]["E_match_max_rel"] <= 1e-9),
            "S0_insensitive": bool(rec["S0"]["output_rel_diff_max"] <= 1e-9)}


def run_regime(name, T, cvec, P_in, J_in, n_out_ports, n_dirs, mask, lap, nl, Ttr=None, Tnoise=None):
    """Four conditions x two training seeds on one regime. T = evaluation operator (radiance-bin input)."""
    Emap = K.emap(P_in, cvec); Ttr = T if Ttr is None else Ttr
    rt = np.random.default_rng(K.PC["data"]["test_seed"])
    Lr = K.sample_fields(K.PC["data"]["test_random"], P_in, J_in, cvec, rt)
    LA = K.sample_fields(4 * K.PC["data"]["test_pairs"], P_in, J_in, cvec, rt)
    LA = LA[(Emap @ LA.T).sum(0) > 0][:K.PC["data"]["test_pairs"]]; LB = K.pair_fields(LA, P_in, J_in, cvec, rt)
    Lte = np.concatenate([Lr, LA, LB]).T; pairs = (LA.T, LB.T)
    res = {"n_test": int(Lte.shape[1]), "floors": K.eonly_floors(T, Emap, Lte, pairs)}
    if Tnoise is not None:
        res["reference_noise_floor"] = K.rel(Tnoise @ Lte, T @ Lte) / np.sqrt(2)
    for seed in K.PC["data"]["train_seeds"]:
        Ltr = K.sample_fields(K.PC["data"]["train"], P_in, J_in, cvec, np.random.default_rng(seed)).T
        tr_keys = {Ltr[:, i].tobytes() for i in range(Ltr.shape[1])}
        assert not any(Lte[:, i].tobytes() in tr_keys for i in range(Lte.shape[1])), "train / test overlap"
        Ytr = Ttr @ Ltr
        for cond in ("A0", "A1", "B0", "B1"):
            m = mask if cond.endswith("0") else np.ones_like(mask)
            Z = Emap @ Ltr if cond[0] == "A" else Ltr
            t0 = time.perf_counter(); Wm = K.fit(Z, Ytr, m, lap); dt = time.perf_counter() - t0
            Wrad = Wm @ Emap if cond[0] == "A" else Wm
            ev = K.evaluate(Wrad, T, Ltr, Lte, pairs, mask, n_out_ports, n_dirs, nl, Ttr=Ttr)
            ev["train_rel"] = K.rel((Wm @ Z)[m], Ytr[m])
            if Tnoise is not None:
                ev["heldout_rel_vs_replicate"] = K.rel(Wrad @ Lte, Tnoise @ Lte)
            if cvec is K.OMP:
                alb = K.flux_albedo(Wrad, np.tile(K.OMP, P_in), np.tile(K.OMP, n_out_ports)); ev["max_flux_albedo"] = float(alb.max())
                ev["n_cols_albedo_gt1"] = int((alb > 1 + 1e-9).sum())
            ev["params"] = int(Wm.size); ev["fit_s"] = dt
            res[f"{cond}_seed{seed}"] = ev
    return res


def effects(res, key):
    out = {}
    for seed in K.PC["data"]["train_seeds"]:
        g = lambda c: res[f"{c}_seed{seed}"][key]  # noqa: E731
        if g("A0") is None:
            continue
        out[f"seed{seed}"] = {"A0->A1": g("A1") - g("A0"), "A1->B1": g("B1") - g("A1"), "B0->B1": g("B1") - g("B0"),
                              "A0->B0": g("B0") - g("A0"), "interaction": (g("B1") - g("B0")) - (g("A1") - g("A0"))}
    return out


def stage_c(out, mir_npz=None):
    rec = {}
    nl = np.ones((K.P * K.J, K.P * K.J), bool)
    for q in range(K.P):
        nl[q * K.J:(q + 1) * K.J, q * K.J:(q + 1) * K.J] = False
    vis = np.zeros(K.J, bool); vis[[0, 1, 4, 5]] = True; mask = np.tile(vis, K.P)
    lap = K.lap_synthetic()
    for name, T in (("S0", K.oracle_s0()), ("S1", K.oracle_s1())):
        rec[name] = run_regime(name, T, K.OMP, K.P, K.J, K.P, K.J, mask, lap, nl)
    if mir_npz:
        d = np.load(mir_npz); T1, T2 = d["T1"], d["T2"]          # [4 ports * 8 dirs, 2 ports * 8 dirs]
        nd = len(K.MIR_T); vis = np.isin(K.MIR_T, K.PM["visible_dirs"]); mask = np.tile(vis, 4)
        nlm = np.ones_like(T1, bool)
        for p in range(2):
            nlm[p * nd:(p + 1) * nd, p * nd:(p + 1) * nd] = False
        rec["MIR"] = run_regime("MIR", T1, np.ones(nd), 2, nd, 4, nd, mask, K.lap_chain(4, nd), nlm, Tnoise=T2)
    for name in rec:
        rec[name]["effects_heldout"] = effects(rec[name], "heldout_rel")
        rec[name]["effects_hidden"] = effects(rec[name], "hidden_rel")
        rec[name]["criteria"] = crit(rec[name], name)
    (out / "stage_c.json").write_text(json.dumps(rec, indent=1))
    for name in rec:
        print(name, json.dumps(rec[name]["criteria"]))
        for c in ("A0", "A1", "B0", "B1"):
            e = rec[name][f"{c}_seed4701"]
            print(f"  {c} held {e['heldout_rel']:.4f} hid {e['hidden_rel']:.4f} vis {e['visible_rel']:.4f} dir {e['directional_rel']:.4f} "
                  f"nl {e['nonlocal_rel']:.4f} disc {e['pair_discrimination']:.3f} pde {e['pair_diff_err']:.3f} neg {e['neg_frac']:.3f} p {e['params']}")


def crit(r, name):
    out = {}
    floor = r.get("reference_noise_floor")
    for seed in K.PC["data"]["train_seeds"]:
        g = lambda c, k: r[f"{c}_seed{seed}"][k]  # noqa: E731
        lim = 0.05 if floor is None else max(0.05, 2 * floor)
        out[f"seed{seed}"] = {
            "directional_benefit": bool(g("B1", "heldout_rel") <= 0.5 * g("A1", "heldout_rel") and g("B1", "pair_discrimination") >= 0.8),
            "learned": bool(g("B1", "heldout_rel") <= lim),
            "supervision_benefit_A": bool(g("A1", "hidden_rel") <= 0.5 * g("A0", "hidden_rel")),
            "supervision_benefit_B": bool(g("B1", "hidden_rel") <= 0.5 * g("B0", "hidden_rel")),
            "irradiance_sufficient": bool(g("A1", "heldout_rel") <= 0.05 and r["floors"]["eonly_linear_floor"] <= 0.05),
        }
    return out


if __name__ == "__main__":
    o = Path(sys.argv[2]); o.mkdir(parents=True, exist_ok=True)
    stage_a(o) if sys.argv[1] == "a" else stage_c(o, sys.argv[3] if len(sys.argv) > 3 else None)
