"""WL104K-C Stage A: baseline and numerical qualification (protocol_stageA.json).

  python stageA104kc.py shape <out_dir>        batch-layout attribution: per-unit forward / gradient pieces at the WL104K-B starts and
                                               at the WL104K-B O0 final states in layouts L72 / L72swap / L72perm / L36 / L9, and the
                                               first 500 steps of the original O0 trajectory in each layout (per-step objectives)
  python stageA104kc.py compare <root>         reproduction comparisons (R4 vs archived WL104K P, R8 vs WL104K-B O0, S_O0r / S_Dr vs
                                               WL104K-B S_O0 / S_D) + numerical batch-size sensitivity on the WL104K-B DEV set vs
                                               across-start variability; qualification per protocol_stageA.json; writes stageA.json

The reproduction fits themselves are run by the chain with the UNCHANGED fit104kb.py (run_kb.py wrapper).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common104kc as C

KB, CM, FT = C.KB, C.CM, C.FT
import fit104kb as FK  # noqa: E402
import eval104kb as EB  # noqa: E402

DEV = C.DEV
N = C.N_MAT
K8 = KB.START_KEYS
K4 = KB.START_KEYS[:4]
SWAP = ["104918", "104919", "104920", "104921"]           # replacement content for start positions 5-8 (layout L72swap)
STEPS_TRAJ = 500
METRIC_KEYS = ("el1", "ealb", "eshape", "e_core", "e_shoulder", "core_l1", "shoulder_l1", "tail_l1", "el1_grazing", "e_prof",
               "e_h", "s_h")


def x_of(keys, n=N):
    y0 = KB.starts_y(n, keys)
    return torch.as_tensor(np.concatenate([y0, np.ones((len(keys) * n, 120))], 1), device=DEV, dtype=torch.float32)


def pieces(data, x):
    """Per-unit forward pieces and gradient (same operations as fit104kb.Data.terms + adam's backward)."""
    U = x.shape[0]
    xr = x.detach().clone().requires_grad_(True)
    y, gt = xr[:, :58], xr[:, 58:].reshape(U, 8, 3, 5)
    p = FK.GF.decode(y)
    rep = data.P * data.L
    pp = {k: v[:, None].expand(U, rep, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
    gg = gt[:, None].expand(U, rep, 8, 3, 5).reshape(-1, 8, 3, 5)
    pred = CM.eval_sdm(pp, gg, data.wi, data.wo)
    t = FT.F.group_terms(pred, data.ref, data.w, data.L)
    obj = (t["e1"] + t["e2"]).reshape(U, data.P).mean(-1)
    (gr,) = torch.autograd.grad(obj.sum(), xr)
    dec = torch.cat([v.reshape(U, -1) for v in p.values()], 1)
    return {"decoded": dec.detach().cpu().numpy(), "pred": pred.detach().reshape(U, rep, 3).cpu().numpy(),
            "e1": t["e1"].detach().reshape(U, data.P).cpu().numpy(), "e2": t["e2"].detach().reshape(U, data.P).cpu().numpy(),
            "objective": obj.detach().cpu().numpy(), "grad": gr.detach().cpu().numpy()}


def cmp(a, b):
    """Per-unit comparison of two arrays [U, ...]: bitwise-equal flag and max relative difference."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    eq = (a == b).reshape(len(a), -1).all(1)
    rel = (np.abs(a - b) / np.maximum(np.abs(b), 1e-30)).reshape(len(a), -1).max(1)
    return {"bitwise_equal_units": int(eq.sum()), "units": int(len(a)), "max_rel": float(rel.max()),
            "per_unit_max_rel": rel.tolist()}


def layout_rows(layout, R_ref=8):
    """For a layout, the row index of every compared (start, material) unit and the matching row in L72."""
    if layout == "L72swap":
        starts = range(4)
        return [r * N + m for r in starts for m in range(N)], [r * N + m for r in starts for m in range(N)]
    if layout == "L72perm":
        return [(7 - r) * N + m for r in range(8) for m in range(N)], [r * N + m for r in range(8) for m in range(N)]
    if layout == "L36":
        return [r * N + m for r in range(4) for m in range(N)], [r * N + m for r in range(4) for m in range(N)]
    if layout == "L9":
        return list(range(N)), list(range(N))
    raise ValueError(layout)


LAYOUT_KEYS = {"L72": K8, "L72swap": K4 + SWAP, "L72perm": K8[::-1], "L36": K4, "L9": K8[:1]}


def layout_x_final(layout, xO0):
    """O0 final states (WL104K-B O0, 72 rows start-major) arranged as the layout (L72swap: starts 5-8 replaced by starts 1-4 of
    the reversed order, i.e. different content)."""
    if layout == "L72":
        return xO0
    if layout == "L72swap":
        x = xO0.clone()
        x[4 * N:] = xO0[:4 * N].flip(0)
        return x
    if layout == "L72perm":
        return torch.cat([xO0[r * N:(r + 1) * N] for r in range(7, -1, -1)])
    if layout == "L36":
        return xO0[:4 * N]
    if layout == "L9":
        return xO0[:N]
    raise ValueError(layout)


def trajectory(data, x0, steps_total=12000, steps=STEPS_TRAJ):
    """First `steps` steps of fit104kb.adam (identical operations and lr schedule of the 12,000-step fit); per-step objectives."""
    OPT = FK.OPT
    y = torch.nn.Parameter(x0[:, :58].clone())
    g = torch.nn.Parameter(x0[:, 58:].clone())
    opt = torch.optim.Adam([y, g], lr=OPT["lr"])
    rec = []
    for step in range(steps + 1):
        sc = data.f(torch.cat([y, g], 1))
        rec.append(sc.detach().clone())
        if step < steps:
            opt.param_groups[0]["lr"] = FT.TR.lr_at(step, steps_total, OPT["lr"], OPT["lr_end"], OPT["warmup"])
            opt.zero_grad(set_to_none=True)
            sc.sum().backward()
            opt.step()
            with torch.no_grad():
                g.clamp_(0.0, 1.0)
    return torch.stack(rec).cpu().numpy(), torch.cat([y.detach(), g.detach()], 1).cpu().numpy()


def first_divergence(A, B):
    """A, B [steps + 1, units] -> per unit first step with A != B (None if never)."""
    ne = A != B
    out = []
    for u in range(A.shape[1]):
        idx = np.flatnonzero(ne[:, u])
        out.append(int(idx[0]) if idx.size else None)
    return out


def run_shape(out, steps=STEPS_TRAJ):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    seen = C.authenticate_kb(["primary/fits/O0/states.npz", "primary/fits/O0/fit.json"])
    ds = KB.training_set("original", N)
    datas = {L: FK.Data(ds, N, len(k), "original") for L, k in LAYOUT_KEYS.items()}
    xO0 = torch.as_tensor(C.flat_x(C.load_states(KB.OUT / "primary/fits/O0/states.npz", 8)), device=DEV, dtype=torch.float32)
    res = {"layouts": {L: {"start_keys": k, "units": len(k) * N} for L, k in LAYOUT_KEYS.items()}, "inputs_sha256": seen}
    for where in ("starts", "O0_final"):
        base = pieces(datas["L72"], x_of(K8) if where == "starts" else xO0)
        res[where] = {}
        for L in ("L72swap", "L72perm", "L36", "L9"):
            xl = x_of(LAYOUT_KEYS[L]) if where == "starts" else layout_x_final(L, xO0)
            pc = pieces(datas[L], xl)
            rows, ref = layout_rows(L)
            if where == "O0_final" and L == "L72swap":
                rows, ref = list(range(4 * N)), list(range(4 * N))
            res[where][L] = {k: {q: v for q, v in cmp(pc[k][rows], base[k][ref]).items() if q != "per_unit_max_rel"}
                             for k in ("decoded", "pred", "e1", "e2", "objective", "grad")}
            print(where, L, {k: (v["bitwise_equal_units"], v["units"], v["max_rel"]) for k, v in res[where][L].items()}, flush=True)
    traj = {}
    finals = {}
    for L, k in LAYOUT_KEYS.items():
        traj[L], finals[L] = trajectory(datas[L], x_of(k), steps=steps)
        print("trajectory", L, round(time.monotonic() - t0, 1), flush=True)
    res["trajectory"] = {"steps": steps, "schedule_total_steps": 12000}
    fj = KB.read(KB.OUT / "primary/fits/O0/fit.json")
    hs = fj["history_steps"]
    H = np.asarray(fj["history_objective"], float)          # [U, logged steps]
    logged = [i for i, s in enumerate(hs) if s <= steps]
    res["trajectory"]["L72_equals_WL104KB_O0_history"] = bool(
        all(np.array_equal(traj["L72"][hs[i]].astype(np.float64), H[:, i]) for i in logged))
    for L in ("L72swap", "L72perm", "L36", "L9"):
        rows, ref = layout_rows(L)
        fd = first_divergence(traj[L][:, rows], traj["L72"][:, ref])
        rel = np.abs(traj[L][-1, rows].astype(np.float64) - traj["L72"][-1, ref]) / np.abs(traj["L72"][-1, ref])
        res["trajectory"][L] = {"units": len(rows), "units_identical_all_steps": int(sum(v is None for v in fd)),
                                "first_divergence_step_min": min([v for v in fd if v is not None], default=None),
                                "first_divergence_step_median": (float(np.median([v for v in fd if v is not None]))
                                                                 if any(v is not None for v in fd) else None),
                                "final_step_objective_max_rel": float(rel.max()), "final_step_objective_median_rel": float(np.median(rel)),
                                "final_states_bitwise_equal_units": int((finals[L][rows] == finals["L72"][ref]).all(1).sum()),
                                "per_unit_first_divergence": fd}
        print("trajectory compare", L, {q: v for q, v in res["trajectory"][L].items() if q != "per_unit_first_divergence"}, flush=True)
    res["seconds"] = time.monotonic() - t0
    res["commit"] = C.commit()
    res["torch"] = torch.__version__
    res["gpu"] = torch.cuda.get_device_name()
    C.write(out / "shape.json", res)


# ------------------------------------------------------------------ reproduction + numerical sensitivity
def states_equal(a, b):
    """lists over starts of {'y','g'} -> per (start, material) bitwise equality of y and g."""
    eq = []
    for sa, sb in zip(a, b):
        for m in range(N):
            eq.append(bool(np.array_equal(sa["y"][m], sb["y"][m]) and np.array_equal(sa["g"][m], sb["g"][m])))
    return eq


def final_obj(fit_dir):
    return np.asarray(KB.read(fit_dir / "fit.json")["final_terms"]["objective"], float)


def rep_block(name, eq, fa, fb):
    rel = np.abs(fa - fb) / np.abs(fb)
    return {"comparison": name, "units": len(eq), "states_bitwise_equal_units": int(sum(eq)),
            "final_objective_max_rel": float(rel.max()), "passed": bool(all(eq))}


def per_unit(M, k):
    """[R, n, P] -> mean over probes [R, n]."""
    return M[k].mean(-1)


def run_compare(root):
    t0 = time.monotonic()
    A = C.PA
    out = root / "stageA"
    seen = C.authenticate_kb(["primary/fits/O0/states.npz", "primary/fits/O0/fit.json", "primary/fits/S_O0/states.npz",
                              "primary/fits/S_O0/fit.json", "primary/fits/S_D/states.npz", "primary/fits/S_D/fit.json",
                              "primary/amendment1/fits/R4a/states.npz", "primary/data/dev.npz", "primary/data/dev_peak.npz"])
    hist, kfj, seen_k = C.archived_P()
    seen.update(seen_k)
    fits = root / "stageA/fits"
    r4, r8 = C.load_states(fits / "R4/states.npz", 4), C.load_states(fits / "R8/states.npz", 8)
    o0 = C.load_states(KB.OUT / "primary/fits/O0/states.npz", 8)
    so0r, sdr = C.load_states(fits / "S_O0r/states.npz", 4), C.load_states(fits / "S_Dr/states.npz", 4)
    so0, sd = C.load_states(KB.OUT / "primary/fits/S_O0/states.npz", 4), C.load_states(KB.OUT / "primary/fits/S_D/states.npz", 4)
    r4a = C.load_states(KB.OUT / "primary/amendment1/fits/R4a/states.npz", 4)
    arch_obj = np.asarray(kfj["P"]["final_objective"], float).reshape(-1)
    q = {"Q1_historical_R4_vs_WL104K_P": rep_block("R4 vs archived WL104K P", states_equal(r4, hist), final_obj(fits / "R4"), arch_obj),
         "Q2_control_R8_vs_WL104KB_O0": rep_block("R8 vs WL104K-B O0", states_equal(r8, o0), final_obj(fits / "R8"),
                                                 final_obj(KB.OUT / "primary/fits/O0")),
         "Q4a_synthetic_S_O0r_vs_WL104KB_S_O0": rep_block("S_O0r vs WL104K-B S_O0", states_equal(so0r, so0), final_obj(fits / "S_O0r"),
                                                         final_obj(KB.OUT / "primary/fits/S_O0")),
         "Q4b_synthetic_S_Dr_vs_WL104KB_S_D": rep_block("S_Dr vs WL104K-B S_D", states_equal(sdr, sd), final_obj(fits / "S_Dr"),
                                                       final_obj(KB.OUT / "primary/fits/S_D"))}
    shape = C.read(out / "shape.json")
    tr = shape["trajectory"]["L72swap"]
    q["Q3_content_invariance_L72swap"] = {"units": tr["units"], "units_identical_all_steps": tr["units_identical_all_steps"],
                                          "final_states_bitwise_equal_units": tr["final_states_bitwise_equal_units"],
                                          "passed": bool(tr["units_identical_all_steps"] == tr["units"]
                                                         and tr["final_states_bitwise_equal_units"] == tr["units"])}
    q["Q3b_L72_trajectory_equals_WL104KB_O0_history"] = {"passed": bool(shape["trajectory"]["L72_equals_WL104KB_O0_history"])}
    qualified = all(v["passed"] for v in q.values())
    # explanation of the WL104K-B O0-vs-archive discrepancy
    o0_obj, r4_obj = final_obj(KB.OUT / "primary/fits/O0")[:4 * N], final_obj(fits / "R4")
    rel_o0_arch = np.abs(o0_obj - arch_obj) / arch_obj
    rel_r8_r4 = np.abs(final_obj(fits / "R8")[:4 * N] - r4_obj) / r4_obj
    expl = {"O0_vs_archive_units_differing": int((rel_o0_arch > 0).sum()),
            "R8_vs_R4_units_differing": int((rel_r8_r4 > 0).sum()),
            "O0_vs_archive_equals_R8_vs_R4": bool(np.array_equal(rel_o0_arch, rel_r8_r4)),
            "R4_equals_WL104KB_R4a_units": int(sum(states_equal(r4, r4a))),
            "unexplained_units": int(sum(not a for a in states_equal(r4, hist))) + int(sum(not a for a in states_equal(r8, o0))),
            "max_rel_final_objective_batch4_vs_batch8": float(rel_r8_r4.max()),
            "median_rel_final_objective_batch4_vs_batch8": float(np.median(rel_r8_r4))}
    # numerical batch-size sensitivity on the WL104K-B DEV set vs across-start variability
    ds, pk = EB.load_sets(KB.OUT / "primary", "dev", N)
    M4, _ = EB.evaluate("P", r4, ds, pk)
    M8, _ = EB.evaluate("P", r8, ds, pk)
    sens = {}
    for k in METRIC_KEYS:
        a4, a8 = per_unit(M4, k), per_unit(M8, k)                 # [4, n], [8, n]
        d = a4 - a8[:4]                                            # batch-size difference per (start, material)
        sd = a8.std(0, ddof=1)                                     # across-start SD per material (8 starts, batch 8)
        ratio = np.abs(d) / np.maximum(sd[None], 1e-30)
        pooled = (M4[k].sum((1, 2)) / M8[k][:4].sum((1, 2)) - 1) if k != "s_h" else None
        sens[k] = {"batch_diff_per_unit": d.tolist(), "across_start_sd_per_material": sd.tolist(),
                   "abs_diff_over_sd_median": float(np.median(ratio)), "abs_diff_over_sd_max": float(ratio.max()),
                   "units_diff_over_sd_above_0.5": int((ratio > 0.5).sum()),
                   "pooled_rel_change_batch4_vs_batch8_per_start": None if pooled is None else pooled.tolist(),
                   "pooled_rel_change_batch4_vs_batch8_median": None if pooled is None else float(np.median(pooled))}
    inst8 = EB.instability(M8["s_h"][:4])
    inst4 = EB.instability(M4["s_h"])
    sens["instability_starts1to4"] = {"batch8": inst8.tolist(), "batch4": inst4.tolist(),
                                      "pooled_rel_change": float(inst4.mean() / inst8.mean() - 1)}
    res = {"protocol": "protocol_stageA.json", "protocol_sha256": C.sha(C.PROTOCOL_A), "qualification": q, "qualified": qualified,
           "verdict": "BASELINE_QUALIFIED" if qualified else "BASELINE_NOT_QUALIFIED", "explanation": expl,
           "shape_attribution": {w: {L: {k: (v["bitwise_equal_units"], v["units"], v["max_rel"]) for k, v in d.items()}
                                     for L, d in shape[w].items()} for w in ("starts", "O0_final")},
           "trajectory": {L: {k: v for k, v in shape["trajectory"][L].items() if k != "per_unit_first_divergence"}
                          for L in ("L72swap", "L72perm", "L36", "L9")},
           "numerical_sensitivity_dev": sens, "materials": [C.NAMES[r] for r in KB.read(fits / "R8/fit.json")["materials"]],
           "fit_seconds": {k: KB.read(fits / k / "fit.json")["seconds"] for k in ("R4", "R8", "S_O0r", "S_Dr")},
           "fit_states_sha256": {k: C.sha(fits / k / "states.npz") for k in ("R4", "R8", "S_O0r", "S_Dr")},
           "inputs_sha256": seen, "shape_json_sha256": C.sha(out / "shape.json"), "commit": C.commit(),
           "seconds": time.monotonic() - t0}
    C.write(out / "stageA.json", res)
    print(res["verdict"], {k: v["passed"] for k, v in q.items()}, expl, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("shape", "compare"))
    ap.add_argument("path", type=Path)
    ap.add_argument("--steps", type=int, default=STEPS_TRAJ, help="trajectory steps (protocol: 500; smaller only for debugging)")
    a = ap.parse_args()
    if a.mode == "shape":
        run_shape(a.path, a.steps)
    else:
        run_compare(a.path)


if __name__ == "__main__":
    main()
