"""WL104K-A Stage B: matched optimization extension (12,000 steps, constant lr 3e-4, fresh Adam moments) of every WL104K 12k state,
stationarity accounting and angular evaluation of 12k vs extended states.

Usage: python extend104ka.py fit <out_dir>          (writes a WL104K-format fit dir: fit.json + states.npz)
       python extend104ka.py eval <ext_dir> <out_json>
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104ka as KA

CM, FT, EV, IO = KA.CM, KA.FT, KA.EV, KA.IO
DEV = CM.DEV
STEPS = int(os.environ.get("WL104KA_DEV_STEPS", 12000))      # dev smoke only; evidence = 12000
LR = 3e-4
LOG = 100


def train_const(data, arm, init, steps):
    st = FT.tensors(init)
    params = {k: torch.nn.Parameter(st[k].clone()) for k in FT.TRAINABLE[arm]}
    opt = torch.optim.Adam(list(params.values()), lr=LR)
    hist = {}
    for step in range(steps + 1):
        sc = data.score(arm, {**st, **params})
        if step % LOG == 0 or step == steps:
            v = sc.detach().cpu().numpy()
            if not np.isfinite(v).all():
                raise IO.Stop(f"NONFINITE {arm} {step}")
            hist[step] = v
        if step < steps:
            opt.zero_grad(set_to_none=True)
            sc.sum().backward()
            opt.step()
            if "g" in params:
                with torch.no_grad():
                    params["g"].clamp_(0.0, 1.0)
    term = {k: (params[k].detach().cpu().numpy().astype(np.float64) if k in params else np.asarray(v, np.float64)) for k, v in init.items()}
    return term, hist


def stationarity(data, arm, st):
    """Full-batch gradient norm and KKT-projected gradient norm (per unit [R, n] summed), active constraint counts."""
    t = FT.tensors(st)
    params = {k: t[k].clone().requires_grad_(True) for k in FT.TRAINABLE[arm]}
    sc = data.score(arm, {**t, **params})
    grads = torch.autograd.grad(sc.sum(), list(params.values()))
    g = dict(zip(params, grads))
    out = {"grad_norm": float(torch.sqrt(sum((v ** 2).sum() for v in g.values())).item())}
    if "g" in g:
        q = params["g"].detach()
        pg = g["g"].clone()
        pg[(q <= 0) & (pg > 0)] = 0.0
        pg[(q >= 1) & (pg < 0)] = 0.0
        out["projected_grad_norm"] = float(torch.sqrt((g["y"] ** 2).sum() + (pg ** 2).sum()).item())
        out["gate_knots_at_0"] = float((q <= 1e-7).float().mean().item())
        out["gate_knots_at_1"] = float((q >= 1 - 1e-7).float().mean().item())
    else:
        out["projected_grad_norm"] = out["grad_norm"]
    return out


def run_fit(out):
    out.mkdir(parents=True, exist_ok=False)
    seen = KA.authenticate_k()
    rows, raw, _, _, _, seen2 = CM.inputs()
    seen.update(seen2)
    n = len(rows)
    fit = CM.load_set("fit_set", n)
    with np.load(KA.K_ROOT / "fit_primary/states.npz") as z:
        S = {k: z[k] for k in z.files}
    fj = KA.read(KA.K_ROOT / "fit_primary/fit.json")
    R = len(fj["start_keys"])
    data = FT.FitData(fit, n, R)
    log = {"start_keys": fj["start_keys"], "restart_semantics": "fresh Adam moments (12k optimizer state not archived), constant lr 3e-4, "
                                                                 "12,000 steps, gate projection after each step"}
    o = {}
    # stationarity at the common physical start (reference scale)
    y0 = FT.starts_y(n, fj["start_keys"])
    for arm in CM.ARMS:
        init = {"y": S[f"{arm}_y"]} if arm == "A" else {"y": S[f"{arm}_y"], "g": S[f"{arm}_g"]}
        start_ref = {"y": y0} if arm == "A" else {"y": y0, "g": CM.identity_gates(R * n, arm)}
        st0 = stationarity(data, arm, start_ref)
        st12 = stationarity(data, arm, init)
        obj12 = FT.objective(data, arm, init)
        torch.cuda.synchronize(); t0 = time.monotonic()
        term, hist = train_const(data, arm, init, STEPS)
        torch.cuda.synchronize(); secs = time.monotonic() - t0
        st24 = stationarity(data, arm, term)
        steps = np.array(sorted(hist)); H = np.stack([hist[s] for s in steps])
        k = 2000 // LOG
        last = np.abs(H[-1] - H[-1 - k]) / np.abs(H[-1])
        log[arm] = {"history_steps": steps.tolist(), "history": H.tolist(), "initial_objective": obj12.tolist(),
                    "final_objective": H[-1].tolist(), "last_2000_relative_change": last.tolist(),
                    "rule_a_fraction": float((last <= 0.005).mean()), "rule_a_pass": bool((last <= 0.005).mean() >= 0.9),
                    "seconds": secs, "stationarity": {"start": st0, "12k": st12, "extended": st24},
                    "relative_objective_change_12k_to_extended": ((H[-1] - obj12) / obj12).tolist()}
        if arm != "A":
            g = term["g"]
            log[arm]["gate_stats"] = {"fraction_at_1": float((g >= 1 - 1e-7).mean()), "fraction_at_0": float((g <= 1e-7).mean())}
        for kk, v in term.items():
            o[f"{arm}_{kk}"] = v
        print(arm, "ext", round(secs, 1), "rule a", log[arm]["rule_a_fraction"], flush=True)
    np.savez(out / "states.npz", **o)
    log.update({"commit": CM.commit(), "protocol_sha256": CM.sha(KA.HERE / "protocol.json"), "inputs_sha256": seen,
                "states_sha256": CM.sha(out / "states.npz"), "trainable_per_material": {"A": 58, "T": 73, "P": 178},
                "optimization_qualification": "PASS" if all(log[a]["rule_a_pass"] for a in CM.ARMS) else "FAIL"})
    KA.write(out / "fit.json", log)


def run_eval(ext, out):
    rows, raw, _, _, ref, seen = CM.inputs()
    n = len(rows)
    ev, _ = CM.eval_reference(ref, n)
    coh = {c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}
    fj = KA.read(ext / "fit.json")
    if CM.sha(ext / "states.npz") != fj["states_sha256"]:
        raise IO.Stop("EXT_STATES_HASH")
    s12, R, _ = KA.states(KA.K_ROOT / "fit_primary", n)
    s24, _, _ = KA.states(ext, n)
    res = {}
    lanes = {}
    for tag, st in (("12k", s12), ("extended", s24)):
        el = {arm: np.stack([EV.metrics(CM.predict(arm, s, ev["wi"], ev["wo"]), ev, ev["rad"] if "rad" in ev else None)["el1"] for s in st[arm]])
              for arm in CM.ARMS}
        el_a = {arm: np.stack([CM.wl82_terms(CM.predict(arm, s, ev["wi"], ev["wo"]), ev["a"], ev["w"])["el1"] for s in st[arm]]) for arm in CM.ARMS}
        el_b = {arm: np.stack([CM.wl82_terms(CM.predict(arm, s, ev["wi"], ev["wo"]), ev["b"], ev["w"])["el1"] for s in st[arm]]) for arm in CM.ARMS}
        rng = np.random.default_rng(KA.P["seeds"]["bootstrap"])
        con = {}
        for name, (x, y) in {"T-A": ("T", "A"), "P-T": ("P", "T"), "P-A": ("P", "A")}.items():
            con[name] = {c: {**EV.contrast(el[x], el[y], ms, rng), "half_a": EV.contrast(el_a[x], el_a[y], ms, rng)["median"],
                             "half_b": EV.contrast(el_b[x], el_b[y], ms, rng)["median"]} for c, ms in coh.items()}
        elA, elP = np.median(el["A"], 0), np.median(el["P"], 0)
        mat = (elP.mean(1) / elA.mean(1) - 1).tolist()
        grp = {c: (elP[ms].reshape(len(ms), CM.G, 2).sum((0, 2)) / elA[ms].reshape(len(ms), CM.G, 2).sum((0, 2)) - 1).tolist() for c, ms in coh.items()}
        res[tag] = {"contrasts": con, "pooled_el1": {arm: float(np.median(el[arm], 0).mean()) for arm in CM.ARMS},
                    "per_start_el1_mean": {arm: el[arm].mean((1, 2)).tolist() for arm in CM.ARMS},
                    "material_change_P_vs_A": mat, "group_change_P_vs_A": grp}
        lanes[tag] = {arm: [CM.predict(arm, s, ev["wi"], ev["wo"]) @ CM.LUMA for s in st[arm]] for arm in CM.ARMS}
    # response change 12k -> extended on evaluation lanes
    w = ev["w"]
    res["response_change_rel_rms"] = {arm: [float(np.sqrt((w * (a - b) ** 2).sum() / (w * b ** 2).sum())) for a, b in zip(lanes["extended"][arm], lanes["12k"][arm])]
                                      for arm in CM.ARMS}
    stab = {}
    for name in ("T-A", "P-T", "P-A"):
        for c in coh:
            a, b = res["12k"]["contrasts"][name][c]["median"], res["extended"]["contrasts"][name][c]["median"]
            stab[f"{name}|{c}"] = {"12k": a, "extended": b, "change_pp": 100 * (b - a), "same_sign": bool(np.sign(a) == np.sign(b))}
    res["stability"] = stab
    res["angular_stability"] = "STABLE" if all(abs(v["change_pp"]) <= 3 and v["same_sign"] for v in stab.values()) else "NOT_STABLE"
    cpa = res["extended"]["contrasts"]["P-A"]
    res["wl104k_gain_criteria_on_extended_states_descriptive"] = bool(
        all(cpa[c]["median"] <= -0.10 and cpa[c]["hi"] < 0 and cpa[c]["half_a"] < 0 and cpa[c]["half_b"] < 0 and sum(x < 0 for x in cpa[c]["per_start"]) >= 3 for c in coh)
        and max(res["extended"]["material_change_P_vs_A"]) <= 0.05 and all(max(v) <= 0.05 for v in res["extended"]["group_change_P_vs_A"].values()))
    res.update({"optimization_qualification": fj["optimization_qualification"], "commit": CM.commit(), "inputs_sha256": seen})
    KA.write(out, res)
    print("ext eval", res["angular_stability"], {k: round(v["change_pp"], 2) for k, v in stab.items()}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        run_fit(Path(sys.argv[2]))
    else:
        run_eval(Path(sys.argv[2]), Path(sys.argv[3]))
