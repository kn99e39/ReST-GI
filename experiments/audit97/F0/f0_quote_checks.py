"""WL97 audit, track F0: cheap raw-number checks behind the inference-chain review.

Read-only on every existing file. numpy only. Writes results/audit97/F0/f0_quote_checks.json.
Run from the repository root:  python experiments/audit97/F0/f0_quote_checks.py
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "audit97", "F0", "f0_quote_checks.json")


def P(*a):
    return os.path.join(ROOT, *a)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


res = {"inputs": {}}


def load(rel):
    p = P(*rel.split("/"))
    res["inputs"][rel] = sha(p)
    return p


# ---- 1. WL86: the F2 failures on E2 region C relative to the 0.06 rule and the stored noise bound
d = json.load(open(load("results/ecro86/results/final.json")))
w86 = {}
for sd in ("ecro_s8601", "ecro_s8602"):
    for t in ("T0", "T1"):
        c = d["seeds"][sd]["fidelity"]["E2"][t]["C"]
        w86[f"{sd}|E2|{t}|C"] = {
            "E_H": c["E_H"],
            "noise_bound": c["noise_bound"],
            "margin_over_0.06": c["E_H"] - 0.06,
            "margin_in_noise_bounds": (c["E_H"] - 0.06) / c["noise_bound"],
        }
res["WL86_E2_C_vs_rule"] = w86
res["WL86_status"] = d["STATUS"]

# ---- 2. WL87: state-realization distribution on E2 T1 region C (EXP and ECRO on the same states)
d = json.load(open(load("results/gm1_explicit87/results/stage2_87.json")))
w87 = {}
for arm, rows in d["per_edit"]["E2"].items():
    if not (isinstance(rows, dict) and "rows" in rows):
        continue
    v = np.array([r["T1"]["C"]["E_H"] for r in rows["rows"]])
    s = np.array([r["T1"]["C"]["signed"] for r in rows["rows"]])
    w87[arm] = {
        "n": int(len(v)),
        "E_H_mean": float(v.mean()),
        "E_H_sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "E_H_min": float(v.min()),
        "E_H_max": float(v.max()),
        "pass_rate_le_0.06": float((v <= 0.06).mean()),
        "signed_mean": float(s.mean()),
        "first_row_seed": rows["rows"][0].get("seed"),
        "first_row_E_H": float(v[0]),
    }
_e = d["per_edit"]["E2"]
if "EXP" in _e and "ECRO" in _e and len(_e["EXP"]["rows"]) == len(_e["ECRO"]["rows"]):
    same = all(a_.get("seed") == b_.get("seed") for a_, b_ in zip(_e["EXP"]["rows"], _e["ECRO"]["rows"]))
    for t in ("T0", "T1"):
        pd_ = np.array(
            [b_[t]["C"]["E_H"] - a_[t]["C"]["E_H"] for a_, b_ in zip(_e["EXP"]["rows"], _e["ECRO"]["rows"])]
        )
        ve = np.array([b_[t]["C"]["E_H"] for b_ in _e["ECRO"]["rows"]])
        vx = np.array([a_[t]["C"]["E_H"] for a_ in _e["EXP"]["rows"]])
        w87[f"paired_ECRO_minus_EXP_{t}"] = {
            "same_seed_order": bool(same),
            "mean": float(pd_.mean()),
            "se": float(pd_.std(ddof=1) / np.sqrt(len(pd_))),
            "n_ECRO_worse": int((pd_ > 0).sum()),
            "n": int(len(pd_)),
            "ECRO_mean": float(ve.mean()),
            "ECRO_pass_rate": float((ve <= 0.06).mean()),
            "EXP_mean": float(vx.mean()),
            "EXP_pass_rate": float((vx <= 0.06).mean()),
        }
res["WL87_E2_T1_C_realizations"] = w87

# ---- 3. WL92B: the independently qualified held-out edit E5 (every model vs the 0.06 rule; explicit control)
d = json.load(open(load("results/ecro_cf92b/results/final92b.json")))
e5 = d["e5"]
w92b = {"admitted": d["outcome"]["e5_admitted"], "chosen": e5["chosen"], "models": {}, "controls": {}}
for m, v in e5["models"].items():
    w92b["models"][m] = {
        t: {
            "C_E_H": v["fidelity"][t]["C"]["E_H"],
            "C_noise_bound": v["fidelity"][t]["C"]["noise_bound"],
            "panel_E_H": v["fidelity"][t]["obj_panel"]["E_H"],
        }
        for t in ("T0", "T1")
    }
for c, v in e5["controls"].items():
    w92b["controls"][c] = {"C_E_H": v["C"]["E_H"], "panel_E_H": v["obj_panel"]["E_H"]}
w92b["max_C_E_H_any_learned_model"] = max(x[t]["C_E_H"] for x in w92b["models"].values() for t in x)
w92b["frozen_ecro_minus_explicit_T1_C"] = {
    m: w92b["models"][m]["T1"]["C_E_H"] - w92b["controls"]["X_EXP_T1"]["C_E_H"] for m in ("ecro_s8601", "ecro_s8602")
}
w92b["frozen_ecro_minus_explicit_T0_C"] = {
    m: w92b["models"][m]["T0"]["C_E_H"] - w92b["controls"]["X_EXP_T0"]["C_E_H"] for m in ("ecro_s8601", "ecro_s8602")
}
w92b["C_edit_slope_corr"] = e5["criteria"]["DYNAMIC"]["C_edit"]
w92b["hypothesis_labels"] = {k: v.get("label") for k, v in d["hypotheses"].items()}
w92b["outcome"] = d["outcome"]
res["WL92B_E5"] = w92b

# ---- 4. WL95A: delivery draws on box A (patches 1130..1183), exactly-2-event S1-mediated exitance
z = np.load(load("results/ecro_attr95a/deliv.npz"))
box = slice(1130, 1184)
ref = 0.5 * (z["ref_a"] + z["ref_b"])[box]
n0 = 0.5 * np.abs(z["ref_a"][box] - z["ref_b"][box]).sum() / ref.sum()
w95 = {"n0_boxA": float(n0), "arms": {}}
for arm in ("A0_s8601", "A0_s8602", "A1", "A2", "A3", "A4", "M64_spool", "q256", "X1c", "X1"):
    dr = z["draws|" + arm][:, box]
    dep = z["dep|" + arm][box]
    r = dr.sum(1) / ref.sum()
    l1 = np.abs(dr - ref).sum(1) / ref.sum()
    pm = dr.mean(0)
    w95["arms"][arm] = {
        "deployed_ratio": float(dep.sum() / ref.sum()),
        "deployed_rel_l1": float(np.abs(dep - ref).sum() / ref.sum()),
        "draw_ratio_mean": float(r.mean()),
        "draw_ratio_sd": float(r.std(ddof=1)),
        "draw_ratio_se": float(r.std(ddof=1) / np.sqrt(len(r))),
        "draw_ratio_min": float(r.min()),
        "draw_ratio_median": float(np.median(r)),
        "draw_ratio_max": float(r.max()),
        "draws_over_delivering": int((r > 1).sum()),
        "n_draws": int(len(r)),
        "deployed_percentile_in_draws": float((r < dep.sum() / ref.sum()).mean()),
        "pooled_rel_l1": float(np.abs(pm - ref).sum() / ref.sum()),
        "per_draw_rel_l1_median": float(np.median(l1)),
        "per_draw_rel_l1_mean": float(l1.mean()),
        # crude sampling-noise scale of the pooled map if the draws were independent
        "pooled_rel_l1_noise_scale": float(np.abs(dr - pm).sum(1).mean() / ref.sum() / np.sqrt(len(r))),
    }
ra = z["draws|A4"][:, box].sum(1) / ref.sum()
rx = z["draws|X1"][:, box].sum(1) / ref.sum()
dd = ra - rx
w95["paired_A4_minus_X1_ratio"] = {
    "mean": float(dd.mean()),
    "se": float(dd.std(ddof=1) / np.sqrt(len(dd))),
    "draws_positive": int((dd > 0).sum()),
    "n": int(len(dd)),
}
a = json.load(open(load("results/ecro_attr95a/an.json")))
w95["an_json_flight"] = a["flight"]
w95["an_json_verdicts"] = a["verdicts"]
m = json.load(open(load("results/ecro_attr95a/main.json")))
w95["operator_metrics"] = m["operator"]
res["WL95A_delivery"] = w95

# ---- 5. WL96A: decision inputs, learned-G ratios, lifecycle shares
s1 = json.load(open(load("results/ecro_util96a/s1.json")))
im = s1["images"]
w96 = {"decision": s1["decision"]}
dI = {k.split("|")[0]: v["dI"] for k, v in im.items() if k.endswith("|ROI_T") and isinstance(v, dict) and "dI" in v}
w96["ROI_T_dI"] = dI
w96["ROI_T_E_H"] = {
    k.split("|")[0]: v["E_H"] for k, v in im.items() if k.endswith("|ROI_T") and isinstance(v, dict) and "E_H" in v
}
w96["learned_G_over_explicit"] = {f"{g}/{x}": dI[g] / dI[x] for g in ("G8601", "G8602") for x in ("XOD", "XD")}
c = json.load(open(load("results/ecro_util96a/cost_s1.0.json")))
r0e, r0s = c["arms"]["R0"]["edit"], c["arms"]["R0"]["source"]
w96["cost"] = {
    "N": c["N"],
    "U": c["U"],
    "R0_edit_wall_ms": 1e3 * r0e["wall_s"],
    "R0_edit_gpu_kernel_ms": r0e["gpu_kernel_ms"],
    "R0_edit_gpu_kernel_share_of_wall": r0e["gpu_kernel_ms"] / (1e3 * r0e["wall_s"]),
    "R0_edit_parts_ms": {k: 1e3 * v for k, v in r0e["parts"].items()},
    "R0_source_wall_ms": 1e3 * r0s["wall_s"],
    "R0_source_parts_ms": {k: 1e3 * v for k, v in r0s["parts"].items()},
    "tail_share_edit": r0e["parts"]["tail_orders"] / r0e["wall_s"],
    "tail_share_source": r0s["parts"]["tail_orders"] / r0s["wall_s"],
    "render_share_edit": r0e["parts"]["render"] / r0e["wall_s"],
    "q_table_share_source": r0s["parts"]["source_q_table"] / r0s["wall_s"],
    # Arithmetic only (NOT a measurement): bound on the frame-inclusive source-update ratio of a path that
    # needs neither the q table nor the prefix / tail sweeps (the WL86 ECRO-style encoding + apply path).
    "arith_ceiling_source_if_q_and_sweeps_free": r0s["wall_s"]
    / (r0s["wall_s"] - r0s["parts"]["source_q_table"] - r0s["parts"]["prefix_order"] - r0s["parts"]["tail_orders"]),
}
f = json.load(open(load("results/ecro_util96a/final.json")))
w96["final"] = {k: f[k] for k in ("S1", "accuracy_confirmed", "cost_met", "LEARNED_RESOLVENT_UTILITY")}
res["WL96A"] = w96

# ---- 6. WL96 ring-0 check (design-only scratch of record)
res["WL96_ring_alloc"] = json.load(open(load("results/wl96_calc/ring_alloc.json")))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="\n") as fh:
    json.dump(res, fh, indent=1, sort_keys=True)
print("wrote", OUT)
for k in ("WL86_E2_C_vs_rule", "WL87_E2_T1_C_realizations"):
    print(k, json.dumps(res[k], indent=1))
print(
    "E5 max learned C",
    w92b["max_C_E_H_any_learned_model"],
    w92b["frozen_ecro_minus_explicit_T1_C"],
    w92b["frozen_ecro_minus_explicit_T0_C"],
)
for arm in ("A4", "q256", "X1", "X1c"):
    print(arm, {k: round(v, 3) for k, v in w95["arms"][arm].items()})
print("paired A4-X1", w95["paired_A4_minus_X1_ratio"])
print(
    "WL96A",
    json.dumps(w96["learned_G_over_explicit"]),
    json.dumps({k: v for k, v in w96["cost"].items() if not isinstance(v, dict)}),
)
