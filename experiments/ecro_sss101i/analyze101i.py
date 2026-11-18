"""WL101I offline analysis: construction qualification contract v2, frozen verdict.

python analyze101i.py RESULTS_DIR WL101G_PRIMARY_A2 WL101H_METRICS REVIEW_DIR
RESULTS_DIR holds final_candidate, final_native, positive_control, regression_native.
"""
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ecro86"))
sys.path.insert(0, str(HERE.parent / "ecro_sss101a"))
import gates101i as Q  # noqa: E402
import tests101i as TT  # noqa: E402

PROTOCOL = HERE / "protocol/contract_v2.json"
P = json.loads(PROTOCOL.read_text())
SMOKE = __import__("os").environ.get("WL101I_SMOKE") == "1"   # code-path check on development seeds only
if SMOKE:
    P["proposals_per_slot"] = 64
    P["final_seeds"] = [10204101, 10204102, 10204103, 10204104]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(d):
    man = json.loads((d / "run_manifest.json").read_text())
    pks = {}
    for name, rec in man["files"].items():
        assert sha(d / name) == rec["sha256"], name
        pks[int(name.split("_")[1].split(".")[0])] = dict(np.load(d / name))
    return man, pks


def pool(pks, seeds):
    keys = pks[seeds[0]].keys()
    return {k: sum(pks[s][k] for s in seeds) for k in keys}


def proxy_from_wl101h(metrics_path, z):
    rows = json.loads(Path(metrics_path).read_text())["replay"]["candidate"]
    near = [r for r in rows if 0.5 <= r["r_mm"] <= 2.0]
    band = [r for r in rows if 0.5 <= r["r_mm"] <= 12.0]
    L = float(np.mean([r["L"] for r in near]))
    spread = float(max(r["L"] for r in band) - min(r["L"] for r in band))
    se = float(max(r["L_se"] for r in band))
    return {"L": L, "model": spread + z * se, "rows_used_mm": [r["r_mm"] for r in band], "spread": spread, "max_se": se}


def gates_for(man, pks, seeds, geom, h_min, semantics, proxy, z, fb, fj):
    res = {}
    s1 = [{"seed": m["seed"], **{k: m["membership"].get(k, 0) for k in ("invalid", "support_conflict")},
           "pass": m["membership"]["invalid"] == 0 and m["membership"].get("support_conflict", 0) == 0} for m in man["replicas"]]
    s2 = [{"seed": m["seed"], "ring": m["classes"]["ring"], "inside_box": m["classes"]["inside_box"],
           "pass": m["classes"]["ring"] == 0 and m["classes"]["inside_box"] == 0} for m in man["replicas"]]
    res["S1"], res["S2"] = {"rows": s1, "pass": all(r["pass"] for r in s1)}, {"rows": s2, "pass": all(r["pass"] for r in s2)}
    s3, sing, _ = Q.s3([pks[m["seed"]] for m in man["replicas"]], geom, h_min, P["b_m_m"], P["proposals_per_slot"])
    res["S3"] = s3
    pooled = pool(pks, seeds)
    per = {}
    detail = {}
    for level in ("slot", "sub"):
        r = Q.s4(pooled, geom, sing, fb * len(seeds) / len(seeds), fj, P["proposals_per_slot"] * len(seeds), proxy, z, level, semantics)
        per.update(r)
        detail.update(r)
    res["S4"] = Q.summarize_s4(per)
    res["labels"] = [m["labels"] for m in man["replicas"]]
    res["classes"] = [m["classes"] for m in man["replicas"]]
    ev = man.get("unregistered_events", [])
    sem = sum(1 for e in ev for a, b in zip(e["inside_box"], e["conflict"]) if a or b)
    label_registered = sum(1 for e in ev for m, j in zip(e["miss_other"], e.get("junction", [False] * len(e["miss_other"]))) if m and j)
    label_unregistered = sum(1 for e in ev for m, j, a, b in zip(e["miss_other"], e.get("junction", [False] * len(e["miss_other"])), e["inside_box"], e["conflict"]) if m and not j and not a and not b)
    res["D3"] = {"records": len(ev), "semantic_events": sem, "registered_junction_label_events": label_registered,
                 "unregistered_label_events": label_unregistered}
    if "camera_independence" in man:
        ci = man["camera_independence"]
        res["camera_independence"] = {**ci, "pass": all(v["integer_columns_and_counts_identical"] and v["max_relative_real_difference"] <= 1e-5 for v in ci.values())}
    if "consistency" in man:
        c = man["consistency"]
        res["consistency"] = {"views": {v: {"h_rel_l1": x["smoke"]["event_errors"]["h"]["relative_l1"], "low": x["low"]["low_max_abs_zero_vs_learned_table"]}
                                        for v, x in c.items()},
                              "pass": all(x["smoke"]["event_errors"]["h"]["relative_l1"] <= P["smoke_gates"]["smoke_event_relative_l1"] and
                                          x["low"]["low_max_abs_zero_vs_learned_table"] == 0 for x in c.values())}
    # Planar cos normalisation (pooled).
    sb = int((geom["psid"] != 6).sum())
    rows = []
    for k in range(4):
        cc = pooled["c"][:sb * 64].reshape(sb, 64, 16)[:, 16 * k:16 * (k + 1)].sum()
        nn = pooled["n"][:sb * 64].reshape(sb, 64, 16)[:, 16 * k:16 * (k + 1)].sum()
        rows.append({"ring": k, "standard_errors": float((cc / nn - (2 * k + 1) / 8) / (0.25 / math.sqrt(12) / math.sqrt(nn)))})
    res["cos_normalization"] = {"rows": rows, "pass": all(abs(r["standard_errors"]) <= 5 for r in rows)}
    return res, detail, sing


def representative(detail, slots):
    out = {}
    for s in slots:
        row = {}
        for name in ("U", "V", "G"):
            r = detail[name]
            row[name] = {"T": float(r["T"][s]), "b": float(r["b"][s]), "se": float(r["se"][s]), "B": float(r["B"][s]),
                         "theta": float(r["theta"][s]), "state": int(r["state"][s]),
                         "parts_b": {k: float(v[s]) for k, v in r["parts_b"].items()},
                         "parts_B": {k: float(v[s]) for k, v in r["parts_B"].items()}}
        out[str(s)] = row
    return out


def main(results, wl101g, h_metrics, out):
    out.mkdir(parents=True, exist_ok=True)
    geom, A = TT.real_geom()
    n_tests = len(geom["psid"]) * 64 * (3 + 4)
    z = Q.z_family(n_tests)
    proxy = proxy_from_wl101h(h_metrics, z)
    fb = Q.band_fraction_upper(geom, P["b_m_m"])
    fj = Q.junction_fraction_upper(geom, P["b_m_m"], A["geometry"]["B"])
    result = {"protocol_sha256": sha(PROTOCOL), "z_family": z, "n_tests": n_tests, "singular_proxy": proxy,
              "unit_tests": TT.run_all()}
    seeds = P["final_seeds"]
    manC, pkC = load(results / "final_candidate")
    manN, pkN = load(results / "final_native")
    manX, pkX = load(results / "positive_control")
    manR, pkR = load(results / "regression_native")
    h = manC["h_min_m"]
    result["candidate"], detC, sing = gates_for(manC, pkC, seeds, geom, h, "candidate", proxy, z, fb, fj)
    result["native"], detN, _ = gates_for(manN, pkN, seeds, geom, h, "native", proxy, z, fb, fj)
    # Positive control: expanded singular region, judged with its own registered measure.
    hX = manX["h_min_m"]
    sX, singX, _ = Q.s3([pkX[s] for s in manX["seeds"]], geom, hX, P["b_m_m"], P["proposals_per_slot"])
    rX = Q.s4(pool(pkX, manX["seeds"]), geom, singX, fb, fj, P["proposals_per_slot"] * len(manX["seeds"]), proxy, z, "sub", "candidate")["S_G"]
    fp, fs = singX["floor_pid"], singX["floor_subcell"]
    cells = np.arange(fp * 64, fp * 64 + 64)
    result["positive_control"] = {"h_min_m": hX, "r_s_m": singX["r_s_m"], "S3_consistent": sX["pass"],
                                  "contact_subcell_states": np.bincount(rX["state"][cells, fs] + 1, minlength=4).tolist(),
                                  "detected": bool(np.all(rX["state"][cells, fs] == -1)),
                                  "median_measured_bias": float(np.median(rX["b"][cells, fs]))}
    # S5: historical regression of slot totals and subcell redistribution against WL101G.
    old = [dict(np.load(wl101g / f"construction_{i}.npz")) for i in range(len(manR["seeds"]))]
    result["S5_regression"] = Q.s5([pkR[s] for s in manR["seeds"]], old, z) if not SMOKE else {"pass": True, "rows": [], "smoke": True}
    # Representative real-scene cells: contact patch, WL101H S4 worst slots, D4 left-wall slots.
    reps = [155 * 64 + b for b in (0, 16, 32, 48)] + [43016, 80198, 80774, 9859, 54034, 47918, 50807]
    result["representative_candidate"] = representative(detC, reps)
    result["representative_native"] = representative(detN, reps)
    # F6 effect visible in v2 accounting: native vs candidate G/U at the contact patch.
    c = result["candidate"]
    discrimination = (not result["native"]["S2"]["pass"]) and result["native"]["S1"]["pass"] and c["S2"]["pass"]
    global_ok = all([c["S1"]["pass"], c["S2"]["pass"], c["S3"]["pass"], result["S5_regression"]["pass"],
                     c.get("camera_independence", {}).get("pass", False), c.get("consistency", {}).get("pass", False),
                     c["cos_normalization"]["pass"], discrimination, result["positive_control"]["detected"],
                     result["positive_control"]["S3_consistent"], c["D3"]["semantic_events"] == 0])
    s4 = c["S4"]
    non_pass = {k: v["fail"] + v["unresolved"] for k, v in s4.items()}
    if not global_ok:
        verdict = "NOT_QUALIFIED"
    elif sum(non_pass.values()) > 0:
        verdict = "PARTIALLY_QUALIFIED"
    else:
        verdict = "QUALIFIED"
    restrictions = {}
    for name in ("U", "V", "G", "S_G"):
        st = detC[name]["state"]
        idx = np.argwhere((st == -1) | (st == 2))
        restrictions[name] = {"fail": np.argwhere(st == -1).tolist()[:500], "unresolved": np.argwhere(st == 2).tolist()[:500],
                              "by_shape": {}}
        sl = idx[:, 0]
        ps = geom["psid"][sl // 64]
        restrictions[name]["by_shape"] = {int(k): int(np.sum(ps == k)) for k in np.unique(ps)}
    result["restrictions"] = restrictions
    result["global_gates"] = {"S1": c["S1"]["pass"], "S2": c["S2"]["pass"], "S3": c["S3"]["pass"], "S5": result["S5_regression"]["pass"],
                              "camera_independence": c.get("camera_independence", {}).get("pass"), "consistency": c.get("consistency", {}).get("pass"),
                              "cos_normalization": c["cos_normalization"]["pass"], "discrimination": discrimination,
                              "positive_control": result["positive_control"]["detected"], "D3_semantic_events": c["D3"]["semantic_events"]}
    result["verdict"] = verdict
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    # Figure: measured |bias| vs bound per table (candidate, active cells).
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for a, name in zip(ax, ("U", "V", "G", "S_G")):
        r = detC[name]
        m = r["state"] != 0
        x = np.maximum(np.abs(r["b"][m]), 1e-12); y = np.minimum(np.where(np.isfinite(r["B"][m]), r["B"][m], 1.0), 1.0)
        th = r["theta"][m]
        for st, col, lab in ((1, "#2a78d6", "PASS"), (2, "#eda100", "UNRESOLVED"), (-1, "#e34948", "FAIL")):
            k = r["state"][m] == st
            a.scatter(x[k] / th[k], y[k] / th[k], s=3, color=col, label=f"{lab} ({int(k.sum())})", alpha=.6)
        a.axhline(1, color="#58616b", lw=.8); a.axvline(1, color="#58616b", lw=.8)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("|measured bias| / theta"); a.set_title(name)
        a.legend(fontsize=7, markerscale=3)
    ax[0].set_ylabel("conservative bound / theta")
    fig.suptitle("WL101I · candidate per-cell bias accounting (contract v2)")
    fig.savefig(out / "bias_accounting.png", dpi=140); plt.close(fig)
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in result["global_gates"].items())
    s4rows = "".join(f"<tr><td>{k}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in s4.items())
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL101I contract v2</title><style>:root{{--bg:#fcfcfb;--ink:#1f1f1f;--line:#d9d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--line:#3b3b38}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}}
img{{max-width:100%;height:auto;background:#fff}} td{{border-bottom:1px solid var(--line);padding:3px 8px;vertical-align:top;word-break:break-word}}</style></head><body>
<h1>WL101I · construction qualification contract v2</h1><p><b>Verdict:</b> {verdict}</p>
<h2>Global semantic gates</h2><table>{rows}</table><h2>S4 per-cell accounting</h2><table>{s4rows}</table>
<p><img src="bias_accounting.png"></p><p><a href="metrics.json">metrics.json</a></p></body></html>""")
    (out / "manifest101i.json").write_text(json.dumps({"review_files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "manifest101i.json"}}, indent=2) + "\n")
    print(verdict, json.dumps(result["global_gates"]), json.dumps(s4), flush=True)


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:5]])
