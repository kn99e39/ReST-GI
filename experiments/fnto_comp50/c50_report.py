"""WL50 report: Gate 1 first-event metrics from the r1_eval outputs (views.npz), seed-wise L1C - L0C with view bootstrap,
integration validity, observed / never-observed receivers, distance bands, view 14, T0 / T1; Gate 3 status summary;
cost; panels (one absolute error scale) into results/evaluation/50.

    python c50_report.py <results/fnto_comp50> <results/evaluation/50>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CONDS = {"H": "H", "L0C_5011": "L0_5011", "L1C_5011": "L1_5011", "L0C_5012": "L0_5012", "L1C_5012": "L1_5012"}


def psnr_sq(sq):
    return float(10 * np.log10(1 / max(sq, 1e-30)))


def main():
    R, ev = Path(sys.argv[1]), Path(sys.argv[2]); ev.mkdir(parents=True, exist_ok=True)
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"]); apos = A["pos"]
    pp = np.load(C.OUT1 / "stagea/prep2_train/px_port.npy", mmap_mode="r"); pdm = np.load(C.OUT1 / "stagea/prep2_train/px_dom.npy", mmap_mode="r")
    obs = np.bincount(np.asarray(pp)[np.asarray(pdm) == 0], minlength=len(A["obj"])) > 0
    vis = np.load(C.OUT1 / "eval_vis_k44.npz"); V = {c: np.load(R / "eval" / d / "views.npz") for c, d in CONDS.items() if (R / "eval" / d / "views.npz").exists()}
    out = {"first_event": {}, "solver": {}, "cost": {}}; per_view = {}
    for st in ("T0", "T1"):
        px = np.load(C.OUT1 / f"eval_px/heldout_{st}.npz"); ref = np.load(C.OUT1 / f"refs/heldout_{st}_first.npz"); nv = int(px["nv"])
        acc = {c: {k: [0.0, 0] for k in ("yellow", "yellow_lit", "yellow_shadow", "yellow_3_30cm", "yellow_observed", "yellow_never", "nonyellow")} for c in V}
        chan = {c: np.zeros((3, 2)) for c in V}
        for v in range(nv):
            val = px[f"v{v}_valid"].astype(bool); port = px[f"v{v}_port"][val].astype(np.int64); dom = px[f"v{v}_dom"][val]
            yel = (dom == 0) & (port < nA) & (A["mat"][np.clip(port, 0, len(A["mat"]) - 1)] == 0); lit = px[f"v{v}_vis"][val].astype(bool)
            fr = vis[f"heldout_{st}_v{v}_{st}"][:nA] * np.maximum(0, A["nrm"][:nA] @ px[f"v{v}_light"]); lp = apos[:nA][fr > 0]
            dl = cKDTree(lp).query(apos[np.clip(port[yel], 0, nA - 1)])[0] if len(lp) else np.full(int(yel.sum()), 1e9)
            d_all = np.full(len(port), np.nan); d_all[yel] = dl
            r_ = ref[f"v{v}_rgb"][val].astype(np.float64)
            masks = {"yellow": yel, "yellow_lit": yel & lit, "yellow_shadow": yel & ~lit, "yellow_3_30cm": yel & ~lit & (d_all >= 0.03) & (d_all < 0.30),
                     "yellow_observed": yel & obs[np.clip(port, 0, len(obs) - 1)], "yellow_never": yel & ~obs[np.clip(port, 0, len(obs) - 1)], "nonyellow": ~yel}
            for c, Z in V.items():
                p = Z[f"heldout_{st}_v{v}_K_only"][:, :3].astype(np.float64); sq = ((C.ldr3(p) - C.ldr3(r_)) ** 2)
                for k, m in masks.items():
                    acc[c][k][0] += float(sq[m].mean(1).sum()); acc[c][k][1] += int(m.sum())
                chan[c][:, 0] += sq[yel].sum(0); chan[c][:, 1] += int(yel.sum())
                per_view.setdefault((st, c), []).append(psnr_sq(sq[yel].mean()))
        out["first_event"][st] = {c: {**{k: psnr_sq(a[0] / max(a[1], 1)) for k, a in acc[c].items()}, "yellow_RGB": [psnr_sq(chan[c][i, 0] / chan[c][i, 1]) for i in range(3)],
                                      "yellow_per_view": per_view[(st, c)], "view14_yellow": per_view[(st, c)][14]} for c in V}
    # first-event edit response (K_only T1 - T0 vs first_r1 T1 - T0) on Lego / Yellow pixels present in both states
    D0 = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); D1 = np.load(C.OUT1 / "eval_px/heldout_T1.npz")
    f0 = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); f1 = np.load(C.OUT1 / "refs/heldout_T1_first.npz"); ed = {}
    for v in range(int(D0["nv"])):
        va0, va1 = D0[f"v{v}_valid"].astype(bool), D1[f"v{v}_valid"].astype(bool)
        l0, l1 = (D0[f"v{v}_dom"] == 0) & va0, (D1[f"v{v}_dom"] == 0) & va1
        com, i0, i1 = np.intersect1d(D0[f"v{v}_mk"][l0], D1[f"v{v}_mk"][l1], return_indices=True)
        y = A["mat"][np.clip(D0[f"v{v}_port"][l0][i0], 0, len(A["mat"]) - 1)] == 0
        dr = (f1[f"v{v}_rgb"][l1][i1] - f0[f"v{v}_rgb"][l0][i0]) @ LUM
        for c, Z in V.items():
            p0 = Z[f"heldout_T0_v{v}_K_only"][:, :3][np.cumsum(va0)[l0] - 1][i0].astype(np.float64); p1 = Z[f"heldout_T1_v{v}_K_only"][:, :3][np.cumsum(va1)[l1] - 1][i1].astype(np.float64)
            e = ed.setdefault(c, {"all": [[], []], "yellow": [[], []]}); dp = (p1 - p0) @ LUM
            e["all"][0].append(dp); e["all"][1].append(dr); e["yellow"][0].append(dp[y]); e["yellow"][1].append(dr[y])
    out["first_event_edit"] = {c: {k: {"corr": float(np.corrcoef(np.concatenate(a[0]), np.concatenate(a[1]))[0, 1]),
                                      "slope": float((np.concatenate(a[0]) * np.concatenate(a[1])).sum() / (np.concatenate(a[1]) ** 2).sum())} for k, a in e.items()} for c, e in ed.items()}
    bs = np.random.default_rng(50); crit = {}
    for seed in ("5011", "5012"):
        r = {}
        for st in ("T0", "T1"):
            if (st, f"L1C_{seed}") in per_view and (st, f"L0C_{seed}") in per_view:
                d = np.array(per_view[(st, f"L1C_{seed}")]) - np.array(per_view[(st, f"L0C_{seed}")])
                boots = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                r[st] = {"mean": float(d.mean()), "ci95": np.quantile(boots, [0.025, 0.975]).tolist(), "n_pos": int((d > 0).sum())}
        crit[seed] = r
    fe = out["first_event"]["T0"]
    out["gate1"] = {"L1C_minus_L0C": crit,
                    "first_event_advantage": all(crit[s]["T0"]["mean"] >= 0.3 and crit[s]["T0"]["ci95"][0] > 0 for s in crit if "T0" in crit[s]),
                    "integration_valid": {c: fe[c]["yellow_lit"] >= fe["H"]["yellow_lit"] - 1.0 for c in fe if c != "H"},
                    "nonyellow_identical_to_H": {c: abs(fe[c]["nonyellow"] - fe["H"]["nonyellow"]) < 1e-3 for c in fe if c != "H"}}
    for c, d in CONDS.items():
        p = R / "eval" / d / "eval.json"
        if p.exists():
            e = json.loads(p.read_text()); out["solver"][c] = {"rho": {k: v["rho_estimate"] for k, v in e["spectral_radius"].items()},
                                                               "status_counts": {f"{s}:{n}": {st_: sum(x["status"] == st_ for x in L) for st_ in set(x["status"] for x in L)} for s, a in e["per_view"].items() for n, L in a.items()},
                                                               "solve_s_median": {f"{s}:{n}": float(np.median([x["solve_s"] for x in L])) for s, a in e["per_view"].items() for n, L in a.items()}}
            out["cost"][c] = json.loads((R / "eval" / d / "cond.json").read_text())
    (R / "report.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("gate1",)}, indent=1))
    # panels: first-event Yellow, views 8 / 14 (T0), reference / H / L0C / L1C (seed 5011)
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); ref = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); res = int(px["res"])
    show = [("ref", "Cycles first_r1"), ("H", "H (K44)"), ("L0C_5011", "L0C"), ("L1C_5011", "L1C (DASQ)")]
    errs = {}
    for v in (8, 14):
        val = px[f"v{v}_valid"].astype(bool); r_ = ref[f"v{v}_rgb"][val]
        for c, _ in show[1:]:
            errs[(v, c)] = np.abs((V[c][f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64) - r_) @ LUM)
    M = float(max(e.max() for e in errs.values()))
    for v in (8, 14):
        val = px[f"v{v}_valid"].astype(bool); mk = px[f"v{v}_mk"][val]; r_ = ref[f"v{v}_rgb"][val]
        fig, ax = plt.subplots(2, 4, figsize=(14, 7))
        for j, (c, lab) in enumerate(show):
            p = r_ if c == "ref" else V[c][f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)
            im = np.zeros((res * res, 3)); im[mk] = C.ldr3(p); ax[0, j].imshow(np.clip(im.reshape(res, res, 3), 0, 1)); ax[0, j].set_title(f"{lab}, first event", fontsize=8); ax[0, j].axis("off")
            if c != "ref":
                e = np.zeros(res * res); e[mk] = np.sqrt(errs[(v, c)] / M); h = ax[1, j].imshow(e.reshape(res, res), cmap="Blues", vmin=0, vmax=1)
                cb = fig.colorbar(h, ax=ax[1, j], fraction=0.046); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", f"{0.25 * M:.3g}", f"{M:.3g}"]); ax[1, j].set_title("|error| lum (sqrt, shared M)", fontsize=8)
            ax[1, j].axis("off")
        fig.suptitle(f"WL50 complete-K first event (K_only vs first_r1), held-out T0 view {v}; coupled solves of L0C / L1C diverged (no coupled image)", fontsize=9)
        fig.tight_layout(); fig.savefig(ev / f"01_first_event_view{v}.png"); plt.close(fig)
    for src in ("report.json", "train/train.json", "gate0_contract/targets_meta.json", "gate2_single_flight/onestep.json", "gate2_single_flight/refcheck/refcheck.json",
                "gate2_single_flight/refcheck_glossy_confounded/refcheck.json", "gate3_solver/diag.json"):
        if (R / src).exists():
            shutil.copy(R / src, ev / ("d50__" + src.replace("/", "__")))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(ev.iterdir()) if p.name != "manifest.json"}
    (ev / "manifest.json").write_text(json.dumps({"error_scale_M_lum": M, "mapping": "sqrt(|e| / M)", "rule": "one absolute error scale for every error map of worklog 50", "files": files}, indent=1))


if __name__ == "__main__":
    main()
