"""Lab-meeting smoke report (protocol b4_smoke_meeting_v1; RNA venv, CPU).

    python b4_smoke_report.py --res <results/rna_selective_reconstruction_lego> --out <results/evaluation/37/smoke_meeting>

Frozen (SO_T0 on T1) vs short selective reconstruction (B_sel_smoke) on the 16 held-out T1 views under PU: per-view display
MSE per predeclared ROI, paired difference selective - frozen with a 90% bootstrap CI over views (2000 resamples, seed 0);
parameter preservation from train.json; output-change support from eval.json. Figure view = the held-out view with the most
physically changed A pixels; crop = the window with the strongest |GT change| (both from references only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "font.size": 9})
EMAX = 0.08
FR, SE, FU = "SO_T0-PU", "B_sel_smoke-PU", "A_full_smoke-PU"


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(score, w):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    return np.unravel_index(np.argmax(tot), tot.shape)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); res, out = Path(a.res), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ed = res / "smoke/eval_smoke2" if (res / "smoke/eval_smoke2/eval.json").exists() else res / "smoke/eval_smoke"
    ev = json.loads((ed / "eval.json").read_text()); z = np.load(ed / "views.npz")
    tr = json.loads((res / "smoke/B_sel_smoke/train.json").read_text()); orc = json.loads((res / "oracle/oracle.json").read_text())
    D = set(orc["D_star"]); S = ev["sets"]["heldout_T1"]; nv = S["views"]
    regions = {}
    ARMS = [FR, SE] + ([FU] if f"heldout_T1_v0_{FU}" in z.files else [])
    per = {r: {**{a_: [] for a_ in ARMS}, "n": []} for r in ("all", "A_changed", "A_unchanged", "Dstar_owned", "Vstar_owned", "B")}
    for v in range(nv):
        k = f"heldout_T1_v{v}"; ref = z[k + "_ref"].astype(np.float32); own = z[k + "_own"]
        masks = {"all": own >= 0, "A_changed": z[k + "_reg_A_changed"], "A_unchanged": z[k + "_reg_A_unchanged"],
                 "Dstar_owned": (own >= 0) & np.isin(own, list(D)), "Vstar_owned": (own >= 0) & ~np.isin(own, list(D)), "B": own == 16}
        for arm in ARMS:
            sq = ((z[k + "_" + arm].astype(np.float32) - ref) ** 2).mean(-1)
            for r, m in masks.items():
                per[r][arm].append(float(sq[m].sum()))
        for r, m in masks.items():
            per[r]["n"].append(int(m.sum()))
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, nv, nv) for _ in range(2000)]
    for r, d in per.items():
        f, s, n = (np.array(d[x], float) for x in (FR, SE, "n"))
        if n.sum() == 0:
            continue
        mse = lambda arr, i=slice(None): arr[i].sum() / max(n[i].sum(), 1)
        diff = [mse(s, i) - mse(f, i) for i in idx]
        regions[r] = {"px": int(n.sum()), "mse_frozen": mse(f), "mse_selective": mse(s), "psnr_frozen": float(-10 * np.log10(mse(f))),
                      "psnr_selective": float(-10 * np.log10(mse(s))), "mse_diff_sel_minus_frozen": mse(s) - mse(f),
                      "ci90": [float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))], "relative_change": (mse(s) - mse(f)) / mse(f),
                      "views_improved": int(sum(1 for v in range(nv) if n[v] and s[v] / n[v] < f[v] / n[v])), "views_with_px": int((n > 0).sum())}
        if FU in ARMS:
            u = np.array(d[FU], float); dfu = [mse(s, i) - mse(u, i) for i in idx]
            regions[r].update({"mse_full": mse(u), "psnr_full": float(-10 * np.log10(mse(u))), "mse_diff_sel_minus_full": mse(s) - mse(u),
                               "ci90_sel_minus_full": [float(np.quantile(dfu, 0.05)), float(np.quantile(dfu, 0.95))]})
    ach = regions["A_changed"]
    verdict = "positive" if (ach["ci90"][1] < 0 and tr.get("frozen_units_bit_identical")) else "negative"
    rows = S["rows"]
    rec = {"protocol": "b4_smoke_meeting_v1", "verdict": verdict, "regions_PU": regions,
           "whole_image": {arm: {k: rows[arm][k] for k in ("view_mean_psnr", "pooled_psnr", "logl1", "hf_abs")} for arm in (FR, SE, "SO_T0", "B_sel_smoke")},
           "parameters": {"updated_units": tr["init"]["reinitialised_units"], "units_changed": tr.get("units_changed"),
                          "frozen_units_bit_identical": tr.get("frozen_units_bit_identical"), "params_total": tr["parameters_total"],
                          "params_updated": int(sum(tr["parameters_per_unit"][u] for u in tr["init"]["reinitialised_units"]))},
           "training": {"epochs": len(tr["epochs"]), "seconds_fit_incl_data_setup": tr["seconds"]["fit_including_data_setup"],
                        "median_epoch_s": float(np.median([e["seconds"] for e in tr["epochs"]])), "note": "overlapped with a concurrent CPU render (cores 0-23); not a cost measurement"},
           "output_support": S["influence"], "eval_dir": str(ed)}
    if FU in ARMS:
        tf = json.loads((res / "smoke/A_full_smoke/train.json").read_text())
        cnt = np.array(tr["owner_pixel_counts_train"], float)
        rec["recompute_cost"] = {
            "selective": {"fit_s": tr["seconds"]["fit_including_data_setup"], "median_epoch_s": float(np.median([e["seconds"] for e in tr["epochs"]])),
                          "trained_params": rec["parameters"]["params_updated"]},
            "full": {"fit_s": tf["seconds"]["fit_including_data_setup"], "median_epoch_s": float(np.median([e["seconds"] for e in tf["epochs"]])),
                     "trained_params": tf["parameters_total"]},
            "dirty_pixel_share_of_T1_training_data": float(cnt[list(D)].sum() / cnt.sum()),
            "conditions": "same 128-image T1 subset, 250 epochs, seed 0, cores 24-31, both with the same concurrent CPU render on cores 0-23; epoch time includes forward passes of frozen units and per-epoch validation; the oracle's own cost is not included"}
        c = rec["recompute_cost"]; c["ratio_fit_selective_over_full"] = c["selective"]["fit_s"] / c["full"]["fit_s"]
        c["ratio_epoch_selective_over_full"] = c["selective"]["median_epoch_s"] / c["full"]["median_epoch_s"]
        c["ratio_trained_params"] = c["selective"]["trained_params"] / c["full"]["trained_params"]
        rec["whole_image"][FU] = {k: rows[FU][k] for k in ("view_mean_psnr", "pooled_psnr", "logl1", "hf_abs")}
    (out / "smoke_metrics.json").write_text(json.dumps(rec, indent=1))
    # figure
    chg = [int(z[f"heldout_T1_v{v}_reg_A_changed"].sum()) for v in range(nv)]; v = int(np.argmax(chg)); k = f"heldout_T1_v{v}"
    ref = z[k + "_ref"].astype(np.float32); own = z[k + "_own"]; fr = z[k + "_" + FR].astype(np.float32); se = z[k + "_" + SE].astype(np.float32)
    ef = np.abs(fr - ref).mean(-1); es = np.abs(se - ref).mean(-1); ef[own < 0] = np.nan; es[own < 0] = np.nan
    upd = np.ones(own.shape + (3,)) * 0.99; upd[own >= 0] = 0.82; upd[np.isin(own, list(D))] = (0.85, 0.2, 0.15)
    upd[(z[k + "_w"].astype(np.float32) > 0) & ~np.isin(own, list(D)) & z[k + "_reg_Vstar_in_Dstar_band"]] = (0.95, 0.6, 0.1)
    gd = np.abs(z[k + "_gtdelta"].astype(np.float32)).mean(-1); gd[own < 0] = 0
    W = 192; y, x = window(gd, W); sl = (slice(y, y + W), slice(x, x + W))
    pv = lambda arm: S["rows"][arm]["per_view_psnr"][v]
    full = FU in ARMS
    fu = z[k + "_" + FU].astype(np.float32) if full else None
    eu = np.abs(fu - ref).mean(-1) if full else None
    if full:
        eu[own < 0] = np.nan
    ncol = 8 if full else 6
    fig, axs = plt.subplots(2, ncol, figsize=(3.2 * ncol, 7.2))
    panels = [(np.clip(z[k + "_refT0"].astype(np.float32), 0, 1), "T0 reference (same camera/light)"), (np.clip(ref, 0, 1), "T1 path-traced reference"),
              (np.clip(fr, 0, 1), f"C: frozen T0 model\n{pv(FR):.2f} dB (view)"), (np.clip(se, 0, 1), f"B: selective update (8 units)\n{pv(SE):.2f} dB (view)"),
              (ef, "|error| frozen"), (es, "|error| selective")]
    if full:
        panels.insert(4, (np.clip(fu, 0, 1), f"A: full recompute (17 units)\n{pv(FU):.2f} dB (view)")); panels.append((eu, "|error| full"))
    for j, (img, t) in enumerate(panels):
        for i, crop in ((0, False), (1, True)):
            im = img[sl] if crop else img
            if img.ndim == 2:
                h = axs[i, j].imshow(im, cmap=SEQ, vmin=0, vmax=EMAX)
            else:
                axs[i, j].imshow(im)
            axs[i, j].set_title(t if i == 0 else "crop: strongest GT change", fontsize=9)
            off(axs[i, j])
    axs[0, 1].add_patch(plt.Rectangle((x, y), W, W, fill=False, ec="#c0392b", lw=1.2))
    fig.colorbar(h, ax=axs[:, -3:] if full else axs[:, 4:], fraction=0.02, pad=0.01, label=f"|display error| 0-{EMAX}")
    fig.suptitle(f"Held-out T1 view {v} (4096 spp). Updated units D* (red), all other units bit-identical. Smoke budget: 128 T1 images, 250 epochs, cold re-init.", fontsize=10)
    f1 = out / "01_meeting_comparison.png"; fig.savefig(f1, dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(upd); ax[0].set_title("updated units: D* (red), PU band in V* (amber), V* grey", fontsize=9); off(ax[0])
    names = ["A_changed", "Dstar_owned", "Vstar_owned", "A_unchanged", "B", "all"]
    xs = np.arange(len(names)); fz = [regions[r]["mse_frozen"] for r in names]; sz = [regions[r]["mse_selective"] for r in names]
    bw = 0.27 if full else 0.4
    ax[1].bar(xs - bw, fz, bw, color="#86b6ef", label="frozen"); ax[1].bar(xs, sz, bw, color="#c0392b", label="selective")
    if full:
        ax[1].bar(xs + bw, [regions[r]["mse_full"] for r in names], bw, color="#52514e", label="full recompute")
    ax[1].set_xticks(xs, names, rotation=20); ax[1].set_ylabel("display MSE (16 held-out T1 views)"); ax[1].legend(frameon=False)
    ax[1].set_title("frozen vs selective" + (" vs full recompute" if full else "") + " by predeclared region")
    fig.tight_layout(); f2 = out / "02_mask_and_regions.png"; fig.savefig(f2, dpi=120); plt.close(fig)
    files = [f1, f2]
    if "recompute_cost" in rec:
        c = rec["recompute_cost"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for a_, (key, lab, scale) in zip(ax, (("fit_s", "reconstruction wall-clock (min)", 1 / 60), ("median_epoch_s", "median epoch time (s)", 1.0),
                                               ("trained_params", "trained parameters (M)", 1e-6))):
            vals = [c["selective"][key] * scale, c["full"][key] * scale]
            a_.bar([0, 1], vals, color=["#c0392b", "#52514e"], width=0.6); a_.set_xticks([0, 1], ["selective\n(8 D* units)", "full recompute\n(17 units)"])
            a_.set_title(lab, fontsize=9)
            for i_, v_ in enumerate(vals):
                a_.text(i_, v_, f"{v_:.1f}" if scale != 1e-6 else f"{v_:.2f}", ha="center", va="bottom", fontsize=9)
            a_.text(0.5, 0.92, f"ratio {vals[0] / vals[1]:.2f}", transform=a_.transAxes, ha="center", fontsize=9, color=INK2)
        fig.suptitle(f"Same 128-image T1 subset, 250 epochs, same cores and concurrent render; D* owns {100 * c['dirty_pixel_share_of_T1_training_data']:.0f}% of training pixels. Oracle cost not included.", fontsize=9)
        fig.tight_layout(); f3 = out / "03_recompute_cost.png"; fig.savefig(f3, dpi=120); plt.close(fig); files.append(f3)
    man = {"protocol": "experiments/rna_selective_reconstruction_lego/protocol/b4_smoke_meeting_v1.json", "view": v, "crop_yx": [int(y), int(x)], "crop_size": W,
           "files": [{"file": f.name, "sha256_16": sha(f)} for f in files + [out / "smoke_metrics.json"]],
           "sources": [{"file": str(p), "sha256_16": sha(p)} for p in (ed / "eval.json", res / "smoke/B_sel_smoke/train.json", res / "smoke/A_full_smoke/train.json", res / "oracle/oracle.json") if p.exists()]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"verdict": verdict, "A_changed": ach, "all": regions["all"], "Vstar": regions["Vstar_owned"], "A_unchanged": regions["A_unchanged"],
                      "params": rec["parameters"], "recompute_cost": rec.get("recompute_cost")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
