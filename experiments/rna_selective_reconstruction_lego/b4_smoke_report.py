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
FR, SE = "SO_T0-PU", "B_sel_smoke-PU"


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
    ev = json.loads((res / "smoke/eval_smoke/eval.json").read_text()); z = np.load(res / "smoke/eval_smoke/views.npz")
    tr = json.loads((res / "smoke/B_sel_smoke/train.json").read_text()); orc = json.loads((res / "oracle/oracle.json").read_text())
    D = set(orc["D_star"]); S = ev["sets"]["heldout_T1"]; nv = S["views"]
    regions = {}
    per = {r: {FR: [], SE: [], "n": []} for r in ("all", "A_changed", "A_unchanged", "Dstar_owned", "Vstar_owned", "B")}
    for v in range(nv):
        k = f"heldout_T1_v{v}"; ref = z[k + "_ref"].astype(np.float32); own = z[k + "_own"]
        masks = {"all": own >= 0, "A_changed": z[k + "_reg_A_changed"], "A_unchanged": z[k + "_reg_A_unchanged"],
                 "Dstar_owned": (own >= 0) & np.isin(own, list(D)), "Vstar_owned": (own >= 0) & ~np.isin(own, list(D)), "B": own == 16}
        for arm in (FR, SE):
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
           "output_support": S["influence"]}
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
    fig, axs = plt.subplots(2, 6, figsize=(19, 7.2))
    panels = [(np.clip(z[k + "_refT0"].astype(np.float32), 0, 1), "T0 reference (same camera/light)"), (np.clip(ref, 0, 1), "T1 path-traced reference"),
              (np.clip(fr, 0, 1), f"C: frozen T0 model\n{pv(FR):.2f} dB (view)"), (np.clip(se, 0, 1), f"B: selective update (8 units)\n{pv(SE):.2f} dB (view)"),
              (ef, "|error| frozen"), (es, "|error| selective")]
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
    fig.colorbar(h, ax=axs[:, 4:], fraction=0.02, pad=0.01, label=f"|display error| 0-{EMAX}")
    fig.suptitle(f"Held-out T1 view {v} (4096 spp). Updated units D* (red), all other units bit-identical. Smoke budget: 128 T1 images, 250 epochs, cold re-init.", fontsize=10)
    f1 = out / "01_meeting_comparison.png"; fig.savefig(f1, dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].imshow(upd); ax[0].set_title("updated units: D* (red), PU band in V* (amber), V* grey", fontsize=9); off(ax[0])
    names = ["A_changed", "Dstar_owned", "Vstar_owned", "A_unchanged", "B", "all"]
    xs = np.arange(len(names)); fz = [regions[r]["mse_frozen"] for r in names]; sz = [regions[r]["mse_selective"] for r in names]
    ax[1].bar(xs - 0.2, fz, 0.4, color="#86b6ef", label="frozen"); ax[1].bar(xs + 0.2, sz, 0.4, color="#c0392b", label="selective")
    ax[1].set_xticks(xs, names, rotation=20); ax[1].set_ylabel("display MSE (16 held-out T1 views)"); ax[1].legend(frameon=False)
    ax[1].set_title("frozen vs selective by predeclared region")
    fig.tight_layout(); f2 = out / "02_mask_and_regions.png"; fig.savefig(f2, dpi=120); plt.close(fig)
    man = {"protocol": "experiments/rna_selective_reconstruction_lego/protocol/b4_smoke_meeting_v1.json", "view": v, "crop_yx": [int(y), int(x)], "crop_size": W,
           "files": [{"file": f.name, "sha256_16": sha(f)} for f in (f1, f2, out / "smoke_metrics.json")],
           "sources": [{"file": str(p), "sha256_16": sha(p)} for p in (res / "smoke/eval_smoke/eval.json", res / "smoke/B_sel_smoke/train.json", res / "oracle/oracle.json")]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"verdict": verdict, "A_changed": ach, "all": regions["all"], "Vstar": regions["Vstar_owned"], "params": rec["parameters"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
