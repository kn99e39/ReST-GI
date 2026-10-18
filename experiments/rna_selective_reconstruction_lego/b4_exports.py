"""Worklog-37 reviewer exports (RNA venv, CPU).

    python b4_exports.py --res <results/rna_selective_reconstruction_lego> --out <results/evaluation/37> [--pilot-only]

Images are the evaluation metric's display transform (code tonemap) of 4096-spp references and predictions; one error
scale for every arm. View and crop choices are made from the references / geometry only (most physically changed
pixels; strongest |GT delta| window), except the explicitly labelled 'largest B_sel error' window (problem example).
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
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#86b6ef", "#fcfcfb", "#f2a07b", "#9b2915"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})
TAU = 0.005
EMAX = 0.08


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def unit_rgb(lab, dirty):
    rng = np.random.default_rng(3); pal = rng.uniform(0.35, 0.85, (18, 3))
    img = np.ones(lab.shape + (3,)) * 0.99
    m = lab >= 0
    img[m] = pal[lab[m]]
    d = m & np.isin(lab, list(dirty))
    img[d] = 0.55 * img[d] + 0.45 * np.array([0.85, 0.15, 0.1])
    return img


def window(score, w):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    y, x = np.unravel_index(np.argmax(tot), tot.shape)
    return y, x


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def pilot_figure(res, out, files):
    orc = json.loads((res / "oracle/oracle.json").read_text()); z = np.load(res / "oracle/oracle_maps.npz")
    D = set(orc["D_star"]); U = orc["units"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.6))
    x = np.arange(len(U)); r = np.array([u["rms_change"] for u in U]); lo = np.array([u["ci90"][0] for u in U]); hi = np.array([u["ci90"][1] for u in U])
    ax.bar(x, r, color=["#c0392b" if u["dirty"] else "#2a78d6" for u in U], width=0.7)
    ax.errorbar(x, r, yerr=[r - lo, hi - r], fmt="none", ecolor=INK2, lw=1)
    ax.axhline(TAU, color=INK, ls="--", lw=1); ax.text(len(U) - 0.5, TAU * 1.04, "tau = 0.005", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, [("B" if u["object"] == "B" else str(u["unit"])) for u in U])
    ax.set_xlabel("unit (0-15 = L16 on A, B = moved box)"); ax.set_ylabel("RMS display change T0->T1\n(visibility-matched, noise-debiased)")
    ax.set_title(f"Oracle (32 views, 4096 spp): D* = {sorted(D)} (red), V* blue; 90% CI over views", fontsize=10)
    fig.tight_layout(); f = out / "01_oracle_units.png"; fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    views = sorted({int(k.split("_")[0][1:]) for k in z.files})
    fig, axs = plt.subplots(len(views), 5, figsize=(15, 3.1 * len(views)))
    for i, v in enumerate(views):
        row = axs[i] if len(views) > 1 else axs
        row[0].imshow(np.clip(z[f"v{v}_T0_rgb"].astype(np.float32), 0, 1)); row[0].set_title(f"view {v}: T0 reference")
        row[1].imshow(np.clip(z[f"v{v}_T1_rgb"].astype(np.float32), 0, 1)); row[1].set_title("T1 reference")
        dA = np.where(z[f"v{v}_A_cnt"], z[f"v{v}_A_d"], np.nan)
        im = row[2].imshow(dA, cmap=DIV, vmin=-0.05, vmax=0.05); row[2].set_title("A: T1 - T0 (display, counted px)")
        row[3].imshow(np.where(z[f"v{v}_A_vch"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.5); row[3].set_title("A: visibility changed (excluded)")
        row[4].imshow(unit_rgb(z[f"v{v}_T1_lab"], D)); row[4].set_title("T1 units (D* tinted red)")
        for a in row:
            off(a)
    fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01, label="display change")
    f = out / "02_oracle_views.png"; fig.savefig(f, dpi=110); plt.close(fig); files.append(f)


def arms_figures(res, out, files):
    ev = json.loads((res / "eval/eval.json").read_text()); z = np.load(res / "eval/views.npz")
    D = set(ev["D_star"]); S = ev["sets"]["heldout_T1"]
    arms = [("RNA_T1", "original RNA, T1 (control)"), ("A_full-PU", "A: T1 full recompute"), ("B_sel-PU", "B: oracle selective"), ("SO_T0-PU", "C: frozen transfer")]
    nv = S["views"]
    chg = [int(z[f"heldout_T1_v{v}_reg_A_changed"].sum()) for v in range(nv)]
    views = list(np.argsort(chg)[::-1][:2])                                        # chosen from the references only
    for v in views:
        k = f"heldout_T1_v{v}"; ref = z[k + "_ref"].astype(np.float32)
        fig, axs = plt.subplots(3, 5, figsize=(17, 10.5))
        axs[0, 0].imshow(np.clip(z[k + "_refT0"].astype(np.float32), 0, 1)); axs[0, 0].set_title(f"T1 view {v}: T0 reference (same camera/light)")
        axs[1, 0].imshow(np.clip(ref, 0, 1)); axs[1, 0].set_title("T1 path-traced reference (4096 spp)")
        gd = z[k + "_gtdelta"].astype(np.float32).mean(-1); reg = z[k + "_reg_A_changed"]
        im0 = axs[2, 0].imshow(np.where(z[k + "_own"] >= 0, gd, np.nan), cmap=DIV, vmin=-0.05, vmax=0.05); axs[2, 0].set_title("GT delta T1 - T0 (display)")
        for j, (n, t) in enumerate(arms, start=1):
            p = z[k + "_" + n].astype(np.float32)
            psnr = S["rows"][n]["per_view_psnr"][v]
            axs[0, j].imshow(np.clip(p, 0, 1)); axs[0, j].set_title(f"{t}\n{psnr:.2f} dB (view)")
            e = np.abs(p - ref).mean(-1); e[z[k + "_own"] < 0] = np.nan
            im = axs[1, j].imshow(e, cmap=SEQ, vmin=0, vmax=EMAX); axs[1, j].set_title("|error| (display)")
        own = z[k + "_own"]
        axs[2, 1].imshow(unit_rgb(own, D)); axs[2, 1].set_title("units (D* tinted red)")
        ov = np.ones(own.shape + (3,)) * 0.99
        for r, c in (("A_unchanged", (0.75, 0.85, 0.95)), ("A_weak", (0.95, 0.85, 0.55)), ("A_changed", (0.85, 0.2, 0.15)), ("A_uncorresponded", (0.4, 0.4, 0.4))):
            ov[z[k + "_reg_" + r]] = c
        ov[own == 16] = (0.6, 0.1, 0.6)
        axs[2, 2].imshow(ov); axs[2, 2].set_title("physical regions: changed red, weak amber,\nunchanged blue, uncorr. grey, B purple")
        band = (z[k + "_w"].astype(np.float32) > 0)
        bi = np.ones(own.shape + (3,)) * 0.99; bi[own >= 0] = 0.85; bi[band] = (0.95, 0.6, 0.1); bi[z[k + "_reg_Vstar_in_Dstar_band"]] = (0.85, 0.15, 0.1)
        axs[2, 3].imshow(bi); axs[2, 3].set_title("PU bands (amber); V* px in D* band (red)")
        ef = np.abs(z[k + "_B_sel-PU"].astype(np.float32) - z[k + "_A_full-PU"].astype(np.float32)).mean(-1); ef[own < 0] = np.nan
        axs[2, 4].imshow(ef, cmap=SEQ, vmin=0, vmax=EMAX / 2); axs[2, 4].set_title("|B_sel - A_full| (half scale)")
        for a in axs.ravel():
            off(a)
        fig.colorbar(im, ax=axs[1, :], fraction=0.012, pad=0.01, label=f"|error| 0-{EMAX}")
        fig.colorbar(im0, ax=axs[2, 0], fraction=0.04, pad=0.01)
        f = out / f"03_arms_view{v}.png"; fig.savefig(f, dpi=100); plt.close(fig); files.append(f)
    # crops: strongest GT change (bleeding), a V* region far from D*, and the largest B_sel error window (problem example)
    W = 160; rows = []
    v0 = views[0]; k = f"heldout_T1_v{v0}"
    gd = np.abs(z[k + "_gtdelta"].astype(np.float32)).mean(-1); gd[z[k + "_own"] < 0] = 0
    rows.append(("strongest GT change (bleeding)", v0, window(gd, W)))
    own = z[k + "_own"]; vs = (own >= 0) & ~np.isin(own, list(D)) & z[k + "_reg_A_unchanged"]
    rows.append(("V* unchanged region (preserved units)", v0, window(vs.astype(float), W)))
    worst = None
    for v in range(nv):
        kk = f"heldout_T1_v{v}"; e = np.abs(z[kk + "_B_sel-PU"].astype(np.float32) - z[kk + "_ref"].astype(np.float32)).mean(-1); e[z[kk + "_own"] < 0] = 0
        y, x = window(e, W); sc = e[y:y + W, x:x + W].sum()
        if worst is None or sc > worst[0]:
            worst = (sc, v, (y, x))
    rows.append(("largest B_sel error window (problem example)", worst[1], worst[2]))
    fig, axs = plt.subplots(len(rows), 6, figsize=(17, 3.0 * len(rows)))
    for i, (t, v, (y, x)) in enumerate(rows):
        kk = f"heldout_T1_v{v}"; sl = (slice(y, y + W), slice(x, x + W))
        axs[i, 0].imshow(np.clip(z[kk + "_refT0"].astype(np.float32)[sl], 0, 1)); axs[i, 0].set_title(f"{t}\nview {v}: T0 ref", fontsize=8)
        axs[i, 1].imshow(np.clip(z[kk + "_ref"].astype(np.float32)[sl], 0, 1)); axs[i, 1].set_title("T1 ref")
        for j, (n, tt) in enumerate(arms, start=2):
            axs[i, j].imshow(np.clip(z[kk + "_" + n].astype(np.float32)[sl], 0, 1)); axs[i, j].set_title(tt, fontsize=8)
        for a in axs[i]:
            off(a)
    f = out / "04_crops.png"; fig.tight_layout(); fig.savefig(f, dpi=110); plt.close(fig); files.append(f)
    return views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pilot-only", action="store_true")
    args = ap.parse_args()
    res, out = Path(args.res), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = []
    pilot_figure(res, out, files)
    sources = [res / "oracle/oracle.json"]
    views = None
    if not args.pilot_only:
        views = arms_figures(res, out, files); sources.append(res / "eval/eval.json")
    man = {"worklog": "docs/worklogs/37_WORKLOG_ORACLE_SELECTIVE_RECONSTRUCTION_DIFFUSE_LEGO.md", "protocol": "experiments/rna_selective_reconstruction_lego/protocol/b4_v1.json",
           "display": "code tonemap (x/(1+x))^(1/2.2) of 4096-spp references and predictions; error scale 0-0.08 for every arm",
           "view_choice": "oracle views 0-3 (fixed); held-out views with the most physically changed A pixels (references only)",
           "files": [{"file": f.name, "sha256_16": sha(f)} for f in files], "sources": [{"file": str(s), "sha256_16": sha(s)} for s in sources if s.exists()],
           "heldout_views_shown": [int(v) for v in views] if views is not None else None}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("wrote", [f.name for f in files])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
