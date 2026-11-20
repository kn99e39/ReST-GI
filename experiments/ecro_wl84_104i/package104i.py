"""WL104I reviewer package (post-processing only): figures, copies of the decision / stage JSONs and a hash manifest.

Usage: python package104i.py <primary_dir> <review_dir>
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import FuncNorm  # noqa: E402

import common104i as CM  # noqa: E402

IO = CM.IO
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104h"))
from package104h import bin_map, ssqrt_norm, sroot4_norm  # noqa: E402

SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")
BANDS = ("near_normal", "mid", "band_55_65", "theta_70", "grazing", "all")
LUMA = CM.LUMA


def fig_training(fit, rows, out):
    keys = fit["start_keys"]
    steps = np.asarray(fit["arms"]["A1"]["history_steps"])
    fig, ax = plt.subplots(3, 3, figsize=(13, 10), layout="constrained", sharex=True)
    for m, r in enumerate(rows):
        a = ax.flat[m]
        for arm, col in (("A1", "tab:blue"), ("B1", "tab:orange")):
            Hh = np.asarray(fit["arms"][arm]["history"])          # [T, starts, materials]
            for i, k in enumerate(keys):
                a.plot(steps, Hh[:, i, m], color=col, lw=0.8 if k != "generic" else 1.6, alpha=0.8,
                       ls="--" if k == "generic" else "-", label=f"{arm} {k}" if m == 0 else None)
        a.set_yscale("log"); a.set_title(f"{r['tag']} ({r['cohort']})", fontsize=9)
    ax.flat[0].legend(fontsize=6, ncol=2)
    fig.supxlabel("Adam step"); fig.supylabel("WL82 objective el1 + ealb (training lanes)")
    fig.suptitle("Matched fits from identical physical starts: A1 original 64 vs B1 ER-v1-GF 58")
    f = out / "training_histories.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def fig_spectra(idt, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained", sharey=True)
    for key, v in idt.items():
        if not key.startswith("R0/"):
            continue
        ax[0].semilogy(v["original"]["singular_relative"], lw=0.9, label=key[3:])
        ax[1].semilogy(v["candidate"]["singular_relative"], lw=0.9)
    for a, t in zip(ax, ("original 64 parameters", "ER-v1-GF 58 parameters")):
        a.axhline(1e-15, color="k", ls="--", lw=0.6)
        a.set_title(f"{t}: normalised singular values at R0 (coordinate dependent)", fontsize=9)
        a.set_xlabel("index")
    ax[0].legend(fontsize=6)
    ax[0].set_ylabel("sigma / sigma_max")
    f = out / "jacobian_spectra.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def fig_angular(ang, out):
    S = ang["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for a, c in zip(ax, ("development", "held_out")):
        x = np.arange(len(BANDS))
        for j, (arm, key) in enumerate((("A0 (R0)", "A0_mean"), ("A1", "A1_mean"), ("B1", "B1_mean"))):
            a.bar(x + (j - 1) * 0.27, [S[c][b]["el1"][key] for b in BANDS], 0.27, label=arm)
        for i, b in enumerate(BANDS):
            e = S[c][b]["el1"]
            a.text(i, max(e["A1_mean"], e["B1_mean"]) * 1.02, f"{100 * e['point']:+.1f}%\n[{100 * e['lo']:+.1f},{100 * e['hi']:+.1f}]",
                   ha="center", fontsize=6)
        a.set_xticks(x, BANDS, rotation=20); a.set_title(f"{c}: el1 vs independent layered reference (B1 vs A1 relative change, 95% CI)", fontsize=9)
    ax[0].legend(fontsize=7)
    f = out / "angular_el1_bands.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def fig_angular_maps(primary, rows, out):
    with np.load(primary / "angular/lanes.npz") as z:
        L = {k: z[k] for k in z.files}
    s = CM.SEEDS[0]
    probes = [2, 10, 18]
    sets, M = {}, 0.0
    for m, r in enumerate(rows):
        if r["cohort"] != "development":
            continue
        sl = slice(m * 20, (m + 1) * 20)
        w = L["w"][sl]
        ref = 0.5 * (L["ref_a"][sl] + L["ref_b"][sl])
        A = (w[..., None] * ref).sum((1, 2))
        for p in probes:
            g = lambda v: bin_map(L["ref_wi"][sl][p, 0].astype(np.float64), L["ref_wo"][sl][p].astype(np.float64), (v[p] @ LUMA) / A[p])
            mp = {"ref - A1": g(ref - L[f"A1_{s}"][sl]), "ref - B1": g(ref - L[f"B1_{s}"][sl]), "B1 - A1": g(L[f"B1_{s}"][sl] - L[f"A1_{s}"][sl])}
            sets[(r["tag"], p)] = mp
            M = max(M, max(np.nanmax(np.abs(v)) for v in mp.values()))
    tags = sorted({k[0] for k in sets})
    fig, ax = plt.subplots(len(tags) * len(probes), 3, figsize=(8, 2.6 * len(tags) * len(probes)), layout="constrained")
    for i, (t, p) in enumerate([(t, p) for t in tags for p in probes]):
        for j, (k, v) in enumerate(sets[(t, p)].items()):
            im = ax[i, j].imshow(v, origin="lower", extent=[-45, 45, -45, 45], cmap="RdBu_r", norm=ssqrt_norm(M))
            ax[i, j].set_title(f"{t} θi={IO.POLAR[p]:g}° {k}", fontsize=7); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.colorbar(im, ax=ax[i, -1], shrink=0.8)
    fig.suptitle(f"Mirror-centred luma residuals / probe albedo, seed {s}, shared ±{M:.3g} (signed sqrt)", fontsize=9)
    f = out / "angular_residual_maps.png"
    fig.savefig(f, dpi=100); plt.close(fig)
    return f, M


def fig_scenes(primary, scenes, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for a, kind in zip(ax, ("primary_fixed", "secondary_native")):
        labels, pts, lo, hi = [], [], [], []
        for s in SCENES:
            blk = scenes[s][kind]
            if blk is None:
                continue
            for ev, v in blk.items():
                labels.append(f"{s.replace('test83_', '')} {ev}"); pts.append(v["point"]); lo.append(v["lo"]); hi.append(v["hi"])
        x = np.arange(len(pts))
        pts, lo, hi = map(np.asarray, (pts, lo, hi))
        a.errorbar(x, 100 * pts, yerr=[100 * (pts - lo), 100 * (hi - pts)], fmt="o", capsize=3)
        a.axhline(5, color="r", ls="--", lw=0.8); a.axhline(0, color="k", lw=0.5)
        a.set_xticks(x, labels, rotation=60, fontsize=7)
        a.set_title(f"{'fixed R0 proposal (primary)' if kind == 'primary_fixed' else 'native proposals (secondary)'}: "
                    "objects RMS, (B1 − A1)/A1 %", fontsize=9)
    f = out / "scene_relative_rms.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    maps, boxes, M = {}, {}, 0.0
    for s in SCENES:
        with np.load(primary / "scenes" / s / "maps.npz") as z:
            maps[s] = {k: z[k] @ LUMA for k in z.files}
        with np.load(IO.G_RAW / "scenes" / s / "region_masks.npz") as z:
            obj = z["objects"]
        ys, xs = np.nonzero(obj)
        boxes[s] = (max(ys.min() - 6, 0), min(ys.max() + 7, 256), max(xs.min() - 6, 0), min(xs.max() + 7, 256))
        M = max(M, max(float(np.abs(v).max()) for v in maps[s].values()))
    s0 = CM.SEEDS[0]
    cols = ("A0_{}_err", f"A1_{s0}" + "_{}_err", f"B1_{s0}" + "_{}_err", "B1_minus_A1_{}")
    fig, ax = plt.subplots(len(SCENES) * 2, 4, figsize=(13, 3.0 * len(SCENES) * 2), layout="constrained")
    for i, s in enumerate(SCENES):
        y0, y1, x0, x1 = boxes[s]
        for e, ev in enumerate(("e1", "e2")):
            for j, c in enumerate(cols):
                im = ax[2 * i + e, j].imshow(maps[s][c.format(ev)][y0:y1, x0:x1], cmap="RdBu_r", norm=sroot4_norm(M))
                ax[2 * i + e, j].set_title(f"{s.replace('test83_', '')} {ev.upper()} {c.format(ev).replace('_err', ' − ref')}", fontsize=7)
                ax[2 * i + e, j].set_xticks([]); ax[2 * i + e, j].set_yticks([])
            fig.colorbar(im, ax=ax[2 * i + e, -1], shrink=0.8)
    fig.suptitle(f"Fixed-proposal scene errors (luma, mean of 12 replicates), shared ±{M:.3g} (signed fourth root); B1 − A1 averaged over seeds")
    f2 = out / "scene_error_maps.png"
    fig.savefig(f2, dpi=100); plt.close(fig)
    return [f, f2], M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("primary", type=Path)
    ap.add_argument("review", type=Path)
    args = ap.parse_args()
    args.review.mkdir(parents=True, exist_ok=False)
    rows = IO.read(IO.MATERIALS)
    fit = IO.read(args.primary / "fit/fit.json")
    idt = IO.read(args.primary / "ident/ident.json")["results"]
    ang = IO.read(args.primary / "angular/angular.json")
    scenes = {s: IO.read(args.primary / "scenes" / s / "analysis.json") for s in SCENES}
    files = [fig_training(fit, rows, args.review), fig_spectra(idt, args.review), fig_angular(ang, args.review)]
    f, m_ang = fig_angular_maps(args.primary, rows, args.review); files.append(f)
    f, m_sc = fig_scenes(args.primary, scenes, args.review); files += f
    copies = {"decision.json": "decision.json", "equivalence/equivalence.json": "equivalence.json", "ident/ident.json": "ident.json",
              "ident_fit/ident.json": "ident_fitted.json", "fit/fit.json": "fit.json", "angular/angular.json": "angular.json",
              "native/native_audit.json": "native_audit.json", "fit/fit_states.npz": "fit_states.npz",
              "angular/compiled_states.npz": "compiled_states.npz"}
    for src, dst in copies.items():
        if (args.primary / src).exists():
            shutil.copyfile(args.primary / src, args.review / dst)
    for s in SCENES:
        shutil.copyfile(args.primary / "scenes" / s / "analysis.json", args.review / f"scene_{s}.json")
    if (args.primary.parent / "posthoc/rank.json").exists():
        shutil.copyfile(args.primary.parent / "posthoc/rank.json", args.review / "posthoc_rank.json")
    raw = {p.relative_to(args.primary).as_posix(): IO.sha(p) for p in sorted(args.primary.rglob("*"))
           if p.is_file() and p.suffix in (".json", ".npz", ".log", ".sh")}
    IO.write(args.review / "manifest.json", {
        "batch": "WL104I", "branch": "ecro_lr3_adapter", "freeze_commit": "7f710a9", "package_commit": IO.commit(),
        "protocol_sha256": IO.sha(CM.HERE / "protocol.json"),
        "code_sha256": {p.name: IO.sha(p) for p in sorted(CM.HERE.glob("*.py"))},
        "display_scales": {"angular_signed_max": m_ang, "angular_mapping": "signed sqrt, luma / probe albedo normaliser",
                           "scene_signed_max": m_sc, "scene_mapping": "signed fourth root, luma radiance"},
        "raw_root": "results/ecro_wl84_104i/primary", "raw_sha256": raw,
        "review_sha256": {p.name: IO.sha(p) for p in sorted(args.review.iterdir()) if p.is_file()}})
    print("package done", len(files), flush=True)


if __name__ == "__main__":
    main()
