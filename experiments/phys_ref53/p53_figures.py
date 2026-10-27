"""WL53 reviewer figures -> results/evaluation/53 (protocol phys_ref53).

    python p53_figures.py <results dir> <evaluation dir>
Error maps share ONE absolute scale (max error over every map of this worklog), sqrt colour map, colourbar in radiance
units; no per-image normalisation.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p53_fixtures as FX  # noqa: E402

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"             # categorical slots 1-4, fixed order
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 2})


def dirkey(d):
    for k, v in FX.DIRS.items():
        if np.allclose(v, d):
            return k


def fig_analytic(R, E):
    rd, cd, od = R / "render", R / "cycles", R / "oracle"
    cy = json.load(open(cd / "c_g1a.json")); iw = json.load(open(od / "g1a_iw.json"))
    mt = json.load(open(rd / "g1a_mitsuba.json")) if (rd / "g1a_mitsuba.json").exists() else None
    x = np.arange(len(cy)); fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axhline(1, color=INK2, lw=1)
    ax.plot(x, [c["cycles"] / c["analytic"] for c in cy], "o", ms=5, color=INK, label="Cycles volume (reference)")
    ax.plot(x, [c["iw"] / c["analytic"] for c in iw], "s", ms=5, mfc="none", color=S2, label="IW (independent walk)")
    if mt:
        ax.plot(x, [c["mitsuba"] / c["analytic"] for c in mt], "^", ms=5, color=S3, label="Mitsuba volpath (diagnostic)")
    ax.set_xticks(x); ax.set_xticklabels([f"{c['albedo']}\n{int(c['theta0'])}/{int(c['theta'])}" for c in cy], fontsize=6)
    ax.set_ylabel("radiance / exact H-function"); ax.set_title("Semi-infinite isotropic medium (albedo, sun / view angle)", color=INK, loc="left")
    ax.legend(loc="lower left", ncol=3); fig.tight_layout(); p = E / "01_semi_infinite_vs_analytic.png"; fig.savefig(p, dpi=160); plt.close(fig); return p


def fig_profile(R, E):
    rd, od = R / "render", R / "oracle"
    m = FX.slab_mesh(); pc = m["port_center"]; pn = m["port_normal"]; r = np.hypot(pc[:, 0], pc[:, 1])
    z = np.load(od / "g2_pred.npz"); cyc = np.load(rd / "cports_iso_L3.npz")
    mit = np.load(rd / "ports_iso_L3.npz") if (rd / "ports_iso_L3.npz").exists() else None
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (cam, face, k) in zip(axs, (("top", 1, 0), ("bot", -1, 1))):
        sel = (pn[:, 2] == face) & (cyc[f"{cam}__count"] >= 30)
        o = np.argsort(r[sel]); rr = r[sel][o] * 100
        ax.semilogy(rr, cyc[f"{cam}__mean"].mean(0)[sel][o, 0], "o", ms=4, color=INK, label="Cycles (reference)")
        ax.semilogy(rr, z["iso_L3__L_exact"][k][sel][o, 0], "-", color=S1, label="TR oracle (exact source)")
        ax.semilogy(rr, z["iso_L3__L_iw"][k][sel][o, 0], "--", color=S2, label="IW")
        if mit is not None:
            ax.semilogy(rr, mit[f"{cam}__mean"].mean(0)[sel][o, 0], "^", ms=3, color=S3, label="Mitsuba (diagnostic)")
        ax.set_xlabel("distance of port centre from beam axis (cm)"); ax.set_title(f"{'top' if face > 0 else 'bottom'} face, red", color=INK, loc="left")
    axs[0].set_ylabel("port radiance (W m$^{-2}$ sr$^{-1}$)"); axs[0].legend(loc="upper right")
    fig.suptitle("Nonlocal response of the 1 cm slab to a 2 x 2 cm beam (iso_L3)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); p = E / "02_slab_beam_profile.png"; fig.savefig(p, dpi=160); plt.close(fig); return p


def _errmaps(R, items):
    rd, od = R / "render", R / "oracle"; maps = []
    for scene, cam, arm, mode, title in items:
        P = np.load(od / "pred" / f"{scene}__{arm}.npz"); sd = FX.scene_def(scene); key = f"{dirkey(sd['cams'][cam]['dir'])}__{mode}"
        pid = np.load(rd / f"cpid_{scene}_{cam}.npy")
        ref = np.mean([np.load(f) for f in sorted((R / "cycles").glob(f"cimg_{scene}_{cam}_*.npy"))], 0)[..., 0]
        pred = np.where(pid >= 0, P[key][np.clip(pid, 0, None), 0], np.nan)
        cnt = np.load(rd / f"cports_{scene}.npz")[f"{cam}__count"]
        refp = np.zeros(len(cnt)); np.add.at(refp, pid[pid >= 0], ref[pid >= 0]); refp /= np.maximum(cnt, 1)
        err = np.where(pid >= 0, np.abs(pred - refp[np.clip(pid, 0, None)]), np.nan)
        maps.append((np.flipud(err), title))
    return maps


def fig_composition(R, E, ops):
    items = [("gap4", "gapdown", "part_s1", "nee", "gap4: lower slab, gap face"), ("gap4", "gapup", "part_s1", "nee", "gap4: upper slab, gap face"),
             ("floor", "underup", "part_s1", "nee", "floor: slab bottom"), ("lblock", "ocorner", "part_s1", "nee", "L-block: concave corner view"),
             ("lblock_pose1", "otop", "reuse_pose0", "nee", "L-block pose 1, REUSED pose-0 K"), ("dent", "otop", "stale_rest", "bin", "dent, STALE rest K"),
             ("dent", "otop", "current", "bin", "dent, current K"), ("gap4", "gapdown", "pair4dup_full", "bin", "gap4 'dup' double count")]
    maps = _errmaps(R, items); finite = [m[np.isfinite(m)] for m, _ in maps[:-1]]
    vmax = max(float(np.nanmax(f)) for f in finite if f.size)                                   # shared absolute scale (dup arm clipped)
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
    for ax, (m, t) in zip(axs.ravel(), maps):
        im = ax.imshow(np.clip(m, 0, vmax), cmap=SEQ, norm=PowerNorm(0.5, vmin=0, vmax=vmax)); ax.set_title(t, fontsize=8, color=INK, loc="left"); ax.axis("off")
    cb = fig.colorbar(im, ax=axs, shrink=0.8); cb.set_label("|port prediction - Cycles| (W m$^{-2}$ sr$^{-1}$), red, sqrt scale")
    fig.suptitle("Composed oracle K + C vs monolithic Cycles reference (one absolute scale for all panels)", x=0.01, ha="left", color=INK)
    p = E / "03_composition_error_maps.png"; fig.savefig(p, dpi=160); plt.close(fig); return p


def fig_deform(R, E, ops):
    names = [("bend30_s3", "bend 30°"), ("bend90_s3", "bend 90°"), ("dent_s3", "dent")]
    bins = list(ops["bend30_s3"]["energy_by_distance"].keys()); x = np.arange(len(bins)); w = 0.26
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, (nm, lab) in enumerate(names):
        axs[0].bar(x + (i - 1) * w, [ops[nm]["energy_by_distance"][b][0] for b in bins], w - 0.02, color=(S1, S2, S3)[i], label=lab)
    axs[0].set_xticks(x); axs[0].set_xticklabels([b.replace("0.0", "").replace("-", "–") + " m" for b in bins], fontsize=7)
    axs[0].set_ylabel("share of ||ΔT||² (red)"); axs[0].set_title("Where the operator change lives (material distance)", color=INK, loc="left"); axs[0].legend()
    for i, (nm, lab) in enumerate(names):
        axs[1].bar(np.arange(3) + (i - 1) * w, ops[nm]["fro_vs_noise"], w - 0.02, color=(S1, S2, S3)[i], label=lab)
    axs[1].axhline(1, color=INK2, lw=1); axs[1].axhline(3, color=INK2, lw=1, ls="--")
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["R", "G", "B"]); axs[1].set_ylabel("||ΔT|| / ||seed noise||")
    axs[1].set_title("Change magnitude vs MC noise (dashed: 'measured' rule)", color=INK, loc="left")
    fig.tight_layout(); p = E / "04_nonrigid_operator_change.png"; fig.savefig(p, dpi=160); plt.close(fig); return p


def main():
    R, E = Path(sys.argv[1]), Path(sys.argv[2]); E.mkdir(parents=True, exist_ok=True)
    ops = json.load(open(R / "oracle" / "ops.json"))
    out = [fig_analytic(R, E), fig_profile(R, E), fig_composition(R, E, ops), fig_deform(R, E, ops)]
    man = dict(worklog=53, protocol="phys_ref53", source=str(R), files=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out],
               note="error maps: single absolute scale = max over all non-control maps; the 'dup' control panel is clipped to that scale")
    json.dump(man, open(E / "manifest.json", "w"), indent=1); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
