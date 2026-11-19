"""WL106 stop report figures (descriptive; after the frozen synthetic gate stopped the batch before R4 training).

python report106.py DATA105 QUALIFY_JSON SYNTH_JSON OUT
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth106 as S  # noqa: E402
import model106 as M6  # noqa: E402
M = M6.M
BINS = np.array(S.BINS)
C = (0.5 * (BINS[1:-2] + BINS[2:-1]))
COL = {"R1": "#eb6834", "R2": "#1baf7a", "R3": "#6250d6", "R3+C": "#c2185b"}


def nanmean_rows(rows):
    a = np.array([[np.nan if x is None else x for x in r] for r in rows], float)
    return np.nanmean(a, 0)


def main(data, qj, sj, out):
    out.mkdir(parents=True, exist_ok=True)
    q = json.loads(qj.read_text())["results"]
    # Figure 1: Part A coverage profiles (eval: R1-R3 total vs mixing; val: R3 vs frozen-R3 + constraint).
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout="constrained", sharey="row")
    for j, s in enumerate(("4", "16", "32")):
        a_ = ax[0, j]
        for arm in ("R1", "R2", "R3"):
            tot = nanmean_rows([r for v in q["eval"] for r in q["eval"][v][arm][s]["bins_bias"]])[1:-1]
            mix = nanmean_rows([r for v in q["eval"] for r in q["eval"][v][arm][s]["bins_mix"]])[1:-1]
            a_.plot(C, tot, "o-", color=COL[arm], label=f"{arm} total")
            a_.plot(C, mix, ":", color=COL[arm], label=f"{arm} mixing term")
        a_.set_title(f"evaluation views · {s} spp (WL105 target)")
        a_ = ax[1, j]
        for arm in ("R3", "R3+C"):
            tot = nanmean_rows([r for v in q["val"] for r in q["val"][v][arm][s]["bins_bias"]])[1:-1]
            mix = nanmean_rows([r for v in q["val"] for r in q["val"][v][arm][s]["bins_mix"]])[1:-1]
            a_.plot(C, tot, "o-", color=COL[arm], label=f"{arm} total")
            a_.plot(C, mix, ":", color=COL[arm], label=f"{arm} mixing term")
        a_.set_title(f"validation views · {s} spp · R3+C: frozen R3 + constraint", fontsize=9)
        for a_ in ax[:, j]:
            a_.axhline(0, color="#58616b", lw=.8); a_.axhspan(-0.01, 0.01, color="#9aa0a6", alpha=.15); a_.grid(alpha=.25)
            a_.set_xlabel("sphere coverage fraction of the pixel")
    ax[0, 0].set_ylabel("silhouette signed residual / target"); ax[1, 0].set_ylabel("silhouette signed residual / target")
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.suptitle("WL106 Part A · silhouette bias by coverage: total vs mixing term (shaded ±1%)")
    fig.savefig(out / "partA_coverage_profile.png", dpi=130); plt.close(fig)
    # Figure 2: R3 kernel sphere-coverage moment per bin (eval) — the kernels import background, not sphere.
    fig, a_ = plt.subplots(figsize=(6.5, 4), layout="constrained")
    for s, mk in (("4", "o"), ("16", "s"), ("32", "D")):
        for comp, ls in (("LOW", "-"), ("HIGH", "--")):
            mo = nanmean_rows([r for v in q["eval"] for r in q["eval"][v]["R3"][s]["moment_sphere_bins"][comp]])[1:-1]
            a_.plot(C, mo, mk + ls, label=f"R3 {comp} {s} spp", ms=4)
    a_.axhline(0, color="#58616b", lw=.8); a_.grid(alpha=.25); a_.legend(fontsize=7)
    a_.set_xlabel("sphere coverage fraction of the pixel"); a_.set_ylabel("sum_q w_q c_sph,q - c_sph,p")
    a_.set_title("WL106 Part A · R3 kernel sphere-coverage moment (evaluation views)")
    fig.savefig(out / "partA_kernel_moment.png", dpi=130); plt.close(fig)
    # Figure 3: synthetic disk, box kernel K1, const vs rim, U vs C (expected-value signed error maps).
    sc = S.scenes(128)
    rec = sc["disk"]
    cov = torch.from_numpy(rec["cover8"])[None].cuda()
    lg = torch.from_numpy(S.kernels(128, "K1"))[None].cuda()
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4), layout="constrained")
    for i, f in enumerate(("const", "rim")):
        Tt = rec["Tex"][f]
        img = torch.from_numpy(np.repeat(Tt[None], 3, 0).astype(np.float32))[None].cuda()
        with torch.no_grad():
            M.FUSED = False
            eu = M.apply_kernel(img, lg)[0, 0].double().cpu().numpy() - Tt
            ec = M6.constrained_apply(img, lg, cov)[0, 0].double().cpu().numpy() - Tt
        sl = np.s_[20:60, 14:54]
        cap = float(np.abs(eu[sl]).max())
        axs[i, 0].imshow(Tt[sl], cmap="gray"); axs[i, 0].set_title(f"target ({f})", fontsize=9)
        axs[i, 1].imshow(rec["cover8"][6][sl], cmap="gray", vmin=0, vmax=1); axs[i, 1].set_title("A coverage (8x8 guide)", fontsize=9)
        axs[i, 2].imshow(eu[sl], cmap="RdBu_r", vmin=-cap, vmax=cap); axs[i, 2].set_title(f"unconstrained E[y]-T (±{cap:.2f})", fontsize=9)
        axs[i, 3].imshow(ec[sl], cmap="RdBu_r", vmin=-cap, vmax=cap); axs[i, 3].set_title(f"constrained E[y]-T (±{cap:.2f})", fontsize=9)
        for a_ in axs[i]:
            a_.set_axis_off()
    fig.suptitle("WL106 Part B · synthetic disk, 15x15 box kernel: per-surface constants vs rim darkening of A")
    fig.savefig(out / "partB_disk_maps.png", dpi=120); plt.close(fig)
    # Figure 4: synthetic per-coverage-bin bias (K1, K3) for const and rim, gated scenes, Tex.
    syn = json.loads(sj.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained", sharey=True)
    for a_, f in zip(axs, ("const", "rim")):
        for row in syn["rows"]:
            if row["truth"] == "Tex" and row["field"] == f and row["kernel"] in ("K1", "K3") and row["scene"] in ("disk", "edge17"):
                b = np.array([np.nan if x is None else x for x in row["coverage_bins_signed"]])[1:-1]
                a_.plot(C, b, ("-" if row["op"] == "C" else ":") + ("o" if row["kernel"] == "K1" else "s"),
                        color="#c2185b" if row["op"] == "C" else "#6250d6", alpha=.8, ms=3,
                        label=f'{row["scene"]} {row["kernel"]} {"constrained" if row["op"] == "C" else "unconstrained"}')
        a_.axhline(0, color="#58616b", lw=.8); a_.axhspan(-0.01, 0.01, color="#9aa0a6", alpha=.15); a_.grid(alpha=.25)
        a_.set_title(f"field: {f}"); a_.set_xlabel("A coverage fraction")
    axs[0].set_ylabel("signed bias / target (exact-area truth)"); axs[0].legend(fontsize=6)
    fig.suptitle("WL106 Part B · per-coverage bias, generic scenes")
    fig.savefig(out / "partB_coverage_bins.png", dpi=130); plt.close(fig)
    print("REPORT FIGURES DONE")


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:5]])
