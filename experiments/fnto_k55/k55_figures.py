"""WL55 reviewer figures (results/evaluation/55) from the stored analysis JSON; manifest with sha256.
    python k55_figures.py <analysis dir> <out dir>"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C3 = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (R, G, B channels), fixed order
C4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
CH = ("R", "G", "B")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def dotplot(ax, rows, title):
    """rows: list of (label, [r, g, b] in %). One row per arm, one marker per channel; +-3 % band."""
    ax.axvspan(-3, 3, color="#f1f0ea", zorder=0); ax.axvline(0, color=MUTED, lw=0.8)
    for i, (lab, v) in enumerate(rows):
        for c in range(3):
            ax.plot(v[c], i + (c - 1) * 0.18, "o", ms=6, color=C3[c], mec="white", mew=1.2, zorder=3, label=CH[c] if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlabel("bias vs Cycles (%)  (shaded: +-3 % rule)")
    ax.set_title(title, loc="left", color=INK, pad=22)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, handletextpad=0.2)


def main(A, O):
    O.mkdir(parents=True, exist_ok=True); files = []
    R = json.load(open(A / "report_tables.json")); BD = json.load(open(A / "stageBD.json")); E = json.load(open(A / "stageE.json"))
    # 1 readout attribution: gap8 obot slab_lo:-y against the 0.125 mm reference
    ob = R["oblique_vs_0p125mm"]["gap8|obot"]; g = "slab_lo:-y"
    rows = [("WL53 stored (nee)", ob["P10_wl53|nee"][g]), ("1 cm, nee", ob["P10|nee"][g]), ("1 cm, nee + ballistic", ob["P10|ball"][g]),
            ("CAK, nee + ballistic", ob["CAK|ball"][g]), ("continuous oracle, nee", ob["COR|nee"][g]), ("continuous oracle, nee + ballistic", ob["COR|ball"][g])]
    fig, ax = plt.subplots(figsize=(7.0, 3.3)); dotplot(ax, rows, "gap8 obot slab_lo:-y vs 0.125 mm reference")
    fig.tight_layout(); p = O / "01_gap8_oblique_readout.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p)
    # 2 spatial convergence of face-on near-contact groups (nee + KB, amended unit, 0.5 mm references)
    sel = [("gap4", "gapdown", "slab_lo:top"), ("gap4", "gapup", "slab_hi:bot"), ("gap8", "gapup", "slab_hi:bot"), ("floor", "underup", "dom:bot"),
           ("bend90", "underup", "dom:bot"), ("gap6s", "gapup", "slab_hi:bot")]
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 4.8), sharey=True)
    for ax, (sc, cam, gn) in zip(axs.ravel(), sel):
        rr = {r["ch"]: r for r in BD["full"][sc]["rules"] if r["cam"] == cam and r["group"] == gn}
        for c in range(3):
            r = rr.get(c)
            if r is None:
                continue
            y = [100 * r[f"T_{a}"] for a in ("P10", "P05", "P025")]
            ax.plot([0, 1, 2], y, "-o", color=C3[c], lw=2, ms=5, mec="white", label=CH[c]); ax.plot(3, 100 * r["T_CAK"], "D", color=C3[c], ms=6, mec="white")
        ax.axhspan(-3, 3, color="#f1f0ea", zorder=0); ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["1 cm", "5 mm", "2.5 mm", "CAK"]); ax.set_title(f"{sc} {cam} {gn}", loc="left", color=INK, fontsize=8)
    axs[0, 0].set_ylabel("bias vs Cycles (%)"); axs[1, 0].set_ylabel("bias vs Cycles (%)"); axs[0, 0].legend(frameon=False)
    fig.suptitle("Face-on near-contact groups: matched K/C refinement (gap6s held out)", x=0.01, ha="left", color=INK); fig.tight_layout()
    p = O / "02_face_on_convergence.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p)
    # 3 outgoing aggregation test
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for i, geo in enumerate(("gap4", "gap8")):
        bt = BD["btest"][geo]
        axs[0].plot([0, 1, 2], [100 * bt[f"L{l}"]["pooled"][0] for l in (1, 2, 4)], "-o", color=C4[i], lw=2, mec="white", label=geo)
        axs[1].plot([0, 1, 2], [100 * bt[f"L{l}"]["port_p90"][0] for l in (1, 2, 4)], "-o", color=C4[i], lw=2, mec="white", label=geo)
    for ax, t in zip(axs, ("pooled receiver flux error (%), R", "per-port |error| p90 (%), R")):
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1 cm", "5 mm", "2.5 mm"]); ax.set_title(t, loc="left", color=INK, fontsize=9); ax.axhline(0, color=MUTED, lw=0.8)
    axs[0].legend(frameon=False); fig.suptitle("Outgoing aggregation: first-flight exit field of slab_lo delivered across the gap (vs continuous oracle)", x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout(); p = O / "03_outgoing_aggregation.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p)
    # 4 near-contact geometry change floor -> floor_c4 (rest K reused)
    G4 = R["groups_floor_c4"]; rows = []
    for cam, gn in (("underup", "dom:bot"), ("underdown", "floor")):
        for a, lab in (("P10|ball|cyc", "1 cm"), ("CAK|ball|cyc", "CAK"), ("COR|ball|cyc", "oracle")):
            rows.append((f"{gn} ({cam}), {lab}", G4[cam][a][gn]))
    fig, ax = plt.subplots(figsize=(7.0, 3.5)); dotplot(ax, rows, "floor_c4: rigid move to 4 mm, rest K reused")
    fig.tight_layout(); p = O / "04_near_contact_change.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p)
    # 5 operator change vs seed noise by aggregation
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    for ax, geo in zip(axs, ("bend90", "dent")):
        o = E[geo]["operator"]; x = np.arange(3)
        for c in range(3):
            ax.bar(x + (c - 1) * 0.26, [o[t]["fro_vs_noise"][c] for t in ("fine_2p5mm", "cak_cells", "port_1cm")], width=0.24, color=C3[c], label=CH[c], edgecolor="white", linewidth=1)
        ax.axhline(3, color=INK, lw=0.8, ls="--"); ax.text(2.45, 3.15, "detect = 3", color=MUTED, fontsize=7, ha="right")
        ax.set_xticks(x); ax.set_xticklabels(["2.5 mm cells", "CAK cells", "1 cm ports"]); ax.set_title(geo, loc="left", color=INK)
    axs[0].set_ylabel("||K_cur - K_rest|| / ||seed noise||"); axs[0].legend(frameon=False)
    fig.tight_layout(); p = O / "05_operator_change_vs_noise.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p)
    man = [dict(file=f.name, sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(), source="experiments/fnto_k55/k55_figures.py from results/fnto_k55/analysis") for f in files]
    json.dump(dict(worklog=55, figures=man), open(O / "manifest.json", "w"), indent=1); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
