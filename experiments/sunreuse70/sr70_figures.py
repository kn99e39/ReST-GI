"""WL70 reviewer figure (report-only; reads e2e.json, analysis.json and the unchanged WL69 analysis; computes no new metric).
01_latency_and_quality.png  left: T1 edit-to-output latency per stage (A retraced sun, B validated reuse; median over repetitions, K seed
                            s1) next to the WL69 Cycles ruled-view times at QM_B and QM_contract; right: T1 max |group bias| vs latency.
    python sr70_figures.py <root> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C1, C2, C3, C4, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85", "#1f1f1e", "#6b6b66"


def main():
    root = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    e = json.load(open(root / "e2e" / "e2e.json")); a = json.load(open(root / "analysis.json"))
    W = json.load(open(Path.home() / "rest_xbench69" / "analysis.json"))
    reps = e["runs"]["s1"]["reps"]; med = lambda arm, k: float(np.median([x["timing"].get(k, 0.0) for x in reps if x["arm"] == arm]))  # noqa: E731
    stages = [("first-order sun (retrace)", "sun_s", C2), ("validation + reuse", "validate_s", C4), ("CAK levels + aggregation", "aggregate_s", C3), ("C build + solve + readout", "compose_s", C1)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = [("A  retraced sun", "A"), ("B  validated reuse", "B")]
    for y, (lab, arm) in enumerate(rows):
        left = 0.0
        for name, k, col in stages:
            v = med(arm, k) + (med(arm, "levels_s") if k == "aggregate_s" else 0.0)
            if v > 0:
                ax.barh(y, v, left=left, color=col, height=0.55, edgecolor="white", linewidth=2, label=name if y == 0 or k == "validate_s" else None); left += v
        ax.text(left + 0.3, y, f"{med(arm, 'total_s'):.2f} s", va="center", color=INK, fontsize=9)
    cy = a["cy_ruled_T1_seconds"]; q = a["QM"]
    for y, (lab, spp) in enumerate([(f"Cycles {q['B']} spp (QM vs B)", q["B"]), (f"Cycles {q['contract']} spp (QM contract)", q["contract"])], start=2):
        v = max(cy[str(spp)]); ax.barh(y, v, color=GREY, height=0.55, edgecolor="white", linewidth=2); ax.text(v + 0.3, y, f"{v:.2f} s", va="center", color=INK, fontsize=9)
        rows.append((lab, None))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK); ax.invert_yaxis()
    ax.set_xlabel("T1 edit-to-output latency, top + underup + underdown [s]", color=MUTED); ax.set_xlim(0, 28)
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # right: quality vs latency (T1 max |group bias| over the ruled cameras)
    xs = [max(cy[s]) for s in sorted(cy, key=int)]; ys = [100 * max(v["M2"] for v in W["quality"]["CY"][s]) for s in sorted(cy, key=int)]
    bx.plot(xs, ys, "o-", color=GREY, ms=5, lw=1.5, label="Cycles Full Recompute (WL69 ladder; max of 2 realizations)")
    for s, x_, y_ in zip(sorted(cy, key=int), xs, ys):
        bx.annotate(f"{s}", (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=7, color=MUTED)
    for arm, col, mk in (("A", C2, "s"), ("B", C3, "D")):
        bx.plot([a["latency"][arm]["latency"]], [100 * a["arms"][arm]["T1"]["M2"]], mk, color=col, ms=9, mec="white", mew=2,
                label=f"ReST-GI CAK, {'retraced' if arm == 'A' else 'reused'} sun")
    bx.axhline(3, color=MUTED, ls=":", lw=1); bx.text(12, 3.25, "3 % contract", fontsize=8, color=MUTED)
    bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlabel("T1 latency [s]", color=MUTED); bx.set_ylabel("T1 max |group bias| [%]", color=MUTED); bx.set_title("M2 only; quality matching uses M1-M4 (M4 binds at 16 spp)", fontsize=8, color=MUTED)
    bx.legend(fontsize=7, frameon=False, loc="lower left"); bx.grid(color="#e6e6e3", lw=0.8, which="both"); bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.suptitle("WL70 C1 floor -> floor_c4: first-order sun-state reuse (LabServer69, RTX 3090, uncontended)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "01_latency_and_quality.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
