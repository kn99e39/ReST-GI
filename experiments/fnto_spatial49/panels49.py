"""WL49 reviewer panels + manifest into results/evaluation/49.

Images: panel views 8 / 12 (all Yellow pixels valid in T0 and T1), display ldr3 of Cycles reference / K44 / L0 / L1 /
oracle on Yellow only; error maps |pred - ref| (luminance) with ONE absolute scale M for every error map of the worklog
(sqrt mapping, real-unit colour bar); T0 -> T1 change maps with one shared signed scale. Numeric panels for Gates 1-3.

    python panels49.py <results/fnto_spatial49> <results/evaluation/49>
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fnto_lego_r1")); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"; INK, MUTED = "#262625", "#73726c"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
CONDS = [("ref", "Cycles first_r1"), ("K44", "K44 (historical)"), ("L0_s4911", "L0 existing aggregation"), ("L1_s4911", "L1 DASQ"), ("oracle", "Burley oracle")]


def img_of(vals, mk, res):
    out = np.zeros((res * res, vals.shape[-1]) if vals.ndim == 2 else res * res, np.float32); out[mk] = vals; return out.reshape(res, res, -1) if vals.ndim == 2 else out.reshape(res, res)


def main():
    R, ev = Path(sys.argv[1]), Path(sys.argv[2]); ev.mkdir(parents=True, exist_ok=True)
    res = json.loads((C.ROOT.parent / "views_r1.json").read_text())["res"]
    P = np.load(R / "gate4_real_scene/g4_panel_views.npz")
    errs = {(st, v, c): np.abs((P[f"{st}_v{v}_{c}"] - P[f"{st}_v{v}_ref"]) @ LUM) for st in ("T0", "T1") for v in (8, 12) for c, _ in CONDS[1:]}
    M = float(max(e.max() for e in errs.values())); chg = {(v, c): (P[f"T1_v{v}_{c}"] - P[f"T0_v{v}_{c}"]) @ LUM for v in (8, 12) for c, _ in CONDS}
    Mc = float(max(np.abs(x).max() for x in chg.values()))
    for v in (8, 12):
        mk = P[f"T0_v{v}_mk"]; ys, xs = np.unravel_index(mk, (res, res)); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        fig, ax = plt.subplots(3, 5, figsize=(14, 8.6))
        for j, (c, lab) in enumerate(CONDS):
            im = img_of(C.ldr3(P[f"T0_v{v}_{c}"]), mk, res)[y0:y1, x0:x1]
            ax[0, j].imshow(np.clip(im, 0, 1)); ax[0, j].set_title(f"{lab}, T0", color=INK); ax[0, j].axis("off")
            if c != "ref":
                e = img_of(np.sqrt(errs[("T0", v, c)] / M), mk, res)[y0:y1, x0:x1]; h = ax[1, j].imshow(e, cmap="Blues", vmin=0, vmax=1)
                cb = fig.colorbar(h, ax=ax[1, j], fraction=0.046); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", f"{0.25 * M:.3g}", f"{M:.3g}"])
                ax[1, j].set_title(f"|error| lum (sqrt, shared M)", color=INK)
            ax[1, j].axis("off")
            dc = img_of(chg[(v, c)], mk, res)[y0:y1, x0:x1]; h = ax[2, j].imshow(np.sign(dc) * np.sqrt(np.abs(dc) / Mc), cmap="RdBu_r", vmin=-1, vmax=1)
            cb = fig.colorbar(h, ax=ax[2, j], fraction=0.046); cb.set_ticks([-1, 0, 1]); cb.set_ticklabels([f"-{Mc:.3g}", "0", f"{Mc:.3g}"])
            ax[2, j].set_title(f"T1 - T0 change (sqrt, shared)", color=INK); ax[2, j].axis("off")
        fig.suptitle(f"Yellow first-event, held-out view {v} (Yellow pixels only; other pixels black)", color=INK); fig.tight_layout()
        fig.savefig(ev / f"01_yellow_view{v}.png"); plt.close(fig)
    g1 = json.loads((R / "gate1_existing_capacity/gate1.json").read_text()); g1b = json.loads((R / "gate1_existing_capacity/gate1b.json").read_text())
    g2 = json.loads((R / "gate2_candidate/gate2.json").read_text()); g3 = json.loads((R / "gate3_learned_ab/gate3.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); x = np.arange(3); w = 0.13
    series = [("A0 quad (existing)", lambda c: g1["channels"][c]["primary_test"]["A0"]["shadow"], C1),
              ("A1* best fit (existing)", lambda c: g1b["channels"][c]["primary_test"][g1b["channels"][c]["A1_star"]]["shadow"], C2),
              ("DASQ a1.0 quad", lambda c: g2["conditions"]["B2p"]["channels"][c]["primary_test"]["quad"]["shadow"], C3),
              ("DASQ a1.0 best fit", lambda c: g2["conditions"]["B2p"]["channels"][c]["primary_test"][g2["conditions"]["B2p"]["channels"][c]["A1_star"]]["shadow"], C4),
              ("L0 learned (heldout rcv)", lambda c: g3["arms"]["L0"]["seeds"]["4911"]["primary_heldout_test"][c]["shadow"], "#9b9a93"),
              ("L1 learned (heldout rcv)", lambda c: g3["arms"]["L1"]["seeds"]["4911"]["primary_heldout_test"][c]["shadow"], "#185a8e")]
    for i, (lab, f, col) in enumerate(series):
        ax[0].bar(x + (i - 2.5) * w, [f(c) for c in "RGB"], w * 0.95, color=col, label=lab)
    ax[0].scatter(x, [g1["channels"][c]["primary_test"]["collision"]["shadow"] for c in "RGB"], marker="_", s=300, color=INK, label="atom collision (irreducible, existing)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["R", "G", "B"]); ax[0].set_ylabel("in-support shadow rel L2 (test fields)"); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("Gates 1-3: existing aggregation vs DASQ (oracle response)", color=INK)
    bins = ["0.00-0.01", "0.01-0.03", "0.03-0.10", "0.10-0.30"]
    for lab, f, col in (("A0 quad", lambda k: g1["channels"]["R"]["sets"]["view_T0"]["A0"].get(f"shadow_d{k}"), C1),
                        ("DASQ quad", lambda k: g2["conditions"]["B2p"]["channels"]["R"]["sets"]["view_T0"]["quad"].get(f"shadow_d{k}"), C3),
                        ("L0 learned", lambda k: g3["arms"]["L0"]["seeds"]["4911"]["heldout_receivers"]["view_T0"]["R"].get(f"shadow_d{k}"), "#9b9a93"),
                        ("L1 learned", lambda k: g3["arms"]["L1"]["seeds"]["4911"]["heldout_receivers"]["view_T0"]["R"].get(f"shadow_d{k}"), "#185a8e")):
        ax[1].plot(range(4), [f(k) for k in bins], "o-", color=col, label=lab)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([b + " m" for b in bins]); ax[1].set_title("Red, view_T0 shadow error by distance to light", color=INK); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(ev / "02_capacity_and_learning.png"); plt.close(fig)
    for src in ("gate0_reference/gate0.json", "gate1_existing_capacity/gate1.json", "gate1_existing_capacity/gate1b.json", "gate2_candidate/gate2.json",
                "gate3_learned_ab/gate3.json", "gate3_learned_ab/gate3_run1_4000steps.json", "gate4_real_scene/gate4.json", "gate4_real_scene/gate4_run1_refindex_bug.json"):
        if (R / src).exists():
            shutil.copy(R / src, ev / ("d49__" + src.replace("/", "__")))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(ev.iterdir()) if p.name != "manifest.json"}
    (ev / "manifest.json").write_text(json.dumps({"error_scale_M_lum": M, "change_scale_lum": Mc, "mapping": "sqrt(|e| / M); signed sqrt for changes",
                                                  "rule": "one absolute scale for every error map and one for every change map of worklog 49", "files": files}, indent=1))


main()
