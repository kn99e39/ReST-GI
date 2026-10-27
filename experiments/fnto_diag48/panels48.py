"""WL48 reviewer panels (numeric; no image difference maps) + manifest into results/evaluation/48.

    python panels48.py <results/fnto_diag48> <results/evaluation/48>
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

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#262625", "#73726c"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def p_a(R, ev):
    a = json.loads((R / "probe_a/probe_a.json").read_text()); ph = json.loads((R / "posthoc/posthoc.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(3); w = 0.2
    for i, (lab, col, f) in enumerate((("K44 hemispherical (all K parts)", C1, lambda v: a["views"][v]["energy_first_event"]["out_hemispherical_over_in_RGB"]),
                                        ("K44 production (normal)", C2, lambda v: a["views"][v]["energy_first_event"]["out_normal_production_over_in_RGB"]),
                                        ("Burley oracle, reference semantics (post hoc)", C3, lambda v: ph["P1_reference_energy"][v]["oracle_yellow_exitance_flux_over_absorbed_RGB"]))):
        for j, v in enumerate(("view6", "view15")):
            ax[0].bar(x + (i - 1) * w + (j - 0.5) * w / 2, f(v), w / 2, color=col, label=lab if j == 0 else None, alpha=1 if j == 0 else 0.6)
    ax[0].axhline(1, color=INK, lw=1, ls="--"); ax[0].set_xticks(x); ax[0].set_xticklabels(["R", "G", "B"])
    ax[0].set_ylabel("outgoing flux / absorbed sun power"); ax[0].set_title("Probe A: first-event energy (views 6 solid, 15 light)", color=INK); ax[0].legend(frameon=False, fontsize=7)
    groups = ["Yellow_observed_lit", "Yellow_observed_shadow", "Yellow_never_lit", "Yellow_never_shadow"]
    for j, v in enumerate(("view6", "view15")):
        g = a["views"][v]["groups"]
        ax[1].plot(range(4), [g[k]["AI_median"] for k in groups], "o-", color=C1, alpha=1 if j == 0 else 0.5, label=f"AI median (19 bins) {v}")
        ax[1].plot(range(4), [g[k]["req_over_normal_median"] for k in groups], "s-", color=C2, alpha=1 if j == 0 else 0.5, label=f"C-request / normal {v}")
    ax[1].axhline(1, color=INK, lw=1, ls="--"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels([k.replace("Yellow_", "") for k in groups])
    ax[1].set_title("Probe A: K44 Yellow nonlocal anisotropy (Cycles exit max/min <= 1.008)", color=INK); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(ev / "01_probeA_energy_anisotropy.png"); plt.close(fig)


def p_b(R, ev):
    b = json.loads((R / "probe_b/probe_b.json").read_text())
    names = ["S5", "S30", "Q", "Q_G64", "S30_G64", "R64", "R256"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(len(names))
    for i, (c, col) in enumerate(zip("RGB", (C1, C2, C3))):
        ax[0].bar(x + (i - 1) * 0.26, [min(b["channels"][c]["field"][n]["shadow"], 1.2) for n in names], 0.24, color=col, label=c)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel("shadow-receiver field error (rel L2, capped 1.2)")
    ax[0].set_title("Probe B: capacity bounds on the Burley oracle", color=INK); ax[0].legend(frameon=False)
    bins = ["0.00-0.01", "0.01-0.03", "0.03-0.10", "0.10-0.30"]
    for n, col in (("S30", C1), ("Q", C2), ("Q_G64", C3), ("S30_G64", C4)):
        ax[1].plot(range(4), [b["channels"]["R"]["field"][n].get(f"shadow_dist_{k}", np.nan) for k in bins], "o-", color=col, label=n)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([k + " m" for k in bins]); ax[1].set_title("Probe B: red, by distance to nearest lit port", color=INK)
    ax[1].legend(frameon=False); fig.tight_layout(); fig.savefig(ev / "02_probeB_capacity.png"); plt.close(fig)


def p_c(R, ev):
    c = json.loads((R / "probe_c/probe_c.json").read_text()); k = json.loads((R / "kf/kf48.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8)); imgs = [f"{s}_v{v}" for s in ("T0", "T1") for v in range(3)]; x = np.arange(6)
    for i, (cn, col) in enumerate((("C0", C1), ("C1", C2), ("C2", C3), ("C2b", C4))):
        ax[0].bar(x + (i - 1.5) * 0.2, [c["views"][m][cn]["plane_lum_ratio"] for m in imgs], 0.19, color=col, label=cn)
    ax[0].axhline(1, color=INK, lw=1, ls="--"); ax[0].set_xticks(x); ax[0].set_xticklabels(imgs, rotation=30); ax[0].set_title("Probe C: plane luminance / reference", color=INK); ax[0].legend(frameon=False, fontsize=7)
    for i, (cn, col) in enumerate((("C0", C1), ("C1", C2), ("C2", C3), ("C2b", C4))):
        ax[1].bar(np.arange(3) + (i - 1.5) * 0.2, [c["edit"][f"v{v}"][cn]["corr"] for v in range(3)], 0.19, color=col, label=cn)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["view 0", "view 1", "view 2"]); ax[1].set_title("Probe C: plane edit-change correlation", color=INK)
    arms = ["K1", "K2", "K3", "K4"]
    for kap, col in (("kappa0", C1), ("kappa2", C2), ("kappa8", C3)):
        ax[2].plot(range(4), [k["per_kappa"][kap][a]["clean"]["median_abs_rel"] for a in arms], "o-", color=col, label=kap)
    ax[2].axhline(k["per_kappa"]["kappa8"]["oracle_replicate"]["clean"]["median_abs_rel"], color=INK, lw=1, ls="--", label="oracle replicate (kappa 8)")
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["K1 port/port", "K2 hit field", "K3 pixel 128", "K4 pixel 2048"]); ax[2].set_title("KF: median |rel| (clean px)", color=INK); ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(ev / "03_probeC_kf.png"); plt.close(fig)


def main():
    R, ev = Path(sys.argv[1]), Path(sys.argv[2]); ev.mkdir(parents=True, exist_ok=True)
    p_a(R, ev); p_b(R, ev); p_c(R, ev)
    for src in ("probe_a/probe_a.json", "probe_b/probe_b.json", "probe_c/probe_c.json", "kf/kf48.json", "posthoc/posthoc.json", "tests/test_diag48_pre.json", "runtime.json"):
        if (R / src).exists():
            shutil.copy(R / src, ev / ("d48__" + src.replace("/", "__")))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(ev.iterdir()) if p.name != "manifest.json"}
    (ev / "manifest.json").write_text(json.dumps({"source": str(R), "note": "numeric panels only; no image difference maps in worklog 48", "files": files}, indent=1))


main()
