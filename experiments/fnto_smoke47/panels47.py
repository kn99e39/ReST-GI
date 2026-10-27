"""WL47 reviewer panels (matplotlib, static PNG) into results/evaluation/47 plus a manifest.

One absolute scale for every difference map of this worklog (the MIR |A - B| images): M = largest value among them,
sqrt mapping, real-unit colour bar; no per-image normalisation.

    python panels47.py <results/fnto_smoke47> <results/evaluation/47>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke47_core as K  # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"      # categorical slots 1-4 (fixed order)
INK, MUTED = "#262625", "#73726c"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def p1_oracles(root, ev):
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for r, name in enumerate(("S0", "S1")):
        d = np.load(root / "stage_a" / f"stage_a_{name}.npz")
        im = ax[r, 0].imshow(d["T"], cmap="Blues", interpolation="nearest"); fig.colorbar(im, ax=ax[r, 0], fraction=0.046)
        ax[r, 0].set_title(f"{name} operator T (out x in, 4 ports x 12 bins)", color=INK)
        for k in range(1, K.P):
            for f in (ax[r, 0].axhline, ax[r, 0].axvline):
                f(k * K.J - 0.5, color=MUTED, lw=0.5)
        i = 0; LA, LB = d["LA"][i].reshape(K.P, K.J), d["LB"][i].reshape(K.P, K.J)
        x = np.arange(K.P * K.J)
        ax[r, 1].bar(x - 0.2, (LA * K.OMP).reshape(-1), 0.4, color=C1, label="field A (E share per bin)")
        ax[r, 1].bar(x + 0.2, (LB * K.OMP).reshape(-1), 0.4, color=C2, label="field B (same E per port)")
        ax[r, 1].set_title("input: irradiance share per (port, bin)", color=INK); ax[r, 1].legend(frameon=False)
        ax[r, 2].bar(x - 0.2, d["YA"][i], 0.4, color=C1, label="output under A"); ax[r, 2].bar(x + 0.2, d["YB"][i], 0.4, color=C2, label="output under B")
        ax[r, 2].set_title(f"{name} outgoing radiance per (port, bin), pair 0", color=INK); ax[r, 2].legend(frameon=False)
        for a in ax[r, 1:]:
            a.set_xticks(np.arange(K.P) * K.J + K.J / 2 - 0.5); a.set_xticklabels([f"port {p}" for p in range(K.P)])
    fig.tight_layout(); fig.savefig(ev / "01_stageA_oracles_pair.png"); plt.close(fig)


def p2_mir(root, ev):
    F = np.load(root / "mir" / "mir_fields.npz")
    pairs = {"R1": (4, 6), "R2": (3, 4)}                             # camera indices shown (specular directions of A / B)
    diffs = {pn: [np.abs(F[f"{pn}_A_rep0_imgs"][k].astype(float).mean(-1) - F[f"{pn}_B_rep0_imgs"][k].astype(float).mean(-1)) for k in ks] for pn, ks in pairs.items()}
    M = max(float(d.max()) for v in diffs.values() for d in v)
    fig, ax = plt.subplots(4, 4, figsize=(11, 10.5))
    for r, (pn, ks) in enumerate(pairs.items()):
        for c, side in enumerate("AB"):
            v = F[f"{pn}_{side}_rep0_vals"]
            im = ax[2 * r, c].imshow(v, cmap="Blues", aspect="auto"); fig.colorbar(im, ax=ax[2 * r, c], fraction=0.046)
            ax[2 * r, c].set_title(f"{pn} field {side}: port-mean radiance", color=INK)
            ax[2 * r, c].set_xticks(range(4)); ax[2 * r, c].set_xticklabels([f"P{q}" for q in range(4)])
            ax[2 * r, c].set_yticks(range(8)); ax[2 * r, c].set_yticklabels([f"t {t:+.3g}" for t in K.MIR_T]); ax[2 * r, c].set_ylabel("output direction")
        for c, k in enumerate(ks):
            for s, side in enumerate("AB"):
                if c == 0:
                    a = ax[2 * r, 2 + s]; img = F[f"{pn}_{side}_rep0_imgs"][k].astype(float).mean(-1)
                    im = a.imshow(img[::-1], cmap="Blues"); fig.colorbar(im, ax=a, fraction=0.046)
                    a.set_title(f"{pn} {side}, camera t {K.MIR_T[k]:+.3g}", color=INK); a.axis("off")
            a = ax[2 * r + 1, 2 + c]; im = a.imshow(np.sqrt(diffs[pn][c][::-1] / M), cmap="Blues", vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=a, fraction=0.046); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", f"{0.25 * M:.3g}", f"{M:.3g}"])
            a.set_title(f"{pn} |A-B| (sqrt, shared M), t {K.MIR_T[k]:+.3g}", color=INK); a.axis("off")
        a = ax[2 * r + 1, 0]; x = np.arange(4)
        a.bar(x - 0.2, F[f"{pn}_A_rep0_vals"].sum(0), 0.4, color=C1, label="A"); a.bar(x + 0.2, F[f"{pn}_B_rep0_vals"].sum(0), 0.4, color=C2, label="B")
        a.set_xticks(x); a.set_xticklabels([f"P{q}" for q in range(4)]); a.set_title(f"{pn}: spatial (sum over directions)", color=INK); a.legend(frameon=False)
        a = ax[2 * r + 1, 1]; x = np.arange(8)
        a.bar(x - 0.2, F[f"{pn}_A_rep0_vals"].sum(1), 0.4, color=C1, label="A"); a.bar(x + 0.2, F[f"{pn}_B_rep0_vals"].sum(1), 0.4, color=C2, label="B")
        a.set_xticks(x); a.set_xticklabels([f"{t:+.3g}" for t in K.MIR_T], rotation=45); a.set_title(f"{pn}: directional (sum over ports)", color=INK); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ev / "02_MIR_pairs.png"); plt.close(fig)
    return M


def p3_sss(root, ev, sb):
    d = np.load(root / "sss" / "sss.npz"); cams = list(K.PROTO["stage_b_sss"]["cameras"])
    fig, ax = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for r, cam in enumerate(cams):
        x = d[f"x_{cam}"].mean(0) * 100
        for c, ch in enumerate("RGB"):
            a = ax[r, c]
            for side, col in (("A", C1), ("B", C2)):
                p0, p1 = d[f"{side}_rep0_{cam}"].mean(0)[:, c], d[f"{side}_rep1_{cam}"].mean(0)[:, c]
                a.semilogy(x, np.maximum(p0, 1e-6), color=col, lw=2, label=f"{side} ({'0' if side == 'A' else '60'} deg, equal E)")
                a.semilogy(x, np.maximum(p1, 1e-6), color=col, lw=0.8, ls=":", label=f"{side} replicate")
            for lo, hi in K.PROTO["stage_b_sss"]["bands_m"]:
                a.axvspan(lo * 100, hi * 100, color=MUTED, alpha=0.08, lw=0)
            a.axvline(0, color=MUTED, lw=0.6)
            a.set_title(f"{cam}, {ch}", color=INK)
            if r == 2:
                a.set_xlabel("x (cm); shadow for x > 0")
            if c == 0:
                a.set_ylabel("radiance")
            if r == 0 and c == 0:
                a.legend(frameon=False, fontsize=7)
    fig.suptitle("Lego Yellow (Burley SSS 0.98 + glossy 0.02): equal-irradiance fields at 0 deg / 60 deg; shaded = nonlocal bands", color=INK)
    fig.tight_layout(); fig.savefig(ev / "03_SSS_profiles.png"); plt.close(fig)


def p4_stagec(root, ev):
    sc = json.loads((root / "stage_c" / "stage_c.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for i, reg in enumerate(("S0", "S1", "MIR")):
        a = ax[i]; r = sc[reg]; x = np.arange(4)
        for j, (key, col) in enumerate((("heldout_rel", C1), ("hidden_rel", C2), ("directional_rel", C3))):
            if reg == "S0" and key == "directional_rel":
                for xx in x:
                    a.text(xx + 0.26, 0.03, "n/a", ha="center", color=MUTED, fontsize=7)
                continue
            v = [min(r[f"{c}_seed4701"][key], 1.6) for c in ("A0", "A1", "B0", "B1")]
            v2 = [min(r[f"{c}_seed4702"][key], 1.6) for c in ("A0", "A1", "B0", "B1")]
            a.bar(x + (j - 1) * 0.26, v, 0.24, color=col, label={"heldout_rel": "held-out (all outputs)", "hidden_rel": "hidden outputs", "directional_rel": "directional part"}[key])
            a.plot(x + (j - 1) * 0.26, v2, "o", ms=4, mfc="white", mec=INK, label="seed 4702" if j == 0 else None)
        a.axhline(r["floors"]["eonly_linear_floor"], color=INK, lw=1, ls="--", label="E-only linear floor")
        a.set_xticks(x); a.set_xticklabels(["A0", "A1", "B0", "B1"]); a.set_title(reg, color=INK)
        if i == 0:
            a.set_ylabel("relative L2 error (capped 1.6)"); a.legend(frameon=False, fontsize=7)
    fig.text(0.5, -0.02, "S0 directional part: true value 0 (isotropic exit), ratio undefined (n/a)", ha="center", color=MUTED)
    fig.tight_layout(); fig.savefig(ev / "04_stageC_errors.png", bbox_inches="tight"); plt.close(fig)


def main():
    root, ev = Path(sys.argv[1]), Path(sys.argv[2]); ev.mkdir(parents=True, exist_ok=True)
    sb = json.loads((root / "stage_b.json").read_text())
    p1_oracles(root, ev); M = p2_mir(root, ev); p3_sss(root, ev, sb); p4_stagec(root, ev)
    for src in ("stage_a/stage_a.json", "stage_b.json", "stage_c/stage_c.json", "diag_posthoc.json", "tests/test_smoke47_server_amend1.json", "runtime.json"):
        if (root / src).exists():
            shutil.copy(root / src, ev / ("d47__" + src.replace("/", "__")))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(ev.iterdir()) if p.name != "manifest.json"}
    (ev / "manifest.json").write_text(json.dumps({"diff_scale_M_radiance": M, "mapping": "sqrt(|A - B| / M)",
                                                  "rule": "one absolute scale for every difference map of worklog 47; no per-image normalisation",
                                                  "source": str(root), "files": files}, indent=1))


main()
