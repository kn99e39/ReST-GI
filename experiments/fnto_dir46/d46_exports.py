"""WL46 reviewer exports -> results/evaluation/46/ (numpy + matplotlib; nothing re-rendered or re-solved).

Error / difference maps: one absolute scale for the whole worklog, M = the largest |error| / |difference| among all
maps below (display units, ldr3), sqrt mapping (|x| / M) ** 0.5, real-unit colorbars (user rule 2026-10-25).
  01_MS_view{v}_{state}.png  glossy-plane scene: reference | R0m | R3mr | R6a | R6br and their |error| maps
  02_MS_edit_view{v}.png     plane-pixel change T0 -> T1: reference | R0m | R6br | R6br - reference
  03_KF_scatter.png          known directional field: C gather vs Blender, R1 (normal) vs R2 (directional), kappa 0/2/8
  04_LG_view6.png            Lego held-out T0 view 6: full_r1 | R0 | R4 and |error| maps, R4 - R0
JSON results are copied with sha256 in the manifest.

    python d46_exports.py
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

O = C.ROOT / "results/fnto_diag46"; EV = C.ROOT / "results/evaluation/46"; D = C.ldr3; RES = 512
COND = [("R0m", "R0m production (iso sender, cos gathers)"), ("R3mr", "R3mr directional reads"), ("R6a", "R6a iso + GGX IS"), ("R6br", "R6br directional reads + GGX IS")]


def img(mk, vals, res=RES):
    a = np.full((res * res,) + vals.shape[1:], np.nan, np.float32); a[mk] = vals; return a.reshape((res, res) + vals.shape[1:])


def main():
    EV.mkdir(parents=True, exist_ok=True); figs = []
    P = np.load(O / "ms/ms46_pred.npz")
    for v in range(3):
        for st in ("T0", "T1"):
            G = np.load(O / f"ms/geom/geom_{st}.npz"); mk = G[f"v{v}_mk"]; ref = np.load(O / f"ms/refs/ref_{st}_v{v}.npz")["rgb"][mk].astype(np.float64)
            r1 = [("rgb", img(mk, D(ref)), f"reference (Cycles 4096 spp) {st} view {v}")]; r2 = [("blank", None, "")]
            for c, lab in COND:
                k = f"{c}_{st}_v{v}"
                if k not in P.files:
                    continue
                p = P[k].astype(np.float64); r1.append(("rgb", img(mk, D(p)), lab)); r2.append(("err", img(mk, np.abs(D(p) - D(ref)).mean(1)), f"|{c} - ref|"))
            figs.append((f"01_MS_view{v}_{st}.png", [r1, r2]))
        G0 = np.load(O / "ms/geom/geom_T0.npz"); G1 = np.load(O / "ms/geom/geom_T1.npz"); m0, m1 = G0[f"v{v}_mk"], G1[f"v{v}_mk"]
        com, i0, i1 = np.intersect1d(m0, m1, return_indices=True)
        pl = (G0[f"v{v}_dom"][i0] == 2) & (G1[f"v{v}_dom"][i1] == 2); com, i0, i1 = com[pl], i0[pl], i1[pl]
        rf = lambda st, mk_, ii: D(np.load(O / f"ms/refs/ref_{st}_v{v}.npz")["rgb"][mk_][ii].astype(np.float64)).mean(1)  # noqa: E731
        dref = rf("T1", m1, i1) - rf("T0", m0, i0); row = [("diff", img(com, dref), "reference change T0 -> T1 (plane pixels)")]
        for c in ("R0m", "R6br"):
            if f"{c}_T1_v{v}" in P.files:
                dp = D(P[f"{c}_T1_v{v}"][i1].astype(np.float64)).mean(1) - D(P[f"{c}_T0_v{v}"][i0].astype(np.float64)).mean(1)
                row.append(("diff", img(com, dp), f"{c} change"))
                if c == "R6br":
                    row.append(("diff", img(com, dp - dref), "R6br change - reference change"))
        figs.append((f"02_MS_edit_view{v}.png", [row]))
    V0 = np.load(O / "lg/R0/views.npz"); V4 = np.load(O / "lg/R4/views.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); v = 6
    val = px[f"v{v}_valid"].astype(bool); mk = px[f"v{v}_mk"][val]; res = int(px["res"]); fu = np.load(C.OUT1 / "refs/heldout_T0_full.npz")[f"v{v}_rgb"][val].astype(np.float64)
    a = V0[f"heldout_T0_v{v}_S_T0"][:, :3].astype(np.float64); b = V4[f"heldout_T0_v{v}_S_T0"][:, :3].astype(np.float64)
    figs.append(("04_LG_view6.png", [[("rgb", img(mk, D(fu), res), "full_r1 reference"), ("rgb", img(mk, D(a), res), "R0 (production)"), ("rgb", img(mk, D(b), res), "R4 (incoming-aware K_loc sender)")],
                                     [("diff", img(mk, (D(b) - D(a)).mean(1), res), "R4 - R0"), ("err", img(mk, np.abs(D(a) - D(fu)).mean(1), res), "|R0 - ref|"), ("err", img(mk, np.abs(D(b) - D(fu)).mean(1), res), "|R4 - ref|")]]))
    M = max(float(np.nanmax(np.abs(im))) for _, rows in figs for row in rows for kind, im, _ in row if kind in ("err", "diff")); g = 0.5
    ticks = np.array([0, 0.01, 0.03, 0.1, 0.3, 1.0]) * M
    for fn, rows in figs:
        ncol = max(len(r) for r in rows); fig, axs = plt.subplots(len(rows), ncol, figsize=(3.9 * ncol, 4.1 * len(rows)), squeeze=False)
        for ri, row in enumerate(rows):
            for ci in range(ncol):
                ax = axs[ri, ci]; ax.axis("off")
                if ci >= len(row) or row[ci][0] == "blank":
                    continue
                kind, im, title = row[ci]; ax.set_title(title, fontsize=7)
                if kind == "rgb":
                    ax.imshow(np.nan_to_num(np.clip(im, 0, 1)), interpolation="nearest")
                elif kind == "err":
                    h = ax.imshow((np.nan_to_num(im) / M) ** g, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                    cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02); cb.set_ticks((ticks / M) ** g); cb.set_ticklabels([f"{t:.3g}" for t in ticks]); cb.ax.tick_params(labelsize=6)
                else:
                    x = np.nan_to_num(im); h = ax.imshow(np.sign(x) * (np.abs(x) / M) ** g, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                    tt = np.concatenate([-ticks[::-1][:-1], ticks]); cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
                    cb.set_ticks(np.sign(tt) * (np.abs(tt) / M) ** g); cb.set_ticklabels([f"{t:.3g}" for t in tt]); cb.ax.tick_params(labelsize=5)
        fig.suptitle(f"{fn}  (error / difference scale M = {M:.3f} display units, sqrt mapping, shared by every map of worklog 46)", fontsize=8)
        fig.tight_layout(); fig.savefig(EV / fn, dpi=110); plt.close(fig)
    K = np.load(O / "kf/kf46_arrays.npz"); fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    for j, kap in enumerate((0, 2, 8)):
        m = K["kappa"] == kap; e = K["E_in"][m]; lim = float(np.quantile(e, 0.999)) * 1.1
        for i, arm in enumerate(("R1", "R2")):
            ax = axs[i, j]; ax.scatter(e, K[arm][m], s=0.2, alpha=0.25, rasterized=True); ax.plot([0, lim], [0, lim], "k--", lw=0.6)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim * (6 if arm == "R1" and kap == 8 else (2.5 if arm == "R1" and kap == 2 else 1))); ax.set_xlabel("Blender E_in (emit46)", fontsize=7); ax.set_ylabel(f"C gather, {arm}", fontsize=7)
            ax.set_title(f"kappa {kap}: {'R1 sender read at the normal' if arm == 'R1' else 'R2 sender read toward the receiver'}", fontsize=8)
    fig.suptitle("KF: frozen C relation table carrying a known directional field (held-out T0 views 6, 15; 40,000 non-Yellow pixels)", fontsize=9)
    fig.tight_layout(); fig.savefig(EV / "03_KF_scatter.png", dpi=110); plt.close(fig)
    man = {"display": "ldr3 = (x/(1+x))^(1/2.2)", "error_diff_scale_M": M, "mapping": f"sign(x) * (|x| / M) ** {g}",
           "rule": "one absolute scale for every error / difference map of worklog 46 (largest value among them); no per-image normalisation", "files": {}}
    for src in sorted(O.rglob("*.json")):
        rel = src.relative_to(O).as_posix().replace("/", "__")
        if "smoke" in rel:
            continue
        (EV / f"d46_{rel}").write_bytes(src.read_bytes())
    for f in sorted(EV.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            man["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    (EV / "manifest.json").write_text(json.dumps(man, indent=1)); print("M =", M, "files", len(man["files"]))


if __name__ == "__main__":
    main()
