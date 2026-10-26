"""WL45 reviewer exports -> results/evaluation/45/ (numpy + matplotlib; nothing is re-rendered, re-solved or re-trained).

Error / difference maps follow the user's rule (2026-10-25): pass 1 builds every map of the worklog's export set and
takes M = the largest |error| / |difference| among ALL of them (display units, ldr3 space); pass 2 renders every error
map on [0, M] and every signed difference on [-M, M] with sqrt mapping (|x| / M) ** 0.5 and real-unit colorbars.

Panels:
  01 Track A, test views 10 / 15 (first_r1): reference | K_k44P K only | |K only - first| on Yellow pixels only
  02 Track B3a, views 6 / 15: Yellow pixels under the emit45 field (non-Yellow emitters): Blender (real Yellow material)
     | K_k44P one K/C step | signed D(pred) - D(ref); and the same for the 'emitall' and localized 'bump' oracles
  03 Track B4, views 6 / 15 (held-out T1, matched Lego pixels): reference indirect change | updated prediction | signed
     prediction - reference  (indirect change = D(full) - D(first) at T1 minus the same at T0 for the reference,
     D(E_upd) - D(K only) at T1 minus D(S_T0) - D(K only) at T0 for the prediction)
  04 Track B2 scatter (not a map): C-gathered irradiance vs Blender oracle, per material
JSON results of tracks A / B are copied with their sha256 in the manifest.

    python d45_exports.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

EV = C.ROOT / "results/evaluation/45"; R1 = C.OUT1; O45 = C.ROOT / "results/fnto_diag45"; D = C.ldr3
MATS = C.MATERIALS[:8] + ["box"]


def full_img(idx, vals, res):
    img = np.full((res * res,) + vals.shape[1:], np.nan, np.float32); img[idx] = vals
    return img.reshape((res, res) + vals.shape[1:])


def main():
    EV.mkdir(parents=True, exist_ok=True); figs = []
    px = np.load(R1 / "eval_px/heldout_T0.npz"); res = int(px["res"]); first = np.load(R1 / "refs/heldout_T0_first.npz")
    V = np.load(R1 / "eval_K_k44P/views.npz")
    for v in (10, 15):                                                              # 01 Track A
        val = px[f"v{v}_valid"].astype(bool); idx = px[f"v{v}_mk"][val]; y = (px[f"v{v}_dom"][val] == 0) & (px[f"v{v}_mat"][val] == 0)
        fr = first[f"v{v}_rgb"][val].astype(np.float64); kk = V[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)
        e = np.abs(D(kk) - D(fr)).mean(1); e[~y] = np.nan
        figs.append((f"01_trackA_yellow_K_only_error_view{v}.png", [[("rgb", full_img(idx, D(fr), res), "first-event reference (first_r1)"),
                     ("rgb", full_img(idx, D(kk), res), "K_k44P K only"), ("err", full_img(idx, e, res), "|K only - first|, Yellow pixels only")]]))
    for tag, name in (("", "emit"), ("_emitall", "emitall"), ("_bump", "bump")):                       # 02 Track B3a
        f = O45 / f"track_b/b2_b3a_arrays{tag}.npz"
        if not f.exists():
            continue
        Z = np.load(f); rows = []
        for v in (6, 15):
            m = Z["b3_view"] == v
            if not m.any():
                continue
            val = px[f"v{v}_valid"].astype(bool); yel = np.flatnonzero(val & (px[f"v{v}_dom"] == 0) & (px[f"v{v}_mat"] == 0)); idx = px[f"v{v}_mk"][yel]
            ref, pr = Z["b3_ref"][m].astype(np.float64), Z["b3_pred"][m].astype(np.float64)
            rows.append([("rgb", full_img(idx, D(ref), res), f"Blender '{name}' oracle, Yellow pixels, view {v}"), ("rgb", full_img(idx, D(pr), res), "K_k44P one K/C step"),
                         ("diff", full_img(idx, (D(pr) - D(ref)).mean(1), res), "D(pred) - D(oracle)")])
        figs.append((f"02_trackB3a_yellow_response_{name}.png", rows))
    p1 = np.load(R1 / "eval_px/heldout_T1.npz"); Rr = {(s, m): np.load(R1 / f"refs/heldout_{s}_{m}.npz") for s in ("T0", "T1") for m in ("first", "full")}
    for v in (6, 15):                                                               # 03 Track B4
        va0, va1 = px[f"v{v}_valid"].astype(bool), p1[f"v{v}_valid"].astype(bool)
        l0, l1 = (px[f"v{v}_dom"] == 0) & va0, (p1[f"v{v}_dom"] == 0) & va1
        com, i0, i1 = np.intersect1d(px[f"v{v}_mk"][l0], p1[f"v{v}_mk"][l1], return_indices=True)
        r0 = np.cumsum(va0)[l0][i0] - 1; r1 = np.cumsum(va1)[l1][i1] - 1
        g = lambda s, arm, r: D(V[f"heldout_{s}_v{v}_{arm}"][r, :3].astype(np.float64)).mean(1)  # noqa: E731
        rr = lambda s, m, l, i: D(Rr[(s, m)][f"v{v}_rgb"][l][i].astype(np.float64)).mean(1)  # noqa: E731
        ref = (rr("T1", "full", l1, i1) - rr("T1", "first", l1, i1)) - (rr("T0", "full", l0, i0) - rr("T0", "first", l0, i0))
        pred = (g("T1", "E_upd", r1) - g("T1", "K_only", r1)) - (g("T0", "S_T0", r0) - g("T0", "K_only", r0))
        figs.append((f"03_trackB4_indirect_change_view{v}.png", [[("diff", full_img(com, ref, res), "reference indirect change T0 -> T1"),
                     ("diff", full_img(com, pred, res), "K_k44P updated prediction"), ("diff", full_img(com, pred - ref, res), "prediction - reference")]]))
    M = 0.0                                                                         # pass 1
    for _, rows in figs:
        for row in rows:
            for kind, img, _ in row:
                if kind in ("err", "diff"):
                    M = max(M, float(np.nanmax(np.abs(img))))
    gamma = 0.5; ticks = np.array([0, 0.01, 0.03, 0.1, 0.3, 1.0]) * M
    manifest = {"display": "ldr3 = (x/(1+x))^(1/2.2)", "error_diff_scale_M": M, "mapping": f"sign(x) * (|x| / M) ** {gamma}",
                "rule": "one absolute scale for every error / difference map of worklog 45 (largest value among them); no per-image normalisation", "files": {}}
    for fn, rows in figs:                                                           # pass 2
        ncol = max(len(r) for r in rows); fig, axs = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 4.4 * len(rows)), squeeze=False)
        for ri, row in enumerate(rows):
            for ci in range(ncol):
                ax = axs[ri, ci]; ax.axis("off")
                if ci >= len(row):
                    continue
                kind, img, title = row[ci]; ax.set_title(title, fontsize=7)
                if kind == "rgb":
                    ax.imshow(np.nan_to_num(np.clip(img, 0, 1)), interpolation="nearest")
                elif kind == "err":
                    im = ax.imshow((np.nan_to_num(img) / M) ** gamma, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_ticks((ticks / M) ** gamma); cb.set_ticklabels([f"{t:.3g}" for t in ticks]); cb.ax.tick_params(labelsize=6)
                else:
                    x = np.nan_to_num(img); im = ax.imshow(np.sign(x) * (np.abs(x) / M) ** gamma, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                    tt = np.concatenate([-ticks[::-1][:-1], ticks]); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.set_ticks(np.sign(tt) * (np.abs(tt) / M) ** gamma); cb.set_ticklabels([f"{t:.3g}" for t in tt]); cb.ax.tick_params(labelsize=5)
        fig.suptitle(f"{fn}  (error / difference scale M = {M:.3f} display units, sqrt mapping, shared by every map of worklog 45)", fontsize=8)
        fig.tight_layout(); fig.savefig(EV / fn, dpi=110); plt.close(fig)
    f = O45 / "track_b/b2_arrays.npz"                                               # 04 Track B2 scatter
    if f.exists():
        Z = np.load(f); fig, axs = plt.subplots(2, 4, figsize=(14, 7)); lim = float(np.quantile(Z["b2_E_in"], 0.999)) * 1.1
        for k, (ax, i) in enumerate(zip(axs.ravel(), [1, 3, 5, 6, 7, 8, 2, 4])):
            m = Z["b2_mat"] == i
            if m.sum() < 50:
                ax.axis("off"); continue
            ax.scatter(Z["b2_E_in"][m], Z["b2_E_C"][m], s=0.3, alpha=0.3, rasterized=True); ax.plot([0, lim], [0, lim], "k--", lw=0.6)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(f"{MATS[i]} (n={int(m.sum())})", fontsize=8); ax.set_xlabel("Blender E_in (emit45)", fontsize=7); ax.set_ylabel("C gather E_C", fontsize=7)
        fig.suptitle("B2: frozen C gathering the known field f vs Blender single-flight oracle (held-out T0 views 6, 15, 2, 10)", fontsize=9)
        fig.tight_layout(); fig.savefig(EV / "04_trackB2_C_vs_oracle_scatter.png", dpi=110); plt.close(fig)
    for src in sorted(O45.rglob("*.json")):
        (EV / f"d45_{src.parent.name}_{src.name}").write_bytes(src.read_bytes())
    for f in sorted(EV.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    (EV / "manifest.json").write_text(json.dumps(manifest, indent=1)); print("M =", M, "files", len(manifest["files"]))


if __name__ == "__main__":
    main()
