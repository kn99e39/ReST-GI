"""WL44 reviewer exports -> results/evaluation/44/ (PC; numpy + matplotlib; nothing is re-rendered or re-solved).

Error / difference maps follow the user's rule (2026-10-25): no per-image normalisation. Pass 1 builds every map of the
worklog's export set; M = the largest |error| / |difference| value among ALL of them (display units, ldr3 space);
pass 2 renders every error map on [0, M] and every signed difference map on [-M, M] with a square-root mapping
(|x| / M) ** 0.5 so that small values stay visible, each with a colorbar in real units. M and the mapping are written
to the manifest.

Panels (held-out views 6 and 15, regime r1):
  01 K-only vs first_r1, T0: reference | K_r1P | K_k44P | K_r1P_s1, and their |error| maps
  02 indirect component (T0): ref (full_r1 - first_r1) | pred (S_T0 - K_only) for K_r1P and K_k44P, and signed
     differences pred - ref
  03 full scene S_T0 vs full_r1 (T0): K_r1P | K_k44P and their |error| maps
  04 visibility correction on held-out probe 12 (multisun): 4096-spp clean render | pixels whose r1 bits changed under
     the 0.3 mm rule | K_r1P probe |error| vs clean | K_k44P probe |error| vs clean  (if the probe predictions exist)

    python k44_exports.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import k44_common as K  # noqa: E402

EV = K.C.ROOT / "results/evaluation/44"
R1 = K.C.OUT1
TAGS = [("K_r1P", "K_r1P (r1 baseline)"), ("K_k44P", "K_k44P (corrected incident field)"), ("K_r1P_s1", "K_r1P seed 1 (control)")]
D = K.C.ldr3


def load_views():
    out = {}
    for t, _ in TAGS:
        p = R1 / f"eval_{t}/views.npz"
        if p.exists():
            out[t] = np.load(p)
    return out


def full_img(idx, vals, res):
    img = np.full((res * res,) + vals.shape[1:], np.nan, np.float32); img[idx] = vals
    return img.reshape((res, res) + vals.shape[1:])


def main():
    EV.mkdir(parents=True, exist_ok=True)
    V = load_views(); s = "heldout_T0"; px = np.load(R1 / f"eval_px/{s}.npz"); res = int(px["res"])
    first = np.load(R1 / f"refs/{s}_first.npz"); fullr = np.load(R1 / f"refs/{s}_full.npz")
    figs = []                                    # (filename, list of rows: list of (kind, image, title))
    for v in (6, 15):
        val = px[f"v{v}_valid"].astype(bool); idx = px[f"v{v}_mk"][val]
        fr = first[f"v{v}_rgb"][val].astype(np.float64); fu = fullr[f"v{v}_rgb"][val].astype(np.float64)
        row_i, row_e = [("rgb", full_img(idx, D(fr), res), "first-event reference (r1, 1024 spp)")], [("blank", None, "")]
        row_fi, row_fe = [("rgb", full_img(idx, D(fu), res), "full-GI reference (r1, 4096 spp)")], [("blank", None, "")]
        row_ind = [("diff", full_img(idx, (D(fu) - D(fr)).mean(1), res), "reference indirect: D(full) - D(first)")]; row_indd = [("blank", None, "")]
        for t, lab in TAGS:
            if t not in V:
                continue
            kk = V[t][f"{s}_v{v}_K_only"].astype(np.float64)[:, :3]; st = V[t][f"{s}_v{v}_S_T0"].astype(np.float64)[:, :3]
            row_i.append(("rgb", full_img(idx, D(kk), res), f"K only: {lab}")); row_e.append(("err", full_img(idx, np.abs(D(kk) - D(fr)).mean(1), res), f"|K only - first|: {t}"))
            row_fi.append(("rgb", full_img(idx, D(st), res), f"S_T0: {lab}")); row_fe.append(("err", full_img(idx, np.abs(D(st) - D(fu)).mean(1), res), f"|S_T0 - full|: {t}"))
            pind = (D(st) - D(kk)).mean(1); rind = (D(fu) - D(fr)).mean(1)
            row_ind.append(("diff", full_img(idx, pind, res), f"predicted indirect: D(S_T0) - D(K only), {t}"))
            row_indd.append(("diff", full_img(idx, pind - rind, res), f"predicted - reference indirect, {t}"))
        figs += [(f"01_K_only_vs_first_T0_view{v}.png", [row_i, row_e]), (f"02_indirect_component_T0_view{v}.png", [row_ind, row_indd]),
                 (f"03_full_scene_T0_view{v}.png", [row_fi, row_fe])]
    # 04 visibility correction on probe 12
    pr = R1 / "stagea/prep_holdout"; p2 = R1 / "stagea/prep2_holdout"; ck = K.C.ROOT / "results/fnto_lego_k44/clean/clean_00012.npz"
    if p2.exists() and ck.exists():
        i = 12; o = np.load(R1 / f"stagea/holdout/img_{i:05d}.npz"); c = np.load(ck)
        st0, st1 = np.load(pr / "start.npy"), np.load(p2 / "start.npy")
        vb0 = np.load(pr / "px_vbits.npy", mmap_mode="r")[st0[i]:st0[i + 1]]; vb1 = np.load(p2 / "px_vbits.npy", mmap_mode="r")[st1[i]:st1[i + 1]]
        col = np.load(pr / "px_col.npy", mmap_mode="r")[st0[i]:st0[i + 1]]
        keep = np.zeros(len(o["col"]), bool); ptr = 0
        for k in range(len(o["col"])):
            if ptr < len(col) and np.array_equal(o["col"][k], col[ptr]):
                keep[k] = True; ptr += 1
        pix = o["pix"][keep]; r = 512
        chg = (np.asarray(vb0) != np.asarray(vb1)).astype(np.float32)
        rowv = [("rgb", full_img(o["pix"], D(c["col"].astype(np.float64)), r), "probe 12 (multisun), 4096-spp clean render"),
                ("mask", full_img(pix, chg, r), "pixels whose light bits changed (r1 -> 0.3 mm rule)"),
                ("rgb", full_img(pix, D(np.asarray(col, np.float64)) * (1 - chg[:, None]) + chg[:, None] * np.array([1.0, 0.1, 0.1]), r), "changed pixels in red over the 512-spp target")]
        figs.append(("04_visibility_correction_probe12.png", [rowv]))
    # pass 1: global maximum over every error / difference map
    M = 0.0
    for _, rows in figs:
        for row in rows:
            for kind, img, _ in row:
                if kind in ("err", "diff"):
                    M = max(M, float(np.nanmax(np.abs(img))))
    gamma = 0.5
    manifest = {"display": "ldr3 = (x/(1+x))^(1/2.2)", "error_diff_scale_M": M, "mapping": f"sign(x) * (|x| / M) ** {gamma}",
                "rule": "one absolute scale for every error / difference map of worklog 44 (largest value among them); no per-image normalisation",
                "files": {}, "sources": {"evals": [str(R1 / f'eval_{t}') for t, _ in TAGS], "refs": str(R1 / "refs")}}
    ticks = np.array([0, 0.01, 0.03, 0.1, 0.3, 1.0]) * M if M > 0 else np.array([0, 1])
    for fn, rows in figs:
        ncol = max(len(r) for r in rows); fig, axs = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 4.4 * len(rows)), squeeze=False)
        for ri, row in enumerate(rows):
            for ci in range(ncol):
                ax = axs[ri, ci]; ax.axis("off")
                if ci >= len(row) or row[ci][0] == "blank":
                    continue
                kind, img, title = row[ci]; ax.set_title(title, fontsize=7)
                if kind == "rgb":
                    ax.imshow(np.nan_to_num(np.clip(img, 0, 1)), interpolation="nearest")
                elif kind == "mask":
                    ax.imshow(np.nan_to_num(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                elif kind == "err":
                    im = ax.imshow((np.nan_to_num(img) / M) ** gamma, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_ticks((ticks / M) ** gamma); cb.set_ticklabels([f"{t:.3g}" for t in ticks]); cb.ax.tick_params(labelsize=6)
                else:
                    x = np.nan_to_num(img); im = ax.imshow(np.sign(x) * (np.abs(x) / M) ** gamma, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                    tt = np.concatenate([-ticks[::-1][:-1], ticks]); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.set_ticks(np.sign(tt) * (np.abs(tt) / M) ** gamma); cb.set_ticklabels([f"{t:.3g}" for t in tt]); cb.ax.tick_params(labelsize=5)
        fig.suptitle(f"{fn}  (error / difference scale M = {M:.3f} display units, sqrt mapping, shared by every map of worklog 44)", fontsize=8)
        fig.tight_layout(); fig.savefig(EV / fn, dpi=110); plt.close(fig)
    for src in sorted((K.C.ROOT / "results/fnto_lego_k44").glob("*.json")):
        (EV / f"k44_{src.name}").write_bytes(src.read_bytes())
    for f in sorted(EV.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    (EV / "manifest.json").write_text(json.dumps(manifest, indent=1)); print("M =", M, "files", len(manifest["files"]))


if __name__ == "__main__":
    main()
