"""Worklog-39 reviewer exports (Windows venv; matplotlib).

    python spt_exports.py --out <results/evaluation/39>

Display = the evaluation metric's code tonemap (x/(1+x))^(1/2.2) of 4096-spp references and predictions; one error
scale (0-0.08) for every arm. Views 6 and 15 are worklog 37's reference-chosen views (most physically changed pixels).
RNA / Surface-Owned images are worklog 37's stored display predictions; SPT images come from spt_eval (kept views).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import spt_common as C  # noqa: E402

EMAX = 0.08
RES = 1024


def ldr(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def ref_img(set_name, v):
    f1 = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); f2 = h5py.File(C.B4 / f"data/{set_name}_s2_val.h5", "r")
    g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))  # noqa: E731
    return (0.5 * (g(f1, "color") + g(f2, "color"))).reshape(RES, RES, 3), g(f1, "alpha")[:, 0].reshape(RES, RES) > 0, g(f1, "diffuse_direct")[:, 0].reshape(RES, RES)


def window(score, w):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    return np.unravel_index(np.argmax(tot), tot.shape)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); files = []
    S = np.load(C.OUT / "eval/views.npz"); B = np.load(C.B4 / "eval/views.npz")
    # 01 static failure at T0 (where the error is)
    for v in (6, 15):
        ref, m, vis = ref_img("heldout_T0", v); s = S[f"heldout_T0_v{v}_S_T0"].astype(np.float32)
        rna = B[f"heldout_T0_v{v}_RNA_T0"].astype(np.float32)
        fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
        tiles = [("T0 reference (4096 spp)", ldr(ref)), ("original RNA_T0", rna), ("SPT-1 S_T0 (persistent + T0 state)", ldr(s[..., :3])),
                 ("SPT-1 direct term (x visibility)", ldr(s[..., 3:6])), ("reference on shadowed pixels (v = 0)", ldr(ref * (vis[..., None] == 0))),
                 ("SPT-1 indirect term (state x transfer)", ldr(s[..., 6:9]))]
        for i, (t, im) in enumerate(tiles):
            ax.ravel()[i].imshow(np.clip(im, 0, 1)); ax.ravel()[i].set_title(t, fontsize=10)
        for j, (t, im) in enumerate([("|error| RNA_T0", rna), ("|error| SPT-1 S_T0", ldr(s[..., :3]))]):
            e = np.abs(im - ldr(ref)).mean(-1); e[~m] = np.nan
            h = ax.ravel()[6 + j].imshow(e, cmap="magma", vmin=0, vmax=EMAX); ax.ravel()[6 + j].set_title(t, fontsize=10)
        for x in ax.ravel():
            off(x)
        fig.colorbar(h, ax=ax[1, 2:], fraction=0.02, label=f"|display error| 0-{EMAX}")
        fig.suptitle(f"Held-out T0 view {v}: SPT-1 static representability (shadowed yellow bucket = subsurface transport)", fontsize=12)
        f = out / f"01_static_T0_view{v}.png"; fig.savefig(f, dpi=80); plt.close(fig); files.append(f)
    # relation-change mask on held-out T1 pixels
    r0, r1 = np.load(C.OUT / "rel_T0.npz"), np.load(C.OUT / "rel_T1.npz")
    changed = (r0["hit"] != r1["hit"]).any(1) | (r0["vis"] != r1["vis"]).any(1)
    idx = np.load(C.OUT / "pixels/heldout_T1_s1_val.idx.npy"); st = np.load(C.OUT / "pixels/heldout_T1_s1_val.start.npy")
    for v in (6, 15):
        ref, m, vis = ref_img("heldout_T1", v)
        sv = {n: S[f"heldout_T1_v{v}_{n}"].astype(np.float32) for n in ("C_upd", "D_stale", "C_full")}
        ii = idx[st[v]:st[v + 1]]; pm = np.zeros(RES * RES, bool); pm[np.flatnonzero(m.ravel())] = (changed[np.maximum(ii, 0)] & (ii >= 0)).any(1)
        fig, ax = plt.subplots(3, 5, figsize=(22, 13.5))
        top = [("T0 reference (same camera/light)", B[f"heldout_T1_v{v}_refT0"].astype(np.float32)), ("T1 reference", ldr(ref)),
               ("RNA_T1 (trained on T1)", B[f"heldout_T1_v{v}_RNA_T1"].astype(np.float32)), ("Surface-Owned A_full (T1-trained)", B[f"heldout_T1_v{v}_A_full-PU"].astype(np.float32)),
               ("Surface-Owned frozen (T0 state)", B[f"heldout_T1_v{v}_SO_T0-PU"].astype(np.float32))]
        mid = [("SPT-1 D_stale (T0 relations)", ldr(sv["D_stale"][..., :3])), ("SPT-1 C_upd (T1 relations, same params)", ldr(sv["C_upd"][..., :3])),
               ("SPT-1 C_full (retrained on T1)", ldr(sv["C_full"][..., :3]))]
        for i, (t, im) in enumerate(top):
            ax[0, i].imshow(np.clip(im, 0, 1)); ax[0, i].set_title(t, fontsize=10)
        for i, (t, im) in enumerate(mid):
            ax[1, i].imshow(np.clip(im, 0, 1)); ax[1, i].set_title(t, fontsize=10)
        dind = np.abs(sv["C_upd"][..., 6:9] - sv["D_stale"][..., 6:9]).mean(-1); dind[~m] = np.nan
        ax[1, 3].imshow(dind, cmap="viridis", vmin=0, vmax=0.05); ax[1, 3].set_title("|indirect(C_upd) - indirect(D_stale)| (state change)", fontsize=10)
        pmi = np.full((RES, RES, 3), 0.99); pmi[m] = 0.8; pmi[pm.reshape(RES, RES)] = (0.85, 0.2, 0.15)
        ax[1, 4].imshow(pmi); ax[1, 4].set_title("pixels whose anchors' relations changed (red)", fontsize=10)
        for j, (t, im) in enumerate([("|err| SO frozen", top[4][1]), ("|err| SPT D_stale", mid[0][1]), ("|err| SPT C_upd", mid[1][1]),
                                     ("|err| SPT C_full", mid[2][1]), ("|err| Surface-Owned A_full", top[3][1])]):
            e = np.abs(im - ldr(ref)).mean(-1); e[~m] = np.nan
            h = ax[2, j].imshow(e, cmap="magma", vmin=0, vmax=EMAX); ax[2, j].set_title(t, fontsize=10)
        for x in ax.ravel():
            off(x)
        fig.colorbar(h, ax=ax[2, :], fraction=0.012, label=f"|display error| 0-{EMAX}")
        fig.suptitle(f"Held-out T1 view {v} (box moved +0.44 m; SPT-1 learned parameters identical in D_stale and C_upd)", fontsize=12)
        f = out / f"02_edit_T1_view{v}.png"; fig.savefig(f, dpi=72); plt.close(fig); files.append(f)
    # 03 crops: strongest physical change (stationary plate) and the shadowed yellow bucket, view 6
    v = 6; ref, m, vis = ref_img("heldout_T1", v); W = 192
    gd = np.abs(B[f"heldout_T1_v{v}_gtdelta"].astype(np.float32)).mean(-1); gd[~m] = 0
    yx_c = window(gd, W)
    yel = (vis == 0) & m & (ldr(ref).mean(-1) > 0.45); yx_y = window(yel.astype(float), W)
    sv = {n: S[f"heldout_T1_v{v}_{n}"].astype(np.float32) for n in ("C_upd", "D_stale", "C_full")}
    cols = [("T0 ref", B[f"heldout_T1_v{v}_refT0"].astype(np.float32)), ("T1 ref", ldr(ref)), ("RNA_T1", B[f"heldout_T1_v{v}_RNA_T1"].astype(np.float32)),
            ("SO A_full", B[f"heldout_T1_v{v}_A_full-PU"].astype(np.float32)), ("SO frozen", B[f"heldout_T1_v{v}_SO_T0-PU"].astype(np.float32)),
            ("SPT D_stale", ldr(sv["D_stale"][..., :3])), ("SPT C_upd", ldr(sv["C_upd"][..., :3])), ("SPT C_full", ldr(sv["C_full"][..., :3]))]
    fig, ax = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 6))
    for r, (lab, (y, x)) in enumerate((("stationary plate, strongest GT change", yx_c), ("shadowed yellow bucket (subsurface)", yx_y))):
        for c, (t, im) in enumerate(cols):
            ax[r, c].imshow(np.clip(im[y:y + W, x:x + W], 0, 1)); ax[r, c].set_title(t if r == 0 else "", fontsize=9); off(ax[r, c])
        ax[r, 0].set_ylabel(lab, fontsize=9)
        ax[r, 0].text(-0.08, 0.5, lab, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout(); f = out / "03_crops_T1_view6.png"; fig.savefig(f, dpi=90); plt.close(fig); files.append(f)
    man = {"worklog": "docs/worklogs/39_WORKLOG_SPATIAL_PERSPECTIVE_TRANSPORT_FACTORIZATION_SPT1.md", "protocol": "experiments/spt_lego/protocol/spt_v1.json",
           "display": "code tonemap (x/(1+x))^(1/2.2); error scale 0-0.08 for every arm", "views": "held-out views 6 and 15 (worklog 37's reference-chosen views)",
           "crops": {"plate_view6_yx": [int(yx_c[0]), int(yx_c[1])], "bucket_view6_yx": [int(yx_y[0]), int(yx_y[1])], "size": W},
           "files": [{"file": p.name, "sha256_16": C.sha16(p)} for p in files] + [{"file": "spt_verdict.json", "sha256_16": C.sha16(out / "spt_verdict.json"), "script": "experiments/spt_lego/spt_verdict.py"}],
           "sources": {p: C.sha16(C.OUT / p) for p in ("eval/eval.json", "eval/views.npz", "SPT_T0/state.pt", "SPT_T1/state.pt", "rel_T0.npz", "rel_T1.npz")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1)); print([p.name for p in files])


if __name__ == "__main__":
    main()
