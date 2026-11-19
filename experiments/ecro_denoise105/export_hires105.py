"""POST HOC reviewer re-export of the WL105 view figures at native resolution (added 2026-11-19 after the owner
reported the original view_*.png panels too small to inspect). Reads the frozen WL105 evaluation outputs and
targets; computes nothing new; changes no decision. Original figures are kept unchanged.

Every panel is placed with figimage (1 image pixel = 1 output pixel, no resampling); crops are enlarged by an
integer nearest-neighbour factor.

python export_hires105.py DATA EVAL OUT
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])
ARMS = ("R0", "R1", "R2", "R3")
LABEL = {"R0": "R0 raw ECRO", "R1": "R1 RGB KPN", "R2": "R2 coupled", "R3": "R3 separated", "T": "target (2 x 8192 spp)"}
GAP, HEAD = 8, 26


def tone(x, exposure):
    y = np.clip(x * exposure, 0, None)
    y = y / (1 + y)
    return np.clip(y ** (1 / 2.2), 0, 1).transpose(1, 2, 0)


def diverging(e, cap):
    t = np.clip(e / cap, -1, 1)
    rgb = np.ones(e.shape + (3,))
    pos, neg = t > 0, t < 0
    rgb[pos] = np.stack([np.ones(pos.sum()), 1 - t[pos], 1 - t[pos]], -1)        # red: over-prediction
    rgb[neg] = np.stack([1 + t[neg], 1 + t[neg], np.ones(neg.sum())], -1)        # blue: under-prediction
    return rgb


def sheet(path, rows, title, scale=1):
    """rows: list of (row label, [(panel label, HxWx3 array)])."""
    tiles = [[(lab, np.repeat(np.repeat(a, scale, 0), scale, 1)) for lab, a in r] for _, r in rows]
    h, w = tiles[0][0][1].shape[:2]
    ncol = max(len(r) for r in tiles)
    W = ncol * w + (ncol + 1) * GAP
    H = len(tiles) * (h + HEAD) + (len(tiles) + 1) * GAP + HEAD
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    fig.text(GAP / W, 1 - 6 / H, title, va="top", fontsize=11, weight="bold")
    for i, r in enumerate(tiles):
        y_top = HEAD + GAP + i * (h + HEAD + GAP)
        for j, (lab, a) in enumerate(r):
            x0 = GAP + j * (w + GAP)
            fig.text((x0) / W, 1 - (y_top + 4) / H, lab, va="top", fontsize=9)
            fig.figimage(a, xo=x0, yo=H - (y_top + HEAD) - h, origin="upper")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(data, ev, out):
    out.mkdir(parents=True, exist_ok=True)
    eman = json.loads((ev / "eval_manifest.json").read_text())
    index = []
    for name in eman["views"]:
        d = data / "eval" / name
        with np.load(d / "aux.npz") as a:
            sph = a["cover"][6]
            cover = a["cover"]
        h = [np.load(d / f"target_half{i}.npz")["comp"].astype(np.float64) for i in range(2)]
        T = 0.5 * (h[0] + h[1])
        Y = T[0:3] + T[3:6] + T[6:9]
        YL = np.tensordot(LUMA, Y, 1)
        exposure = 1.0 / max(np.percentile(YL, 99), 1e-3)
        ys, xs = np.nonzero((sph > 0) & (sph < 1))
        crops = {}
        if len(ys):
            for tag, k in (("left silhouette", np.argmin(xs)), ("top silhouette", np.argmin(ys)), ("right silhouette", np.argmax(xs))):
                cy, cx = int(ys[k]), int(xs[k])
                y0, x0 = min(max(cy - 48, 0), 512 - 96), min(max(cx - 48, 0), 512 - 96)
                crops[tag] = np.s_[y0:y0 + 96, x0:x0 + 96]
        mb = np.nonzero((cover.max(0) < 1) & (sph == 0))
        if len(mb[0]):
            k = len(mb[0]) // 2
            y0, x0 = min(max(int(mb[0][k]) - 48, 0), 512 - 96), min(max(int(mb[1][k]) - 48, 0), 512 - 96)
            crops["material boundary"] = np.s_[y0:y0 + 96, x0:x0 + 96]
        for s in (4, 16, 32):
            with np.load(ev / f"{name}_spp{s}_r0.npz") as f:
                outs = {a: f[a].astype(np.float64) for a in ARMS}
            err = {a: np.tensordot(LUMA, outs[a] - Y, 1) for a in ARMS}
            cap = float(np.percentile(np.abs(err["R1"]), 99.5))
            rows = [("image", [(LABEL[a], tone(outs[a], exposure)) for a in ARMS] + [(LABEL["T"], tone(Y, exposure))]),
                    ("signed", [(f"{a} signed error (red +, blue -; +/-{cap:.3g})", diverging(err[a], cap)) for a in ARMS]
                     + [("sphere coverage", np.repeat(sph[..., None], 3, -1))])]
            p = out / f"{name}_spp{s}_full.png"
            sheet(p, rows, f"WL105 POST HOC native-resolution re-export · {name} · {s} spp · replicate 0 · 512 x 512 (1:1 pixels) · "
                           f"common exposure; error scale = R1's 99.5th percentile |luma error| (R0 clipped)")
            index.append(p.name)
            for tag, sl in crops.items():
                rows = [("image", [(LABEL[a], tone(outs[a][(slice(None),) + sl], exposure)) for a in ARMS] + [(LABEL["T"], tone(Y[(slice(None),) + sl], exposure))]),
                        ("signed", [(f"{a} signed (+/-{cap:.3g})", diverging(err[a][sl], cap)) for a in ARMS]
                         + [("sphere coverage", np.repeat(sph[sl][..., None], 3, -1))])]
                p = out / f"{name}_spp{s}_crop_{tag.replace(' ', '_')}.png"
                sheet(p, rows, f"WL105 POST HOC · {name} · {s} spp · {tag} · 96 x 96 crop, x4 nearest-neighbour", scale=4)
                index.append(p.name)
        print("exported", name, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print("EXPORT DONE", len(index))


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:4]])
