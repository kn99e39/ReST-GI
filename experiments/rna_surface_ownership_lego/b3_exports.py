"""Worklog-36 reviewer exports (RNA venv, GPU).

    python b3_exports.py --out results/evaluation/36

held-out views 0 and 8 (4096-spp references): reference | A | B (hard) | C (B + PU) and |ldr error| maps;
ownership map (unit hue by cell, intra-component seams black, PU band amber); 128x128 crops at the window
with the most seam pixels and at the highest reference-gradient window (both chosen from geometry /
reference only); REINIT (unit 9) vs B under C: changed pixels vs predicted influence; key numbers + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_compose as C  # noqa: E402
import b3_model as B  # noqa: E402
import rso_model as M  # noqa: E402

P = "/mnt/c/Projects/NeuralShaderDemo"
R = f"{P}/results/rna_surface_ownership_lego"
H5 = f"{P}/results/rna_reproduction/work/data/training_data/b3_heldout_4096_val.h5"
A_CKPT = f"{P}/results/rna_reproduction/work/checkpoints/tog-lego/version_2/checkpoints/last.ckpt"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def ldr(x):
    x = np.clip(x, 0, None)
    return ((x / (1 + x)) ** (1 / 2.2)).mean(-1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(score, w):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    return np.unravel_index(np.argmax(tot), tot.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    import rna
    import rna.config
    import torch

    sys.modules.setdefault("neumat", rna); sys.modules.setdefault("neumat.config", rna.config)
    from rna import config, models

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    conf = config.get_neumat_config(config_filename=f"{R}/b3_lego_cpudata.yml")
    owner = B.LegoOwner(f"{R}/partition_L16.npz", f"{R}/mesh/lego_mesh.npz")
    sd = torch.load(A_CKPT, map_location="cuda", weights_only=False)["state_dict"]
    A = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).cuda(); A.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}); A.eval(); A.set_iteration_index(10 ** 9)
    comps = {}
    for arm in ("B", "REINIT"):
        d = Path(f"{R}/{arm}")
        specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((d / "specs.json").read_text())]
        st = torch.load(d / "state.pt", map_location="cuda")
        m = M.OwnedTriplaneSet(specs, (st["dmin"].cpu().numpy(), st["dmax"].cpu().numpy())).cuda(); m.load_state_dict(st); m.eval(); m.set_iteration_index(10 ** 9)
        comps[arm] = C.LegoPUComposer.from_partition(m, f"{R}/partition_L16.npz", 0.0045)
    f = h5py.File(H5, "r"); res = int(f.attrs["resolution"][0])
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda").unsqueeze(0)
    files = []
    hue = plt.get_cmap("tab20")(np.arange(16))[:, :3]  # 16 unit identities (exceeds 8 categorical slots: ownership map only, labelled)
    data = {}
    for v in (0, 8):
        g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
        alpha = g("alpha")[:, 0]; m = alpha > 0
        Pp, N, Cd, Lt, vis, ref = g("position"), g("normal"), g("camera_dir"), g("light_dir"), g("diffuse_direct")[:, 0] > 0, g("color")
        lab, _, (tri, comp, onsurf) = owner(Pp[m].astype(np.float64), N[m].astype(np.float64), return_tri=True)
        pos = (Pp[m] - aabb[0]) / (aabb[1] - aabb[0])
        with torch.no_grad():
            pr = torch.cat([A(T(pos[s:s + (1 << 19)]), T(Cd[m][s:s + (1 << 19)]), T(Lt[m][s:s + (1 << 19)]), T(N[m][s:s + (1 << 19)])).squeeze(0) for s in range(0, len(pos), 1 << 19)])
        full = lambda x: (lambda o: (o.__setitem__(m, x), o)[1])(np.zeros((res * res, 3), np.float32)).reshape(res, res, 3)
        pa = torch.where(torch.tensor(vis[m], device="cuda")[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
        cb, pb = comps["B"](Pp[m], Cd[m], Lt[m], N[m], vis[m], lab, comp, onsurf, return_parts=True)
        cr, _ = comps["REINIT"](Pp[m], Cd[m], Lt[m], N[m], vis[m], lab, comp, onsurf, return_parts=True)
        o = -np.ones(res * res, int); o[m] = lab; cc = -np.ones(res * res, int); cc[m] = comp
        w = np.zeros(res * res); w[m] = pb["w"]; nb = -np.ones(res * res, int); nb[m] = pb["nbr"]
        data[v] = dict(ref=full(ref[m]), A=full(pa), B=full(pb["hard"]), C=full(cb), R=full(cr), own=o.reshape(res, res), comp=cc.reshape(res, res),
                       w=w.reshape(res, res), nbr=nb.reshape(res, res), m=m.reshape(res, res))
    for v, d in data.items():
        cols = [("4096-spp reference", d["ref"]), ("A native RNA HQ", d["A"]), ("B independent L16", d["B"]), ("C = B + PU", d["C"])]
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.6))
        for c, (t, img) in enumerate(cols):
            axs[0, c].imshow(disp(img)); axs[0, c].set_title(t, fontsize=9); off(axs[0, c])
            if c:
                im = axs[1, c].imshow(np.where(d["m"], np.abs(ldr(img) - ldr(d["ref"])), 0), cmap=SEQ, vmin=0, vmax=0.08)
            off(axs[1, c])
        o = d["own"]; rgb = np.ones(o.shape + (3,)) * np.array(matplotlib.colors.to_rgb(SURF))
        rgb[o >= 0] = 0.55 * hue[o[o >= 0]] + 0.45 * ldr(d["ref"])[o >= 0, None]
        seam = np.zeros_like(o, bool)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            so = np.roll(o, (dy, dx), (0, 1)); sc = np.roll(d["comp"], (dy, dx), (0, 1))
            seam |= (so != o) & (so >= 0) & (o >= 0) & (sc == d["comp"])
        rgb[d["w"] > 0] = 0.5 * rgb[d["w"] > 0] + 0.5 * np.array(matplotlib.colors.to_rgb("#eda100")); rgb[seam] = matplotlib.colors.to_rgb(INK)
        axs[1, 0].imshow(rgb); axs[1, 0].set_title("16 units (hue), same-brick seams (black),\nPU band (amber)", fontsize=8)
        fig.colorbar(im, ax=axs[1, 1:], fraction=0.02, shrink=0.8).set_label("|ldr error| (0-0.08)", fontsize=7)
        fig.suptitle(f"Native Lego held-out view {v} (4096-spp reference; code tonemap error)", fontsize=9)
        name = f"01_heldout_view{v}.png"; fig.savefig(out / name, dpi=130, bbox_inches="tight"); plt.close(fig); files.append(name)
        data[v]["seam"] = seam
    fig, axs = plt.subplots(4, 6, figsize=(15, 10.5)); r = 0
    for v, d in data.items():
        lum = ldr(d["ref"]); grad = np.zeros_like(lum)
        for dy, dx in ((0, 1), (1, 0)):
            grad = np.maximum(grad, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))) * d["m"])
        for kind, score in (("most same-brick seam pixels", d["seam"].astype(float)), ("highest reference gradient", grad)):
            y, x = window(score, 128); sl = (slice(y, y + 128), slice(x, x + 128))
            cols = [("reference", disp(d["ref"])[sl], None), ("A", disp(d["A"])[sl], None), ("B", disp(d["B"])[sl], None), ("C = B+PU", disp(d["C"])[sl], None),
                    ("|B - ref|", np.abs(ldr(d["B"]) - lum)[sl], "e"), ("|C - ref|", np.abs(ldr(d["C"]) - lum)[sl], "e")]
            for c, (t, img, k) in enumerate(cols):
                ax = axs[r, c]; ax.imshow(img, cmap=SEQ if k else None, vmin=0 if k else None, vmax=0.08 if k else None, interpolation="nearest"); off(ax)
                if r == 0:
                    ax.set_title(t, fontsize=8)
            if kind.startswith("most"):
                axs[r, 0].contour(d["seam"][sl], levels=[0.5], colors="#e34948", linewidths=0.6)
            axs[r, 0].set_ylabel(f"view {v}\n{kind}\n@({y},{x})", fontsize=7); r += 1
    fig.suptitle("128x128 crops, windows chosen from geometry / reference only (red contour = same-brick ownership seams); error scale 0-0.08", fontsize=9)
    fig.savefig(out / "02_crops.png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append("02_crops.png")
    d = data[0]
    changed = (np.abs(d["R"] - d["C"]).sum(-1) > 0) & d["m"]
    infl = ((d["own"] == 9) | ((d["nbr"] == 9) & (d["w"] > 0))) & d["m"]
    rgb = np.ones(d["own"].shape + (3,)) * np.array(matplotlib.colors.to_rgb(SURF)); rgb[d["m"]] = 0.85
    rgb[(d["own"] == 9)] = matplotlib.colors.to_rgb("#2a78d6"); rgb[infl & (d["own"] != 9)] = matplotlib.colors.to_rgb("#eda100")
    rgb[changed & ~infl] = matplotlib.colors.to_rgb("#e34948")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
    axs[0].imshow(rgb); off(axs[0]); axs[0].set_title("predicted influence of unit 9: owned (blue),\nneighbour PU band (amber), unexpected (red)", fontsize=8)
    axs[1].imshow(changed, cmap="Greys"); off(axs[1]); axs[1].set_title(f"pixels changed REINIT vs B under C: {int(changed.sum())}\noutside prediction: {int((changed & ~infl).sum())}", fontsize=8)
    fig.suptitle("Unit-9 full reconstruction (REINIT) on native Lego, held-out view 0", fontsize=9)
    fig.savefig(out / "03_reinit_influence.png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append("03_reinit_influence.png")
    key = {"eval": json.loads(Path(f"{R}/eval/eval.json").read_text()), "partition": json.loads(Path(f"{R}/partition_L16.json").read_text()),
           "tests": json.loads(Path(f"{R}/tests/tests.json").read_text()), "train_B": {k: v for k, v in json.loads(Path(f"{R}/B/train.json").read_text()).items() if k not in ("epochs", "final_digest")},
           "train_REINIT": {k: v for k, v in json.loads(Path(f"{R}/REINIT/train.json").read_text()).items() if k not in ("epochs", "final_digest", "init_digest")}}
    (out / "04_key_numbers.json").write_text(json.dumps(key, indent=1, default=str)); files.append("04_key_numbers.json")
    (out / "manifest.json").write_text(json.dumps({"worklog": "docs/worklogs/36_WORKLOG_NATIVE_LEGO_SURFACE_OWNERSHIP_GENERALIZATION.md",
                                                   "generator": "experiments/rna_surface_ownership_lego/b3_exports.py", "source": "results/rna_surface_ownership_lego",
                                                   "files": {x: hashlib.sha256((out / x).read_bytes()).hexdigest() for x in files}}, indent=1))
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
