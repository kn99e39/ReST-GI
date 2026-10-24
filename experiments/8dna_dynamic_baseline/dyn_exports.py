"""Reviewer exports of protocol 8dna_dyn_v1 into $EDNA_OUT/exports (copied to results/evaluation/43 with its manifest).
Presentation only: every number comes from eval.json. CPU only.

    python dyn_exports.py
"""

from __future__ import annotations

import hashlib

import numpy as np

import dynlib as L
from dyn_eval import Exp, disp
from dyn_stages import STATES

OUTD = L.OUT / "exports"


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def labelled(img, text):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    dr_ = ImageDraw.Draw(im)
    dr_.rectangle([0, 0, 8 + 6 * len(text), 14], fill=(0, 0, 0))
    dr_.text((3, 1), text, fill=(255, 255, 255))
    return np.array(im)


def cmap(x, name, vmin, vmax):
    import matplotlib

    return u8(matplotlib.colormaps[name](np.clip((x - vmin) / (vmax - vmin), 0, 1))[..., :3])


def png(name, img):
    from PIL import Image

    OUTD.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(OUTD / name)


def gif(name, frames, ms=700):
    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(OUTD / name, save_all=True, append_images=ims[1:], duration=ms, loop=0)


def lum(x):
    return x @ np.array([0.2126, 0.7152, 0.0722])


def change_row(E, t0, t1, mask, title):
    dg, dn = lum(E.G[t1] - E.G[t0]), lum(E.N[t1] - E.N[t0])
    v = float(np.percentile(np.abs(dg[mask]), 99)) if mask.any() else 1.0
    v = max(v, 1e-4)
    m3 = mask[..., None]
    tiles = [labelled(np.where(m3, cmap(dg, "RdBu_r", -v, v), 40), f"{title} reference dG (lum, +-{v:.3g})"),
             labelled(np.where(m3, cmap(dn, "RdBu_r", -v, v), 40), "fixed 8DNA + PT dN"),
             labelled(np.where(m3, cmap(dn - dg, "RdBu_r", -v, v), 40), "dN - dG")]
    return np.concatenate(tiles, 1)


def main() -> None:
    files = []
    for exp in ("E1", "E2", "E3"):
        E = Exp(exp)
        for t in E.G:
            err = np.abs(disp(E.N[t]) - disp(E.G[t])).mean(-1)
            row = np.concatenate([labelled(u8(disp(E.G[t])), f"{exp} {t} path-traced reference"),
                                  labelled(u8(disp(E.N[t])), "released 8DNA + path tracing"),
                                  labelled(cmap(err, "inferno", 0, 0.2), "|display diff| 0..0.2")], 1)
            png(f"{exp}_{t}_static.png", row)
            files.append(f"{exp}_{t}_static.png")
        t0 = "T0_red"
        for st in STATES[exp][1:]:
            t1 = f"{st}_red"
            mask = E.receivers(t0, t1, "absent") if "absent" in E.cls else E.receivers(t0, t1)
            png(f"{exp}_{st}_change.png", change_row(E, t0, t1, mask, f"{exp} T0->{st}"))
            files.append(f"{exp}_{st}_change.png")
            if exp != "E3":
                mx = E.receivers(t1, "absent")
                fam = np.concatenate([change_row(E, "absent", f"{st}_black", mx, f"{st} occlusion"),
                                      change_row(E, f"{st}_black", t1, mx, f"{st} box reflection")], 0)
                png(f"{exp}_{st}_families.png", fam)
                files.append(f"{exp}_{st}_families.png")
            for src, name in ((E.G, "reference"), (E.N, "8dna")):
                gif(f"{exp}_{st}_{name}_flicker.gif", [labelled(u8(disp(src[t0])), f"{name} T0"), labelled(u8(disp(src[t1])), f"{name} {st}")])
                files.append(f"{exp}_{st}_{name}_flicker.gif")

    # official demonstrations
    for key in ("seal_scene2", "teaset_get_scene"):
        d = L.OUT / "static" / key
        if (d / "ref_1.npy").exists():
            g = 0.5 * (np.load(d / "ref_0.npy") + np.load(d / "ref_1.npy"))
            n = np.load(d / "neural_0.npy")
            png(f"B_{key}.png", np.concatenate([labelled(u8(disp(g)), f"{key} reference"), labelled(u8(disp(n)), "released 8DNA"),
                                                 labelled(cmap(np.abs(disp(n) - disp(g)).mean(-1), "inferno", 0, 0.2), "|diff| 0..0.2")], 1))
            files.append(f"B_{key}.png")

    # ladder plot
    ev = L.read_json(L.OUT / "eval" / "eval.json")
    if ev.get("ladder"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, exp in zip(axs, ("E1", "E2")):
            rows = ev["ladder"][exp]["rows"]
            for pre, lab in (("neural_", "8DNA + PT"), ("ref_", "path tracing")):
                ks = sorted([k for k in rows if k.startswith(pre)], key=lambda k: int(k.split("_")[1]))
                ax.loglog([rows[k]["seconds"] for k in ks], [rows[k]["mse_receivers"] for k in ks], "o-", label=lab)
            ax.set_title(f"{exp} T1_out: receiver display MSE vs time (raw)")
            ax.set_xlabel("seconds")
            ax.set_ylabel("display MSE vs reference")
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTD / "ladder.png", dpi=110)
        files.append("ladder.png")

    import platform

    manifest = {"protocol": L.PROTOCOL["protocol"], "worklog": 43, "commit": L.git("rev-parse", "HEAD"), "machine": platform.node(),
                "dry_run": L.DRY, "sources": str(L.OUT), "eval_json_sha256": L.sha256(L.OUT / "eval" / "eval.json"),
                "placements_sha256": L.sha256(L.OUT / "design" / "placements.json") if (L.OUT / "design" / "placements.json").exists() else None,
                "classes_sha256": L.sha256(L.OUT / "design" / "classes.npz"),
                "files": {f: hashlib.sha256((OUTD / f).read_bytes()).hexdigest() for f in files},
                "note": "presentation only; every number is in eval/eval.json; masks are the eroded receivers of dyn_eval"}
    L.write_json(OUTD / "manifest.json", manifest)
    print(len(files), "exports")


if __name__ == "__main__":
    main()
