"""FNTO Batch 6 reviewer exports -> results/evaluation/41/ (8dna26 .venv-cu128; matplotlib optional -> PIL).

    python fnto_exports.py <fnto eval dir> [<stagea eval dir>]

All panels use the worklog-37 display transform x/(1+x) ^ (1/2.2); error maps are |display error| on one fixed scale
(0 .. 0.15) for every arm. Historical arms are read from their stored records (worklog-37 eval/views.npz for RNA and
Surface-Owned, worklog-39 eval/views.npz for SPT-1); nothing is re-rendered.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import fnto_common as C

EV = C.ROOT / "results/evaluation/41"
ESCALE = 0.15


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def err_rgb(e):
    t = np.clip(e / ESCALE, 0, 1)
    return np.stack([t, np.clip(1.5 * t - 0.25, 0, 1) * 0.8, 1 - t], -1) * (e[..., None] > 0) + (e[..., None] == 0) * 0.0


def panel(imgs, labels, path, scale=0.5):
    h, w = imgs[0].shape[:2]
    hs, ws = int(h * scale), int(w * scale)
    canvas = Image.new("RGB", (ws * len(imgs), hs + 22), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        canvas.paste(Image.fromarray(to_u8(im)).resize((ws, hs), Image.BILINEAR), (i * ws, 22))
        d.text((i * ws + 4, 4), lb, fill=(0, 0, 0))
    canvas.save(path)


def grid(rows, path, scale=0.5):
    ims = []
    for imgs, labels in rows:
        tmp = EV / "_row.png"; panel(imgs, labels, tmp, scale); ims.append(Image.open(tmp).copy()); tmp.unlink()
    W = max(i.width for i in ims); H = sum(i.height for i in ims)
    canvas = Image.new("RGB", (W, H), (255, 255, 255)); y = 0
    for i in ims:
        canvas.paste(i, (0, y)); y += i.height
    canvas.save(path)


def main(evdir, sdir=None):
    evdir = Path(evdir); EV.mkdir(parents=True, exist_ok=True)
    F = np.load(evdir / "views.npz"); B = np.load(C.B4 / "eval/views.npz"); Sp = np.load(C.SPT / "eval/views.npz")
    px = {s: np.load(C.OUT / f"eval_px/{s}.npz") for s in ("heldout_T0", "heldout_T1")}
    res = 1024
    manifest = {"display": "x/(1+x)^(1/2.2)", "error_scale": ESCALE, "files": {}}

    def full(set_name, v, vals):
        img = np.zeros((res * res, vals.shape[-1]), np.float32); img[px[set_name][f"v{v}_mk"]] = vals
        return img.reshape(res, res, -1)

    def ref(set_name, v):
        return full(set_name, v, C.ldr3(px[set_name][f"v{v}_tgt"]))

    def fn(set_name, v, arm, part=None):
        a = F[f"{set_name}_v{v}_{arm}"].astype(np.float32)
        return C.ldr3(a[..., :3] if part is None else a[..., 3 + 3 * part: 6 + 3 * part])

    def stored(set_name, v, arm):
        return B[f"{set_name}_v{v}_{arm}"].astype(np.float32).reshape(res, res, 3)

    def spt(set_name, v, arm):
        return C.ldr3(Sp[f"{set_name}_v{v}_{arm}"].astype(np.float32)[..., :3])

    def err(a, r, mask):
        e = np.abs(a - r).mean(-1) * mask
        return err_rgb(e)

    for v in (6, 15):
        m0 = (np.abs(ref("heldout_T0", v)).sum(-1) > 0).astype(np.float32)
        r0 = ref("heldout_T0", v)
        arms0 = [("RNA_T0", stored("heldout_T0", v, "RNA_T0")), ("FNTO S_T0", fn("heldout_T0", v, "S_T0")), ("SPT-1 S_T0", spt("heldout_T0", v, "S_T0"))]
        grid([([r0] + [a for _, a in arms0], ["T0 reference (4096 spp)"] + [n for n, _ in arms0]),
              ([np.zeros_like(r0)] + [err(a, r0, m0) for _, a in arms0], ["|error| 0..0.15"] + [n for n, _ in arms0])],
             EV / f"01_static_T0_view{v}.png")
        m1 = (np.abs(ref("heldout_T1", v)).sum(-1) > 0).astype(np.float32); r1 = ref("heldout_T1", v)
        arms1 = [("FNTO D_stale", fn("heldout_T1", v, "D_stale")), ("FNTO E_upd (no retraining)", fn("heldout_T1", v, "E_upd")),
                 ("RNA_T1 (T1-trained)", stored("heldout_T1", v, "RNA_T1")), ("SO A_full-PU (T1-trained)", stored("heldout_T1", v, "A_full-PU")),
                 ("SO B_sel-PU (T1-trained)", stored("heldout_T1", v, "B_sel-PU")), ("SPT-1 C_upd", spt("heldout_T1", v, "C_upd"))]
        grid([([r1] + [a for _, a in arms1[:3]], ["T1 reference"] + [n for n, _ in arms1[:3]]),
              ([np.zeros_like(r1)] + [err(a, r1, m1) for _, a in arms1[:3]], ["|error| 0..0.15"] + [n for n, _ in arms1[:3]]),
              ([a for _, a in arms1[3:]] + [np.zeros_like(r1)], [n for n, _ in arms1[3:]] + [""]),
              ([err(a, r1, m1) for _, a in arms1[3:]] + [np.zeros_like(r1)], [n for n, _ in arms1[3:]] + [""])],
             EV / f"02_edit_T1_view{v}.png")
        # coupling update: where the FNTO output moved between stale and updated, vs where the reference moved T0 -> T1
        fu, fs = F[f"heldout_T1_v{v}_E_upd"].astype(np.float32)[..., :3], F[f"heldout_T1_v{v}_D_stale"].astype(np.float32)[..., :3]
        dl = np.abs(C.ldr3(fu) - C.ldr3(fs)).mean(-1)
        rel1 = np.load(C.OUT / "rel_T1_inc.npz")["hit"]; rel0 = np.load(C.OUT / "rel_T0.npz")["hit"]
        chg_port = (rel0 != rel1).any(1)
        pmap = np.zeros(res * res, np.float32); p1 = px["heldout_T1"]
        pmap[p1[f"v{v}_mk"]] = chg_port[p1[f"v{v}_port"]].astype(np.float32); pmap = pmap.reshape(res, res)
        gt = B[f"heldout_T1_v{v}_gtdelta"].astype(np.float32) if f"heldout_T1_v{v}_gtdelta" in B.files else None
        gtd = np.abs(gt).mean(-1).reshape(res, res) if gt is not None else np.zeros((res, res))
        parts = [fn("heldout_T1", v, "E_upd", k) for k in range(4)]
        grid([([np.stack([pmap] * 3, -1), err_rgb(dl * 3), err_rgb(gtd * 3)], ["ports with an updated relation row", "|FNTO updated - stale| x3", "|reference T1 - T0| x3 (worklog 37)"]),
              (parts, ["K_loc delta sun", "K_loc indirect (gathered)", "K_glob (low rank)", "K_band + K_mid"])],
             EV / f"03_coupling_update_and_parts_view{v}.png")
        for name, (y0, y1, x0, x1) in {"plate_box": (520, 900, 250, 750), "upper": (150, 530, 300, 800)}.items():
            crop = lambda im: im[y0:y1, x0:x1]  # noqa: E731
            grid([([crop(r1)] + [crop(a) for _, a in arms1[:3]], ["T1 reference"] + [n for n, _ in arms1[:3]]),
                  ([np.zeros_like(crop(r1))] + [crop(err(a, r1, m1)) for _, a in arms1[:3]], ["|error| 0..0.15"] + [n for n, _ in arms1[:3]])],
                 EV / f"04_crop_{name}_T1_view{v}.png", scale=1.0)
    if sdir:
        Sd = json.loads((Path(sdir) / "stagea_eval.json").read_text())
        (EV / "stagea_eval.json").write_text(json.dumps(Sd, indent=1))
    for f in sorted(EV.glob("*.png")):
        manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    manifest["sources"] = {"fnto_eval": str(evdir), "b4_views": str(C.B4 / "eval/views.npz"), "spt_views": str(C.SPT / "eval/views.npz")}
    (EV / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
