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


if __name__ == "__main__" and sys.argv[1] != "outcome":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)


def outcome(arrays_path):
    """Panels for the measured outcome (the full FNTO composition diverged on every held-out view; K-only evidence)."""
    Z = np.load(arrays_path); B = np.load(C.B4 / "eval/views.npz"); Sp = np.load(C.SPT / "eval/views.npz")
    EV.mkdir(parents=True, exist_ok=True); res = 1024
    manifest = {"display": "x/(1+x)^(1/2.2)", "error_scale": ESCALE, "files": {}, "sources": {"arrays": str(arrays_path)}}

    def img(set_name, v, key, ch=3):
        x = np.zeros((res * res, ch), np.float32); x[Z[f"{set_name}_v{v}_mk"]] = Z[f"{set_name}_v{v}_{key}"].astype(np.float32).reshape(len(Z[f"{set_name}_v{v}_mk"]), -1)[:, :ch]
        return x.reshape(res, res, ch)
    for set_name, tag in (("heldout_T0", "T0"), ("heldout_T1", "T1")):
        for v in (6, 15):
            first, kon, full = (C.ldr3(img(set_name, v, k)) for k in ("first", "konly", "full"))
            m = (full.sum(-1) > 0).astype(np.float32)
            rem = C.ldr3(np.clip(img(set_name, v, "full") - img(set_name, v, "first"), 0, None) * 2)
            grid([([first, kon, err(kon, first, m) if False else err_rgb(np.abs(kon - first).mean(-1) * m)], ["first-event reference (1024 spp)", "FNTO K only (frozen, coupling off)", "|error| 0..0.15"]),
                  ([full, rem, np.zeros_like(full)], ["full-GI reference (4096 spp)", "reference remainder (full - first) x2", "FNTO full composition: diverged (no image)"])],
                 EV / f"01_K_only_vs_first_event_{tag}_view{v}.png")
    alb = img("heldout_T0", 6, "albedo", 1)[..., 0]; chg = img("heldout_T1", 6, "relchg", 1)[..., 0]
    acol = np.stack([np.clip(alb / 3, 0, 1), np.clip(1 - np.abs(alb - 1) * 2, 0, 1) * 0.6, np.clip(1 - alb / 1.0, 0, 1)], -1) * (alb[..., None] > 0)
    parts = img("heldout_T0", 6, "parts", 12)
    grid([([acol, np.stack([chg] * 3, -1), C.ldr3(parts[..., 0:3])], ["port albedo under uniform light (blue <1, green ~1, red >=3)", "ports with an updated relation row (T1)", "K_loc delta sun"]),
          ([C.ldr3(parts[..., 6:9]), C.ldr3(parts[..., 9:12]), C.ldr3(img("heldout_T0", 6, "first"))], ["K_glob (sun only)", "K_band + K_mid (sun only)", "first-event reference"])],
         EV / "02_passivity_and_components_T0_view6.png")
    rows = []
    for j in sorted({int(k.split("_")[1]) for k in Z.files if k.startswith("stagea_")}):
        pix = Z[f"stagea_{j}_pix"]
        def to_img(a):
            x = np.zeros((512 * 512, 3), np.float32); x[pix] = a.astype(np.float32); return x.reshape(512, 512, 3)
        r, p = C.ldr3(to_img(Z[f"stagea_{j}_ref"])), C.ldr3(to_img(Z[f"stagea_{j}_pred"]))
        m = (r.sum(-1) > 0).astype(np.float32)
        rows.append(([r, p, err_rgb(np.abs(p - r).mean(-1) * m)], [f"Stage A holdout probe {j} (512 spp)", "FNTO K prediction", "|error| 0..0.15"]))
    grid(rows, EV / "03_stage_a_holdout_probes.png", scale=0.8)
    dg = json.loads((C.OUT / "diag/diag.json").read_text())["D2_growth_T0_view6"]
    W_, H_ = 900, 420; canvas = Image.new("RGB", (W_, H_), (255, 255, 255)); d = ImageDraw.Draw(canvas)
    cols = [(0, 0, 0), (60, 120, 220), (220, 60, 60), (40, 160, 70), (160, 60, 200), (230, 140, 20)]
    ymax = 13.0; x0, y0, x1, y1 = 60, 20, 860, 380
    d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0)); d.line([x0, y1 - (1 / ymax) * (y1 - y0), x1, y1 - (1 / ymax) * (y1 - y0)], fill=(150, 150, 150))
    d.text((x0 - 50, y1 - (1 / ymax) * (y1 - y0) - 6), "1.0", fill=(0, 0, 0)); d.text((x0, y1 + 8), "Jacobi iteration (T0 coupling graph, held-out T0 view 6)", fill=(0, 0, 0))
    d.text((x0 - 55, y0), f"{ymax:.0f}x", fill=(0, 0, 0))
    for k, (name, g) in enumerate(dg.items()):
        pts = [(x0 + i * (x1 - x0) / 11, y1 - min(r, ymax) / ymax * (y1 - y0)) for i, r in enumerate(g) if r is not None]
        if len(pts) > 1:
            d.line(pts, fill=cols[k], width=3)
        d.text((x1 - 230, y0 + 10 + 16 * k), f"{name}" + (" (diverged)" if None in g[1:] else ""), fill=cols[k])
    d.text((x0 + 10, y0 + 6), "growth of incident energy per iteration (ratio)", fill=(0, 0, 0))
    canvas.save(EV / "04_multibounce_growth_by_component.png")
    r0 = C.ldr3(np.zeros((res, res, 3)))
    t0 = np.zeros((res * res, 3), np.float32); t0[Z["heldout_T0_v6_mk"]] = Z["heldout_T0_v6_full"].astype(np.float32); t0 = C.ldr3(t0.reshape(res, res, 3))
    grid([([t0, B["heldout_T0_v6_RNA_T0"].astype(np.float32).reshape(res, res, 3), C.ldr3(Sp["heldout_T0_v6_S_T0"].astype(np.float32)[..., :3]), r0],
           ["T0 reference", "RNA_T0 (historical)", "SPT-1 S_T0 (historical, NOT SUPPORTED)", "FNTO S_T0: diverged"])], EV / "05_static_T0_view6_arms.png")
    for f in sorted(EV.glob("*.png")):
        manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    for f in ("verdict.json", "eval.json"):
        src = C.OUT / "eval_main" / f
        if src.exists():
            (EV / f).write_bytes(src.read_bytes()); manifest["files"][f] = C.sha16(src)
    for f in ("diag/diag.json", "eval_main/stagea/stagea_eval.json", "eval_main/passivity.json", "contracts.json"):
        src = C.OUT / f; (EV / Path(f).name).write_bytes(src.read_bytes()); manifest["files"][Path(f).name] = C.sha16(src)
    (EV / "manifest.json").write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))


if __name__ == "__main__" and sys.argv[1] == "outcome":
    outcome(sys.argv[2])
