"""WL42 reviewer exports -> results/evaluation/42/ (PC, 8dna26 .venv-cu128: numpy + PIL; nothing is re-rendered or re-solved).

    python r1_exports.py <eval dir, e.g. results/fnto_lego_r1/eval_K_r1>

Inputs: <eval dir>/views.npz (written by r1_eval.run: per set / view / arm the fp16 radiance + 4 component parts at the
valid pixels, key <set>_v<v>_<arm> [n, 15], and <set>_v<v>_vidx = flat pixel indices of those pixels), eval.json,
verdict.json (root), the historical worklog-37 views.npz (RNA_T0 / RNA_T1 / A_full-PU images, reference T1 - T0 delta),
eval_px (historical references, masks), refs/ (r1 references). Display transform x/(1+x)^(1/2.2) as worklogs 37-41;
|error| maps on one fixed scale 0..0.15; signed maps on a symmetric scale +-0.10 (red +, blue -).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import r1_common as C

EV = C.ROOT / "results/evaluation/42"
ESCALE = 0.15
SSCALE = 0.10
ARM_LABEL = {"S_T0": "FNTO r1 S_T0", "D_stale": "FNTO r1 D_stale (T0 state)", "E_upd": "FNTO r1 E_upd (incremental C, K frozen)",
             "E_relfull": "FNTO r1 E_relfull (full C rebuild)", "K_only": "FNTO r1 K only (coupling off)"}


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def err_rgb(e):
    t = np.clip(e / ESCALE, 0, 1)
    return np.stack([t, np.clip(1.5 * t - 0.25, 0, 1) * 0.8, 1 - t], -1) * (e[..., None] > 0)


def signed_rgb(d, mask):
    t = np.clip(d / SSCALE, -1, 1)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    img = np.stack([1 - neg, 1 - pos - neg + pos * 0.2, 1 - pos], -1)
    return img * mask[..., None] + (1 - mask[..., None]) * 0.0


def panel(imgs, labels, path, scale=0.5):
    h, w = imgs[0].shape[:2]; hs, ws = int(h * scale), int(w * scale)
    canvas = Image.new("RGB", (ws * len(imgs), hs + 22), (255, 255, 255)); d = ImageDraw.Draw(canvas)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        canvas.paste(Image.fromarray(to_u8(im)).resize((ws, hs), Image.BILINEAR), (i * ws, 22)); d.text((i * ws + 4, 4), lb, fill=(0, 0, 0))
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


def main(evdir):
    evdir = Path(evdir); EV.mkdir(parents=True, exist_ok=True)
    F = np.load(evdir / "views.npz"); B = np.load(C.B4 / "eval/views.npz")
    px = {s: np.load(C.OUT1 / f"eval_px/{s}.npz") for s in ("heldout_T0", "heldout_T1")}
    refs = {s: (np.load(C.OUT1 / f"refs/{s}_full.npz") if (C.OUT1 / f"refs/{s}_full.npz").exists() else None) for s in px}
    firsts = {s: (np.load(C.OUT1 / f"refs/{s}_first.npz") if (C.OUT1 / f"refs/{s}_first.npz").exists() else None) for s in px}
    res = int(px["heldout_T0"]["res"])
    manifest = {"display": "x/(1+x)^(1/2.2)", "error_scale": ESCALE, "signed_scale": SSCALE, "files": {}}

    def full(idx, vals):
        img = np.zeros((res * res, vals.shape[-1]), np.float32); img[idx] = vals; return img.reshape(res, res, -1)

    def ref_hist(s, v):
        return full(px[s][f"v{v}_mk"], C.ldr3(px[s][f"v{v}_tgt"]))

    def ref_r1(s, v):
        return full(px[s][f"v{v}_mk"], C.ldr3(refs[s][f"v{v}_rgb"])) if refs[s] is not None else None

    def fn(s, v, arm, part=None):
        a = F[f"{s}_v{v}_{arm}"].astype(np.float32); idx = F[f"{s}_v{v}_vidx"]
        return full(idx, C.ldr3(a[:, :3] if part is None else a[:, 3 + 3 * part: 6 + 3 * part]))

    def lin(s, v, arm):
        return full(F[f"{s}_v{v}_vidx"], F[f"{s}_v{v}_{arm}"].astype(np.float32)[:, :3])

    def stored(s, v, arm):
        return B[f"{s}_v{v}_{arm}"].astype(np.float32).reshape(res, res, 3)

    def err(a, r, m):
        return err_rgb(np.abs(a - r).mean(-1) * m)

    for v in (6, 15):
        # 01 static T0: historical regime and r1 regime side by side
        r0 = ref_hist("heldout_T0", v); m0 = (np.abs(r0).sum(-1) > 0).astype(np.float32); s0 = fn("heldout_T0", v, "S_T0")
        rows = [([r0, stored("heldout_T0", v, "RNA_T0"), s0], ["T0 reference (hist, clamped, 4096 spp)", "RNA_T0 (worklog 37)", ARM_LABEL["S_T0"]]),
                ([np.zeros_like(r0), err(stored("heldout_T0", v, "RNA_T0"), r0, m0), err(s0, r0, m0)], ["|error| vs hist 0..0.15", "RNA_T0", "FNTO r1 S_T0"])]
        rr = ref_r1("heldout_T0", v)
        if rr is not None:
            rows.append(([rr, err(s0, rr, m0), err(r0, rr, m0)], ["T0 reference (r1 regime, unclamped)", "|FNTO S_T0 - r1 ref|", "|hist ref - r1 ref| (clamp effect)"]))
        grid(rows, EV / f"01_static_T0_view{v}.png")
        # 02 edit T1
        r1 = ref_hist("heldout_T1", v); m1 = (np.abs(r1).sum(-1) > 0).astype(np.float32)
        arms = [(ARM_LABEL["D_stale"], fn("heldout_T1", v, "D_stale")), (ARM_LABEL["E_upd"], fn("heldout_T1", v, "E_upd")),
                ("RNA_T1 (T1-trained, worklog 37)", stored("heldout_T1", v, "RNA_T1")), ("SO A_full-PU (T1-trained, worklog 37)", stored("heldout_T1", v, "A_full-PU"))]
        grid([([r1] + [a for _, a in arms], ["T1 reference (hist)"] + [n for n, _ in arms]),
              ([np.zeros_like(r1)] + [err(a, r1, m1) for _, a in arms], ["|error| 0..0.15"] + [n for n, _ in arms])], EV / f"02_edit_T1_view{v}.png")
        # 03 signed change: reference T1 - T0 vs predicted E_upd - S_T0 vs stale D_stale - S_T0 (luminance of display values)
        lum = lambda x: C.ldr3(x) @ np.array([0.2126, 0.7152, 0.0722], np.float32)  # noqa: E731
        gt = B[f"heldout_T1_v{v}_gtdelta"].astype(np.float32).reshape(res, res, -1) if f"heldout_T1_v{v}_gtdelta" in B.files else None
        both = (full(F[f"heldout_T0_v{v}_vidx"], np.ones((len(F[f"heldout_T0_v{v}_vidx"]), 1)))[..., 0] > 0) & (full(F[f"heldout_T1_v{v}_vidx"], np.ones((len(F[f"heldout_T1_v{v}_vidx"]), 1)))[..., 0] > 0)
        bothf = both.astype(np.float32)
        d_ref = (lum(lin("heldout_T1", v, "E_upd") * 0 + full(px["heldout_T1"][f"v{v}_mk"], px["heldout_T1"][f"v{v}_tgt"])) - lum(full(px["heldout_T0"][f"v{v}_mk"], px["heldout_T0"][f"v{v}_tgt"])))
        d_upd = lum(lin("heldout_T1", v, "E_upd")) - lum(lin("heldout_T0", v, "S_T0")); d_stale = lum(lin("heldout_T1", v, "D_stale")) - lum(lin("heldout_T0", v, "S_T0"))
        chg = B[f"heldout_T1_v{v}_reg_A_changed"].reshape(res, res).astype(np.float32)
        rel1 = np.load(C.OUT1 / "rel_T1_inc.npz")["hit"]; rel0 = np.load(C.OUT1 / "rel_T0.npz")["hit"]; chg_port = (rel0 != rel1).any(1)
        pmap = np.zeros(res * res, np.float32); p1 = px["heldout_T1"]; pmap[p1[f"v{v}_mk"]] = chg_port[np.clip(p1[f"v{v}_port"], 0, None)] * (p1[f"v{v}_port"] >= 0); pmap = pmap.reshape(res, res)
        grid([([signed_rgb(d_ref, bothf), signed_rgb(d_upd, bothf), signed_rgb(d_stale, bothf)],
               ["reference T1 - T0 (hist, signed +-0.10)", "FNTO E_upd(T1) - S_T0(T0)", "FNTO D_stale(T1) - S_T0(T0)"]),
              ([np.stack([chg] * 3, -1), np.stack([pmap] * 3, -1), signed_rgb(d_upd - d_ref, bothf)], ["A_changed region (worklog 37)", "pixels whose port got an updated relation row", "predicted - reference change"])],
             EV / f"03_signed_change_view{v}.png")
        # 05 K only vs first-event r1 reference
        if firsts["heldout_T0"] is not None:
            fr = full(px["heldout_T0"][f"v{v}_mk"], C.ldr3(firsts["heldout_T0"][f"v{v}_rgb"])); ko = fn("heldout_T0", v, "K_only")
            grid([([fr, ko, err(ko, fr, m0)], ["first-event reference (r1 regime, 1024 spp)", ARM_LABEL["K_only"], "|error| 0..0.15"])], EV / f"05_K_only_vs_first_T0_view{v}.png")
    # 04 components of S_T0 at view 6
    parts = [fn("heldout_T0", 6, "S_T0", k) for k in range(4)]
    grid([(parts, ["K_loc delta sun", "K_loc indirect (gathered)", "K_glob (low rank)", "K_band + K_mid"])], EV / "04_components_T0_view6.png")
    # json copies
    for src, name in ((evdir / "eval.json", "eval.json"), (C.OUT1 / "verdict.json", "verdict.json"), (evdir.parent / f"K{evdir.name[6:]}" / "rho.json", "rho.json"),
                      (evdir.parent / f"K{evdir.name[6:]}" / "train.json", "train.json"), (C.OUT1 / "gate1/gate1.json", "gate1.json"), (C.OUT1 / "gate2/quadrature_a1.json", "gate2_quadrature_a1.json"),
                      (C.OUT1 / "gate2/coverage.json", "gate2_coverage.json"), (C.OUT1 / "gate3/light_table_radiometry.json", "gate3_light_table.json"), (C.OUT1 / "relations.json", "relations.json")):
        if src.exists():
            d = json.loads(src.read_text())
            if name == "train.json":
                d.pop("log", None)
            (EV / name).write_text(json.dumps(d, indent=1))
    for f in sorted(EV.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    manifest["sources"] = {"fnto_r1_eval": str(evdir), "b4_views": str(C.B4 / "eval/views.npz"), "eval_px": str(C.OUT1 / "eval_px"), "refs": str(C.OUT1 / "refs")}
    (EV / "manifest.json").write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
