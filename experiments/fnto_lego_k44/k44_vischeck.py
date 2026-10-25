"""WL44 A5 - why do 'shadowed' opaque probe pixels carry radiance? (LabServer63 bpy, CPU; reads r1 prep rows read-only)

For one held-out Stage A probe: opaque Lego pixels (Gray / Sand / Brown / Light_brown) whose r1 light bits are all 0
(Mitsuba: no light visible) but whose 4096-spp clean radiance is > thr. For each such pixel and each light of the
configuration, Blender's own BVH (scene.ray_cast on the evaluated scene incl. masks; the box only when present) is cast
from the pixel position (offset along the camera-facing normal) to the light, recording: the facing term n.w, the
distance, Blender visibility, the hit object when blocked. Also the same counts for a random sample of pixels whose
bits are > 0 (control). Prints a summary and writes results/fnto_lego_k44/A5_vischeck_<i>.json.

    RNA_DIR=<rna repo> python k44_vischeck.py -- <i> [thr=0.05] [n=400]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego")); sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import r1_common as C  # noqa: E402
import r1_stagea_render as SR  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    i = int(a[0]); thr = float(a[1]) if len(a) > 1 else 0.05; nmax = int(a[2]) if len(a) > 2 else 400
    cfg = {c["i"]: c for c in json.loads((C.OUT1 / "stagea/holdout/configs.json").read_text())}[i]
    d = C.OUT1 / "stagea/prep_holdout"; start = np.load(d / "start.npy"); s, e = start[i], start[i + 1]
    px = {k: np.load(d / f"px_{k}.npy", mmap_mode="r")[s:e] for k in ("pos", "nrm", "col", "dom", "port", "vbits")}
    A = np.load(C.OUT1 / "anchors.npz"); mat = A["mat"][np.asarray(px["port"])]
    oz = np.load(C.OUT1 / f"stagea/holdout/img_{i:05d}.npz"); cz = np.load(Path.home() / f"rest_fnto_r1/k44_clean/clean_{i:05d}.npz")
    rows_col = np.asarray(px["col"]); keep = np.zeros(len(oz["col"]), bool); ptr = 0
    for k in range(len(oz["col"])):
        if ptr < len(rows_col) and np.array_equal(oz["col"][k], rows_col[ptr]):
            keep[k] = True; ptr += 1
    clean = cz["col"][keep].mean(1)
    dom = np.asarray(px["dom"]); vb = np.asarray(px["vbits"]); pos = np.asarray(px["pos"]).astype(np.float64); nrm = np.asarray(px["nrm"]).astype(np.float64)
    opaque = np.isin(mat, [1, 5, 6, 7]) & (dom == 0)
    sel_bad = np.flatnonzero(opaque & (vb == 0) & (clean > thr)); sel_ctl = np.flatnonzero(opaque & (vb > 0))
    rng = np.random.default_rng(0)
    sel_bad = rng.choice(sel_bad, min(nmax, len(sel_bad)), replace=False) if len(sel_bad) else sel_bad
    sel_b0 = np.flatnonzero(opaque & (vb == 0)); sel_b0 = rng.choice(sel_b0, min(nmax, len(sel_b0)), replace=False) if len(sel_b0) else sel_b0
    lego = np.flatnonzero(dom == 0); sel_all = rng.choice(lego, min(nmax, len(lego)), replace=False)
    sel_ctl = rng.choice(sel_ctl, min(nmax, len(sel_ctl)), replace=False) if len(sel_ctl) else sel_ctl
    scn = SR.Scene(64, 1); scn.apply(dict(cfg)); dg = bpy.context.evaluated_depsgraph_get(); sc = bpy.context.scene
    box_present = cfg["box_offset"] is not None
    rec = {"i": i, "family": cfg["family"], "n_lights": len(cfg["lights"]), "n_opaque_bits0_bright": int((opaque & (vb == 0) & (clean > thr)).sum()),
           "n_opaque_bits0": int((opaque & (vb == 0)).sum()), "n_opaque": int(opaque.sum())}
    for name, sel in (("bits0_bright", sel_bad), ("control_bits_gt0", sel_ctl), ("bits0_random", sel_b0), ("lego_random_all_materials", sel_all)):
        stats = {"pixels": int(len(sel)), "pix_any_light_blender_visible": 0, "pix_bits_vs_blender_disagree": 0, "facing_neg_but_blender_visible": 0, "blockers": {}}
        for k in sel:
            p = pos[k]; n = nrm[k] / np.linalg.norm(nrm[k]); anyvis = False; dis = False
            for jl, L in enumerate(cfg["lights"]):
                if L["type"] == "sun":
                    w = np.asarray(L["dir"], np.float64); w /= np.linalg.norm(w); dist = 1e3
                else:
                    v = np.asarray(L["pos"], np.float64) - p; dist = float(np.linalg.norm(v)); w = v / dist
                    if L["type"] == "spot" and float(-w @ np.asarray(L["axis"])) < L["cos_half"]:
                        continue
                o = p + 2e-4 * n
                hit, loc, hn, idx, obj, _ = sc.ray_cast(dg, Vector(o), Vector(w), distance=dist - 1e-4)
                if hit and obj is not None and obj.name.startswith("B_") and not box_present:
                    hit = False
                vis_b = not hit; facing = float(n @ w)
                bit = bool((int(vb[k]) >> jl) & 1)
                if vis_b and facing > 0:
                    anyvis = True
                if vis_b and facing <= 0:
                    stats["facing_neg_but_blender_visible"] += 1
                if (vis_b and facing > 0) != bit:
                    dis = True
                    pl = stats.setdefault("per_light", {}).setdefault(str(jl), {"bit1_blender_blocked_or_backfacing": 0, "bit0_blender_visible": 0})
                    pl["bit0_blender_visible" if not bit else "bit1_blender_blocked_or_backfacing"] += 1
                if hit and obj is not None:
                    stats["blockers"][obj.name] = stats["blockers"].get(obj.name, 0) + 1
            stats["pix_any_light_blender_visible"] += int(anyvis); stats["pix_bits_vs_blender_disagree"] += int(dis)
            stats.setdefault("clean_radiance_sum", 0.0); stats["clean_radiance_sum"] += float(clean[k])
            if dis:
                stats.setdefault("clean_radiance_sum_disagree", 0.0); stats["clean_radiance_sum_disagree"] += float(clean[k])
        rec[name] = stats
    out = C.ROOT / "results/fnto_lego_k44"; out.mkdir(parents=True, exist_ok=True)
    (out / f"A5_vischeck_{i}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
