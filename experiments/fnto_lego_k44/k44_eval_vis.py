"""WL44 candidate evaluation input: held-out port sun fractions with the 0.3 mm visibility rule (protocol fnto_k44).

Same keys and order as r1_eval.prep's eval_vis.npz (r1_relations.sun_vis), but occluder hits closer than T_MIN to the
offset sample origin are ignored. Writes results/fnto_lego_r1/eval_vis_k44.npz (r1's eval_vis.npz untouched); the
pixel visibility of the held-out views stays the historical renderer's own (diffuse_direct > 0).

    python k44_eval_vis.py          (Mitsuba cuda: PC, or LabServer63 with mi_env)
"""
import time

import numpy as np

import k44_common as K  # noqa: F401
import r1_common as C
import fnto_relations as R
import r1_relations as RR
from k44_prep import T_MIN

SETS = {"heldout_T0": "T0", "heldout_T1": "T1"}


def sun_vis_t(A, geo, w, chunk=400000):
    off = RR.offs(A, geo)
    sp = A["spos"].astype(np.float64) + off[:, None]; sn = A["snrm"].astype(np.float64)
    valid = np.isfinite(sp[..., 0]) & ((sn @ np.asarray(w)) > 0)
    o = (sp + C.EPS1 * sn)[valid] + T_MIN * np.asarray(w, np.float64); vis = np.zeros(o.shape[0], bool)
    mi = geo.mi; f = lambda a: mi.Float(np.ascontiguousarray(a.astype(np.float32)))  # noqa: E731
    for s in range(0, len(o), chunk):
        oo = o[s:s + chunk]; dd = np.broadcast_to(np.asarray(w, np.float64), oo.shape)
        ray = mi.Ray3f(mi.Point3f(f(oo[:, 0]), f(oo[:, 1]), f(oo[:, 2])), mi.Vector3f(f(dd[:, 0]), f(dd[:, 1]), f(dd[:, 2])))
        vis[s:s + chunk] = ~np.array(geo.scene.ray_test(ray))
    frac = np.zeros(valid.shape); frac[valid] = vis
    return frac.sum(1) / np.maximum(np.isfinite(A["spos"][..., 0]).sum(1), 1)


def main():
    A = RR.load(); old = np.load(C.OUT1 / "eval_vis.npz"); px = {s: np.load(C.OUT1 / f"eval_px/{s}.npz") for s in SETS}
    geos = {st: R.Geo(*C.offsets(st)) for st in ("T0", "T1")}; out = {}; changed = 0; total = 0
    for s, state in SETS.items():
        for v in range(int(px[s]["nv"])):
            light = px[s][f"v{v}_light"]
            for st in (("T0", "T1") if s == "heldout_T1" else ("T0",)):
                t = time.time(); fr = sun_vis_t(A, geos[st], light)
                out[f"{s}_v{v}_{st}"] = fr.astype(np.float16); out[f"{s}_v{v}_{st}_s"] = np.float32(time.time() - t)
                d = np.abs(out[f"{s}_v{v}_{st}"].astype(np.float32) - old[f"{s}_v{v}_{st}"].astype(np.float32)); changed += int((d > 1e-3).sum()); total += len(d)
        print(s, "done", flush=True)
    out["_meta_changed_frac"] = np.float32(changed / total)
    np.savez_compressed(C.OUT1 / "eval_vis_k44.npz", **out); print("ports with a changed sun fraction:", changed / total)


if __name__ == "__main__":
    main()
