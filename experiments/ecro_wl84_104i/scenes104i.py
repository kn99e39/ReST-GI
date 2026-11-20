"""WL104I Stage E scenes.

fixed  : exact replay of the archived WL104G launches; every state's F at E1 (fixed light direction) and at E2 (frozen native R0
         proposal, same actual pdf, fixed R0 terminal); states A0 (R0), A1 x seeds, B1 x seeds.
native : SECONDARY native renders of A1 / B1 (own proposal at every event), same seeds and launch layout.
analyze: stable RMS / signed energy per ROI and event; paired B1 - A1 contrasts.

Usage: python scenes104i.py fixed|native|analyze <scene> <angular_dir> <out_dir> [--reps a:b] [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import common104i as CM

IO, GF = CM.IO, CM.GF
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104h"))
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104e"))
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104b"))
import scenes104h as S8  # noqa: E402
from metrics104e import event_metrics  # noqa: E402
from stats104b import jk_ci  # noqa: E402

H, C, RT, mi, dr = S8.H, S8.C, S8.RT, S8.mi, S8.dr
SC = CM.P["stage_E_scenes"]
EVENTS = ("e1", "e2")
GATES = {"precision_energy_half_width": 0.0025, "precision_rms_half_width": 0.005}
ROIS = ("objects", "sphere", "cube", "highlight_core", "highlight_ring", "grazing", "sphere_lower")


def state_set(angular):
    with np.load(angular / "compiled_states.npz") as z:
        st = {k: z[k] for k in z.files}
    names = ["A0"] + [f"{a}_{s}" for a in ("A1", "B1") for s in CM.SEEDS]
    return {k: st[k] for k in names}


class FixedPass(S8.Pass):
    def __init__(self, tag, states, i):
        key = f"wl104i_{tag}_A0"
        H.RR.REG[key] = ("ER", states["A0"][i])
        RT.Ctx.__init__(self, mi.load_dict(C.scene_dict(mi, RT.T, {"type": "cmp83", "key": key}, with_sensor=True)))
        names = [m[1] for m in self.meshes]
        self.sph, self.cub = names.index("sphere"), names.index("cube")
        self.cl = {k: H.CD.DrClosure83("ER", v[i]) for k, v in states.items()}

    def chunk(self, sd, offset, k, light, with_ac=None):
        n = C.RES * C.RES * k
        rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(offset))
        ray, pix = self.rays(k, rng)
        si, sid, _, ok = self.surface(ray, mi.Bool(True))
        active = ok & (si.wi.z > 1e-6)
        er = self.uses_er(sid)
        native = si.bsdf()
        lanes, chk = {}, {}

        def put(name, ev, col):
            for c in range(3):
                lanes[f"{name}|{ev}|{c}"] = col[c]
        wo1, vis, lf = self.light_term(si, light, active)
        m1 = vis & er
        f_nat1 = native.eval(self.bctx, si, wo1, vis)
        put("N", "e1", dr.select(vis & ~er, f_nat1 * lf, 0.0))
        for name, cl in self.cl.items():
            put(name, "e1", dr.select(m1, cl.eval(si.wi, wo1, m1) * lf, 0.0))
        chk["e1_A0_minus_native"] = dr.max(dr.select(m1, dr.max(dr.abs(f_nat1 - self.cl["A0"].eval(si.wi, wo1, m1))), 0.0))
        chk["e1_native_peak"] = dr.max(dr.select(m1, dr.max(f_nat1), 0.0))
        bs, w = native.sample(self.bctx, si, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active)
        used = active & (bs.pdf > 0) & (dr.max(w) > 0)
        pdf = dr.select(used, bs.pdf, 1.0)
        wo2 = dr.select(used, bs.wo, mi.Vector3f(0, 0, 1))
        other, osid, _, ok2 = self.surface(si.spawn_ray(si.to_world(bs.wo)), used)
        act2 = used & ok2 & (other.wi.z > 1e-6)
        wl2, vis2, lf2 = self.light_term(other, light, act2)
        recv = dr.select(vis2, other.bsdf().eval(self.bctx, other, wl2, vis2) * lf2, 0.0)
        scale2 = recv * dr.rcp(pdf)
        m2 = used & er
        f_nat2 = native.eval(self.bctx, si, wo2, used)
        put("N", "e2", dr.select(used & ~er, f_nat2 * scale2, 0.0))
        for name, cl in self.cl.items():
            put(name, "e2", dr.select(m2, cl.eval(si.wi, wo2, m2) * scale2, 0.0))
        chk["e2_weight_pdf_minus_eval"] = dr.max(dr.select(used, dr.max(dr.abs(w * bs.pdf - f_nat2)), 0.0))
        chk["e2_native_peak"] = dr.max(dr.select(used, dr.max(f_nat2), 0.0))
        acc = self.accum(lanes, pix, C.RES * C.RES)
        dr.eval(*chk.values())
        return acc, {q: float(np.asarray(v).ravel()[0]) for q, v in chk.items()}


def setup(scene, angular):
    i, row = S8.scene_row(scene)
    if "held" in SC_ROLE(scene):
        C.CAM = IO.read(CM.HERE.parent / "ecro_wl84_104d/protocol.json")["population"]["held_out_view"]
    return i, row, state_set(angular)


def SC_ROLE(scene):
    return CM.P["stage_E_scenes"].get("roles", {}).get(scene) or IO.P["frozen_inputs"]["scenes"][scene]["role"]


def spp_of(scene):
    return SC["spp"]["held_out" if "held" in SC_ROLE(scene) else "development"]


def run_fixed(scene, angular, out, reps, smoke, k=128):
    out.mkdir(parents=True, exist_ok=True)
    i, row, states = setup(scene, angular)
    ctx = FixedPass(row["tag"] + "_" + scene, states, i)
    light = RT.point_lights(row["light"])
    seed0 = SC["scenes"][scene]
    if smoke:
        nat = H.HA.render(ctx, scene, 4, seed0, "cam", b=4)
        img, chk = ctx.replicate(seed0, 4, 2, light, None, launch=4)
        res = {"e1": S8.l1(img["N|e1"] + img["A0|e1"], nat["e1"]), "e2": S8.l1(img["N|e2"] + img["A0|e2"], nat["e2"]), "chk": chk}
        tp = time.monotonic()
        sd = dr.opaque(mi.UInt64, seed0 * 1000003)
        for c in range(S8.LAUNCH // k):
            ctx.chunk(sd, c * C.RES * C.RES * k, k, light)
        res["full_launch_seconds"] = time.monotonic() - tp
        IO.write(out / "fixed_smoke.json", res)
        print(res, flush=True)
        return
    spp = spp_of(scene)
    log = []
    for r in range(*reps):
        t = time.monotonic()
        img, chk = ctx.replicate(seed0 + r, spp, k, light, None)
        chk["seconds"] = time.monotonic() - t
        log.append(chk)
        np.savez_compressed(out / f"fixed_r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in img.items()})
        print("fixed", scene, r, round(chk["seconds"], 1), flush=True)
    IO.write(out / f"fixed_{reps[0]:02d}_{reps[1]:02d}.json", {"checks": log, "replicates": list(range(*reps)), "chunk_spp": k,
                                                               "commit": IO.commit()})


def run_native(scene, angular, out, reps, names):
    out.mkdir(parents=True, exist_ok=True)
    i, row, states = setup(scene, angular)
    seed0 = SC["scenes"][scene]
    spp = spp_of(scene)
    for name in names:
        key = f"wl104i_native_{scene}_{name}"
        H.RR.REG[key] = ("ER", states[name][i])
        ctx = RT.Ctx(mi.load_dict(C.scene_dict(mi, RT.T, {"type": "cmp83", "key": key}, with_sensor=True)))
        t = time.monotonic()
        for r in range(*reps):
            ev = H.HA.render(ctx, scene, spp, seed0 + r, "cam", b=S8.LAUNCH)
            np.savez_compressed(out / f"native_{name}_r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in ev.items()})
        IO.write(out / f"native_{name}_{reps[0]:02d}_{reps[1]:02d}.json", {"seconds": time.monotonic() - t, "spp": spp,
                                                                           "replicates": list(range(*reps)), "commit": IO.commit()})
        print("native", scene, name, round(time.monotonic() - t, 1), flush=True)


def paired(metrics, event, roi="objects"):
    """Seed-mean relative difference (B1 - A1) / A1 of stable RMS with replicate jackknife."""
    da, db = [], []
    for s in CM.SEEDS:
        a, b = metrics[f"A1_{s}"][roi]["rms"][event], metrics[f"B1_{s}"][roi]["rms"][event]
        da.append((a["point"], np.asarray(a["loo"]))); db.append((b["point"], np.asarray(b["loo"])))
    A = np.mean([x[0] for x in da]); B = np.mean([x[0] for x in db])
    Al = np.mean([x[1] for x in da], 0); Bl = np.mean([x[1] for x in db], 0)
    rel = jk_ci((B - A) / A, (Bl - Al) / Al)
    rel["A1_rms"], rel["B1_rms"] = float(A), float(B)
    rel["per_seed_rel"] = [float((db[j][0] - da[j][0]) / da[j][0]) for j in range(len(da))]
    return rel


def analyze(scene, out):
    arch, masks, seen = S8.archived(scene, None)
    ref = {"e1": arch["reference_e1"], "e2": arch["reference_e2"], "h": arch["reference_h"]}
    fx = {}
    for r in range(12):
        with np.load(out / f"fixed_r{r:03d}.npz") as z:
            for q in z.files:
                fx.setdefault(q, []).append(z[q])
    fx = {q: np.stack(v).astype(np.float64) for q, v in fx.items()}
    logs = [IO.read(p) for p in sorted(out.glob("fixed_[0-9]*_*.json"))]
    chk = [c for lg in logs for c in lg["checks"]]
    ident = {"A0_e1": max(S8.l1(fx["N|e1"][r] + fx["A0|e1"][r], arch["R0_e1"][r]) for r in range(12)),
             "A0_e2": max(S8.l1(fx["N|e2"][r] + fx["A0|e2"][r], arch["R0_e2"][r]) for r in range(12)),
             "lane_e1_rel": max(c["e1_A0_minus_native"] / max(c["e1_native_peak"], 1e-30) for c in chk),
             "lane_e2_weight_pdf_rel": max(c["e2_weight_pdf_minus_eval"] / max(c["e2_native_peak"], 1e-30) for c in chk)}
    ok = ident["A0_e1"] <= 2e-5 and ident["A0_e2"] <= 2e-5 and ident["lane_e1_rel"] <= 2e-6 and ident["lane_e2_weight_pdf_rel"] <= 2e-6
    names = [q.split("|")[0] for q in fx if q.endswith("|e1") and not q.startswith("N")]
    roi_masks = {k: masks[k] for k in ROIS if k in masks and masks[k].any()}
    fixed = {}
    for name in names:
        buf = {"e1": fx["N|e1"] + fx[f"{name}|e1"], "e2": fx["N|e2"] + fx[f"{name}|e2"], "h": arch["R0_h"]}
        fixed[name], _ = event_metrics(buf, ref, roi_masks, GATES)
    native = {}
    var = {}
    for name in names:
        if name == "A0":
            buf = {"e1": arch["R0_e1"], "e2": arch["R0_e2"], "h": arch["R0_h"]}
        else:
            files = [out / f"native_{name}_r{r:03d}.npz" for r in range(12)]
            if not all(f.exists() for f in files):
                continue
            buf = {e: [] for e in ("e1", "e2", "h")}
            for f in files:
                with np.load(f) as z:
                    for e in buf:
                        buf[e].append(z[e])
            buf = {e: np.stack(v).astype(np.float64) for e, v in buf.items()}
        native[name], _ = event_metrics(buf, ref, roi_masks, GATES)
        var[name] = float(np.var(buf["e2"][:, roi_masks["objects"]], axis=0, ddof=1).mean())
    _, precision = event_metrics({"e1": arch["R0_e1"], "e2": arch["R0_e2"], "h": arch["R0_h"]}, ref, roi_masks, GATES)
    prim = {ev: paired(fixed, ev) for ev in ("e1", "e2", "total")}
    sec = {ev: paired(native, ev) for ev in ("e1", "e2", "h", "total")} if all(f"B1_{s}" in native for s in CM.SEEDS) else None
    var_ratio = (float(np.mean([var[f"B1_{s}"] for s in CM.SEEDS]) / np.mean([var[f"A1_{s}"] for s in CM.SEEDS]))
                 if sec is not None else None)
    maps = {}
    for name in ("A0", f"A1_{CM.SEEDS[0]}", f"B1_{CM.SEEDS[0]}"):
        for ev in EVENTS:
            maps[f"{name}_{ev}_err"] = (fx[f"N|{ev}"] + fx[f"{name}|{ev}"] - ref[ev]).mean(0).astype(np.float32)
    maps["B1_minus_A1_e1"] = np.mean([fx[f"B1_{s}|e1"] - fx[f"A1_{s}|e1"] for s in CM.SEEDS], 0).mean(0).astype(np.float32)
    maps["B1_minus_A1_e2"] = np.mean([fx[f"B1_{s}|e2"] - fx[f"A1_{s}|e2"] for s in CM.SEEDS], 0).mean(0).astype(np.float32)
    np.savez_compressed(out / "maps.npz", **maps)
    res = {"scene": scene, "identity": ident, "identity_passed": bool(ok), "baseline_precision_failures": precision,
           "fixed": fixed, "native": native, "primary_fixed": prim, "secondary_native": sec, "native_e2_variance_ratio_B_over_A": var_ratio,
           "native_e2_variance": var, "inputs_sha256": seen, "commit": IO.commit()}
    IO.write(out / "analysis.json", res)
    print("analysis", scene, ok, {e: (round(v["point"], 4), round(v["lo"], 4), round(v["hi"], 4)) for e, v in prim.items()}, flush=True)
    if not ok:
        raise IO.Stop("SCENE_REPLAY_IDENTITY")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fixed", "native", "analyze"))
    ap.add_argument("scene")
    ap.add_argument("angular", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", default="0:12")
    ap.add_argument("--names", default=",".join(f"{a}_{s}" for a in ("A1", "B1") for s in CM.SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    reps = tuple(int(x) for x in args.reps.split(":"))
    if args.mode == "fixed":
        run_fixed(args.scene, args.angular, args.out, reps, args.smoke)
    elif args.mode == "native":
        run_native(args.scene, args.angular, args.out, reps, args.names.split(","))
    else:
        analyze(args.scene, args.out)


if __name__ == "__main__":
    main()
