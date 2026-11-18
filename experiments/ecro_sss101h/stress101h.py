"""WL101H Q1: targeted boundary stress test of native and candidate origins at the sphere/floor contact.

Derives h_min (the smallest half-gap at which a contact-valid free-space origin is numerically
reliable) by the rule declared in protocol/qualification.json: h_min = 4 x the largest half-gap at
which any check fails with the candidate installed at h_min = 0.

python stress101h.py WL101_ROOT OUT_DIR
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402
import instrument101h as I  # noqa: E402
import semantics101h as SEM  # noqa: E402

C, RAD = I.SPHERE_C, I.SPHERE_R


def main():
    root, out = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    reader, *_ = RG.F101.inputs(root)
    R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
    out.mkdir(parents=True, exist_ok=False)
    light = I.P(rt.lt.pos[0]).reshape(-1, 3)[0]
    Pf = lambda a: mi.Point3f(*[R.F(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    Vf = lambda a: mi.Vector3f(*[R.F(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    result = {}

    def surface(points, normals, scene_shape):
        """Native hit at requested points through the single-shape attachment scene (materialised)."""
        from oracle_field import ExposedOracle
        oracle = ExposedOracle(R, rt)
        ray = mi.Ray3f(Pf(points + 0.002 * normals), Vf(-normals))
        ray.maxt = 0.003
        hit = oracle.shape_scenes[scene_shape].ray_intersect(ray)
        dr.eval(hit)
        return hit

    for semantics in ("native", "candidate"):
        if semantics == "candidate":
            contact = SEM.install(R, rt, 0.0)
            spawn, spawn_to = (lambda si, d: contact.spawn(si, d)[0]), (lambda si, t: contact.spawn_to(si, t)[0])
        else:
            spawn, spawn_to = I.native_spawn, I.native_spawn_to
        # ---- floor points around the tangency
        radii = np.geomspace(1e-8, 2e-2, 64)
        az = np.radians(np.arange(16) * 22.5 + 3.0)
        rr, aa = np.meshgrid(radii, az, indexing="ij")
        pts = np.stack([0.9 + rr.ravel() * np.cos(aa.ravel()), 0.4 + rr.ravel() * np.sin(aa.ravel()), 0 * rr.ravel()], 1)
        nrm = np.tile([[0., 0., 1.]], (len(pts), 1))
        hit = surface(pts, nrm, 0)
        ok = np.asarray(hit.is_valid())
        hp = I.P(hit.p)
        r_true = np.hypot(hp[:, 0] - 0.9, hp[:, 1] - 0.4)
        half_gap = 0.5 * (C[2] - np.sqrt(np.maximum(RAD ** 2 - r_true ** 2, 0)) - hp[:, 2])
        # NEE
        sray = spawn_to(hit, rt.lt.pos[0])
        sh = rt.scene.ray_intersect(sray)
        wl = mi.Vector3f(dr.normalize(rt.lt.pos[0] - hit.p))
        de = rt.sphere_exit(sh.p + wl * 1e-6, wl)
        so = I.P(sray.o)
        chord = I.sphere_chord(hp, np.broadcast_to(light, hp.shape))
        nee = {"origin_free": (~I.inside_sphere(so)) & (so[:, 2] > 0),
               "first_hit_sphere": np.asarray(sh.is_valid() & dr.eq(rt.shape_id(sh), 6)),
               "entry": np.asarray(dr.dot(sh.n, wl) < 0),
               "de_rel_err": np.abs(np.asarray(de) - chord) / chord}
        nee["pass"] = nee["origin_free"] & nee["first_hit_sphere"] & nee["entry"] & (nee["de_rel_err"] < 1e-3)
        # continuation / visibility for 8 upward directions
        cont_pass = np.ones(len(pts), bool)
        cont_free = np.ones(len(pts), bool)
        for c in (0.2, 0.5, 0.8, 0.98):
            for al in (0.0, np.pi):
                w = np.stack([np.sqrt(1 - c * c) * np.cos(al + aa.ravel()), np.sqrt(1 - c * c) * np.sin(al + aa.ravel()),
                              np.full(len(pts), c)], 1)
                ray = spawn(hit, Vf(w))
                y = rt.scene.ray_intersect(ray)
                o = I.P(ray.o)
                free = (~I.inside_sphere(o)) & (o[:, 2] > 0)
                front = np.asarray(y.is_valid() & dr.eq(rt.shape_id(y), 6) & (y.to_local(-Vf(w)).z > 1e-6))
                cont_free &= free
                cont_pass &= free & front
        floor = {"r_m": r_true, "half_gap_m": half_gap, "attached": ok, "nee_pass": nee["pass"],
                 "nee_origin_free": nee["origin_free"], "cont_origin_free": cont_free, "cont_pass": cont_pass}
        # ---- sphere cap points: outward visibility and SSS exit toward the floor
        heights = np.geomspace(1e-9, 1e-3, 64)
        hh, aa2 = np.meshgrid(heights, az, indexing="ij")
        theta = np.arccos(np.clip((C[2] - hh.ravel()) / RAD, -1, 1))       # polar angle from -z axis
        nrm2 = np.stack([np.sin(theta) * np.cos(aa2.ravel()), np.sin(theta) * np.sin(aa2.ravel()), -np.cos(theta)], 1)
        pts2 = C + RAD * nrm2
        hit2 = surface(pts2, nrm2, 6)
        hp2 = I.P(hit2.p)
        cap_ok = np.asarray(hit2.is_valid())
        vis_pass = np.ones(len(pts2), bool)
        vis_free = np.ones(len(pts2), bool)
        for tilt in (0.3, 0.7):
            w = nrm2 * np.sqrt(1 - tilt * tilt)
            w[:, 2] -= tilt
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            ray = spawn(hit2, Vf(w))
            y = rt.scene.ray_intersect(ray)
            o = I.P(ray.o)
            free = (~I.inside_sphere(o)) & (o[:, 2] > 0)
            front = np.asarray(y.is_valid() & dr.eq(rt.shape_id(y), 0) & (y.to_local(-Vf(w)).z > 1e-6))
            vis_free &= free
            vis_pass &= free & front
        ew = Vf(np.tile([[0.2, 0.0, -0.98]], (len(pts2), 1)) / np.linalg.norm([0.2, 0, -0.98]))
        if semantics == "candidate":
            eo, _, _ = contact.exit(hit2.p, ew)
        else:
            eo = hit2.p + ew * 1e-4
        y = rt.scene.ray_intersect(mi.Ray3f(eo, ew))
        e = I.P(eo)
        exit_free = (~I.inside_sphere(e)) & (e[:, 2] > 0)
        exit_pass = exit_free & np.asarray(y.is_valid() & dr.eq(rt.shape_id(y), 0) & (y.to_local(-ew).z > 1e-6))
        cap_half_gap_vis = 0.5 * hp2[:, 2] / np.maximum(-nrm2[:, 2], 1e-12)
        cap_half_gap_exit = 0.5 * hp2[:, 2] / (0.98 / np.linalg.norm([0.2, 0, -0.98]))
        cap = {"height_m": hp2[:, 2], "attached": cap_ok, "half_gap_vis_m": cap_half_gap_vis,
               "half_gap_exit_m": cap_half_gap_exit, "vis_origin_free": vis_free, "vis_pass": vis_pass,
               "exit_origin_free": exit_free, "exit_pass": exit_pass}
        np.savez_compressed(out / f"stress_{semantics}.npz", **{f"floor_{k}": v for k, v in floor.items()},
                            **{f"cap_{k}": v for k, v in cap.items()})

        def worst(pass_mask, gaps, valid):
            fail = valid & ~pass_mask
            return float(gaps[fail].max()) if fail.any() else 0.0
        result[semantics] = {
            "floor_nee_fail_max_half_gap_m": worst(floor["nee_pass"], half_gap, ok),
            "floor_cont_fail_max_half_gap_m": worst(floor["cont_pass"], half_gap, ok),
            "cap_vis_fail_max_half_gap_m": worst(cap["vis_pass"], cap_half_gap_vis, cap_ok),
            "cap_exit_fail_max_half_gap_m": worst(cap["exit_pass"], cap_half_gap_exit, cap_ok),
            "floor_nee_pass_fraction": float(floor["nee_pass"][ok].mean()),
            "floor_cont_pass_fraction": float(floor["cont_pass"][ok].mean()),
            "cap_vis_pass_fraction": float(cap["vis_pass"][cap_ok].mean()),
            "cap_exit_pass_fraction": float(cap["exit_pass"][cap_ok].mean()),
            "attached_floor": int(ok.sum()), "attached_cap": int(cap_ok.sum()), "samples": int(len(pts))}
        if semantics == "candidate":
            SEM.uninstall(rt)
    worst_c = max(v for k, v in result["candidate"].items() if k.endswith("_half_gap_m"))
    result["h_min_rule"] = "4 x largest candidate failing half-gap (h_min = 0 installed)"
    result["candidate_worst_failing_half_gap_m"] = worst_c
    result["h_min_m"] = 4 * worst_c
    (out / "stress.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
