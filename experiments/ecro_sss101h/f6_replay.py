"""WL101H stage A: read-only per-sample GPU replay of the native sphere/floor contact semantics (F6).

python f6_replay.py WL101_ROOT OUT_DIR [native|candidate]

Part I  : the exact WL101G construction proposals of replica 0 (seed 10203101) on floor patch 155,
          re-traced with the unchanged launch layout so that per-sample radiance must reproduce the
          stored construction packet.
Part II : a designed radial grid around the tangency (0.9, 0.4, 0) with fixed outgoing directions,
          NEE intermediates, continuation and visibility outcomes and repeated probe traces.
"""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402  (inserts WL101A-F paths, WL101F input validation)
import gpu101g as GG  # noqa: E402
from oracle_field import ExposedOracle  # noqa: E402
import instrument101h as I  # noqa: E402

ROOT, OUT = Path(argv[1]).resolve(), Path(argv[2]).resolve()
MODE = argv[3] if len(argv) > 3 else "native"
PATCH = 155
SEED, LAUNCH_SPP = 10203101, 32
RADII_MM = [0.3, 0.6, 1, 2, 4, 6, 8, 10, 11, 11.5, 12, 12.5, 14, 20, 40, 80]
AZ = np.radians(np.arange(8) * 45 + 11.25)
COS = [0.2, 0.5, 0.8, 0.95]
ALPHA = np.radians([0, 90, 180, 270])
REPEATS = 256


def main():
    reader, learned, *_ = RG.F101.inputs(ROOT)
    R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
    OUT.mkdir(parents=True, exist_ok=False)
    if MODE == "candidate":
        import semantics101h as SEM
        SEM.install(R, rt, json.loads((HERE / "protocol/qualification.json").read_text())["candidate"]["h_min_m"])
        spawn_to, spawn = SEM.candidate_spawn_to(R, rt), SEM.candidate_spawn(R, rt)
    else:
        spawn_to, spawn = I.native_spawn_to, I.native_spawn
    oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
    tol = 3e-5
    summary = {"mode": MODE, "patch": PATCH}

    # ---------------------------------------------------------------- Part I
    rows = {k: [] for k in ("p", "n", "w", "bin", "s", "t", "L", "f2", "fh", "vis_front", "vis_hit", "vis_shape",
                            "vis_o", "cont_o", "through", "through_orig", "shadow_o", "sh_shape", "first_hit_dot", "de",
                            "cross", "clear", "f", "light_d2")}
    sums = np.zeros((64, 4, 4, 3))
    counts = np.zeros((64, 4, 4))
    for launch in range(128):
        slots, ray, hits, rng, sd, info = oracle.probes(LAUNCH_SPP, SEED, launch)
        out, count = oracle.trace(ray, hits, rng, sd, info["valid"], "probe")
        sel = info["valid"] & (slots // 64 == PATCH)
        ids = dr.compress(sel)
        g = lambda x: dr.gather(type(x), x, ids)
        hit, r, req = g(hits), g(ray), dr.gather(R.U, slots, ids)
        dr.eval(hit, r, req)
        q = SimpleNamespace(point=hit.p, ray=r, pid=req // 64, bn=req % 64)
        slot, s, t, _ = reader.addresses(q, tol)
        w = -r.d
        f2 = I.P([dr.gather(R.F, out[f"f2{c}"], ids) for c in range(3)])
        fh = I.P([dr.gather(R.F, out[f"fh{c}"], ids) for c in range(3)])
        # Native k = 1 state of the probe trace at the injected hit.
        wi = hit.to_local(w)
        active = mi.Bool(True) & (wi.z > 1e-6)
        mid = dr.gather(R.U, rt.PMATU, q.pid)
        cl = R.PT.CD.DrClosure83("ER", rt.mats.states, mid)
        not_sss = ~(active & dr.eq(mid, 2))
        _, through, rec = I.record_direct(R, rt, hit, wi, cl, rt.lt, active, not_sss, spawn_to)
        _, through_orig = rt.direct(hit, wi, cl, rt.lt, active, not_sss)
        vray = spawn(hit, w)
        y = rt.scene.ray_intersect(vray)
        rows["p"].append(I.P(hit.p)); rows["n"].append(I.P(hit.n)); rows["w"].append(I.P(w))
        rows["bin"].append(np.asarray(q.bn)); rows["s"].append(np.asarray(s)); rows["t"].append(np.asarray(t))
        rows["f2"].append(f2); rows["fh"].append(fh); rows["L"].append(f2 + fh)
        rows["vis_hit"].append(np.asarray(y.is_valid()))
        rows["vis_front"].append(np.asarray(y.is_valid() & (y.to_local(-w).z > 1e-6)))
        rows["vis_shape"].append(np.asarray(rt.shape_id(y)))
        rows["vis_o"].append(I.P(vray.o)); rows["cont_o"].append(I.P(spawn(hit, hit.n).o))
        rows["through"].append(I.P(through)); rows["through_orig"].append(I.P(through_orig))
        for k in ("shadow_o", "sh_shape", "first_hit_dot", "de", "cross", "clear", "f", "light_d2"):
            rows[k].append(rec[k])
        Ls = rows["L"][-1]
        np.add.at(sums, (rows["bin"][-1], rows["s"][-1], rows["t"][-1]), Ls)
        np.add.at(counts, (rows["bin"][-1], rows["s"][-1], rows["t"][-1]), 1)
    part1 = {k: np.concatenate(v) for k, v in rows.items()}
    part1["sums"], part1["counts"] = sums, counts
    np.savez_compressed(OUT / "part1_construction_samples.npz", **part1)
    a, b = part1["through"], part1["through_orig"]
    pattern = bool(np.array_equal(a == 0, b == 0))
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    if MODE == "native":
        # Same branch decisions; values equal to float32 kernel-fusion rounding.
        assert pattern and rel <= 1e-5, ("record_direct differs from pt100.direct", pattern, rel)
    summary["part1"] = {"samples": int(len(part1["bin"])), "through_zero_pattern_equal": pattern,
                        "through_max_relative_difference": rel}

    # ---------------------------------------------------------------- Part II
    pts, dirs, meta = [], [], []
    for ri, r_mm in enumerate(RADII_MM):
        for ai, az in enumerate(AZ):
            p = np.array([0.9 + 1e-3 * r_mm * np.cos(az), 0.4 + 1e-3 * r_mm * np.sin(az), 0.0])
            for ci, c in enumerate(COS):
                for bi, al in enumerate(ALPHA):
                    sn = np.sqrt(1 - c * c)
                    pts.append(p); dirs.append([sn * np.cos(al + az), sn * np.sin(al + az), c])
                    meta.append((ri, ai, ci, bi))
    pts, dirs, meta = np.array(pts, np.float32), np.array(dirs, np.float32), np.array(meta)
    n_cfg = len(pts)
    lane = np.repeat(np.arange(n_cfg), REPEATS)
    Pf = lambda a: mi.Point3f(*[R.F(np.ascontiguousarray(a[:, k])) for k in range(3)])
    Vf = lambda a: mi.Vector3f(*[R.F(np.ascontiguousarray(a[:, k])) for k in range(3)])
    p_all, w_all = pts[lane], dirs[lane]
    nrm = np.tile(np.array([[0, 0, 1]], np.float32), (len(lane), 1))
    attach = mi.Ray3f(Pf(p_all + 0.002 * nrm), Vf(-nrm))
    attach.maxt = 0.003
    hits = oracle.shape_scenes[0].ray_intersect(attach)
    dr.eval(hits)            # one OptiX pipeline per kernel: materialise the single-shape attachment first
    ok = hits.is_valid() & (hits.to_local(Vf(w_all)).z > 1e-6)
    dr.eval(ok)
    ray = mi.Ray3f(Pf(p_all + 0.002 * w_all), Vf(-w_all))
    sd = dr.opaque(mi.UInt64, 10204001)
    rng = mi.PCG32(size=len(lane), initstate=sd, initseq=mi.UInt64(0))
    out, okp = oracle.trace(ray, hits, rng, sd, ok, "probe")
    f2 = I.P([out[f"f2{c}"] for c in range(3)]); fh = I.P([out[f"fh{c}"] for c in range(3)])
    first = np.arange(n_cfg) * REPEATS
    idx = R.U(first.astype(np.uint32))
    g = lambda x: dr.gather(type(x), x, idx)
    h1 = g(hits); w1 = Vf(dirs)
    dr.eval(h1, w1)
    wi = h1.to_local(w1)
    active = mi.Bool(True) & (wi.z > 1e-6)
    mid = dr.gather(R.U, rt.PMATU, rt.patch_bin(h1, R.U(np.zeros(n_cfg, np.uint32)), w1)[0])
    cl = R.PT.CD.DrClosure83("ER", rt.mats.states, mid)
    not_sss = ~(active & dr.eq(mid, 2))
    _, through, rec = I.record_direct(R, rt, h1, wi, cl, rt.lt, active, not_sss, spawn_to)
    vray = spawn(h1, w1)
    y = rt.scene.ray_intersect(vray)
    cray = spawn(h1, w1)          # continuation origin for the same outgoing direction
    y2 = rt.scene.ray_intersect(cray)
    light = I.P(rt.lt.pos[0]).reshape(-1, 3)[0]
    part2 = {"meta": meta, "points": pts, "dirs": dirs, "radii_mm": np.array(RADII_MM),
             "f2_mean": f2.reshape(n_cfg, REPEATS, 3).mean(1), "fh_mean": fh.reshape(n_cfg, REPEATS, 3).mean(1),
             "f2_sd": f2.reshape(n_cfg, REPEATS, 3).std(1), "fh_sd": fh.reshape(n_cfg, REPEATS, 3).std(1),
             "valid": np.asarray(okp).reshape(n_cfg, REPEATS).mean(1),
             "hit_p": I.P(h1.p), "vis_o": I.P(vray.o), "vis_hit": np.asarray(y.is_valid()),
             "vis_front": np.asarray(y.is_valid() & (y.to_local(-w1).z > 1e-6)), "vis_shape": np.asarray(rt.shape_id(y)),
             "cont_hit_shape": np.asarray(rt.shape_id(y2)), "cont_front": np.asarray(y2.is_valid() & (y2.to_local(-w1).z > 1e-6)),
             "through": I.P(through), "light": light, **{f"nee_{k}": v for k, v in rec.items()}}
    part2["origin_inside_sphere_vis"] = I.inside_sphere(part2["vis_o"])
    part2["origin_inside_sphere_nee"] = I.inside_sphere(part2["nee_shadow_o"])
    part2["chord_physical"] = I.sphere_chord(part2["hit_p"], np.broadcast_to(light, part2["hit_p"].shape))
    np.savez_compressed(OUT / "part2_radial_grid.npz", **part2)
    summary["part2"] = {"configurations": n_cfg, "repeats": REPEATS, "light": light.tolist()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
