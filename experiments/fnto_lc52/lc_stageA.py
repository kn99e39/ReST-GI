"""WL52 Stage A - cost attribution of the production explicit C (LabServer69, RTX 3090).

    python lc_stageA.py relations <out dir>   (8dna26 venv: Mitsuba 3.5.1 cuda_ad_rgb)
        production gather on a deterministic port subset, split into ray setup (numpy) / Mitsuba trace (incl. host
        transfer) / hit -> port mapping; an on-device trace-throughput probe (rays generated and traced in drjit, no
        host copies) = the irreducible ray cost; the production incremental T0 -> T1 update (select / trace + map)
    python lc_stageA.py solve <out dir>       (rna venv: torch)
        one held-out T0 view (6), K_k44P: per-iteration split into C read (aggregates: gather sum / bin scatter),
        C write (I[hit] gather into the gather table), K (outgoing), full step; complete solve; table memory
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in ("fnto_lego_r1", "fnto_lego"):
    sys.path.insert(0, str(HERE.parent / p))
import r1_common as C  # noqa: E402


def relations(out):
    import fnto_relations as R
    import r1_relations as RR
    A = RR.load(); geo = R.Geo(*C.offsets("T0")); mi, dr = geo.mi, geo.dr
    idx = np.sort(np.random.default_rng(5200).choice(len(A["obj"]), 100000, replace=False)); rec = {"ports_subset": len(idx), "ports_total": int(len(A["obj"])), "K": C.K_GATHER}
    tt = {"setup": 0.0, "trace": 0.0, "map": 0.0}; off = RR.offs(A, geo)
    for s in range(0, len(idx), 40000):
        ii = idx[s:s + 40000]
        t = time.time(); o, d = RR.ray_setup(A, ii); o = (o + off[ii][:, None]).reshape(-1, 3); d = d.reshape(-1, 3); tt["setup"] += time.time() - t
        t = time.time(); valid, p, prim, sid, tv = geo.intersect(o, d); tt["trace"] += time.time() - t
        t = time.time(); RR.map_hits(A, geo, valid, p, prim, sid, d); tt["map"] += time.time() - t
    rays = len(idx) * C.K_GATHER
    rec["production_gather_subset_s"] = tt; rec["production_gather_full_extrapolated_s"] = {k: v * len(A["obj"]) / len(idx) for k, v in tt.items()}
    rec["production_rays_per_s_trace_incl_host"] = rays / tt["trace"]
    # on-device probe: random origins on the scene bbox, cosine-free random directions; trace only
    n = 1 << 24; bb = geo.scene.bbox(); lo, hi = np.array(bb.min), np.array(bb.max)
    sampler_o = mi.Point3f(*[mi.Float(np.random.default_rng(k).uniform(lo[k], hi[k], n).astype(np.float32)) for k in range(3)])
    v = np.random.default_rng(9).normal(size=(n, 3)).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
    dvec = mi.Vector3f(*[mi.Float(v[:, k]) for k in range(3)]); ray = mi.Ray3f(sampler_o, dvec)
    si = geo.scene.ray_intersect(ray); dr.eval(si.t); dr.sync_thread()
    t = time.time()
    for _ in range(5):
        si = geo.scene.ray_intersect(ray); dr.eval(si.t, si.prim_index); dr.sync_thread()
    dt = (time.time() - t) / 5; rec["device_trace_probe"] = {"rays": n, "s": dt, "rays_per_s": n / dt, "full_table_extrapolated_s": C.K_GATHER * len(A["obj"]) / (n / dt)}
    z0 = np.load(C.OUT1 / "rel_T0.npz"); t = time.time()
    _, _, inc = RR.incremental(A, R.Geo(*C.offsets("T1")), z0["hit"], z0["t"], 1, C.offsets("T0")[1], C.offsets("T1")[1])
    inc["total_s"] = time.time() - t; rec["incremental_T0_T1"] = inc
    rec["table_bytes"] = {"hit_int32": int(z0["hit"].nbytes), "t_float32": int(z0["t"].nbytes)}
    (Path(out) / "stageA_relations.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


def solve(out):
    import torch
    import r1_solve as SV
    from r1_eval import load_kernels
    dev = torch.device("cuda"); KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); v = 6
    S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]); S.solve()
    sync = torch.cuda.synchronize; reps = 10; r = {}

    def tm(f):
        sync(); t = time.time()
        for _ in range(reps):
            f()
        sync(); return (time.time() - t) / reps
    r["C_read_aggregates_s"] = tm(lambda: S.aggregates())
    Eind, h = S.aggregates(); I = S.outgoing(Eind, h).half()
    r["K_outgoing_s"] = tm(lambda: S.outgoing(Eind, h))

    def write():
        for s in range(0, T.N, 1 << 16):
            sl = slice(s, s + (1 << 16)); S.L[sl] = I[S.Hc[sl]] * S.valid[sl][..., None] + S.spec[sl]
    r["C_write_gather_s"] = tm(write)
    r["step_s"] = tm(lambda: S.step())
    S2 = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]); sync(); t = time.time(); S2.solve(); sync()
    r["solve_s"] = time.time() - t; r["solve_iters"] = S2.iters
    r["state_bytes"] = {"L_fp16": S.L.numel() * 2, "spec_fp16": S.spec.numel() * 2, "Hc_int64": S.Hc.numel() * 8, "valid_bool": S.valid.numel()}
    r["ports"] = T.N; r["gpu"] = torch.cuda.get_device_name(0)
    (Path(out) / "stageA_solve.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r, indent=1))


if __name__ == "__main__":
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    {"relations": relations, "solve": solve}[sys.argv[1]](sys.argv[2])
