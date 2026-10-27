"""WL52 Stage G - matched-hardware cost and scaling of explicit C0 (efficient GPU path) vs learned C1 (protocol fnto_lc52).

Per configuration and scale: C0 update = Mitsuba scene / BVH build + 128-gather relation table (trace + hit -> port map, on
device); C1 update = patch encoder + dense receiver x patch transfer F; per-iteration application (C0 table gather-sum vs C1
patch means + F @ Lbar); complete coupled solve with the Lambertian K; frame = update + solve + camera port lookup; memory of
the C state. Scales: object count (2 / 4 / 8 / 16 boxes, base ports) and port resolution (PORT x 2, x 1, x 1/2 with the same
configuration). Plus the production Lego mesh BVH build (2.1 M triangles) for context. 3 warm-up + 5 timed repetitions,
torch.cuda.synchronize around every timed region.

    python lc_bench.py <ckpt> <out dir>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import lc_eval as LE  # noqa: E402
import lc_model as LM  # noqa: E402
import lc_scene as LS  # noqa: E402
import lc_trace as TR  # noqa: E402

DEV = torch.device("cuda")


def timed(f, reps=5, warm=3):
    for _ in range(warm):
        f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        r = f()
    torch.cuda.synchronize(); return (time.time() - t) / reps, r


def bench_cfg(model, cfg):
    S = LS.build(cfg); r = {"N": S.N, "M": S.M, "tris": int(len(S.F)), "n_objects": len(cfg["objects"])}
    r["C0_bvh_build_s"], G = timed(lambda: TR.Geo(S, DEV))
    r["C0_gather128_s"], (H0, _) = timed(lambda: TR.gather(G, 128, 3))
    r["C1_encode_transfer_s"], F1 = timed(lambda: LE.learned_F(model, G))
    with torch.no_grad():
        z = model.encode(G.pcen, G.pnrm, G.parea)
    r["C1_encode_only_s"], _ = timed(lambda: model.encode(G.pcen, G.pnrm, G.parea))
    L = torch.rand(S.N, 3, device=DEV)
    r["C0_apply_s"], _ = timed(lambda: TR.apply_table(H0, L), reps=20)
    r["C1_apply_s"], _ = timed(lambda: F1 @ TR.patch_mean(G, L), reps=20)
    Es = TR.sun_irr(G, cfg["sun"]); r["sun_irr_s"], _ = timed(lambda: TR.sun_irr(G, cfg["sun"]))
    r["C0_solve_s"], s0 = timed(lambda: TR.solve(G, Es, lambda L: TR.apply_table(H0, L)), reps=3, warm=1)
    r["C1_solve_s"], s1 = timed(lambda: TR.solve(G, Es, lambda L: F1 @ TR.patch_mean(G, L)), reps=3, warm=1)
    r["C0_iters"], r["C1_iters"] = s0["iters"], s1["iters"]
    r["lookup_s"], _ = timed(lambda: TR.camera_ports(G, *TR.CAMS[0], res=512))
    r["C0_update_s"] = r["C0_bvh_build_s"] + r["C0_gather128_s"]; r["C1_update_s"] = r["C1_encode_transfer_s"]
    r["C0_path_s"] = r["C0_update_s"] + r["C0_iters"] * r["C0_apply_s"]; r["C1_path_s"] = r["C1_update_s"] + r["C1_iters"] * r["C1_apply_s"]
    # frame: C1 still needs the BVH for sun visibility and the camera lookup (only C's relation table is replaced)
    r["C0_frame_s"] = r["C0_bvh_build_s"] + r["sun_irr_s"] + r["C0_gather128_s"] + r["C0_solve_s"] + r["lookup_s"]
    r["C1_frame_s"] = r["C0_bvh_build_s"] + r["sun_irr_s"] + r["C1_encode_transfer_s"] + r["C1_solve_s"] + r["lookup_s"]
    r["mem_bytes"] = {"C0_table_int64": int(H0.numel() * 8), "C1_F_fp32": int(F1.numel() * 4)}
    r["speedup_C_path"] = r["C0_path_s"] / r["C1_path_s"]; r["speedup_frame"] = r["C0_frame_s"] / r["C1_frame_s"]
    return r


def main(ckpt, outd):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True); model = LM.GLT().to(DEV); model.load_state_dict(torch.load(ckpt, map_location=DEV)); model.eval()
    rec = {"gpu": torch.cuda.get_device_name(0), "object_scale": {}, "port_scale": {}}
    for nb in (2, 4, 8, 16):
        cfg = LS.sample_config(700000 + nb, n_box=(nb, nb), n_arm=1, n_sheet=1)
        rec["object_scale"][nb] = bench_cfg(model, cfg); print("boxes", nb, json.dumps({k: v for k, v in rec["object_scale"][nb].items() if k != "mem_bytes"}), flush=True)
    base = LS.PORT
    for f in (2.0, 1.0, 0.5):
        LS.PORT = base * f; cfg = LS.sample_config(700100, n_box=(3, 3), n_arm=1, n_sheet=1)
        try:
            rec["port_scale"][f] = bench_cfg(model, cfg)
        except torch.OutOfMemoryError as e:
            rec["port_scale"][f] = {"error": "OOM", "detail": str(e)[:200]}
        print("port x", f, json.dumps({k: v for k, v in rec["port_scale"][f].items() if k != "mem_bytes"}), flush=True); torch.cuda.empty_cache()
    LS.PORT = base
    import r1_common as C
    m = np.load(C.MESH_A); mi, dr = TR.mi_init()

    def lego():
        me = mi.Mesh("A", len(m["V"]), len(m["F"])); p = mi.traverse(me); p["vertex_positions"] = mi.Float(m["V"].astype(np.float32).ravel())
        p["faces"] = mi.UInt32(m["F"].astype(np.uint32).ravel()); p.update(); sc = mi.load_dict({"type": "scene", "A": me}); dr.eval(); return sc
    rec["lego_bvh_build_s"], _ = timed(lego, reps=3, warm=1); rec["lego_tris"] = int(len(m["F"]))
    rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9; (outd / "bench.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({"lego_bvh_build_s": rec["lego_bvh_build_s"]}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
