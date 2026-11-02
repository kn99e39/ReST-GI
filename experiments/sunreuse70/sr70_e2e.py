"""WL70 Stage 5 end-to-end (protocol sunreuse70_v1): the WL69 C1 CAK update T0 floor -> T1 floor_c4 with
  A  historical retraced sun (k55_compose.sun_term, unchanged) + CAK levels + aggregation + compose        (= WL69 x_rg path)
  B  resident cached T0 sun state: contract validation (sr70_state.validate) + reuse + the same levels / aggregation / compose
for both rest K seeds (slab_L4_s1/_s2, resident). CUDA-synchronised per-stage timing; interleaved repetitions A B B A B B A B; one-time
costs (K load, cache construction at T0) reported separately; predictions saved per arm (first repetition) with every repetition hashed.

    cd <repo>/experiments/fnto_k55 && python ../sunreuse70/sr70_e2e.py <out_dir>
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sr70_state as SS  # noqa: E402
import k55_compose as KP  # noqa: E402
import k55_fixtures as KF  # noqa: E402
import k55_stageB as SB  # noqa: E402

WL69_RG = Path.home() / "rest_xbench69" / "rg"
ORDER = ("A", "B", "B", "A", "B", "B", "A", "B")


def keys_of(sd, scene):
    return sorted(set(SB.cam_dirs_of(sd, SB.SCENE_CAMS[scene])))


def pred_arrays(r):
    return {f"{key}|{ro}": v[ro] for key, v in r["pred"].items() for ro in ("nee", "ball", "bin")}


def phash(p):
    h = hashlib.sha256()
    for k in sorted(p):
        h.update(k.encode()); h.update(np.ascontiguousarray(p[k]).tobytes())
    return h.hexdigest()


def update(arm, sd, keys, Kf, state):
    """One T1 update; returns (prediction arrays, timing dict, validation report)."""
    tm = {}; torch.cuda.reset_peak_memory_stats(); t_all = SS.sync(); rep = None
    if arm == "A":
        t = SS.sync(); sun = KP.sun_term(sd, keys); tm["sun_s"] = SS.sync() - t
    else:
        t = SS.sync(); acc, rep = SS.validate(state, sd, 0, keys); tm["validate_s"] = SS.sync() - t
        t = SS.sync()
        sun = {0: SS.reuse(state, keys)} if acc else KP.sun_term(sd, keys)
        tm["reuse_s" if acc else "fallback_sun_s"] = SS.sync() - t
    t = SS.sync(); levels = SB.cak_levels(sd); tm["levels_s"] = SS.sync() - t
    t = SS.sync(); Kops = {ci: KP.arm_operator(Kf, m, V, levels[ci]) for ci, (n, k, m, V) in enumerate(sd["comps"]) if k == "medium"}; tm["aggregate_s"] = SS.sync() - t
    t = SS.sync(); r = KP.compose(sd, levels, Kops, sun, keys, label=f"wl70/{arm}"); tm["compose_s"] = SS.sync() - t
    tm["total_s"] = SS.sync() - t_all; tm["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    tm["meta"] = {k: r["meta"].get(k) for k in ("t_C", "t_solve", "iters", "converged", "cells", "C_nnz", "rho")}
    p = pred_arrays(r); del Kops, r, sun; torch.cuda.empty_cache()
    return p, tm, rep


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rec = dict(protocol="sunreuse70_v1", runs={})
    sd0 = KF.scene_def("floor"); sd1 = KF.scene_def("floor_c4"); k0, k1 = keys_of(sd0, "floor"), keys_of(sd1, "floor_c4")
    for opname in ("slab_L4_s1", "slab_L4_s2"):
        tag = opname[-2:]; R = rec["runs"][tag] = dict(reps=[])
        t = SS.sync(); Kf = SB.load_op(opname); R["t_load_K_s"] = SS.sync() - t
        # one-time cache construction at T0 (the T0 sun term through the recording replica) and the T0 prediction
        torch.cuda.reset_peak_memory_stats(); t = SS.sync(); sun0, states = SS.sun_term_rec(sd0, k0); R["t_cache_build_s"] = SS.sync() - t
        R["cache_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9; st = states[0]; R["cache_bytes"] = SS.state_bytes(st); R["cache_out_hash"] = st["out_hash"]
        lev0 = SB.cak_levels(sd0); Kops0 = {ci: KP.arm_operator(Kf, m, V, lev0[ci]) for ci, (n, k, m, V) in enumerate(sd0["comps"]) if k == "medium"}
        p0 = pred_arrays(KP.compose(sd0, lev0, Kops0, sun0, k0, label="wl70/T0")); del Kops0; torch.cuda.empty_cache()
        np.savez(out / f"floor__CAK_{tag}.npz", **p0)
        w = np.load(WL69_RG / f"floor__CAK_{tag}.npz"); R["T0_maxrel_vs_wl69"] = max(float(np.abs(p0[k] - w[k]).max() / max(np.abs(w[k]).max(), 1e-30)) for k in w.files)
        first = {}
        for i, arm in enumerate(ORDER):
            p, tm, rep = update(arm, sd1, k1, Kf, st)
            R["reps"].append(dict(i=i, arm=arm, timing=tm, validation=rep, pred_hash=phash(p)))
            print(time.strftime("%H:%M:%S"), tag, i, arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm.items() if k != "meta"}), flush=True)
            if arm not in first:
                first[arm] = p; np.savez(out / f"floor_c4__CAK{arm}_{tag}.npz", **p)
        w = np.load(WL69_RG / f"floor_c4__CAK_{tag}.npz")
        R["T1_A_maxrel_vs_wl69"] = max(float(np.abs(first["A"][k] - w[k]).max() / max(np.abs(w[k]).max(), 1e-30)) for k in w.files)
        R["T1_B_vs_A_maxrel"] = {k: float(np.abs(first["B"][k] - first["A"][k]).max() / max(np.abs(first["A"][k]).max(), 1e-30)) for k in first["A"]}
        R["reps_identical"] = {a: len({x["pred_hash"] for x in R["reps"] if x["arm"] == a}) == 1 for a in ("A", "B")}
        del Kf, st, states, sun0; torch.cuda.empty_cache()
    json.dump(rec, open(out / "e2e.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
