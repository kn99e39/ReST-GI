"""WL69 Stage 4 arm RG (protocol xbench69_v1 + amendment 1): ReST-GI WL55 CAK (and the 1 cm P10 control) on C1, re-executed with
per-component timing. The 2.5 mm rest oracle K (slab_L4_s1 / _s2, WL55 ops, read-only) is the resident persistent state; for the
rigid T1 (floor_c4) it is reused unchanged (WL53/55: rigid K invariance). The update after an edit is:
  sun   first-order sun term of the medium domain on the current geometry (k55_compose.sun_term, WL55 seed),
  lev   CAK geometry-only level rule on the current geometry,
  agg   aggregation of the resident K onto the current cells (k55_compose.arm_operator),
  comp  explicit C build + coupled solve + ballistic camera readout (k55_compose.compose).
Predictions are compared with WL55's stored CAK / P10 predictions (same code, same seeds).

    cd <repo>/experiments/fnto_k55 && WL55=<root with ops/ link> python ../xbench69/x_rg.py <out_dir>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55"))
import k55_common as KCm  # noqa: E402
import k55_compose as KP  # noqa: E402
import k55_fixtures as KF  # noqa: E402
import k55_stageB as SB  # noqa: E402

WL55_OUT = Path.home() / "rest_wl55" / "out" / "stageB"
ARMS = {"CAK": "cak", "P10": 1}


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rec = dict(protocol="xbench69_v1", arm="RG", runs={})
    for opname in ("slab_L4_s1", "slab_L4_s2"):
        t0 = sync(); Kf = SB.load_op(opname); t_load = sync() - t0
        for scene in ("floor", "floor_c4"):
            sd = KF.scene_def(scene); keys = sorted(set(SB.cam_dirs_of(sd, SB.SCENE_CAMS[scene])))
            torch.cuda.reset_peak_memory_stats(); t0 = sync(); sun = KP.sun_term(sd, keys); t_sun = sync() - t0
            for arm, lev in ARMS.items():
                t0 = sync(); levels = SB.cak_levels(sd) if lev == "cak" else SB.uniform_levels(sd, lev); t_lev = sync() - t0
                t0 = sync(); Kops = {ci: KP.arm_operator(Kf, m, V, levels[ci]) for ci, (n, k, m, V) in enumerate(sd["comps"]) if k == "medium"}; t_agg = sync() - t0
                t0 = sync(); r = KP.compose(sd, levels, Kops, sun, keys, label=f"xb69/{scene}/{arm}"); t_comp = sync() - t0
                tag = f"{arm}_{opname[-2:]}"
                pred = {f"{key}|{ro}": v[ro] for key, v in r["pred"].items() for ro in ("nee", "ball", "bin")}
                np.savez(out / f"{scene}__{tag}.npz", **pred)
                ref = np.load(WL55_OUT / scene / f"{arm}_{opname[-2:]}.npz") if (WL55_OUT / scene / f"{arm}_{opname[-2:]}.npz").exists() else None
                dev = None
                if ref is not None:
                    dev = max(float(np.abs(pred[k] - ref[k]).max() / max(np.abs(ref[k]).max(), 1e-30)) for k in ref.files if k in pred)
                rec["runs"][f"{scene}/{tag}"] = dict(t_load_K_s=t_load, t_sun_s=t_sun, t_levels_s=t_lev, t_aggregate_s=t_agg, t_compose_s=t_comp,
                                                     t_update_s=t_sun + t_lev + t_agg + t_comp, meta={k: r["meta"].get(k) for k in ("t_C", "t_solve", "iters", "converged", "cells", "C_nnz")},
                                                     peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, max_rel_dev_vs_wl55=dev)
                print(time.strftime("%H:%M:%S"), scene, tag, json.dumps({k: v for k, v in rec["runs"][f"{scene}/{tag}"].items() if k != "meta"}), flush=True)
                del Kops, r; torch.cuda.empty_cache()
            del sun
        del Kf; torch.cuda.empty_cache()
    KCm.dump(rec, out / "rg.json"); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
