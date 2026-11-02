"""WL70 Stage 2 (physical identity) and Stage 3 (invalidation controls), protocol sunreuse70_v1.

Runs (medium comp 0, n_sun 16M, k55 Dout 64, level-4 domain; seeds are the absolute run_sun seeds):
  cache   T0 5591 through the recording replica (= the cached state B; historical seed 5591 + ci)
  T0      7001, 7002, 7003 (independent replicas of the cached state)
  T1      5591 historical TR.run_sun path (= arm A of WL69) and 5591 recording replica (replica equivalence), 7011, 7012
  controls N1_sun, N2_occluder, N3_medium, N4_rotate, P1_offpath: 7021, 7022 each
Then: contract decision of the cached state for T1 and every control (sr70_state.validate), predeclared statistics (sr70_stats), the
same-seed T1-vs-cache comparison, replica equivalence and the reproduction of WL55's stored sun terms (floor, floor_c4).

    cd <repo>/experiments/fnto_k55 && python ../sunreuse70/sr70_identity.py <out_dir>
"""
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
import sr70_scenes as SC  # noqa: E402
import sr70_stats as ST  # noqa: E402
import k55_compose as KP  # noqa: E402
import k55_ops as KO  # noqa: E402
import k55_stageB as SB  # noqa: E402

WL55_SUN = Path.home() / "rest_wl55" / "out" / "stageB"


def keys_of(sd):
    return sorted(set(SB.cam_dirs_of(sd, SB.SCENE_CAMS["floor_c4"])))


def summary(sun, rec, sd):
    """Summary arrays of comp 0: fine-port flux, coarse-port x out-bin flux, L0 and fine-port source power."""
    n, k, m, V = sd["comps"][0]; fm, _, par = KO.refined(m, V, 4)
    o = sun[0]; Pf = len(par); Pc = len(m["port_area"])
    phf = o["phi_f"].double().view(Pf, KP.DOUT, 3)
    cb = torch.zeros(Pc, KP.DOUT, 3, dtype=torch.float64).index_add_(0, torch.as_tensor(par), phf)
    out = dict(phi_fine=phf.sum(1).numpy(), phi_cbin=cb.reshape(-1, 3).numpy(), L0=o["L0"].double().numpy(), stats=o["stats"])
    if rec is not None:
        tp = torch.as_tensor(fm["tri_port"], device="cuda")[rec["tri"].long()]
        src = torch.zeros(Pf, dtype=torch.float64, device="cuda").index_add_(0, tp, rec["ok"].double() * rec["w"][0])
        assert len(set(rec["w"])) == 1
        out["src"] = np.repeat(src.cpu().numpy()[:, None], 3, 1)
    return out


def run(name, seed, hist=False):
    sd, med = SC.state(name); keys = keys_of(sd); t = SS.sync()
    if hist:
        sun = KP.sun_term(sd, keys, seed=seed); st = None; rec = None
    else:
        sun, sts = SS.sun_term_rec(sd, keys, seed=seed, medium=med); st = sts[0]; rec = st["rec"]
    dt = SS.sync() - t; s = summary(sun, rec, sd); s["seconds"] = dt
    print(time.strftime("%H:%M:%S"), name, seed, "hist" if hist else "rec", f"{dt:.1f}s", flush=True)
    return sun, st, s


def sets_for(A1, A2, B1, B2, labels_fine, labels_cbin, L0_groups):
    S = {}
    S["phi_fine"] = dict(r=ST.r_test(A1["phi_fine"], A2["phi_fine"], B1["phi_fine"], B2["phi_fine"]),
                         groups=ST.group_test(A1["phi_fine"], A2["phi_fine"], B1["phi_fine"], B2["phi_fine"], labels_fine))
    S["phi_cbin"] = dict(r=ST.r_test(A1["phi_cbin"], A2["phi_cbin"], B1["phi_cbin"], B2["phi_cbin"]))
    S["src"] = dict(r=ST.r_test(A1["src"], A2["src"], B1["src"], B2["src"]),
                    groups=ST.group_test(A1["src"], A2["src"], B1["src"], B2["src"], labels_fine))
    gl = []; rl = []
    for ki, (lab, sel) in enumerate(L0_groups):
        a1, a2, b1, b2 = (x["L0"][ki][sel] for x in (A1, A2, B1, B2))
        rl += [dict(cam=ki, **r) for r in ST.r_test(a1, a2, b1, b2)]
        gl += [dict(cam=ki, **g) for g in ST.group_test(a1, a2, b1, b2, lab[sel], mean=True)]
    S["L0"] = dict(r=rl, groups=gl)
    return S


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); res = dict(protocol="sunreuse70_v1", runs={}, decisions={}, tests={})
    torch.cuda.reset_peak_memory_stats()
    sun_c, cache, s_c = run("T0", 5591)
    res["cache"] = dict(out_hash=cache["out_hash"], mesh_hash=cache["mesh_hash"], bytes=SS.state_bytes(cache), seconds=s_c["seconds"],
                        lights_mat=cache["lights_mat"], cams_mat=cache["cams_mat"], estimator=cache["estimator"], pose_residual=cache["pose"][2])
    S = {("T0", 5591): s_c}
    for seed in (7001, 7002, 7003):
        S[("T0", seed)] = run("T0", seed)[2]
    sun_h, _, S[("T1", "5591h")] = run("T1", 5591, hist=True)
    sun_r, _, S[("T1", 5591)] = run("T1", 5591)
    for seed in (7011, 7012):
        S[("T1", seed)] = run("T1", seed)[2]
    for name in SC.CONTROLS:
        for seed in (7021, 7022):
            S[(name, seed)] = run(name, seed)[2]
    for (name, seed), s in S.items():
        res["runs"][f"{name}/{seed}"] = dict(seconds=s["seconds"], stats=s["stats"])
    np.savez_compressed(out / "summaries.npz", **{f"{n}|{sd}|{k}": v for (n, sd), s in S.items() for k, v in s.items() if isinstance(v, np.ndarray)})
    # contract decisions of the cached state
    for name in ("T1",) + SC.CONTROLS:
        sd, med = SC.state(name); acc, rep = SS.validate(cache, sd, 0, keys_of(sd), medium=med)
        res["decisions"][name] = dict(accept=bool(acc), expected=SC.EXPECT[name], **rep); print(name, "accept" if acc else "reject", rep["checks"], flush=True)
    # statistics (labels from the T0 geometry; material indices are pose independent)
    sd0, _ = SC.state("T0"); n, k, m, V = sd0["comps"][0]; _, _, par = KO.refined(m, V, 4)
    lab_f = np.asarray(m["port_face"])[par]; lab_c = np.repeat(np.asarray(m["port_face"]), KP.DOUT)
    keys = keys_of(sd0); L0g = [(np.asarray(m["port_face"]), (S[("T0", 7001)]["L0"][ki] > 0).all(1) & (S[("T0", 7002)]["L0"][ki] > 0).all(1)) for ki in range(len(keys))]
    B1, B2 = S[("T0", 7001)], S[("T0", 7002)]
    res["stats"] = {"null_T0": sets_for(S[("T0", 7003)], S[("T0", 5591)], B1, B2, lab_f, lab_c, L0g)}
    res["stats"]["T1"] = sets_for(S[("T1", 7011)], S[("T1", 7012)], B1, B2, lab_f, lab_c, L0g)
    for name in SC.CONTROLS:
        res["stats"][name] = sets_for(S[(name, 7021)], S[(name, 7022)], B1, B2, lab_f, lab_c, L0g)
    res["verdicts"] = {k_: ST.verdict(v) for k_, v in res["stats"].items()}
    res["null_calibrated"] = all(ST.null_ok(s["r"]) for s in res["stats"]["null_T0"].values())
    res["cam_keys"] = keys
    # same-seed T1 historical vs cache, replica equivalence, WL55 reproduction
    def maxrel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    res["tests"]["same_seed_T1hist_vs_cache"] = {k_: maxrel(S[("T1", "5591h")][k_], s_c[k_]) for k_ in ("phi_fine", "phi_cbin", "L0")}
    res["tests"]["replica_equals_historical_T1_5591"] = {k_: maxrel(sun_r[0][k_].numpy(), sun_h[0][k_].numpy()) for k_ in ("phi_f", "L0")}
    for scene, sun in (("floor", sun_c), ("floor_c4", sun_h)):
        p = WL55_SUN / scene / "sun.pt"
        if p.exists():
            w = torch.load(p, map_location="cpu")[0]
            res["tests"][f"reproduces_wl55_sun_{scene}"] = {k_: maxrel(sun[0][k_].numpy(), w[k_].numpy()) for k_ in ("phi_f", "L0")}
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(res, open(out / "identity.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(dict(verdicts=res["verdicts"], null=res["null_calibrated"], decisions={k_: v["accept"] for k_, v in res["decisions"].items()}, tests=res["tests"])), flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
