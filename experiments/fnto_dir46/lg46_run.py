"""WL46 LG (class III, existing FNTO regression; frozen K_k44P): one condition over the historical held-out suite.

Conditions (protocol fnto_dir46): R0 (nl 'n', loc 'E' = production), R3 (nl 'basis', loc 'E'), R4 (nl 'n', loc 'dir' at the
normal), R5 (nl 'basis', loc 'dir' toward the ray), R3x (nl 'exact', loc 'E'; diagnostic, view subset).
Arms as r1_eval: heldout_T0 S_T0 = (rel_T0, T0 port sun fractions); heldout_T1 E_upd = (rel_T1_inc, T1 fractions) and
D_stale = (rel_T0, T0 fractions) rendered with the T1 pixels. Port sun fractions: eval_vis_k44 (worklog-44 rule).
Per solve: status, iterations, seconds, timing split, n_below; spectral radius of the homogeneous map on rel_T0 (power
iteration); first-bounce Yellow sent flux (area x pi x mean over 64 cosine directions of the condition's sender value,
L = 0 state of held-out T0 view 6, vs the whole Lego's sun power) - the physical flux, not pi L(n).
Predictions: <out>/<cond>/views.npz with the r1_eval key layout (rows = valid pixels): heldout_T0_v{v}_S_T0,
heldout_T1_v{v}_E_upd, heldout_T1_v{v}_D_stale.

    python lg46_run.py <cond> <out dir> [sets=T0,T1] [views=all] [rho_iters=20]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dir46_core as D  # noqa: E402
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

CONDS = {"R0": dict(nl="n", loc_in="E"), "R3": dict(nl="basis", loc_in="E"), "R4": dict(nl="n", loc_in="dir", loc_out="n"),
         "R5": dict(nl="basis", loc_in="dir", loc_out="ray"), "R3x": dict(nl="exact", loc_in="E")}
LUM = np.array([0.2126, 0.7152, 0.0722])


@torch.no_grad()
def sent_flux(T, S, ports, k=64):
    """Area-weighted physical flux pi A mean_j L_out(q, w_j) over k cosine-distributed directions (cosine quadrature)."""
    tot = torch.zeros(3, device=T.dev)
    for s in range(0, len(ports), 1 << 12):
        q = ports[s:s + (1 << 12)]; W = D.cosine_dirs(T.nrm[q], k)
        acc = torch.zeros(len(q), 3, device=T.dev)
        for j in range(k):
            w = W[:, j].contiguous(); acc += S.ray_values(q, -w) + S.sun_lobe(q, w)
        tot += (math.pi * acc / k * T.area[q][:, None]).sum(0)
    return tot.cpu().numpy()


@torch.no_grad()
def main(cond, out, sets=("T0", "T1"), views=None, rho_iters=20):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out) / cond; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); mode = CONDS[cond]
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    rel = {k: np.load(C.OUT1 / f"rel_{k}.npz")["hit"] for k in ("T0", "T1_inc")}; vis = np.load(C.OUT1 / "eval_vis_k44.npz")
    rec = {"cond": cond, "mode": mode, "solves": {}, "gpu": torch.cuda.get_device_name(0)}; preds = {}
    t = time.time(); rec["spectral_radius_T0"] = DS.spectral_radius(T, rel["T0"], iters=rho_iters, **mode); rec["spectral_radius_seconds"] = time.time() - t
    print(cond, "rho", rec["spectral_radius_T0"]["rho_estimate"], flush=True); torch.cuda.empty_cache()
    # first-bounce Yellow flux (view 6 sun)
    px6 = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); An = np.load(C.OUT1 / "anchors.npz")
    S = DS.DirState(T, rel["T0"], vis["heldout_T0_v6_T0"].astype(np.float32), px6["v6_light"], **mode); S.prepare(True)
    yp = torch.as_tensor(np.flatnonzero((An["mat"] == 0) & (An["obj"] == 0)), device=dev); lego = An["obj"] == 0
    fl = sent_flux(T, S, yp); sunp = float((S.Esun.cpu().numpy()[lego] * An["area"][lego]).sum())
    rec["yellow_first_bounce_flux_over_lego_sun_power_RGB_view6"] = (fl / sunp).tolist(); del S; torch.cuda.empty_cache()
    print(cond, "yellow flux / lego sun", rec["yellow_first_bounce_flux_over_lego_sun_power_RGB_view6"], flush=True)
    plan = {"T0": ("heldout_T0", {"S_T0": ("T0", "T0")}), "T1": ("heldout_T1", {"E_upd": ("T1_inc", "T1"), "D_stale": ("T0", "T0")})}
    for st in sets:
        set_name, arms = plan[st]; px = np.load(C.OUT1 / f"eval_px/{set_name}.npz"); nv = int(px["nv"])
        for v in (range(nv) if views is None else views):
            val = px[f"v{v}_valid"].astype(bool)
            pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
            for arm, (rk, vk) in arms.items():
                t = time.time()
                S = DS.DirState(T, rel[rk], vis[f"{set_name}_v{v}_{vk}"].astype(np.float32), px[f"v{v}_light"], **mode).solve()
                pr, parts = SV.render_pixels(T, S, pxd)
                preds[f"{set_name}_v{v}_{arm}"] = torch.cat([pr, parts.reshape(len(pr), -1)], 1).cpu().numpy().astype(np.float16)
                rec["solves"][f"{set_name}_v{v}_{arm}"] = {"status": S.status, "iters": S.iters, "seconds": time.time() - t, "timing": dict(S.timing),
                                                           "n_below": S.n_below, "hist": [float(x) for x in S.hist]}
                print(cond, set_name, v, arm, S.status, S.iters, round(time.time() - t, 1), flush=True)
                del S; torch.cuda.empty_cache()
            np.savez(out / "views.npz", **preds)
            (out / "run.json").write_text(json.dumps(rec, indent=1))
    rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0; rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
    (out / "run.json").write_text(json.dumps(rec, indent=1)); print("done", cond, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], tuple(a[2].split(",")) if len(a) > 2 else ("T0", "T1"),
         None if len(a) <= 3 or a[3] == "all" else [int(x) for x in a[3].split(",")], int(a[4]) if len(a) > 4 else 20)
