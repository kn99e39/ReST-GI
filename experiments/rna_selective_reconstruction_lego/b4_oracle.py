"""Architecture Batch 4 (worklog 37): GT-only pilot (P1-P5) and oracle dirty-unit set D* (protocol b4_v1; RNA venv, CPU).

    python b4_oracle.py --data <dir with oracle_{T0,T1,T1b}_s{1,2}_val.h5> --out <dir>

Counted pixels of unit i (A: pair T0/T1 view v; B: pair T0/T1b view v): alpha >= 0.999 in both states and both
realisations, same owner unit in both, canonical positions within 1 mm, visibility equal in both states and uniform in
its 3x3 neighbourhood in both (no shadow-edge pixels).  d_k = T(L_T1,k) - T(L_T0,k), T = code tonemap, realisation
k = 1, 2.  MS_i = sum(d_1 d_2) / (3 n_i) (unbiased squared change), RMS_i = sqrt(max(MS_i, 0)); bootstrap over views.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_model as E  # noqa: E402

TAU = 0.005
MIN_PX = 2000
POS_TOL = 1e-3
NEAR = 0.15
TILE = 8
TILE_MIN = 32
NBOOT = 2000
K = 17


def T(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def uniform3(vis, res):
    """True where the 3x3 neighbourhood (inside the image) has one visibility value."""
    v = vis.reshape(res, res).astype(np.int8); mn = v.copy(); mx = v.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.roll(v, (dy, dx), (0, 1)); mn = np.minimum(mn, s); mx = np.maximum(mx, s)
    u = mn == mx; u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = False
    return u.ravel()


def accumulate(d1, d2, unit, cnt, S, NOISE, NAIVE, NPX):
    """Per-unit sums for one view: cross product d1*d2 (unbiased squared change), noise of the realisation mean,
    naive squared mean change (biased by noise), element count (pixels x 3 channels)."""
    for arr, w in ((S, (d1 * d2).sum(1)), (NOISE, (((d1 - d2) / 2) ** 2).sum(1)), (NAIVE, (((d1 + d2) / 2) ** 2).sum(1))):
        np.add.at(arr, unit[cnt], w[cnt])
    np.add.at(NPX, unit[cnt], 3)


def depth_edge(P, m, res):
    """Amendment b4_v1.1: a covered pixel is a depth-edge pixel when a 4-neighbour is uncovered or its AOV position is
    more than 3 x the view's median nearest-neighbour spacing away (the pixel filter then averages positions of
    different surfaces, so the AOV position lies off every surface)."""
    Pi = P.reshape(res, res, 3); ai = m.reshape(res, res); nb = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        an = np.roll(ai, (dy, dx), (0, 1))
        nb.append(np.where(an, np.linalg.norm(Pi - np.roll(Pi, (dy, dx), (0, 1)), axis=-1), 1e3))
    nb = np.stack(nb); fp = np.median(nb.min(0)[ai])
    return (nb.max(0) > 3 * fp).ravel()


def box_dist(p, lo, hi):
    return np.linalg.norm(p - np.clip(p, lo, hi), axis=1)


def boot_ci(num, den, nboot=NBOOT, seed=0, f=lambda n, d: n / np.maximum(d, 1)):
    """num, den: (views, ...) per-view sums -> (point, lo5, hi95) of f(sum num, sum den) under view resampling."""
    rng = np.random.default_rng(seed); nv = num.shape[0]
    pt = f(num.sum(0), den.sum(0))
    bs = np.stack([f(num[i].sum(0), den[i].sum(0)) for i in (rng.integers(0, nv, nv) for _ in range(nboot))])
    return pt, np.quantile(bs, 0.05, axis=0), np.quantile(bs, 0.95, axis=0), bs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="oracle"); ap.add_argument("--keep-views", type=int, nargs="*", default=[0, 1, 2, 3])
    args = ap.parse_args()
    import h5py

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    H = {(s, k): h5py.File(data / f"{args.prefix}_{s}_s{k}_val.h5", "r") for s in ("T0", "T1", "T1b") for k in (1, 2)}
    rec = {"protocol": "b4_v1", "tau": TAU, "files": {f"{s}_s{k}": str(data / f"{args.prefix}_{s}_s{k}_val.h5") for (s, k) in H}}
    # pairing check: identical lights and cameras across states and realisations
    ld0 = H[("T0", 1)]["light_dir"][:, 0, :]
    cams = {f"{s}_s{k}": json.loads((data / f"{args.prefix}_{s}_s{k}_val.json").read_text())["cameras"] for (s, k) in H}
    c0 = np.array([c["mtx"] for c in cams["T0_s1"]])
    rec["pairing"] = {n: {"light_max_abs_diff": float(np.abs(H[key]["light_dir"][:, 0, :].astype(np.float32) - ld0.astype(np.float32)).max()),
                          "camera_max_abs_diff": float(np.abs(np.array([c["mtx"] for c in cams[n]]) - c0).max())}
                      for n, key in ((f"{s}_s{k}", (s, k)) for (s, k) in H)}
    nv = H[("T0", 1)]["color"].shape[0]; res = int(H[("T0", 1)].attrs["resolution"][0])
    t0 = time.time()
    owners = {s: E.make_owner(s) for s in ("T0", "T1", "T1b")}
    rec["owner_init_s"] = time.time() - t0
    scene = E.SCENE; lo0, hi0 = map(np.array, scene["B_mesh"]["aabb_T0"]); dlt = np.array(scene["delta"])
    # per-view, per-unit accumulators
    S = np.zeros((nv, K)); NPX = np.zeros((nv, K)); NOISE = np.zeros((nv, K)); NAIVE = np.zeros((nv, K))
    VISCH = np.zeros((nv, K)); CORR = np.zeros((nv, K)); CH = np.zeros((nv, K, 3)); CHN = np.zeros((nv, K))
    near = {"T1": np.zeros((nv, 2)), "T0": np.zeros((nv, 2))}
    ident = {"A": np.zeros((nv, 2)), "B": np.zeros((nv, 2))}
    dists = {s: [] for s in owners}
    dists_int = {s: [] for s in owners}  # amendment b4_v1.1: non-depth-edge pixels
    tiles = {i: [] for i in range(K)}
    keep = {}
    for v in range(nv):
        st = {}
        for s in ("T0", "T1", "T1b"):
            g = lambda k, key: np.nan_to_num(H[(s, k)][key][v].astype(np.float32))
            a1, a2 = g(1, "alpha")[:, 0], g(2, "alpha")[:, 0]
            m = (a1 >= 0.999) & (a2 >= 0.999)
            P, N, vis = g(1, "position").astype(np.float64), g(1, "normal").astype(np.float64), g(1, "diffuse_direct")[:, 0] > 0
            lab = -np.ones(res * res, np.int64); can = np.full((res * res, 3), np.nan)
            l_, d_ = owners[s](P[m], N[m]); lab[m] = l_; can[m] = owners[s].canonical(P[m], l_); dists[s].append(d_)
            dists_int[s].append(d_[~depth_edge(P, m, res)[m]])
            st[s] = {"m": m, "lab": lab, "can": can, "vis": vis, "u3": uniform3(vis, res), "c1": g(1, "color"), "c2": g(2, "color"), "P": P}
        for obj, s1 in (("A", "T1"), ("B", "T1b")):
            a, b = st["T0"], st[s1]
            isobj = (lambda L: (L >= 0) & (L < E.B_UNIT)) if obj == "A" else (lambda L: L == E.B_UNIT)
            both = a["m"] & b["m"] & isobj(a["lab"]) & isobj(b["lab"])
            corr = both & (a["lab"] == b["lab"]) & (np.linalg.norm(np.nan_to_num(a["can"] - b["can"], nan=1.0), axis=1) < POS_TOL)
            ident[obj][v] = (corr.sum(), both.sum())
            cnt = corr & (a["vis"] == b["vis"]) & a["u3"] & b["u3"]
            vch = corr & (a["vis"] != b["vis"])
            d1 = T(b["c1"]) - T(a["c1"]); d2 = T(b["c2"]) - T(a["c2"])
            lin = 0.5 * ((b["c1"] - a["c1"]) + (b["c2"] - a["c2"]))
            u = a["lab"]
            accumulate(d1, d2, u, cnt, S[v], NOISE[v], NAIVE[v], NPX[v])
            np.add.at(VISCH[v], u[vch], 1); np.add.at(CORR[v], u[corr], 1)
            np.add.at(CH[v], u[cnt], lin[cnt]); np.add.at(CHN[v], u[cnt], 1)
            if obj == "A":
                for nm, lo, hi in (("T1", lo0 + dlt, hi0 + dlt), ("T0", lo0, hi0)):
                    sel = cnt & (box_dist(np.nan_to_num(a["can"]), lo, hi) < NEAR)
                    near[nm][v] = ((lin[sel, 0] - lin[sel, 1]).sum(), sel.sum())
            # 8x8 tiles
            ty, tx = np.divmod(np.arange(res * res), res); tid = (ty // TILE) * (res // TILE) + tx // TILE
            key = tid[cnt] * K + u[cnt]
            ts = np.bincount(key, weights=(d1 * d2).sum(1)[cnt], minlength=(res // TILE) ** 2 * K)
            tn = np.bincount(key, minlength=(res // TILE) ** 2 * K)
            ok = tn >= TILE_MIN
            for kk in np.flatnonzero(ok):
                tiles[int(kk % K)].append(float(np.sqrt(max(ts[kk] / (3 * tn[kk]), 0.0))))
            if v in args.keep_views:
                keep[f"v{v}_{obj}_cnt"] = cnt.reshape(res, res); keep[f"v{v}_{obj}_d"] = (0.5 * (d1 + d2)).mean(1).reshape(res, res).astype(np.float32)
                keep[f"v{v}_{obj}_vch"] = vch.reshape(res, res)
        if v in args.keep_views:
            for s in ("T0", "T1", "T1b"):
                keep[f"v{v}_{s}_lab"] = st[s]["lab"].reshape(res, res).astype(np.int16)
                keep[f"v{v}_{s}_rgb"] = T(0.5 * (st[s]["c1"] + st[s]["c2"])).reshape(res, res, 3).astype(np.float16)
        print("view", v, round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "oracle_maps.npz", **keep)
    rms = lambda n, d: np.sqrt(np.clip(n / np.maximum(d, 1), 0, None))
    pt, lo5, hi95, _ = boot_ci(S, NPX, f=rms)
    naive = np.sqrt(NAIVE.sum(0) / np.maximum(NPX.sum(0), 1)); noise = np.sqrt(NOISE.sum(0) / np.maximum(NPX.sum(0), 1))
    px = (NPX.sum(0) / 3).astype(int)
    under = px < MIN_PX
    dirty = (pt > TAU) | under
    units = []
    for i in range(K):
        tl = np.array(tiles[i]) if tiles[i] else np.zeros(1)
        q99 = float(np.quantile(tl, 0.99))
        units.append({"unit": i, "object": "B" if i == E.B_UNIT else "A", "counted_px": int(px[i]), "rms_change": float(pt[i]), "ci90": [float(lo5[i]), float(hi95[i])],
                      "naive_rms_incl_noise": float(naive[i]), "noise_rms_of_mean_diff": float(noise[i]),
                      "dirty": bool(dirty[i]), "under_observed": bool(under[i]), "ambiguous": bool(lo5[i] <= TAU <= hi95[i]),
                      "dirty_at_tau_half": bool(pt[i] > TAU / 2 or under[i]), "dirty_at_2tau": bool(pt[i] > 2 * TAU or under[i]),
                      "vis_change_fraction": float(VISCH[:, i].sum() / max(CORR[:, i].sum(), 1)),
                      "tile_rms_q99": q99, "tiles": len(tiles[i]), "locally_changed_but_valid": bool((not dirty[i]) and q99 > 4 * TAU),
                      "mean_linear_change_rgb": (CH[:, i].sum(0) / max(CHN[:, i].sum(), 1)).tolist(),
                      "energy": float(max(S[:, i].sum(), 0.0))})
    # pilot criteria
    A = np.arange(E.B_UNIT)
    msA, loA, hiA, bsA = boot_ci(S[:, A].sum(1), NPX[:, A].sum(1))
    zA = float(msA / max(np.std(bsA), 1e-30))
    near_res = {}
    for nm in ("T1", "T0"):
        p_, l_, h_, _ = boot_ci(near[nm][:, 0], near[nm][:, 1])
        near_res[nm] = {"mean_linear_R_minus_G": float(p_), "ci90": [float(l_), float(h_)], "px": int(near[nm][:, 1].sum())}
    idr = {o: float(ident[o][:, 0].sum() / max(ident[o][:, 1].sum(), 1)) for o in ident}
    dq = {s: float(np.quantile(np.concatenate(dists[s]), 0.999)) for s in dists}
    P = {"P1_indirect_change": {"pooled_ms": float(msA), "bootstrap_z": zA, "pass": zA >= 4},
         "P2_above_noise_and_tolerance": {"units_ci_lower_above_tau": [int(i) for i in A if lo5[i] > TAU], "pass": bool((lo5[A] > TAU).any())},
         "P3_nonlocal_colour_bleeding": {**near_res, "pass": near_res["T1"]["ci90"][0] > 0 and near_res["T0"]["ci90"][1] < 0},
         "P4_identity": {"owner_dist_q999_m": dq, "correspondence_rate": idr, "pass": max(dq.values()) <= 1.125e-3 and min(idr.values()) >= 0.9},
         "P5_coverage": {"units_with_min_px": int((px >= MIN_PX).sum()), "pass": int((px >= MIN_PX).sum()) >= 15}}
    P["all_pass"] = all(v["pass"] for v in P.values() if isinstance(v, dict))
    dqi = {s: float(np.quantile(np.concatenate(dists_int[s]), 0.999)) for s in dists_int}
    P11 = {k: v for k, v in P.items() if k != "all_pass"}
    P11["P4_identity"] = {"owner_dist_q999_m_non_depth_edge": dqi, "correspondence_rate": idr, "pass": max(dqi.values()) <= 1.125e-3 and min(idr.values()) >= 0.9,
                          "amendment": "b4_v1.1 (user decision 2026-10-18 after the pilot): distance criterion on non-depth-edge pixels"}
    P11["all_pass"] = all(v["pass"] for v in P11.values() if isinstance(v, dict))
    Dstar = [u["unit"] for u in units if u["dirty"]]
    pa = np.load(E.ROOT / "results/rna_surface_ownership_lego/partition_L16.npz")
    area = np.concatenate([pa["unit_area"].astype(float), [scene["B_mesh"]["area"]]])
    allpx = px.sum()
    rec.update({"views": nv, "resolution": res, "pilot": P, "pilot_b4_v1_1": P11, "units": units, "D_star": Dstar, "V_star": [i for i in range(K) if i not in Dstar],
                "dirty_area_fraction": float(area[Dstar].sum() / area.sum()), "dirty_pixel_fraction": float(px[Dstar].sum() / max(allpx, 1)),
                "energy_fraction_in_D_star": float(sum(units[i]["energy"] for i in Dstar) / max(sum(u["energy"] for u in units), 1e-30)),
                "ambiguous_units": [u["unit"] for u in units if u["ambiguous"]],
                "sensitivity": {"D_star_tau_half": [u["unit"] for u in units if u["dirty_at_tau_half"]], "D_star_2tau": [u["unit"] for u in units if u["dirty_at_2tau"]]},
                "missed_candidates": [u["unit"] for u in units if (not u["dirty"]) and (u["ci90"][1] > TAU or u["locally_changed_but_valid"])],
                "seconds": time.time() - t0})
    (out / "oracle.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"pilot": {k: v.get("pass") if isinstance(v, dict) else v for k, v in P.items()}, "pilot_b4_v1_1": {k: v.get("pass") if isinstance(v, dict) else v for k, v in P11.items()}, "D_star": Dstar}, indent=None))
    for u in units:
        print(u["unit"], u["object"], u["counted_px"], round(u["rms_change"], 5), [round(x, 5) for x in u["ci90"]], "dirty" if u["dirty"] else "valid",
              "visch", round(u["vis_change_fraction"], 3), "q99tile", round(u["tile_rms_q99"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
