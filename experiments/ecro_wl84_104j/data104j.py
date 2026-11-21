"""WL104J Stage 0: authenticate frozen inputs, generate the fit and transfer sets (fresh directions and walks), qualify precision,
support and isolation against the WL104D Amendment 1 evaluation reference.

Usage: python data104j.py <out_dir> [--smoke]
"""
import argparse
import time
from pathlib import Path

import numpy as np

import common104j as CM

IO = CM.IO
from data104d import proposal  # noqa: E402  (WL104D frozen proposal)
from common104d import DA  # noqa: E402

P = CM.P["data"]
WL104D_STREAMS = [(104460, 0, 735870976), (10441901, 12288000000, 12363497472), (104470, None, None)]


def make_set(desc, polar, az, lanes, walks, dseed, wseed, pair0):
    n = len(desc)
    inc = CM.incident_directions(np.asarray(polar, float), np.asarray(az, float))
    npb = len(inc)
    wi, wo, q, use = proposal(np.repeat(desc, npb, axis=0), np.tile(inc, (n, 1)), lanes, np.random.default_rng(dseed))
    mid = np.broadcast_to(np.repeat(np.arange(n), npb)[:, None], use.shape)
    act = use.ravel()
    t0 = time.monotonic()
    a, b = DA.ref_eval(desc, mid.ravel()[act], wi.reshape(-1, 3)[act], wo.reshape(-1, 3)[act], walks, wseed, pair0=pair0)
    secs = time.monotonic() - t0
    A = np.zeros((act.size, 3)); B = np.zeros((act.size, 3))
    A[act], B[act] = a, b
    shp = use.shape + (3,)
    d = {"wi": np.ascontiguousarray(wi), "wo": wo, "q": q, "use": use, "a": A.reshape(shp), "b": B.reshape(shp),
         "mid": np.ascontiguousarray(mid)}
    seq = (pair0 * walks, pair0 * walks + int(act.sum()) * walks)
    return d, {"seconds_walks": secs, "pairs_walked": int(act.sum()), "walks": int(act.sum()) * walks, "sequence_range": seq,
               "polar": [float(x) for x in polar], "azimuth": [float(x) for x in az]}


def precision(d, n, polar_per_probe):
    w = (d["use"] / d["q"])
    t = CM.wl82_terms(d["a"], d["b"], w)
    sh = t["eshape"].reshape(n, -1)
    al = t["ealb"].reshape(n, -1)
    band = (polar_per_probe >= 55) & (polar_per_probe <= 65)
    graze = polar_per_probe >= 80
    rows = []
    for i in range(n):
        r = {"band_shape": float(sh[i, band].mean() * .5) if band.any() else None,
             "grazing_shape": float(sh[i, graze].mean() * .5) if graze.any() else None,
             "albedo": float(al[i].mean() * .5), "all_shape": float(sh[i].mean() * .5),
             "el1_half_noise": float(t["el1"].reshape(n, -1)[i].mean() * .5)}
        r["passed"] = bool((r["band_shape"] is None or r["band_shape"] <= .015) and
                           (r["grazing_shape"] is None or r["grazing_shape"] <= .025) and r["albedo"] <= .008)
        rows.append(r)
    return rows


def support(d):
    rad = CM.mirror_radius(d["wi"], d["wo"])
    act = d["use"] & (d["q"] > 0)
    core = (act & (rad < 3)).sum(-1)
    sh = (act & (rad >= 3) & (rad < 12)).sum(-1)
    return {"min_core_active": int(core.min()), "min_shoulder_active": int(sh.min()),
            "passed": bool(core.min() >= 16 and sh.min() >= 16)}


def isolation(fit, ev, n):
    """Exact-equality and nearest-distance check of fit outgoing directions vs evaluation outgoing directions (same material, probe)."""
    """Only positive-weight lanes count: rejected draws of both sets carry the zero-weight placeholder (0, 0, 1)."""
    fw = fit["wo"].reshape(n, CM.NP, -1, 3)
    ew = ev["wo"].reshape(n, CM.NP, -1, 3)
    fu = fit["use"].reshape(n, CM.NP, -1)
    eu = ev["use"].reshape(n, CM.NP, -1)
    exact, dmin = 0, np.inf
    for m in range(n):
        for p in range(CM.NP):
            a, b = fw[m, p][fu[m, p]], ew[m, p][eu[m, p]]
            dots = a @ b.T
            exact += int((np.abs(a[:, None, :] - b[None]).max(-1) == 0).sum())
            dmin = min(dmin, float(np.degrees(np.arccos(np.clip(dots.max(), -1, 1)))))
    return {"exact_equal_outgoing_active": exact, "min_angle_deg_active": dmin}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, desc, ref, wl104i, seen = CM.inputs()
    n = len(rows)
    ref_sha = CM.sha(IO.ANGULAR)
    fs, ts = P["fit_set"], P["transfer_set"]
    lanes = 64 if args.smoke else fs["lanes_per_probe"]
    walks = 512 if args.smoke else fs["walks_per_lane"]
    fit, fit_info = make_set(desc, CM.ANGLES, CM.AZ, lanes, walks, fs["direction_seed"], fs["walk_seed"], fs["walk_pair0"])
    pol_t = np.array(ts["polar_transfer_deg"], float)
    tr1, tr1_info = make_set(desc, pol_t, ts["polar_transfer_azimuth_deg"], lanes, walks, ts["direction_seed"], ts["walk_seed"],
                             ts["walk_pair0"])
    tr2, tr2_info = make_set(desc, CM.ANGLES, ts["azimuth_transfer_azimuth_deg"], lanes, walks, ts["azimuth_transfer_direction_seed"],
                             ts["walk_seed"], ts["walk_pair0"] + tr1_info["pairs_walked"] + 1)
    report = {"fit_set": fit_info, "polar_transfer": tr1_info, "azimuth_transfer": tr2_info}
    # precision / support
    pol_fit = CM.POLAR
    pol_tr1 = np.repeat(pol_t, len(ts["polar_transfer_azimuth_deg"]))
    pol_tr2 = np.repeat(CM.ANGLES, len(ts["azimuth_transfer_azimuth_deg"]))
    report["precision"] = {"fit_set": precision(fit, n, pol_fit), "polar_transfer": precision(tr1, n, pol_tr1),
                           "azimuth_transfer": precision(tr2, n, pol_tr2)}
    report["evaluation_reference_precision"] = IO.reference_precision(ref, n)[0]
    ev_noise = CM.wl82_terms(ref["a"], ref["b"], ref["use"] / ref["q"])
    report["evaluation_reference_el1_half_noise"] = (ev_noise["el1"].reshape(n, -1).mean(1) * .5).tolist()
    report["support"] = {"fit_set": support(fit), "polar_transfer": support(tr1), "azimuth_transfer": support(tr2)}
    # isolation
    iso = isolation(fit, ref, n)
    ranges = [fit_info["sequence_range"], tr1_info["sequence_range"], tr2_info["sequence_range"]]
    overl = any(a0 < b1 and b0 < a1 for (a0, a1) in ranges for (b0, b1) in [(0, 735870976), (12288000000, 12363497472)])
    overl |= any(ranges[i][0] < ranges[j][1] and ranges[j][0] < ranges[i][1] for i in range(3) for j in range(i + 1, 3))
    seeds_new = {fs["walk_seed"], ts["walk_seed"]}
    report["isolation"] = {**iso, "sequence_ranges": ranges, "sequence_overlap": bool(overl),
                           "seed_collision_with_wl104d": bool(seeds_new & {s for s, _, _ in WL104D_STREAMS}),
                           "evaluation_reference_sha256_before": ref_sha}
    np.savez(args.out / "fit_set.npz", **fit)
    np.savez(args.out / "polar_transfer_set.npz", **tr1)
    np.savez(args.out / "azimuth_transfer_set.npz", **tr2)
    report["isolation"]["evaluation_reference_sha256_after"] = CM.sha(IO.ANGULAR)
    prec_ok = all(r["passed"] for k in report["precision"].values() for r in k)
    sup_ok = all(v["passed"] for v in report["support"].values())
    iso_ok = (iso["exact_equal_outgoing_active"] == 0 and not overl and not report["isolation"]["seed_collision_with_wl104d"]
              and report["isolation"]["evaluation_reference_sha256_after"] == ref_sha)
    report["qualified"] = {"precision": prec_ok, "support": sup_ok, "isolation": iso_ok, "all": bool(prec_ok and sup_ok and iso_ok)}
    report.update({"materials": rows, "smoke": args.smoke, "lanes_per_probe": lanes, "walks_per_lane": walks,
                   "commit": CM.commit(), "protocol_sha256": CM.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
                   "outputs_sha256": {f: CM.sha(args.out / f) for f in ("fit_set.npz", "polar_transfer_set.npz",
                                                                       "azimuth_transfer_set.npz")},
                   "seconds": time.monotonic() - t0})
    CM.write(args.out / "data.json", report)
    print("data", report["qualified"], {k: round(v["seconds_walks"], 1) for k, v in
                                         (("fit", fit_info), ("t1", tr1_info), ("t2", tr2_info))}, flush=True)
    if not report["qualified"]["all"] and not args.smoke:
        raise IO.Stop(f"DATA_UNQUALIFIED {report['qualified']}")


if __name__ == "__main__":
    main()
