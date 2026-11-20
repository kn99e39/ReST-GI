"""WL104I Stage E angular: A0 (R0), A1 and B1 terminal states on the WL104D Amendment 1 independent layered reference.

Usage: python angular104i.py <fit_dir> <out_dir>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import common104i as CM

GF, IO = CM.GF, CM.IO
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104d"))
from response104d import terms as metric_terms  # noqa: E402

LUMA = CM.LUMA
BANDS = {"near_normal": [5, 20], "mid": [35, 50], "band_55_65": [57.5, 60, 62.5], "theta_70": [70], "grazing": [80, 86],
         "all": None}
MARGIN = 0.05


def mirror_radius(wi, wo):
    m = np.stack([-wi[..., 0], -wi[..., 1], wi[..., 2]], -1)
    return np.degrees(np.arccos(np.clip((m * wo).sum(-1), -1, 1)))


def evaluate(states, ref, n):
    """states: name -> compiled [n, 69] -> per state arrays [n, 20] of el1/ealb/eshape (+ halves), regions, F lanes."""
    w = (ref["use"] / ref["q"])[..., None]
    rmean = 0.5 * (ref["a"] + ref["b"])
    rad = mirror_radius(ref["wi"], ref["wo"])
    masks = {"core": rad < 3, "shoulder": (rad >= 3) & (rad < 12), "tail": rad >= 12}
    out, lanes = {}, {}
    for name, st in states.items():
        pred = np.empty_like(rmean)
        for m in range(n):
            sl = slice(m * 20, (m + 1) * 20)
            pred[sl] = CM.drjit_eval(st[m], ref["wi"][sl].reshape(-1, 3), ref["wo"][sl].reshape(-1, 3)).reshape(20, -1, 3)
        t = metric_terms(pred, rmean, w)
        ta, tb = metric_terms(pred, ref["a"], w), metric_terms(pred, ref["b"], w)
        pl, rl = (pred @ LUMA) * w[..., 0], (rmean @ LUMA) * w[..., 0]
        reg = {k: (((pl - rl) * mk).sum(1) / np.maximum((rl * mk).sum(1), 1e-20)).reshape(n, 20) for k, mk in masks.items()}
        out[name] = {"el1": t["el1"].reshape(n, 20), "ealb": t["ealb"].reshape(n, 20), "eshape": t["eshape"].reshape(n, 20),
                     "el1_a": ta["el1"].reshape(n, 20), "el1_b": tb["el1"].reshape(n, 20), "regions": reg}
        lanes[name] = pred
    return out, lanes, w


def spread(lanes, names, w, n):
    """Across-seed response spread per material: mean pairwise RMS difference / RMS of the mean response (WL104H metric)."""
    A = (w * 0.5).sum((1, 2))
    res = []
    for m in range(n):
        sl = slice(m * 20, (m + 1) * 20)
        W2 = w[sl] / (w[sl].sum((1, 2), keepdims=True))
        F = [lanes[k][sl] for k in names]
        Fm = sum(F) / len(F)
        den = np.sqrt((W2 * Fm ** 2).sum())
        d = [np.sqrt((W2 * (F[i] - F[j]) ** 2).sum()) / den for i in range(len(F)) for j in range(i + 1, len(F))]
        res.append(float(np.mean(d)))
    return np.array(res)


def boot(A, B, units, rng, nboot=10000):
    """A, B [U, P] paired per (material, seed) unit -> relative increase of sums and percentile CI."""
    pt = B.sum() / A.sum() - 1
    idx = rng.integers(0, len(units), size=(nboot, len(units)))
    sa, sb = A.sum(1)[idx].sum(1), B.sum(1)[idx].sum(1)
    r = sb / sa - 1
    return {"point": float(pt), "lo": float(np.quantile(r, .025)), "hi": float(np.quantile(r, .975))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, seen = CM.inputs()
    fit = IO.read(args.fit / "fit.json")
    if IO.sha(args.fit / "fit_states.npz") != fit["fit_states_sha256"]:
        raise IO.Stop("FIT_STATES_HASH")
    with np.load(args.fit / "fit_states.npz") as z:
        fs = {k: z[k] for k in z.files}
    keys = fit["start_keys"]
    n = len(rows)
    states = {"A0": compiled["R0"]}
    for k in keys:
        states[f"A1_{k}"] = GF.compile_original(fs[f"A1_{k}"])
        states[f"B1_{k}"] = GF.compile_states(fs[f"B1_{k}"])
    ref = IO.reference()
    met, lanes, w = evaluate(states, ref, n)
    seeds = [str(s) for s in CM.SEEDS]
    cohorts = {c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}
    rng = np.random.default_rng(104931)
    pol = IO.POLAR
    summary = {}
    for c, ms in cohorts.items():
        units = [(m, s) for m in ms for s in seeds]
        blk = {}
        for band, ang in BANDS.items():
            pc = np.arange(20) if ang is None else np.nonzero(np.isin(pol, ang))[0]
            row = {}
            for metric in ("el1", "ealb", "eshape"):
                A = np.stack([met[f"A1_{s}"][metric][m, pc] for m, s in units])
                B = np.stack([met[f"B1_{s}"][metric][m, pc] for m, s in units])
                st = boot(A, B, units, rng)
                st["A1_mean"], st["B1_mean"] = float(A.mean()), float(B.mean())
                st["A0_mean"] = float(np.mean([met["A0"][metric][m, pc] for m in ms]))
                if metric == "el1":
                    st["local_regression_flag"] = bool(st["lo"] > MARGIN)
                row[metric] = st
            for rg in ("core", "shoulder", "tail"):
                A = np.mean([np.abs(met[f"A1_{s}"]["regions"][rg][m, pc]).mean() for m, s in units])
                B = np.mean([np.abs(met[f"B1_{s}"]["regions"][rg][m, pc]).mean() for m, s in units])
                row[f"abs_signed_energy_{rg}"] = {"A1": float(A), "B1": float(B)}
            blk[band] = row
        # across-seed variation
        blk["seed_std_el1"] = {arm: float(np.median([np.std([met[f"{arm}_{s}"]["el1"][m].mean() for s in seeds]) for m in ms]))
                               for arm in ("A1", "B1")}
        sp_a = spread(lanes, [f"A1_{s}" for s in seeds], w, n)[ms]
        sp_b = spread(lanes, [f"B1_{s}" for s in seeds], w, n)[ms]
        blk["response_spread"] = {"A1": sp_a.tolist(), "B1": sp_b.tolist(), "median_ratio_B_over_A": float(np.median(sp_b / sp_a))}
        blk["reference_half_noise_el1"] = float(np.mean([np.abs(met["A0"]["el1_a"][m] - met["A0"]["el1_b"][m]).mean() / 2 for m in ms]))
        summary[c] = blk
    repro = {"A1_generic_minus_R0_mean_el1": [float(met["A1_generic"]["el1"][m].mean() - met["A0"]["el1"][m].mean()) for m in range(n)],
             "B1_generic_minus_A1_generic_mean_el1": [float(met["B1_generic"]["el1"][m].mean() - met["A1_generic"]["el1"][m].mean())
                                                      for m in range(n)]}
    per = {k: {q: v[q].tolist() for q in ("el1", "ealb", "eshape", "el1_a", "el1_b")} | {"regions": {r: a.tolist() for r, a in v["regions"].items()}}
           for k, v in met.items()}
    np.savez_compressed(args.out / "lanes.npz", **{k: v.astype(np.float32) for k, v in lanes.items()},
                        **{f"ref_{q}": ref[q].astype(np.float32) for q in ("wi", "wo", "a", "b")}, w=w[..., 0].astype(np.float32))
    np.savez(args.out / "compiled_states.npz", **states)
    IO.write(args.out / "angular.json", {"summary": summary, "reproduction": repro, "per_state": per, "materials": rows,
                                         "theta": pol.tolist(), "margin": MARGIN, "commit": IO.commit(),
                                         "protocol_sha256": IO.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
                                         "fit_states_sha256": fit["fit_states_sha256"], "seconds": time.monotonic() - t0})
    print("angular done", {c: summary[c]["all"]["el1"] for c in summary}, flush=True)


if __name__ == "__main__":
    main()
