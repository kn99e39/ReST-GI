"""WL104I Stage B: response-family equivalence, validity, boundary, orbit-invariance and derivative tests of ER-v1-GF.

Usage: python equiv104i.py <out_dir> [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104i as CM

GF, K, CL, IO = CM.GF, CM.K, CM.CL, CM.IO
TOL = CM.P["stage_B_equivalence"]["tolerances"]
DEV = CM.DEV
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104h"))
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104d"))
from tests104h import directions, mirror_near  # noqa: E402
from response104d import terms as metric_terms, metric_arrays  # noqa: E402
import params104h as PP  # noqa: E402


def boundary_states(g):
    """Constructed boundary states from one generic raw state g [64]."""
    out = {}
    def put(name, fn):
        z = g.copy(); fn(z); out[name] = z
    lam = lambda c, r: 21 + 9 * c + r
    put("block_to_zero", lambda z: [z.__setitem__(lam(c, r), -12.0) for c in range(3) for r in (0, 4, 5, 6, 7)])
    put("outer_all_mass", lambda z: [z.__setitem__(lam(c, r), 12.0) for c in range(3) for r in (1, 2, 3)])
    put("absorption_dominant", lambda z: [z.__setitem__(lam(c, 8), 12.0) for c in range(3)])
    put("knots_high", lambda z: z.__setitem__(slice(48, 63), 12.0))
    put("knots_low", lambda z: z.__setitem__(slice(48, 63), -12.0))
    put("eta_high", lambda z: z.__setitem__(63, 12.0))
    put("eta_low", lambda z: z.__setitem__(63, -12.0))
    put("f0_extreme", lambda z: [z.__setitem__(3 * k + 2, 12.0 if k % 2 else -12.0) for k in range(7)])
    put("rough_high", lambda z: [z.__setitem__(3 * k + q, 12.0) for k in range(7) for q in (0, 1)])
    put("rough_low", lambda z: [z.__setitem__(3 * k + q, -12.0) for k in range(7) for q in (0, 1)])
    put("inner_dominant", lambda z: [z.__setitem__(lam(c, 5), 8.0) for c in range(3)])
    put("outer_dominant", lambda z: [z.__setitem__(lam(c, 2), 8.0) for c in range(3)])
    return out


def collect_states(rows, raw):
    n = len(rows)
    states = {}
    for arm in ("R0", "A", "C", "ER_L"):
        for m in range(n):
            states[f"hist/{arm}/{rows[m]['tag']}"] = raw[arm][m]
    starts = CM.training_starts(n)
    for key, z in starts.items():
        for m in range(n):
            states[f"start/{key}/{rows[m]['tag']}"] = z[m]
    rng = np.random.default_rng(104921)
    for i in range(48):
        states[f"synthetic/random{i:02d}"] = 3.0 * rng.standard_normal(64)
    for name, z in boundary_states(starts["generic"][0]).items():
        states[f"synthetic/{name}"] = z
    return states


def check_state(z, wi, wo, x, y, fd):
    rec = {}
    zt = torch.as_tensor(z[None], device=DEV, dtype=torch.float64)
    yv = GF.from_original(z[None])
    yt = torch.as_tensor(yv, device=DEV, dtype=torch.float64)
    po, pg = CL.decode("ER", "cert", zt), GF.decode(yt)
    fo, fg = K.native_torch(po, x, y), K.native_torch(pg, x, y)
    pk = float(fo.abs().max())
    rec["f64_response"] = float((fo - fg).abs().max()) / max(pk, 1e-300)
    to, tg = K.terms_torch(po, x, y), K.terms_torch(pg, x, y)
    comp = {"O0": "O0", "O1": "O1", "O2": "O2"}
    worst = 0.0
    for k in ("O0", "O1", "O2"):
        worst = max(worst, float((to[k] - tg[k]).abs().max()))
    for k in ("L", "I0", "I1", "I2", "I3"):
        worst = max(worst, float((to["S"] * to[k] - tg["S"] * tg[k]).abs().max()))
    rec["f64_terms"] = worst / max(pk, 1e-300)
    # round trip and section
    zb = GF.to_original(yv)
    rec["round_trip_raw"] = float(np.abs(GF.from_original(zb) - yv).max())
    fs = K.native_torch(CL.decode("ER", "cert", torch.as_tensor(zb, device=DEV, dtype=torch.float64)), x, y)
    rec["section_response"] = float((fs - fg).abs().max()) / max(pk, 1e-300)
    # reciprocity of the candidate (well-posed lanes)
    good = (x[:, 2] > .05) & (y[:, 2] > .05)
    a = K.native_torch(pg, x[good], y[good]) / y[good, 2:3]
    b = K.native_torch(pg, y[good], x[good]) / x[good, 2:3]
    rec["reciprocity"] = float((a - b).abs().max() / a.abs().max().clamp_min(1e-300))
    # compiled runtime parity and validity
    so, sg = GF.compile_original(z[None])[0], GF.compile_states(yv)[0]
    do, dg = CM.drjit_eval(so, wi, wo), CM.drjit_eval(sg, wi, wo)
    rec["compiled_drjit"] = float(np.abs(do - dg).max() / max(np.abs(do).max(), 1e-300))
    lam = sg[:24].reshape(3, 8).astype(np.float64)
    probs, knots, eta = sg[45:53], sg[53:68], sg[68]
    rec["compiled_sum_lambda_max"] = float(lam.sum(1).max())
    rec["valid"] = bool(np.isfinite(sg).all() and lam.min() >= 0 and lam.sum(1).max() <= 1 + 1e-6 and knots.min() >= 0
                        and knots.max() <= 1 and 1 <= eta <= 2.5 and probs[0] > 0 and abs(probs.sum() - 1) < 1e-6
                        and probs.min() >= 0 and np.isfinite(dg).all() and dg.min() >= 0)
    # gauge orbit invariance: infinitesimal (exact JVP through the torch map) and finite moves
    gv = GF.original_gauge_vectors(z)
    worst_inf = 0.0
    for v in gv:
        _, d = torch.func.jvp(CM.from_original_t, (zt,), (torch.as_tensor(v[None], device=DEV, dtype=torch.float64),))
        worst_inf = max(worst_inf, float(d.abs().max()) / float(np.abs(v).max()))
    rec["orbit_infinitesimal"] = worst_inf
    worst_fin, moves = 0.0, 0
    for c in range(3):
        for kind, ts in (("shift", (-3.0, 2.5)), ("scale", (-0.05, -0.01, 0.01, 0.05))):
            for t in ts:
                try:
                    z2 = GF.gauge_orbit_point(z, kind, c, t)
                except ValueError:
                    continue
                moves += 1
                worst_fin = max(worst_fin, float(np.abs(GF.from_original(z2[None]) - yv).max()))
    rec["orbit_finite"] = worst_fin
    rec["orbit_finite_moves"] = moves
    if fd:
        J = torch.func.jacfwd(CM.f_candidate(x[:600], y[:600]))(yt[0])
        rec["fd_relative"] = max(PP.fd_errors(CM.f_candidate(x[:600], y[:600]), J, yt[0]))
        v = torch.as_tensor(np.random.default_rng(7).standard_normal(GF.N_GF), device=DEV, dtype=torch.float64)
        _, jv = torch.func.jvp(CM.f_candidate(x[:600], y[:600]), (yt[0],), (v,))
        rec["jvp_relative"] = float((jv - J @ v).abs().max() / (J @ v).abs().max().clamp_min(1e-300))
    return rec


def angular_parity(rows, raw, ref):
    """Historical states: WL82 metrics on all reference lanes, original compiled vs candidate compiled (Dr.Jit)."""
    w = (ref["use"] / ref["q"])[..., None]
    rmean = 0.5 * (ref["a"] + ref["b"])
    worst, alb = 0.0, 0.0
    for arm in ("R0", "A", "C", "ER_L"):
        po, pg = np.empty_like(rmean), np.empty_like(rmean)
        so, sg = GF.compile_original(raw[arm]), GF.compile_states(GF.from_original(raw[arm]))
        for m in range(len(rows)):
            sl = slice(m * 20, (m + 1) * 20)
            x, y = ref["wi"][sl].reshape(-1, 3), ref["wo"][sl].reshape(-1, 3)
            po[sl] = CM.drjit_eval(so[m], x, y).reshape(20, -1, 3)
            pg[sl] = CM.drjit_eval(sg[m], x, y).reshape(20, -1, 3)
        mo = metric_arrays(metric_terms(po, rmean, w), len(rows))
        mg = metric_arrays(metric_terms(pg, rmean, w), len(rows))
        worst = max(worst, max(float(np.abs(mo[k] - mg[k]).max()) for k in ("el1", "ealb", "eshape")))
        ao, ag = (po * w).sum(1), (pg * w).sum(1)
        alb = max(alb, float((np.abs(ao - ag) / np.maximum(ao, 1e-30)).max()))
    return {"angular_metric_abs": worst, "rgb_albedo_relative": alb}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, seen = CM.inputs()
    rng = np.random.default_rng(104801)
    wi = np.concatenate([directions(3000, rng), directions(500, rng, True)])
    wo = np.concatenate([directions(1500, rng), mirror_near(wi[1500:3000], rng, 2.0), directions(500, rng, True)])
    x, y = (torch.as_tensor(v, device=DEV, dtype=torch.float64) for v in (wi, wo))
    states = collect_states(rows, raw)
    if args.smoke:
        states = {k: v for i, (k, v) in enumerate(states.items()) if i % 25 == 0}
    recs = {}
    for name, z in states.items():
        recs[name] = check_state(np.asarray(z, np.float64), wi, wo, x, y, fd=name.startswith(("hist/R0", "synthetic/")))
    ref = IO.reference()
    ang = angular_parity(rows, raw, ref)
    keys = ("f64_response", "f64_terms", "section_response", "round_trip_raw", "compiled_drjit", "reciprocity",
            "orbit_infinitesimal", "orbit_finite", "fd_relative", "jvp_relative")
    worst = {k: max((r[k] for r in recs.values() if k in r), default=None) for k in keys}
    limits = {"f64_response": TOL["float64_response_relative_to_peak"], "f64_terms": TOL["float64_term_relative_to_peak"],
              "section_response": TOL["float64_response_relative_to_peak"], "round_trip_raw": TOL["round_trip_raw_abs"],
              "compiled_drjit": TOL["compiled_drjit_relative_to_peak"], "reciprocity": TOL["reciprocity_relative"],
              "orbit_infinitesimal": TOL["orbit_invariance_abs"], "orbit_finite": TOL["orbit_invariance_abs"],
              "fd_relative": TOL["fd_relative"], "jvp_relative": TOL["jvp_relative"]}
    failures = [k for k in keys if worst[k] is not None and not worst[k] <= limits[k]]
    invalid = [n for n, r in recs.items() if not r["valid"]]
    if ang["angular_metric_abs"] > TOL["angular_metric_abs"]:
        failures.append("angular_metric_abs")
    worst_by_group = {}
    for g in ("hist", "start", "synthetic"):
        worst_by_group[g] = {k: max((r[k] for n, r in recs.items() if n.startswith(g) and k in r), default=None) for k in keys}
    result = {"passed": not failures and not invalid, "failures": failures, "invalid_states": invalid, "worst": worst,
              "limits": limits, "worst_by_group": worst_by_group, "angular": ang, "n_states": len(recs),
              "orbit_moves_total": int(sum(r["orbit_finite_moves"] for r in recs.values())), "states": recs,
              "commit": IO.commit(), "protocol_sha256": IO.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
              "seconds": time.monotonic() - t0, "smoke": args.smoke}
    IO.write(args.out / "equivalence.json", result)
    print("EQUIVALENCE", "PASS" if result["passed"] else "FAIL", failures, invalid[:5], {k: v for k, v in worst.items()}, flush=True)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
