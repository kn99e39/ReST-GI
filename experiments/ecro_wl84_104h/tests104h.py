"""WL104H focused tests (run before evidence). Writes a JSON report; exits non-zero on any failure.

Usage: python tests104h.py <out_json>
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

import io104h as IO
import comp104h as K

TOL = IO.P["decomposition"]["identity_tolerances"]
DEV = K.H.dev


def directions(n, rng, grazing=False):
    if grazing:
        mu = 10 ** rng.uniform(-6.5, -2, n)
    else:
        mu = rng.uniform(0.0, 1.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu ** 2)
    return np.stack([st * np.cos(ph), st * np.sin(ph), mu], 1)


def mirror_near(wi, rng, deg):
    m = np.stack([-wi[:, 0], -wi[:, 1], wi[:, 2]], 1)
    v = m + np.radians(deg) * rng.normal(size=m.shape)
    v[:, 2] = np.abs(v[:, 2])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    out = Path(sys.argv[1])
    t0 = time.monotonic()
    rows, raw, compiled, _ = IO.inputs()
    rng = np.random.default_rng(104801)
    report, fails = {}, []

    def check(name, value, limit):
        report[name] = {"value": float(value), "limit": float(limit), "passed": bool(value <= limit)}
        if not value <= limit:
            fails.append(name)

    wi = np.concatenate([directions(3000, rng), directions(500, rng, True)])
    wo = np.concatenate([directions(1500, rng), mirror_near(wi[1500:3000], rng, 2.0), directions(500, rng, True)])
    for arm in ("R0", "A", "C"):
        worst_t, worst_d, worst_td, worst_state = 0.0, 0.0, 0.0, 0.0
        for m in range(len(rows)):
            # Torch float64 (decoded from the raw float64 state): exact algebra
            p64 = IO.decode64(raw[arm][m])
            x, y = (torch.as_tensor(v, device=DEV, dtype=torch.float64) for v in (wi, wo))
            t = K.terms_torch(p64, x, y)
            nat = K.native_torch(p64, x, y)
            worst_t = max(worst_t, float((K.native(t) - nat).abs().max() / nat.abs().max()))
            # Dr.Jit runtime terms (float32 compiled state)
            d = K.terms_dr_numpy(compiled[arm][m], wi, wo)
            pk = np.abs(d["native"]).max()
            worst_d = max(worst_d, np.abs(d["native"] - d["native_order_sum"]).max() / pk,
                          np.abs(d["native"] - K.native({k: d[k] for k in d})).max() / pk)
            # Torch (float32-decoded params in float64, float32 directions) vs Dr.Jit per term
            z32 = torch.as_tensor(np.asarray(raw[arm][m], np.float32)[None], device=DEV)
            p32 = {k: v.double() for k, v in K.CL.decode("ER", "cert", z32).items()}
            x32, y32 = (torch.as_tensor(v.astype(np.float32).astype(np.float64), device=DEV) for v in (wi, wo))
            tt = K.terms_torch(p32, x32, y32)
            for key in (*K.BASE_TERMS, "U0", "U1", "U2", "U3", "S"):
                a = tt[key].cpu().numpy()
                b = d[key]
                scale = max(np.abs(a).max(), 1e-30)
                worst_td = max(worst_td, np.abs(a - b).max() / scale)
            # VALID_FAMILY interventions == native evaluator with the equivalently modified state
            f0 = K.native(d)
            for name, spec in K.INTERVENTIONS.items():
                st = K.state_equivalent(name, compiled[arm][m])
                if st is None:
                    continue
                fx = spec["f"](d)
                ref = K.terms_dr_numpy(st, wi, wo)["native"]
                worst_state = max(worst_state, np.abs(fx - ref).max() / max(np.abs(ref).max(), np.abs(f0).max()))
        check(f"{arm}.torch_float64_identity", worst_t, TOL["torch_float64_relative_to_peak"])
        check(f"{arm}.drjit_identity", worst_d, TOL["drjit_native_vs_term_sum_relative_to_peak"])
        check(f"{arm}.torch_vs_drjit_terms", worst_td, TOL["torch_vs_drjit_term_relative_to_term_peak"])
        check(f"{arm}.state_equivalent_interventions", worst_state, TOL["state_equivalent_relative_to_peak"])
    # eta = 1 counterfactual == closure82 arm 'E' with the same parameters (Torch float64)
    m = 0
    p64 = IO.decode64(raw["R0"][m])
    x, y = (torch.as_tensor(v, device=DEV, dtype=torch.float64) for v in (wi, wo))
    t = K.terms_torch(p64, x, y)
    eta_off = K.INTERVENTIONS["ETA_OFF"]["f"](t)
    n = x.shape[0]
    armE = K.CL.eval_closure("E", {k: v.expand(n, *v.shape[1:]) for k, v in p64.items()}, x, y)
    check("eta_off_equals_arm_E", float((eta_off - armE).abs().max() / armE.abs().max()), 1e-12)
    # attenuation interaction algebra
    d = {k: K.INTERVENTIONS[k]["f"](t) - K.native(t) for k in ("ATTEN_OFF", "ATTEN_IN_OFF", "ATTEN_OUT_OFF", "ETA_OFF", "ETA_ATTEN_OFF")}
    X = t["L"] + t["I0"] + t["I1"] + t["I2"] + t["I3"]
    inter = d["ATTEN_OFF"] - d["ATTEN_IN_OFF"] - d["ATTEN_OUT_OFF"]
    check("atten_interaction_algebra", float((inter - (1 - t["si"]) * (1 - t["so"]) * X).abs().max() / X.abs().max()), 1e-12)
    Usum = t["U0"] + t["U1"] + t["U2"] + t["U3"]
    Isum = t["I0"] + t["I1"] + t["I2"] + t["I3"]
    inter2 = d["ETA_ATTEN_OFF"] - d["ETA_OFF"] - d["ATTEN_OFF"]
    check("eta_atten_interaction_algebra", float((inter2 - (1 - t["S"]) * (Usum - Isum)).abs().max() / Usum.abs().max()), 1e-12)
    # reciprocity: VALID_FAMILY interventions reciprocal, DIAGNOSTIC_ONLY not
    good = (wi[:, 2] > .05) & (wo[:, 2] > .05)
    xg, yg = x[good], y[good]
    ta, tb = K.terms_torch(p64, xg, yg), K.terms_torch(p64, yg, xg)
    worst_valid, min_diag = 0.0, np.inf
    for name, spec in K.INTERVENTIONS.items():
        fa = spec["f"](ta) / yg[:, 2:3]
        fb = spec["f"](tb) / xg[:, 2:3]
        r = float((fa - fb).abs().max() / fa.abs().max())
        if spec["validity"] == "VALID_FAMILY":
            worst_valid = max(worst_valid, r)
        else:
            min_diag = min(min_diag, r)
    check("valid_family_reciprocity", worst_valid, 1e-10)
    report["diagnostic_only_nonreciprocity_min"] = float(min_diag)
    if not min_diag > 1e-6:
        fails.append("diagnostic_only_nonreciprocity")
    # invalid directions: every term zero and finite
    bad_wi = np.array([[0.3, 0.1, -0.5], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.0, 0.6, 0.0]])
    bad_wo = np.array([[0.1, 0.1, 0.9], [0.0, 0.0, -1.0], [-0.6, 0.0, -0.8], [0.0, -0.6, 0.8]])
    bad_wi /= np.linalg.norm(bad_wi, axis=1, keepdims=True)
    bad_wo /= np.linalg.norm(bad_wo, axis=1, keepdims=True)
    dbad = K.terms_dr_numpy(compiled["R0"][0], bad_wi, bad_wo)
    worst_bad = max(float(np.abs(dbad[k]).max()) for k in (*K.BASE_TERMS, "U0", "U1", "U2", "U3"))
    finite = all(np.isfinite(dbad[k]).all() for k in dbad if k != "valid")
    check("invalid_directions_zero", worst_bad if finite else np.inf, 0.0)
    # grazing stability (cos down to ~3e-7): finite, non-negative, identity holds
    gwi, gwo = directions(4000, rng, True), directions(4000, rng, True)
    worst_g, neg = 0.0, 0.0
    for m in range(len(rows)):
        dg = K.terms_dr_numpy(compiled["R0"][m], gwi, gwo)
        ok = all(np.isfinite(dg[k]).all() for k in dg if k != "valid")
        pk = max(np.abs(dg["native"]).max(), 1e-30)
        worst_g = max(worst_g, np.abs(dg["native"] - K.native(dg)).max() / pk if ok else np.inf)
        neg = min(neg, min(float(dg[k].min()) for k in (*K.BASE_TERMS, "U0", "U1", "U2", "U3")))
    check("grazing_identity_finite", worst_g, TOL["drjit_native_vs_term_sum_relative_to_peak"])
    check("terms_nonnegative", -neg, 0.0)
    # eta dependency: robust refraction at eta -> 1 reproduces the unrefracted inner lobes
    st = np.array(compiled["R0"][0], np.float32, copy=True)
    st[68] = 1.0
    d1 = K.terms_dr_numpy(st, wi, wo)
    check("eta_one_limit_I_equals_U", max(np.abs(d1[f"I{k}"] - d1[f"U{k}"]).max() / max(np.abs(d1[f"U{k}"]).max(), 1e-30)
                                          for k in range(4)), 2e-6)
    # small Jacobian FD check (R0 development material 0, 600 lanes)
    import params104h as PP
    xx, yy = (torch.as_tensor(v[:600], device=DEV, dtype=torch.float64) for v in (wi, wo))
    z0 = torch.as_tensor(raw["R0"][0], device=DEV, dtype=torch.float64)
    J = PP.jacobian(z0, xx, yy)
    check("jacobian_fd_small", max(PP.fd_errors(PP.response_fn(xx, yy), J, z0)), 1e-6)
    groups, lobes = K.raw_groups()
    report["raw_groups_dims"] = {k: int(v.shape[1]) for k, v in groups.items()}
    report["seconds"] = time.monotonic() - t0
    report["failures"] = fails
    report["commit"] = IO.commit()
    IO.write(out, report)
    print("FAIL" if fails else "PASS", fails, flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
