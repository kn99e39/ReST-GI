"""WL45 Track A1a - fit the Burley length per channel to the measured half-plane step response.

Model (Christensen-Burley, normalised over the plane): R(r) = (exp(-r/d) + exp(-r/(3d))) / (8 pi d r).
Shadow-side radiance at distance s from a lit half-plane (constant irradiance) is proportional to
G(s) = integral_{x > s} integral_y R(sqrt(x^2 + y^2)) dy dx; the lit-side far value is proportional to 1. The measured
profile is normalised by its lit plateau (x < -0.45 m) and d is fitted on the shadow side (x > 0) per channel.
Also reported: the Cycles 3.x parameterisation d = (0.25 / pi) r / (1.9 - A + 3.5 (A - 0.8)^2) for the material's
radius r and colour A, and worklog 44's reading (d ~ r).

    python a1a_fit.py <dir with a1a_profile.npz>      -> <dir>/a1a_fit.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def edge_response(s, d):
    """Fraction of the profile's mass at distance > s from the receiver along one axis (half-plane, s >= 0)."""
    # R integrated over y gives the 1-D marginal; integrate the marginal from s to infinity numerically
    def marg(x):
        f = lambda y: (np.exp(-np.hypot(x, y) / d) + np.exp(-np.hypot(x, y) / (3 * d))) / (8 * np.pi * d * np.hypot(x, y))  # noqa: E731
        return 2 * quad(f, 0, 40 * d, limit=200)[0]
    return quad(marg, s, 40 * d, limit=200)[0]


def main(dd):
    dd = Path(dd); z = np.load(dd / "a1a_profile.npz"); meta = json.loads((dd / "a1a_meta.json").read_text())
    x, rgb = z["x"], z["rgb"].astype(np.float64)
    lit = rgb[x < -0.45].mean(0); prof = rgb / lit
    rec = {"meta": meta, "lit_plateau_rgb": lit.tolist(), "fit": {}}
    sh = (x > 0.003) & (x < 0.8)
    for c, name in enumerate("RGB"):
        y = prof[sh, c]; xs = x[sh]
        def loss(logd):
            d = np.exp(logd); g = np.array([edge_response(s, d) for s in xs[::8]])
            return float(((np.log(np.maximum(y[::8], 1e-6)) - np.log(np.maximum(g, 1e-6))) ** 2 * (y[::8] > 1e-4)).mean())
        r = minimize_scalar(loss, bounds=(np.log(5e-4), np.log(2.0)), method="bounded")
        rad = meta["radius"][c] * meta["scale"]; A = meta["color"][c]
        d_cycles = 0.25 / np.pi * rad / (1.9 - A + 3.5 * (A - 0.8) ** 2)
        frac = lambda R, d: 1 - (np.exp(-R / d) + 3 * np.exp(-R / (3 * d))) / 4  # noqa: E731
        dfit = float(np.exp(r.x))
        rec["fit"][name] = {"d_fit_m": dfit, "fit_loss": float(r.fun), "d_cycles_formula_m": float(d_cycles), "material_radius_m": rad,
                            "shadow_value_at_1cm_5cm_30cm": [float(np.interp(s, x, prof[:, c])) for s in (0.01, 0.05, 0.30)],
                            "profile_mass_within_5cm_30cm_100cm": [float(frac(R, dfit)) for R in (0.05, 0.30, 1.0)]}
    (dd / "a1a_fit.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["fit"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
