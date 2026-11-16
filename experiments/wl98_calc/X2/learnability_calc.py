"""WL98 critic X2 scratch calculations (design input only, not evidence of record).

1. Seed-count power: minimum detectable multiplicative effect between two trained arms
   with n training seeds each, given seed spreads of the magnitude measured in WL97/WL93D.
2. Lobe-position learnability under a downstream L2 loss: width of the gradient basin
   for a normalised 2D angular Gaussian lobe of width w ~ 2*alpha.
3. Coverage arithmetic for CK2-like grazing sharp lightings under natural light sampling.
numpy only; seed 98200.
"""
import json, math, os
import numpy as np

rng = np.random.default_rng(98200)
out = {}

# ---- 1. seed power ------------------------------------------------------------------
def tq(df, q, n=2_000_000):
    return float(np.quantile(rng.standard_t(df, size=n), q))

sig = {
    "own_boxA_delivery_two_seeds": abs(math.log(1.63) - math.log(0.52)) / math.sqrt(2.0),  # J1/J7: 0.52x vs 1.63x
    "wl93d_seed_drift_cv0.29": 0.29,   # WL93D seed drift 26-32 % (as log-SD, approx.)
    "moderate_0.10": 0.10,
}
power = {}
for name, s in sig.items():
    rows = {}
    for n in (2, 3, 5, 8, 12, 20):
        df = 2 * n - 2
        mde = (tq(df, 0.975) + tq(df, 0.80)) * s * math.sqrt(2.0 / n)
        rows[str(n)] = {"mde_log": round(mde, 3), "mde_factor": round(math.exp(mde), 2)}
    power[name] = {"sigma_log": round(s, 3), "per_arm_seeds": rows}
out["seed_power_two_sided_a0.05_power0.8"] = power

# ---- 2. lobe basin ------------------------------------------------------------------
# L(d) = 2c(1 - exp(-d^2/(4 w^2)));  |dL/dd| ∝ (d/(2w^2)) exp(-d^2/(4w^2)), max at d = sqrt(2) w
basin = {}
for alpha in (0.03, 0.1, 0.3):
    w = 2.0 * alpha
    d = np.linspace(1e-6, math.pi / 2, 200001)
    g = (d / (2 * w * w)) * np.exp(-d * d / (4 * w * w))
    g /= g.max()
    beyond = d[np.argmax((d > math.sqrt(2) * w) & (g < 1e-2))]
    beyond6 = d[np.argmax((d > math.sqrt(2) * w) & (g < 1e-6))]
    basin[str(alpha)] = {
        "w_deg": round(math.degrees(w), 2),
        "grad_below_1pct_beyond_deg": round(math.degrees(float(beyond)), 1),
        "grad_below_1e-6_beyond_deg": round(math.degrees(float(beyond6)), 1),
        "grad_rel_at_80deg_offset": float(np.interp(math.radians(80), d, g)),
    }
out["lobe_position_basin_L2"] = basin

# ---- 3. coverage arithmetic ----------------------------------------------------------
rate = 4 / 3712  # B-04: lightings with ring-0 mirror-to-box proxy >= CK2
cov = {}
for m in (100, 1000, 10000):
    cov[str(m)] = {"natural_lightings_needed": int(math.ceil(m / rate)),
                   "multiple_of_P92A_3712": round(m / rate / 3712, 1)}
out["coverage_ck2_like"] = {"natural_rate": rate, "for_M_examples": cov}

os.makedirs("results/wl98_calc/X2", exist_ok=True)
with open("results/wl98_calc/X2/learnability_calc.json", "w", newline="\n") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
