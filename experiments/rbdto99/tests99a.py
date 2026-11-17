"""WL99A focused tests of the Q3 v2 metric on synthetic images (no renderer; independent of every candidate).

S1 smooth ramp + noise          -> C ~ 0 (noise-unbiased), structure test fails.
S2 sinusoid of wavelength lambda_c, relative amplitude 0.10, + noise -> C close to the analytic high-pass response (0.057), passes c_min.
S3 same sinusoid at 2x resolution (same physical scene)      -> C converged within 0.20 (resolved).
S4 step edge sharper than the pixel (sub-pixel)   at 384 vs 768 -> C changes by > 0.20 or scale < 4 px only when unresolved.
S5 noise-only (A, B independent)                   -> C ~ 0, unlike a single-image gradient metric.
S6 silhouette control: a body-mask hole (bar in front) inside a smooth image -> C ~ 0 (erosion + mask-normalised blur).
  python tests99a.py <out_json>
"""
import json, sys
import numpy as np
import q3v2 as Q

rng = np.random.default_rng(99101)
out = {}


def scene(res, kind, amp=0.10, noise=0.05, hole=False):
    """Physical square 0.4 m x 0.4 m imaged at res px (footprint 0.4 / res m/px); returns halves A, B, roi, mb, footprint."""
    f = 0.4 / res
    x = (np.arange(res) + 0.5) * f
    X, Y = np.meshgrid(x, x)
    if kind == 'ramp':
        S = 1.0 + 0.5 * X / 0.4
    elif kind == 'sin':
        S = 1.0 + amp * np.sin(2 * np.pi * X / Q.LAMBDA_C)
    elif kind == 'step':
        S = np.where(X > 0.2 + 0.37 * f, 1.5, 1.0)                     # step at a sub-pixel position
    else:
        S = np.ones_like(X)
    mb = np.ones_like(S)
    if hole:
        mb[(np.abs(Y - 0.2) < 0.01)] = 0.0                              # a 2 cm bar in front of the body
        S = S * (mb > 0)
    A = S + noise * rng.standard_normal(S.shape) * (mb > 0); B = S + noise * rng.standard_normal(S.shape) * (mb > 0)
    to = lambda I: {'Sd': np.repeat(I[..., None], 3, -1)}
    roi = mb >= 0.99
    return to(A), to(B), roi, mb, f


def m(res, kind, **kw):
    A, B, roi, mb, f = scene(res, kind, **kw)
    return Q.measure(A, B, roi, mb, f, res)


r = m(384, 'ramp'); out['S1_ramp'] = r; out['S1_pass'] = r['C'] < 0.01
r = m(384, 'sin'); out['S2_sin384'] = r
# analytic: high-pass gain for a sinusoid of wavelength L with Gaussian sigma s = L / sqrt(12): 1 - exp(-2 pi^2 / 12)
g = 1 - np.exp(-2 * np.pi ** 2 / 12)
pred = 0.10 * g / np.sqrt(2) / np.sqrt(1 + 0.10 ** 2 / 2)
out['S2_pred'] = float(pred); out['S2_pass'] = abs(r['C'] - pred) / pred < 0.1 and r['C'] >= Q.C_MIN
r2 = m(768, 'sin'); out['S3_sin768'] = r2; out['S3_rel_change'] = abs(out['S2_sin384']['C'] - r2['C']) / r2['C']; out['S3_pass'] = out['S3_rel_change'] <= 0.20
a, b = m(384, 'step'), m(768, 'step'); out['S4_step'] = {'384': a, '768': b, 'rel_change': abs(a['C'] - b['C']) / max(b['C'], 1e-12)}
out['S4_note'] = 'a pixel-sharp step: C depends on the pixel grid only through the band at lambda_c (>= 4 px), so it is expected to converge'
out['S4_pass'] = out['S4_step']['rel_change'] <= 0.20
r = m(384, 'flat', noise=0.2); out['S5_noise'] = r; out['S5_pass'] = r['C'] < 0.01
r = m(384, 'ramp', hole=True); out['S6_silhouette'] = r; out['S6_pass'] = r['C'] < 0.01
out['all_pass'] = all(v for k, v in out.items() if k.endswith('_pass'))
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k.endswith('pass') or k.endswith('change') or k == 'S2_pred'}, indent=1, default=float))
print({k: round(out[k]['C'], 4) for k in ('S1_ramp', 'S2_sin384', 'S3_sin768', 'S5_noise', 'S6_silhouette')})
