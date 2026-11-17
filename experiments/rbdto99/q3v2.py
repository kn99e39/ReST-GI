"""WL99A Q3 v2: noise-unbiased, resolution-checked sharp-structure measure of the body self-transport image Sd (protocol rbdto99a_v2).

Physical definition (declared before any v2 value was computed on a candidate):
  * Domain: ROI_self (WL99 definition) eroded by e px (e = 3 at 384 px, scaled with resolution) away from every pixel whose body
    fraction mb < 0.99, so that geometric silhouettes (body outline, occluder bars in front of the body) cannot create structure.
  * Scale: the structure a representation of on-surface resolution lambda averages away, H_lambda(S) = S - G_s(S), with the Gaussian
    matched to a box of width lambda (s = lambda / sqrt(12) / f px), f = median on-surface pixel footprint (m / px) over the ROI from
    the camera geometry. Blurs are mask-normalised (blur(S M) / blur(M)), M = body mask, so outside-body zeros do not leak in.
    (Revision before any candidate value: a first draft used a band-pass G_s - G_2s with s = lambda / 4, whose response to a lambda
    sinusoid is only ~0.29 of its amplitude, so 'C >= tau1' would not have meant 'averaging loses >= tau1 relative RMS'. Found with
    the synthetic test S2 only.)
  * Noise-unbiased amplitude from independent halves A, B:  C(lambda) = sqrt( max(0, sum_ROI H(A).H(B)) / sum_ROI A.B ), i.e. the
    relative RMS error that lambda-scale spatial averaging of Sd would incur on the ROI. E[H(A).H(B)] = |H(S)|^2 because the halves'
    noise is independent; no noise floor is subtracted by hand.
  * lambda_c = 0.02 m: the coarsest spatial cell of the frozen WL99 Stage 1 precomputed comparator ladder (TwinP h_x = 0.02 m), i.e. the
    scale a coarse precomputed representation would average away.  c_min = tau1 = 0.05: the WL99 Stage 1 fidelity tier; structure whose
    relative amplitude is below the tier cannot make an arm that erases it fail the tier, so it does not discriminate sharp transport.
  * Q3v2.structure:   C(lambda_c) >= c_min at T0 and at T1.
  * Q3v2.resolved:    lambda_c spans >= 4 px at the evaluation resolution r, and |C_r - C_2r| / C_2r <= 0.20 (C converged under
                      a 2x finer pixel grid of the same scene, camera and transport).  The workload's evaluation resolution is the
                      smallest r in {384, 768} for which this holds (768 checked against 1536 only if 384 fails and 1536 is rendered).
  * Q3v2.reliable:    block-bootstrap SE of C (32x32-px blocks at 384, scaled) <= 0.25 C.
  * The exact-direction angular part of WL99 Q3 is kept unchanged.
"""
import numpy as np

LAMBDA_C = 0.02
C_MIN = 0.05


def gblur(x, s):
    """Separable Gaussian blur (sigma s px), reflect padding."""
    if s <= 0:
        return x.copy()
    r = int(np.ceil(3 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    y = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode='reflect'), k, 'valid'), 0, x)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode='reflect'), k, 'valid'), 1, y)


def mblur(x, m, s):
    num = gblur(x * m, s); den = gblur(m.astype(np.float64), s)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def highpass(x, m, s):
    return x - mblur(x, m, s)


def erode(m, e):
    out = m.copy()
    for dy in range(-e, e + 1):
        for dx in range(-e, e + 1):
            out &= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def amplitude(A, B, roi, body, s, block=32):
    """C and its block-bootstrap SE. A, B: luminance images of the two halves."""
    bA, bB = highpass(A, body, s), highpass(B, body, s)
    num = (bA * bB) * roi; den = (A * B) * roi
    C = float(np.sqrt(max(num.sum(), 0.0) / max(den.sum(), 1e-300)))
    H, W = A.shape
    nb = [(i, j) for i in range(0, H, block) for j in range(0, W, block)]
    ns = np.array([num[i:i + block, j:j + block].sum() for i, j in nb]); ds = np.array([den[i:i + block, j:j + block].sum() for i, j in nb])
    keep = ds > 0
    ns, ds = ns[keep], ds[keep]
    rng = np.random.default_rng(99001)
    bs = []
    for _ in range(400):
        k = rng.integers(0, len(ns), len(ns))
        bs.append(np.sqrt(max(ns[k].sum(), 0) / max(ds[k].sum(), 1e-300)))
    return C, float(np.std(bs))


def measure(A, B, roi_self, mb, footprint_m, res):
    """A, B: dict images of the two halves ([H,W,3] 'Sd'); roi_self from the WL99 masks at this resolution; footprint_m: median
    on-surface metres per pixel. Returns C(lambda_c), SE, scale in px, ROI size."""
    a, b = A['Sd'].mean(-1).astype(np.float64), B['Sd'].mean(-1).astype(np.float64)
    body = mb >= 0.99
    e = max(1, int(round(3 * res / 384)))
    roi = erode(roi_self & body, e) & erode(body, e)
    s = LAMBDA_C / np.sqrt(12) / footprint_m
    C, se = amplitude(a, b, roi, body, s, block=max(8, int(round(32 * res / 384))))
    return {'C': C, 'se': se, 'scale_px': float(LAMBDA_C / footprint_m), 'sigma_px': float(s), 'roi_px': int(roi.sum()), 'footprint_mm': 1e3 * footprint_m}
