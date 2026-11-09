"""WL80C reference adapter: Mitsuba 3.5.1 `principled` BSDF (reflection only).

The reference is the renderer's own plugin (principled.so), evaluated through the
Mitsuba Python API on the GPU (cuda_ad_rgb; llvm_ad_rgb for cross-checks).
Per-lane materials are supplied through nearest-filtered 1 x N raw bitmap textures,
so one kernel evaluates many materials (uv.x selects the texel).

Conventions (Mitsuba): directions are in the local shading frame (z = normal,
x = tangent = anisotropy axis u); eval() returns f * cos(theta_o); pdf() is a
solid-angle density; sample() returns (bs, weight = eval / pdf).
"""
import os
import numpy as np

_MI = None


def mi_init(variant=None):
    global _MI
    import mitsuba as mi
    if _MI is None:
        mi.set_variant(variant or os.environ.get('C80_MI_VARIANT', 'cuda_ad_rgb'))
        _MI = mi
    return _MI


# Descriptor layout (frozen in the protocol): 9 dims
DESC_NAMES = ['base_r', 'base_g', 'base_b', 'metallic', 'roughness', 'anisotropic',
              'clearcoat', 'clearcoat_gloss', 'sheen']
FIXED = {'specular': 0.5, 'spec_tint': 0.0, 'sheen_tint': 0.5, 'spec_trans': 0.0}


def _bitmap(mi, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return {'type': 'bitmap', 'bitmap': mi.Bitmap(arr), 'raw': True,
            'filter_type': 'nearest', 'wrap_mode': 'clamp'}


def textured_principled(desc):
    """One principled BSDF whose parameters vary per texel (desc: [N, 9])."""
    mi = mi_init()
    d = np.asarray(desc, np.float32)
    n = d.shape[0]
    sc = lambda j: _bitmap(mi, d[:, j].reshape(1, n, 1))
    bd = {'type': 'principled',
          'base_color': _bitmap(mi, d[:, 0:3].reshape(1, n, 3)),
          'metallic': sc(3), 'roughness': sc(4), 'anisotropic': sc(5),
          'clearcoat': sc(6), 'clearcoat_gloss': sc(7), 'sheen': sc(8)}
    bd.update(FIXED)
    return mi.load_dict(bd), n


def uniform_principled(d):
    """Principled BSDF with uniform (scalar) parameters for one descriptor."""
    mi = mi_init()
    d = [float(x) for x in d]
    bd = {'type': 'principled', 'base_color': {'type': 'rgb', 'value': d[0:3]},
          'metallic': d[3], 'roughness': d[4], 'anisotropic': d[5],
          'clearcoat': d[6], 'clearcoat_gloss': d[7], 'sheen': d[8]}
    bd.update(FIXED)
    return mi.load_dict(bd)


def _si(mi, wi, mat_idx, n_mat):
    import drjit as dr
    m = len(wi)
    si = dr.zeros(mi.SurfaceInteraction3f, m)
    si.wi = mi.Vector3f(*[mi.Float(np.ascontiguousarray(wi[:, k], np.float32)) for k in range(3)])
    u = (np.asarray(mat_idx, np.float64) + 0.5) / n_mat
    si.uv = mi.Point2f(mi.Float(u.astype(np.float32)), mi.Float(np.full(m, 0.5, np.float32)))
    si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
    si.n = mi.Normal3f(0, 0, 1)
    return si


def _np3(v):
    return np.stack([np.asarray(v[k]) for k in range(3)], 1)


def ref_eval_pdf(bsdf, n_mat, mat_idx, wi, wo, chunk=1 << 22):
    """Reference eval (f cos_o, RGB) and pdf for lanes (mat_idx, wi, wo)."""
    mi = mi_init()
    out_f = np.empty((len(wi), 3), np.float32)
    out_p = np.empty(len(wi), np.float32)
    ctx = mi.BSDFContext()
    for s in range(0, len(wi), chunk):
        e = slice(s, min(s + chunk, len(wi)))
        si = _si(mi, wi[e], mat_idx[e], n_mat)
        w = mi.Vector3f(*[mi.Float(np.ascontiguousarray(wo[e, k], np.float32)) for k in range(3)])
        f, p = bsdf.eval_pdf(ctx, si, w)
        out_f[e] = _np3(f)
        out_p[e] = np.asarray(p)
    return out_f, out_p


def ref_sample(bsdf, n_mat, mat_idx, wi, u1, u2, chunk=1 << 22):
    """Reference sample: returns wo, pdf, weight(RGB), valid."""
    mi = mi_init()
    n = len(wi)
    wo = np.empty((n, 3), np.float32); pdf = np.empty(n, np.float32)
    wt = np.empty((n, 3), np.float32); ok = np.empty(n, bool)
    ctx = mi.BSDFContext()
    for s in range(0, n, chunk):
        e = slice(s, min(s + chunk, n))
        si = _si(mi, wi[e], mat_idx[e], n_mat)
        bs, w = bsdf.sample(ctx, si, mi.Float(np.ascontiguousarray(u1[e], np.float32)),
                            mi.Point2f(mi.Float(np.ascontiguousarray(u2[e, 0], np.float32)),
                                       mi.Float(np.ascontiguousarray(u2[e, 1], np.float32))))
        wo[e] = _np3(bs.wo); pdf[e] = np.asarray(bs.pdf); wt[e] = _np3(w)
        ok[e] = (pdf[e] > 0) & (wo[e][:, 2] > 0) & np.all(np.isfinite(wt[e]), 1)
    return wo, pdf, wt, ok


# ---------------- direction helpers (numpy, float64 internally) ----------------
def sph(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], -1)


def cosine_dirs(u):
    r = np.sqrt(u[:, 0]); ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.maximum(0, 1 - u[:, 0]))], -1)


def uniform_dirs(u):
    z = u[:, 0]; r = np.sqrt(np.maximum(0, 1 - z * z)); ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def mixture_dirs(bsdf, n_mat, mat_idx, wi, rng, w_ref=0.5, w_cos=0.25):
    """Defensive mixture q = w_ref*p_ref + w_cos*cos/pi + w_uni/(2pi) (w_uni = rest).

    Returns wo [n,3], q [n], f_ref [n,3] (eval at wo), use [n] (bool).
    A failed reference sample is replaced by a uniform direction for data use but
    marked use=False: Monte Carlo estimators must give it zero weight (then
    E[use * g / q] = integral of g exactly, with p_ref sub-normalised).
    """
    n = len(wi)
    c = rng.random(n)
    u = rng.random((n, 2))
    wo = np.empty((n, 3))
    use = np.ones(n, bool)
    sel_ref = c < w_ref
    sel_cos = (c >= w_ref) & (c < w_ref + w_cos)
    sel_uni = ~(sel_ref | sel_cos)
    wo[sel_cos] = cosine_dirs(u[sel_cos])
    wo[sel_uni] = uniform_dirs(u[sel_uni])
    if sel_ref.any():
        ii = np.nonzero(sel_ref)[0]
        w_s, _, _, ok = ref_sample(bsdf, n_mat, mat_idx[ii], wi[ii], rng.random(len(ii)), u[ii])
        w_s = w_s.astype(np.float64)
        bad = ~ok
        w_s[bad] = uniform_dirs(rng.random((bad.sum(), 2)))  # failed sample -> data only
        use[ii[bad]] = False
        wo[ii] = w_s
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    wo[:, 2] = np.maximum(wo[:, 2], 1e-6)
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    f, p = ref_eval_pdf(bsdf, n_mat, mat_idx, wi, wo)
    w_uni = 1 - w_ref - w_cos
    q = w_ref * p.astype(np.float64) + w_cos * wo[:, 2] / np.pi + w_uni / (2 * np.pi)
    return wo.astype(np.float32), q, f, use
