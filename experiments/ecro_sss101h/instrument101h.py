"""WL101H read-only instrumentation of native transport semantics at the sphere/floor contact.

Every quantity is recomputed with the same Mitsuba calls and statement order as the
unchanged pt100.Ctx.direct / trace, and the replicated through-term is asserted
bit-identical to the original direct() output.
"""
import math

import numpy as np

SPHERE_C = np.array([0.9, 0.4, 0.4])
SPHERE_R = 0.4
SIGMA = 12.0


def inside_sphere(o):
    o = np.asarray(o, np.float64)
    return np.linalg.norm(o - SPHERE_C, axis=-1) < SPHERE_R


def sphere_chord(p, target):
    """Float64 length of the segment p -> target inside the sphere (0 if it misses)."""
    p, target = np.asarray(p, np.float64), np.asarray(target, np.float64)
    d = target - p
    L = np.linalg.norm(d, axis=-1)
    u = d / L[..., None]
    q = p - SPHERE_C
    b = np.einsum("...i,...i->...", q, u)
    c = np.einsum("...i,...i->...", q, q) - SPHERE_R ** 2
    disc = b * b - c
    root = np.sqrt(np.maximum(disc, 0))
    t0, t1 = np.clip(-b - root, 0, L), np.clip(-b + root, 0, L)
    return np.where(disc > 0, np.maximum(t1 - t0, 0), 0.0)


def P(v):
    return np.stack([np.asarray(v[k], np.float64) for k in range(3)], -1)


def record_direct(R, rt, si, wi, cl, lights, active, not_sss, spawn_to):
    """Replicate pt100.Ctx.direct (case 'sss', one light) and keep its intermediates."""
    mi, dr = R.mi, R.dr
    assert lights.n == 1
    l = 0
    dl = lights.pos[l] - si.p
    d2 = dr.squared_norm(dl)
    dist = dr.sqrt(d2)
    wl = dl / dr.maximum(dist, 1e-12)
    wo = si.to_local(wl)
    on = active & not_sss & (wo.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
    shadow_ray = spawn_to(si, lights.pos[l])
    sh = rt.scene.ray_intersect(shadow_ray, on)
    clear = on & ~sh.is_valid()
    cross = on & sh.is_valid() & dr.eq(rt.shape_id(sh), 6)
    de = rt.sphere_exit(sh.p + wl * 1e-6, wl)
    sr = mi.Ray3f(sh.p + wl * (de + 1e-4), wl)
    sr.maxt = dr.maximum(dist - sh.t - de - 2e-4, 0.)
    clear2 = cross & ~rt.scene.ray_test(sr, cross)
    f = cl.eval(wi, dr.select(on, wo, mi.Vector3f(0, 0, 1)), on)
    direct = dr.select(clear, f * lights.rgb[l] / dr.maximum(d2, 1e-12), 0.)
    through = dr.select(clear2, f * lights.rgb[l] * dr.exp(-rt.sigma * de) / dr.maximum(d2, 1e-12), 0.)
    first_hit_sense = dr.dot(sh.n, wl)            # < 0 entry from outside, > 0 exit from inside
    rec = {"shadow_o": P(shadow_ray.o), "shadow_d_norm": np.asarray(dr.norm(shadow_ray.d)),
           "on": np.asarray(on), "clear": np.asarray(clear), "cross": np.asarray(cross), "clear2": np.asarray(clear2),
           "sh_shape": np.asarray(rt.shape_id(sh)), "sh_t": np.asarray(sh.t), "sh_p": P(sh.p),
           "first_hit_dot": np.asarray(first_hit_sense), "de": np.asarray(de),
           "f": P(f), "direct": P(direct), "through": P(through), "light_d2": np.asarray(d2)}
    return direct, through, rec


def native_spawn_to(si, target):
    return si.spawn_ray_to(target)


def native_spawn(si, d):
    return si.spawn_ray(d)
