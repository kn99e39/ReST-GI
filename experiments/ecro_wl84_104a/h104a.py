"""WL104A harness: frozen WL104 ECRO tables, the field ('oracle') table of the evaluation light, paired camera renders and ROI masks.

Every WL104 module is imported unchanged (common104, rt104 -> render83 / render84 / pt102 / closure83_dr, model86, train102).
Arms rendered per replicate (one replicate = SPP samples per pixel, pt102 camera estimator, WL81 camera):
  ecro    'ecro' mode, learned table B (WL104 checkpoint)            -> e1, e2, hl   (HIGH = hl = thr * B[patch, bin] at hit 2)
  orcA/B  'ecro' mode, field table F = f2 + fh of the evaluation light (two independent field estimates)  -> hl
  er      'cam' mode D = 6, learned-ER closure                      -> e1, e2, h    (HIGH = h = NEE at hits 3..6)
  lay     'cam' mode D = 6, layered lay81 BSDF, PASS_SEED per launch (WL84 convention) -> e1, e2, h
ecro, orcA, orcB and er share the seed: identical camera rays and first-event samples (paired); lay uses its own seeds.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_wl84_104'))
import common104 as C            # noqa: E402
import rt104 as RT               # noqa: E402
import model86 as M              # noqa: E402
import train102 as TRN           # noqa: E402

mi, dr, P, R81 = RT.mi, RT.dr, RT.P, RT.R81
E104 = C.ROOT / 'evidence'
dev = torch.device('cuda')


def ctx_for(tag, kind):
    RT.register(tag)
    return RT.Ctx(RT.scene(tag, kind))


def learned_table(ctx, image):
    """WL104 eval104 path: frozen checkpoint -> source of the WL84 point light -> B [N, NB, 3] float32."""
    tag, light, _, _ = C.image_spec(image)
    gs, rows = TRN.geostate(E104 / 'data' / tag)
    model = M.ECRO().to(dev); model.mat_states = rows
    model.load_state_dict(torch.load(E104 / 'models' / tag / 'model.pt', map_location=dev)); model.eval()
    s = ctx.source(RT.point_lights(light))
    with torch.no_grad():
        f = model.factors(TRN.input_state(gs))
        B = model.apply(f, gs, torch.as_tensor(s, device=dev)[None])[0]
    return B.detach().cpu().numpy().astype(np.float32), s, model, gs


def field_table(ctx, image, spp, seed, b=64):
    """Field (training-target type) table of the evaluation light: f2 + fh (2..5 events at the probe), bin-averaged, [N, NB, 3]."""
    _, light, _, _ = C.image_spec(image)
    f2, fh, cnt = ctx.field_components(RT.point_lights(light), spp, seed, b=b)
    return (f2 + fh).astype(np.float32), cnt


def render(ctx, image, spp, seed, mode, B=None, b=512, pass_seed=False):
    """pt102.render (same launch layout and RNG); with pass_seed the layered plugin's PASS_SEED is set per launch as render84.render_g
    does. Returns dict of per-pixel means [H, W, 3] float64: e1, e2 and h ('cam') or hl ('ecro')."""
    _, light, _, _ = C.image_spec(image)
    L = RT.point_lights(light)
    Bf = P.fl(B) if B is not None else None
    if not pass_seed:
        acc, _ = ctx.render(L, spp, seed, mode, B=Bf, b=b)
    else:
        kmax = 2 if mode == 'ecro' else P.D
        acc = {}
        done, launch = 0, 0
        W = H = C.RES
        while done < spp:
            bb = min(b, spp - done)
            R81.PASS_SEED['v'] = mi.UInt64(seed * 100003 + launch + 1); dr.make_opaque(R81.PASS_SEED['v'])
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            N = W * H * bb
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            ray, pix, _ = ctx.gen_rays(bb, rng)
            out, _ = ctx.trace(ray, N, rng, sd, 'MI', kmax, L, mode, B=Bf)
            a = ctx.accum(out, pix, W * H)
            for k, v in a.items():
                acc[k] = acc[k] + v if k in acc else v
            done += bb; launch += 1
    pk = P.pack(acc, C.RES, C.RES, spp)
    return {k: v.astype(np.float64) for k, v in pk.items()}


def gbuffer(ctx):
    """Centre-ray first hit: shape name per pixel, world position, shading normal."""
    ray, _, _ = ctx.gen_rays(1, None)
    si = ctx.scene.ray_intersect(ray)
    sid = np.asarray(dr.select(si.is_valid(), mi.Int32(ctx.shape_id(si)), mi.Int32(-1))).reshape(C.RES, C.RES)
    pos = np.stack([np.asarray(si.p[k]) for k in range(3)], -1).reshape(C.RES, C.RES, 3)
    nrm = np.stack([np.asarray(si.sh_frame.n[k]) for k in range(3)], -1).reshape(C.RES, C.RES, 3)
    names = [m[1] for m in ctx.meshes]
    return sid, pos, nrm, names


def masks(ctx, cfg):
    """Predeclared ROIs (protocol rois): objects, sphere, cube, sphere_lower, floor_near, full."""
    sid, pos, nrm, names = gbuffer(ctx)
    is_ = lambda n: sid == names.index(n)
    sph, cub, flo = is_('sphere'), is_('cube'), is_('floor')
    c = np.asarray(C.SPHERE['center'])
    d_sph = np.linalg.norm(pos[..., :2] - c[:2], axis=-1)
    # cube footprint: centre (0.45, 0.1), yaw 35 deg, half 0.25 -> 2-D distance outside the rotated square
    yaw = np.radians(35.0); ca, sa = np.cos(yaw), np.sin(yaw)
    q = pos[..., :2] - np.array([0.45, 0.1])
    u = np.stack([ca * q[..., 0] + sa * q[..., 1], -sa * q[..., 0] + ca * q[..., 1]], -1)
    d_cube = np.linalg.norm(np.maximum(np.abs(u) - 0.25, 0.0), axis=-1)
    r = cfg['floor_near']
    m = {'objects': sph | cub, 'sphere': sph, 'cube': cub,
         'sphere_lower': sph & (nrm[..., 2] < cfg['sphere_lower_nz_max']),
         'floor_near': flo & ((d_sph <= r['sphere_axis_m']) | (d_cube <= r['cube_footprint_m'])),
         'full': np.ones_like(sph)}
    return m


def lum(x):
    return x @ C.LUMA if hasattr(C, 'LUMA') else x @ np.array([0.2126, 0.7152, 0.0722])
