"""WL104 runtime: WL84 materials and scenes, the WL102 ECRO adapter with the WL81 camera, and the WL84 GPU reference path.

Imports unchanged modules: render83 (registers the learned-closure BSDF plugin 'cmp83' and the layered reference 'lay81' via render81),
render84.render_g (WL84 GPU-resident Mitsuba 'path' renderer, WL81 camera and jitter), pt102 (ECRO transport on mesh patches, physical
dispatcher = the scene's own BSDF), model86 (ECRO v0). Nothing in those files is modified.
"""
import numpy as np
import torch

import common104 as C
import render83 as RR            # noqa: E402  registers cmp83 / lay81, initialises Mitsuba 3.5.1 cuda_ad_rgb
import render84 as R84           # noqa: E402  render_g
import pt102 as P                # noqa: E402
import closure82 as CL           # noqa: E402
import closure83_dr as CD        # noqa: E402

R81, mi, dr = RR.R81, RR.mi, RR.dr
assert P.mi is mi
T = R81.T
PT = P.PT
NB, K, D = P.NB, P.K, P.D


def material(tag):
    """(lay81 descriptor, learned-ER compiled state) of a WL83 test material; state hash = WL84 audit."""
    _, _, st, k = C.image_spec(tag + '_L1')
    desc = np.load(C.W83 / 'data' / 'materials83.npz')[st][k]
    z = np.load(C.W83 / 'fit' / f'fit_{st}_L_ER_s8301.npz')['z'][k:k + 1]
    state = CD.compiled_state('ER', CL.decode('ER', 'cert', torch.from_numpy(z).cuda()))[0]
    import hashlib
    h = hashlib.sha256(np.asarray(state, np.float32).tobytes()).hexdigest()
    return desc, np.asarray(state, np.float32), h


def register(tag):
    desc, state, h = material(tag)
    R81.REG[f'lay_{tag}'] = desc
    RR.REG[f'ER_L_{tag}'] = ('ER', state)
    return desc, state, h


def scene(tag, kind, light=None, sphere='mesh', with_sensor=True):
    """kind 'er': learned-ER closure (ECRO LOW material); 'lay': layered physical reference BSDF."""
    mat = {'type': 'cmp83', 'key': f'ER_L_{tag}'} if kind == 'er' else {'type': 'lay81', 'key': f'lay_{tag}'}
    return mi.load_dict(C.scene_dict(mi, T, mat, light, sphere, with_sensor))


def point_lights(light, intensity=None):
    pos, I = C.LIGHTS[light]
    return PT.Lights([{'pos': pos, 'rgb': [intensity or I] * 3}])


class Ctx(P.Ctx):
    """WL102 adapter context; camera rays = the WL81 / WL84 camera formula (centre + uniform jitter), not the scene sensor."""

    def gen_rays(self, b, rng):
        o, fwd, right, up, s = C.cam_basis()
        W = H = C.RES
        n = W * H
        lane = dr.arange(P.U, n * b)
        pix = lane % n
        ii, jj = pix // W, pix % W
        jx, jy = (rng.next_float32(), rng.next_float32()) if rng is not None else (P.F(0.5), P.F(0.5))
        x = (P.F(jj) + jx) * (2.0 / W) - 1.0
        y = 1.0 - (P.F(ii) + jy) * (2.0 / H)
        d = dr.normalize(mi.Vector3f(*[float(fwd[k]) + s * x * float(right[k]) + s * y * float(up[k]) for k in range(3)]))
        return mi.Ray3f(mi.Point3f(*[P.F(float(o[k])) for k in range(3)]), d), pix, (W, H)


def lambert_row(albedo):
    s = np.zeros(69, np.float32)
    for c in range(3):
        s[c * 8] = albedo
    s[24:38] = .3; s[38:45] = .5; s[45] = 1; s[53:] = 1
    return s                      # = common100.lambert (WL100), the encoder row of a Lambertian material


def encoder_rows(ctx, state):
    """One 69-float encoder row per patch (WL102 convention, PMAT = patch id): floor / wall = Lambert 0.8, sphere / cube = the learned-ER
    compiled state itself (the native ECRO material row: the generator's material input is the ER closure state)."""
    names = [m[1] for m in ctx.meshes]
    rows = np.zeros((ctx.N, 69), np.float32)
    for i in range(ctx.N):
        nm = names[ctx.G.POBJ[i]]
        rows[i] = state if nm in ('sphere', 'cube') else lambert_row(0.8)
    return rows


def render_ref(scene_, spp, seed, max_depth, batch=64):
    """WL84 render_g (GPU rays + float32 per-launch scatter, float64 across launches). Returns mean [RES, RES, 3], sums, sums of squares, timing."""
    return R84.render_g(scene_, spp, seed, batch, max_depth=max_depth)
