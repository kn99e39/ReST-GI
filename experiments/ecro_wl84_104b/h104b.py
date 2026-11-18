"""WL104B shared harness: frozen WL83 / WL84 material states, response evaluators, probe geometry, ROIs.

Arms (same response convention eval(wi, wo) = f(wi, wo) cos(theta_o), wi = camera / view side, wo = light side, shading-frame local):
  LAY    physical layered BSDF: data81.ref_eval (layered81.layered_eval random walks, two independent halves); in scenes the lay81 plugin
  ER_L   learned-ER closure, compiler seed 8301 (z from fit_<set>_L_ER_s8301.npz): closure82.eval_closure (torch) / closure83_dr (Dr.Jit) /
         the cmp83 plugin in scenes
  ER_DF  learned-ER closure family fitted directly per material (fit_<set>_ER_cert_metric_12000_generic.npz, WL83 DF)
Nothing in WL81-WL84 code is modified.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_wl84_104a'))
sys.path.insert(0, str(HERE.parent / 'ecro_wl84_104'))
import common104 as C            # noqa: E402
import h104a as HA               # noqa: E402  (render with PASS_SEED, ctx, masks)
import rt104 as RT               # noqa: E402
import closure82 as CL           # noqa: E402
import closure83                 # noqa: E402,F401  (WL83 robust-refraction patch of closure82, as eval83 / render83 use)
import eval82 as E82             # noqa: E402  (WL82 group_metrics: el1 / ealb / eshape)
import closure83_dr as CD        # noqa: E402
import data81 as DA              # noqa: E402

mi, dr, P, R81, RR = RT.mi, RT.dr, RT.P, RT.R81, RT.RR
dev = torch.device('cuda')
ROOT = C.REPO / 'results' / 'ecro_wl84_104b'
IN63B = ROOT / 'inputs63' / 'rest_compiled83' / 'results'
DF_STATE_SHA = {'test83_iid_0': 'a9889e53ce1bbde5217fef76b9b90f2e04f8df879b52736249b0b96164327f3b',
                'test83_iid_1': 'c60d7f499bb79cb1b4b0893595cf45c37abe3941d923ccf70bb7421fdc8274d0',
                'test83_hard_1': '91b4bde4afcbcdb9684bc1a258e906e58cf805a37767b9972f205b47e7266ea9'}   # WL84 audit84.json ER_DF
LUMA = np.array([0.2126, 0.7152, 0.0722])


def material(tag):
    """desc [10], decoded params (torch) and compiled states for ER_L / ER_DF, with sha256 of the states."""
    _, _, st, k = C.image_spec(tag + '_L1')
    desc = np.load(C.W83 / 'data' / 'materials83.npz')[st][k].astype(np.float64)
    out = {'desc': desc, 'set': st, 'k': k}
    for arm, fn in (('ER_L', f'fit_{st}_L_ER_s8301.npz'), ('ER_DF', f'fit_{st}_ER_cert_metric_12000_generic.npz')):
        z = torch.from_numpy(np.load(C.W83 / 'fit' / fn)['z'][k:k + 1]).to(dev)
        p = CL.decode('ER', 'cert', z)
        stt = CD.compiled_state('ER', p)[0]
        out[arm] = {'z': z, 'p': {a: b.double() for a, b in p.items()}, 'p32': p, 'state': np.asarray(stt, np.float32),
                    'sha256': hashlib.sha256(np.asarray(stt, np.float32).tobytes()).hexdigest()}
    return out


L81 = R81.L


class Lay104b(R81.Lay81):
    """DIAGNOSTIC ONLY (WL104B): render81.Lay81 with an explicit random-walk depth (the shared plugin uses the layered81 default 64;
    the WL83 fitting / evaluation reference data81.ref_eval uses 1024). Same descriptor, RNG seeding and estimator otherwise."""

    def __init__(self, props):
        R81.Lay81.__init__(self, props)
        self.depth = int(props['max_depth'])

    def sample(self, ctx, si, s1, s2, active=True):
        rng = self._rng(si, s2.x + s1, s2.y)
        wo, w, valid = L81.layered_sample(self.m, si.wi, rng, self.depth)
        valid &= mi.Bool(active)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo; bs.pdf = dr.select(valid, 1.0, 0.0); bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection); bs.sampled_component = mi.UInt32(0)
        return bs, mi.Color3f(dr.select(valid, w, 0.0))

    def eval(self, ctx, si, wo, active=True):
        rng = self._rng(si, wo.x, wo.y + wo.z)
        return mi.Color3f(dr.select(mi.Bool(active), L81.layered_eval(self.m, si.wi, wo, rng, self.depth), 0.0))

    def to_string(self):
        return 'Lay104b'


mi.register_bsdf('lay104b', lambda props: Lay104b(props))


def register(tag, mat):
    R81.REG[f'lay_{tag}'] = mat['desc']
    RR.REG[f'ER_L_{tag}'] = ('ER', mat['ER_L']['state'])
    RR.REG[f'ER_DF_{tag}'] = ('ER', mat['ER_DF']['state'])


def scene(tag, arm, light=None, with_sensor=True):
    key = {'LAY': {'type': 'lay81', 'key': f'lay_{tag}'}, 'LAY1024': {'type': 'lay104b', 'key': f'lay_{tag}', 'max_depth': 1024},
           'ER_L': {'type': 'cmp83', 'key': f'ER_L_{tag}'},
           'ER_DF': {'type': 'cmp83', 'key': f'ER_DF_{tag}'}}[arm]
    return mi.load_dict(C.scene_dict(mi, RT.T, key, light, 'mesh', with_sensor))


def er_eval_torch(p, wi, wo, chunk=1 << 21):
    """closure82.eval_closure('ER') (with the closure83 patch) for one material; directions in the dtype of p (float64 p: float64;
    p32: float32 exactly as eval83 evaluated). wi, wo [n, 3] numpy -> [n, 3] float64."""
    dt = next(iter(p.values())).dtype
    out = []
    with torch.no_grad():
        for s in range(0, len(wi), chunk):
            e = slice(s, min(len(wi), s + chunk)); n = e.stop - e.start
            pp = {a: b.expand(n, *b.shape[1:]) for a, b in p.items()}
            out.append(CL.eval_closure('ER', pp, torch.as_tensor(np.asarray(wi[e]), device=dev, dtype=dt),
                                       torch.as_tensor(np.asarray(wo[e]), device=dev, dtype=dt)).double().cpu().numpy())
    return np.concatenate(out)


def er_eval_dr(state, wi, wo):
    """Runtime closure (closure83_dr.DrClosure83, the cmp83 plugin's evaluator) on float32 directions."""
    cl = CD.DrClosure83('ER', state)
    V = lambda a: mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    f = cl.eval(V(wi), V(wo))
    dr.eval(f)
    return np.stack([np.asarray(f[k], np.float64) for k in range(3)], 1)


def lay_eval(desc, wi, wo, nwalk, seed, max_depth=1024):
    """Two independent half means (each nwalk / 2 walks) of the layered eval per pair; max_depth 1024 = WL83 data reference, 64 = the
    in-scene lay81 plugin's walk depth."""
    return DA.ref_eval(np.asarray(desc, np.float64)[None], np.zeros(len(wi), np.int64), np.asarray(wi), np.asarray(wo), nwalk, seed, max_depth=max_depth)


def sph(th, ph):
    return DA.sph(th, ph)


def mirror(wi):
    return np.array([-wi[0], -wi[1], wi[2]])


def polar_grid(center, rmax_deg, dr_deg, n_az):
    """Directions on rings around `center` (unit), with exact spherical-cap ring-segment solid angles; returns dirs [n, 3], dw [n], r_deg, psi."""
    c = center / np.linalg.norm(center)
    t = np.cross(c, [0, 0, 1.0]) if abs(c[2]) < 0.999 else np.array([1.0, 0, 0])
    t /= np.linalg.norm(t); b = np.cross(c, t)
    r_edges = np.radians(np.arange(0, rmax_deg + 1e-9, dr_deg))
    rc = 0.5 * (r_edges[1:] + r_edges[:-1])
    psi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    R, PS = np.meshgrid(rc, psi, indexing='ij')
    d = (np.cos(R)[..., None] * c + np.sin(R)[..., None] * (np.cos(PS)[..., None] * t + np.sin(PS)[..., None] * b)).reshape(-1, 3)
    dw = np.repeat((np.cos(r_edges[:-1]) - np.cos(r_edges[1:])) * 2 * np.pi / n_az, n_az)
    return d, dw, np.degrees(R.ravel()), PS.ravel()


def hemi_grid(dth_deg=1.0, dph_deg=2.0):
    th_e = np.radians(np.arange(0, 90 + 1e-9, dth_deg)); ph_e = np.radians(np.arange(0, 360 + 1e-9, dph_deg))
    thc, phc = 0.5 * (th_e[1:] + th_e[:-1]), 0.5 * (ph_e[1:] + ph_e[:-1])
    T, PH = np.meshgrid(thc, phc, indexing='ij')
    dw = np.repeat((np.cos(th_e[:-1]) - np.cos(th_e[1:]))[:, None] * np.diff(ph_e)[None], 1, 0)
    return sph(T.ravel(), PH.ravel()), dw.ravel(), T, PH


def lum(x):
    return np.asarray(x) @ LUMA
