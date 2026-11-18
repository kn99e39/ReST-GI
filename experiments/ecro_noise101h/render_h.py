"""WL101H renderer-side helpers around the ORIGINAL WL101 optimized runtime (experiments/ecro_dyn101/rt101.py).

Normal arm  : rt101.RT.frame / rt101.render_hq, unchanged (the raw ECRO frame path; no reconstruction anywhere).
Split arm   : SplitRT._split_prog, a diagnostic copy of RT._frame_prog whose ONLY change is the RNG wiring:
              pixel jitter <- stream J, first-event transport (k=1 SSS walk and k=1 closure/exit sample) <- stream P,
              second-event SSS walk (k=2, after the HIGH lookup) <- stream S. Every draw is still one PCG32 uniform, consumed in the
              same order within its stage, so the estimator's distribution is unchanged; only which stream feeds which stage differs.
              The split arm is a conditional diagnostic (pick-freeze stage decomposition), never a replacement for the normal arm.
Geometry    : center-ray / supersampled first-hit id buffers for ROI construction (no radiance).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments' / 'ecro_dyn101'))
import numpy as np  # noqa: E402
import rt101 as R  # noqa: E402  (imports Mitsuba from DYN101_MI_DIR / external/mi39_dyn101, cuda_ad_rgb)
import torch  # noqa: E402

mi, dr, PT, C100 = R.mi, R.dr, R.PT, R.C100
F, U, I32 = mi.Float, mi.UInt32, mi.Int32
ROLE = {'J': 1 << 40, 'P': 2 << 40, 'S': 3 << 40}     # distinct PCG stream selectors per stage
KEYS = ['e10', 'e11', 'e12', 'e20', 'e21', 'e22', 'hl0', 'hl1', 'hl2']   # sorted output keys of pt100.trace in ecro mode


class SplitRT(R.RT):
    _rng_s = None
    _walk_calls = 0

    def sss_walk(self, si, ray, rng, lights, active):
        """pt100.trace calls sss_walk once per event (k = 1, 2). In the split program the k = 2 walk draws from stream S."""
        self._walk_calls += 1
        if self._rng_s is not None and self._walk_calls == 2:
            return super().sss_walk(si, ray, self._rng_s, lights, active)
        return super().sss_walk(si, ray, rng, lights, active)

    def _split_prog(self, scene, tab, lp, Bd, sj, sp, ss, cam_res_spp):
        self._bind(tab)
        cam, res, spp = cam_res_spp
        n = res * res
        N = n * spp
        lt = self.lt
        lt.pos, lt.rgb, lt.axis, lt.cc = list(lp['pos']), list(lp['rgb']), list(lp['axis']), list(lp['cc'])
        rj = R.pcg_046(N, sj, mi.UInt64(ROLE['J']))
        rp = R.pcg_046(N, sp, mi.UInt64(ROLE['P']))
        self._rng_s = R.pcg_046(N, ss, mi.UInt64(ROLE['S']))
        self._walk_calls = 0
        try:
            ray, pix = PT.gen_rays(cam, res, spp, rj)
            out, _ = self.trace(ray, N, rp, sp, 'ER', 2, lt, 'ecro', B=Bd)
        finally:
            self._rng_s = None
        assert self._walk_calls == 2, self._walk_calls
        keys = sorted(out)
        assert keys == KEYS, keys
        if spp == 1:
            return [out[k] for k in keys]
        bufs = []
        for k in keys:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, out[k], pix)
            bufs.append(buf)
        return bufs

    def split_frame(self, Bd, cam, res, spp, sj, sp, ss):
        op = lambda s: dr.opaque(mi.UInt64, int(s) * 1000003)
        out = self._split_prog(self.scene, self.tabs(), self.light_args(), Bd, op(sj), op(sp), op(ss), (dict(cam), res, spp))
        dr.eval(out)
        return out


def build(split=False):
    """Fresh construction of the WL100 SSS test-0 / light-0 state (arm C initialisation path of WL101) + its learned table."""
    cfg = C100_cfg0()
    model = R.load_model('sss')
    cls = SplitRT if split else R.RT
    rt = cls(cfg['geom'], 'sss', model, cfg['lights'][0])
    s = rt.source()
    B, _ = rt.respond(s)
    return rt, s, B


def C100_cfg0():
    import json
    return json.loads((R.ROOT100 / 'configs.json').read_text())['sss']['test'][0]


def camera(res=512):
    return dict(C100.P['datasets']['sss']['camera'], res=res)


def to_np(out):
    """Dr.Jit outputs (9 arrays) -> float32 numpy [9, res*res] (per-pixel SUMS over the frame's spp, native interface)."""
    return torch.stack([o.torch() for o in out]).cpu().numpy()


def compose_contract(out, spp):
    """Raw composed ECRO radiance by the WL101 / WL101B consumer contract: (E1 + E2 + HL) / spp in float32 on the device."""
    t = torch.stack([o.torch() for o in out]).view(3, 3, -1)
    return ((t[0] + t[1] + t[2]) * (1.0 / spp)).cpu().numpy()   # [3, res*res] float32


def gbuffer(rt, cam, res, sub=1):
    """First-hit shape id for a sub x sub grid of fixed subpixel positions (sub = 1: pixel centre, as WL100 map_*.npz)."""
    o, fwd, right, up, s = PT.cam_basis(cam)
    n = res * res * sub * sub
    lane = dr.arange(U, n)
    pix = lane // (sub * sub)
    k = lane % (sub * sub)
    ii, jj = pix // res, pix % res
    jx = (F(k % sub) + 0.5) / sub
    jy = (F(k // sub) + 0.5) / sub
    x = (F(jj) + jx) * (2.0 / res) - 1.0
    y = 1.0 - (F(ii) + jy) * (2.0 / res)
    d = dr.normalize(mi.Vector3f(*[float(fwd[q]) + s * x * float(right[q]) + s * y * float(up[q]) for q in range(3)]))
    ray = mi.Ray3f(mi.Point3f(*[F(float(o[q])) for q in range(3)]), d)
    si = rt.scene.ray_intersect(ray)
    sid = dr.select(si.is_valid(), I32(rt.shape_id(si)), I32(-1))
    p = si.p
    dr.eval(sid, p)
    sid = np.asarray(sid).reshape(res, res, sub * sub)
    pos = np.stack([np.asarray(p.x), np.asarray(p.y), np.asarray(p.z)], -1).reshape(res, res, sub * sub, 3)
    return sid, pos


def coverage(rt, cam, res, sub=8):
    sid, _ = gbuffer(rt, cam, res, sub)
    frac = (sid == 6).mean(-1)
    pure = np.where((sid == sid[..., :1]).all(-1), sid[..., 0], -1)
    return {'sphere_frac': frac, 'pure_sid': pure}


def versions():
    import platform
    return {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__, 'numpy': np.__version__,
            'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0), 'variant': mi.variant(),
            'freeze': R.FREEZE, 'mi_dir': str(R.MI39), 'cuda_runtime': torch.version.cuda}
