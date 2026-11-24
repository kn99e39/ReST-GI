"""WL111 benchmark adapter: full-depth physical path tracer with the SAME scene semantics as the frozen ECRO renderer.

It subclasses the frozen WL101 runtime (experiments/ecro_dyn101/rt101.RT, imported unchanged). Unchanged and inherited:
the scene, the patch/material tables, the frozen ER closure, the SSS first-exit walk, the direct and ballistic-through-sphere
next-event estimation and the camera model. Only the path-continuation loop is new.

Why a new loop: pt100.trace unrolls its bounce loop in Python, so it can only run a small fixed depth (D = 6). This adapter
puts the same per-vertex statements in a symbolic dr.while_loop. That allows full depth (kmax up to 256 events) with
Russian roulette from event `rr` on. Event bookkeeping is the same as pt100 'cam' mode:
  e1 (k = 1), e2 (k = 2), h (k >= 3); a ballistic through-sphere contribution found at vertex k counts as event k + 1.
Guide buffers for the denoiser (first-hit shading normal; first-vertex sampled weight as albedo, SSS exit weight for the
sphere) come out of the same samples, so the practical path tracer pays nothing extra for them.

Nothing here is learned, and nothing here touches ECRO's representation, checkpoint or estimator.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('RESTGI_REPO', str(HERE.parents[1])))
sys.path.insert(0, str(REPO / 'experiments' / 'ecro_dyn101'))
import numpy as np  # noqa: E402
import rt101 as R  # noqa: E402
import torch  # noqa: E402

mi, dr, PT = R.mi, R.dr, R.PT
F, U = mi.Float, mi.UInt32
OUT_KEYS = ['e1_0', 'e1_1', 'e1_2', 'e2_0', 'e2_1', 'e2_2', 'h_0', 'h_1', 'h_2',
            'a_0', 'a_1', 'a_2', 'n_0', 'n_1', 'n_2']


class PT111(R.RT):
    """Frozen WL101 runtime + a full-depth physical frame program (_pt_prog). ECRO frames still use RT.frame / render_hq."""

    def _pt_prog(self, scene, tab, lp, seed, cfg):
        self._bind(tab)
        cam, res, spp, kmax, rr = cfg
        n = res * res
        N = n * spp
        lt = self.lt
        lt.pos, lt.rgb, lt.axis, lt.cc = list(lp['pos']), list(lp['rgb']), list(lp['axis']), list(lp['cc'])
        rng = R.pcg_046(N, mi.UInt64(seed), mi.UInt64(0))
        ray0, pix = PT.gen_rays(cam, res, spp, rng)
        sss = self.case == 'sss'
        z3 = lambda: mi.Color3f(0.0)
        state = (rng, mi.Point3f(ray0.o), mi.Vector3f(ray0.d), mi.Color3f(1.0), mi.Bool(True), U(1),
                 z3(), z3(), z3(), z3(), mi.Vector3f(0.0))

        def cond(rng, o, d, thr, active, k, e1, e2, h, alb, nrm):
            return active & (k <= kmax)

        def body(rng, o, d, thr, active, k, e1, e2, h, alb, nrm):
            ray = mi.Ray3f(o, d)
            si = scene.ray_intersect(ray, active)
            active = active & si.is_valid()
            sid = self.shape_id(si)
            pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active = active & (wi.z > 1e-6)
            wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            first = active & (k == 1)
            nrm = dr.select(first, si.n, nrm)
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            if sss:
                is_sss = active & (mid == 2)
                qs, ep, ew, eb = self.sss_walk(si, ray, rng, lt, is_sss)
            else:
                is_sss = mi.Bool(False)
                qs = mi.Color3f(0.0)
            dd, ballistic = self.direct(si, wi, cl, lt, active, ~is_sss)
            c = thr * (dd + qs)
            e1 = e1 + dr.select(k == 1, c, 0.0)
            e2 = e2 + dr.select(k == 2, c, 0.0)
            h = h + dr.select(k >= 3, c, 0.0)
            cb = thr * ballistic
            more = k < kmax
            e2 = e2 + dr.select(more & (k == 1), cb, 0.0)
            h = h + dr.select(more & (k >= 2), cb, 0.0)
            cont = active & more
            wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()),
                                               cont & ~is_sss)
            newray = si.spawn_ray(si.to_world(wo))
            no, nd = mi.Point3f(newray.o), mi.Vector3f(newray.d)
            if sss:
                no = dr.select(is_sss, ep + ew * 1e-4, no)
                nd = dr.select(is_sss, ew, nd)
                weight = dr.select(is_sss, eb, weight)
                valid = dr.select(is_sss, dr.max(eb) > 0.0, valid)
            valid = valid & cont
            alb = dr.select(first, dr.select(valid, weight, 0.0), alb)
            thr = thr * dr.select(valid, weight, 0.0)
            active = valid
            if rr > 0:
                roul = active & (k >= rr)
                q = dr.minimum(dr.max(thr), 0.95)
                u = rng.next_float32()
                survive = ~roul | (u < q)
                thr = dr.select(roul, thr / dr.maximum(q, 1e-12), thr)
                active = active & survive
            return rng, no, nd, thr, active, k + 1, e1, e2, h, alb, nrm

        rng2, o, d, thr, active, k, e1, e2, h, alb, nrm = dr.while_loop(
            state, cond, body,
            labels=('rng', 'o', 'd', 'thr', 'active', 'k', 'e1', 'e2', 'h', 'alb', 'nrm'), label='WL111_full_depth')
        vals = [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, h.x, h.y, h.z, alb.x, alb.y, alb.z, nrm.x, nrm.y, nrm.z]
        if spp == 1:
            return vals
        bufs = []
        for v in vals:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, pix)
            bufs.append(buf)
        return bufs

    def pt_frame(self, cam, res, spp, seed, kmax=256, rr=4):
        """One full-depth physical frame: 15 device arrays of per-pixel SUMS over spp (OUT_KEYS order). seed -> stream seed*1000003."""
        key = ('pt', res, spp, kmax, rr)
        if key not in self._frames:
            cam_c = dict(cam)
            self._frames[key] = R.maybe_freeze(
                lambda scene, tab, lp, sd: self._pt_prog(scene, tab, lp, sd, (cam_c, res, spp, kmax, rr)))
        out = self._frames[key](self.scene, self.tabs(), self.light_args(), dr.opaque(mi.UInt64, int(seed) * 1000003))
        dr.eval(out)
        return out


def pt_hq(rt, cam, res, spp, seed, kmax=256, rr=4, b=32):
    """Reference-style accumulation: launches of b spp (stream seed*1000003 + launch), float64 host sums -> [15, res*res]."""
    key = ('pthq', res, b, kmax, rr)
    if key not in rt._frames:
        cam_c = dict(cam)
        rt._frames[key] = R.maybe_freeze(
            lambda scene, tab, lp, sd: rt._pt_prog(scene, tab, lp, sd, (cam_c, res, b, kmax, rr)))
    acc = None
    for k in range(spp // b):
        out = rt._frames[key](rt.scene, rt.tabs(), rt.light_args(), dr.opaque(mi.UInt64, int(seed) * 1000003 + k))
        dr.eval(out)
        t = torch.stack([o.torch().double() for o in out])
        acc = t if acc is None else acc + t
    return acc.cpu().numpy()


def to_np(out):
    return torch.stack([o.torch() for o in out]).cpu().numpy()
