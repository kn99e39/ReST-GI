"""WL101 arm B: optimized ECRO dynamic-update runtime (same representation, checkpoint, estimator and ownership as arm A).

Runtime stack: Mitsuba 3.9.1 / Dr.Jit 1.5.0 from external/mi39_dyn101 (a separate --target install; the frozen venv keeps
Mitsuba 3.5.1 / Dr.Jit 0.4.6 for arm A) + the same torch 2.8. The frozen sources (pt86, pt100, closure83_dr, scene86, model86,
common100) are IMPORTED UNCHANGED; two shims make them run on Dr.Jit 1.x:
  * dr.eq / dr.neq (removed in 1.0)            -> elementwise == / !=
  * pt100.Ctx.sss_walk uses mi.Loop (removed)   -> RT.sss_walk: the same statements in dr.while_loop (same RNG consumption order)

What is changed (execution organisation only):
  geometry  : incremental patch / rect tables for the moved boxes only (bit-identical to scene86.Geom); constant tables
              (state-ray / source sample offsets, material ids) stay resident; BVH update via params.update() as before
  relation  : GPU-resident state rays (hit, bin, segment); dirty test on the GPU with the frozen rule (origin on a moved box, or
              segment through the old / new 1 mm-padded box); re-trace of the dirty rays only; in-place scatter (no host copies)
  source    : same samples / shadow rays; reduction on the GPU in float64 (frozen: numpy bincount in float64)
  generator : unchanged model86 factors / apply on resident tensors (no CPU GeoState casts)
  hand-over : table B shared torch -> Dr.Jit by DLPack (no host round trip)
  frame     : unchanged LOW + lookup estimator (pt100.trace, mode 'ecro'), image kept on the GPU
  replay    : every Dr.Jit program is wrapped in dr.freeze (kernels recorded once, replayed without Python re-tracing);
              FREEZE=0 runs the same code eagerly (attribution control)
"""
import math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
MI39 = Path(os.environ.get('DYN101_MI_DIR') or (REPO / 'external' / 'mi39_dyn101'))   # Amendment 1: LabServer69 uses Mitsuba 3.8.0 / Dr.Jit 1.3.1
sys.path.insert(0, str(MI39))
import mitsuba as mi  # noqa: E402
mi.set_variant('cuda_ad_rgb')
import drjit as dr  # noqa: E402
assert mi.__version__ == os.environ.get('DYN101_MI_VERSION', '3.9.1') and int(dr.__version__.split('.')[0]) >= 1, (dr.__version__, mi.__version__)
assert hasattr(dr, 'freeze')
dr.eq = lambda a, b: a == b          # shim for the unchanged frozen sources
dr.neq = lambda a, b: a != b
_load_dict = mi.load_dict
mi.load_dict = (lambda d, **kw: _load_dict(d, optimize=False, **kw)) if 'optimize' in (_load_dict.__doc__ or '') else _load_dict   # 3.6+ merges same-BSDF meshes; keep 3.5.1 shapes / ids
for d in ('ecro_material100', 'ecro86'):
    sys.path.insert(0, str(EXP / d))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common100 as C100  # noqa: E402
import pt100  # noqa: E402
PT = pt100.PT
S = PT.S
import model86 as M  # noqa: E402
assert PT.mi is mi and PT.dr is dr
F, U, I32 = mi.Float, mi.UInt32, mi.Int32
NB, K, KAPPA, PSRC = S.NB, S.K, S.KAPPA, S.PSRC
dev = torch.device('cuda')
FREEZE = os.environ.get('FREEZE', '1') == '1'
ROOT100 = REPO / 'results' / 'ecro_material100'


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def maybe_freeze(fn):
    return dr.freeze(fn) if FREEZE else fn


def pcg_046(N, initstate, initseq):
    """mi.PCG32(size=N, initstate, initseq) with the Dr.Jit 0.4.6 seeding (1.x seeds lanes differently): same streams as arm A."""
    r = mi.PCG32(size=N)
    inc = ((initseq + dr.arange(mi.UInt64, N)) << 1) | 1
    r.state = (inc + initstate) * mi.UInt64(0x5851f42d4c957f2d) + inc
    r.inc = inc
    return r


def t2d(t, typ=F):
    """torch (cuda, contiguous) -> Dr.Jit array sharing memory (DLPack)."""
    return typ(t.contiguous().__dlpack__())


# ------------------------------------------------------------------------------------------------ incremental geometry
class IncGeom:
    """scene86.Geom (+ WL100 checker) of the current geometry; after construction only the moved boxes' rows are rewritten.
    Every array is bit-identical to a fresh scene86.Geom(geom) with common100.checker (tested in tests101)."""

    def __init__(self, geom):
        g = S.Geom(geom)
        C100.checker(g)
        self.G = g

    def update(self, geom, moved):
        g = self.G
        g.g = geom
        for b in moved:
            mat = 4 if b == 'A' else 5
            faces = S.box_faces(geom[b], mat)
            r0 = 6 if b == 'A' else 12
            for f, r in enumerate(faces):
                ri = r0 + f
                g.RC[ri], g.RA1[ri], g.RA2[ri] = r[0], r[1], r[2]
                g.RH[ri] = [r[3], r[4]]
                nu, nv = g.RG[ri]
                n = np.cross(g.RA1[ri], g.RA2[ri])
                e1 = g.RA1[ri] * (2 * g.RH[ri, 0] / nu); e2 = g.RA2[ri] * (2 * g.RH[ri, 1] / nv)
                o0 = g.RC[ri] - g.RA1[ri] * g.RH[ri, 0] - g.RA2[ri] * g.RH[ri, 1]
                iu = np.tile(np.arange(nu), nv); iv = np.repeat(np.arange(nv), nu)
                p = g.RBASE[ri] + iu + nu * iv
                po = (o0[None] + iu[:, None] * e1[None]) + iv[:, None] * e2[None]
                g.PO[p] = po; g.PE1[p] = e1; g.PE2[p] = e2
                g.PT[p] = g.RA1[ri]; g.PB[p] = g.RA2[ri]; g.PN[p] = n
                g.PC[p] = po + 0.5 * (e1 + e2)
                g.PAREA[p] = np.linalg.norm(e1) * np.linalg.norm(e2)
        g.BX = np.stack([np.concatenate(S.box_axes(geom[b]['yaw'])) for b in 'AB'])
        return g


# ------------------------------------------------------------------------------------------------ closure table as an input
_CL = {'states': None, 'flat': None}
_cd_init = PT.CD.DrClosure83.__init__


def _cd_init_resident(self, arm, state, idx=None):
    """closure83_dr.DrClosure83.__init__ (per-lane branch, same statements) reading the expanded material table from a resident
    device array (a frozen-function input) instead of re-uploading it from numpy inside the program."""
    if idx is None or _CL['flat'] is None or state is not _CL['states']:
        return _cd_init(self, arm, state, idx)
    self.arm = arm
    flat, M, W = _CL['flat'], _CL['M'], _CL['W']
    vals = [dr.gather(F, flat, idx + k * M) for k in range(W)]
    self.lam = [[vals[c * 8 + r] for r in range(8)] for c in range(3)]
    self.au, self.av, self.f0 = vals[24:31], vals[31:38], vals[38:45]
    self.pr = vals[45:53]
    self.s = [[vals[53 + c * 5 + j] for j in range(5)] for c in range(3)]
    self.eta = vals[68]
    self.iau2, self.iav2, self.dn = vals[69:76], vals[76:83], vals[83:90]
    self.au2, self.av2 = vals[90:97], vals[97:104]


PT.CD.DrClosure83.__init__ = _cd_init_resident


# ------------------------------------------------------------------------------------------------ runtime
class RT(pt100.Ctx):
    """Arm B runtime. Same scene, closure, patch / bin maps and estimator as pt100.Ctx."""

    def __init__(self, geom, case, model, light):
        self._resident = False
        super().__init__(geom, case)          # builds scene, tables (via set_tables) exactly as the frozen Ctx
        self.inc = IncGeom(geom)
        self.geom = geom
        self.model = model
        self.N_ = self.N
        self.box_rows = {b: np.where(self.inc.G.PSID == (S.SID_BOXA if b == 'A' else S.SID_BOXB))[0] for b in 'AB'}
        ray_psid = torch.as_tensor(self.inc.G.PSID, device=dev).repeat_interleave(K)
        self.ray_on_box = {b: ray_psid == (S.SID_BOXA if b == 'A' else S.SID_BOXB) for b in 'AB'}
        self._resident = True
        tab = PT.CD.expand_state(np.atleast_2d(np.asarray(self.mats.states, np.float32)))
        self.CLF = F(np.ascontiguousarray(tab.T).ravel()); dr.eval(self.CLF)
        _CL.update(states=self.mats.states, flat=self.CLF, M=tab.shape[0], W=tab.shape[1])
        self.ptrs = [mi.ShapePtr(s_) for s_ in [p for p in self.ptrs]]
        self._upload_geo()
        self.set_light(light)
        self._f_state = maybe_freeze(self._state_prog)
        self._f_source = maybe_freeze(self._source_prog)
        self._frames = {}
        self.rebuild_state()

    # ---- SSS walk: pt100.Ctx.sss_walk with mi.Loop replaced by dr.while_loop (statements and RNG order unchanged)
    def sss_walk(self, si, ray, rng, lights, active):
        p = mi.Point3f(si.p + ray.d * 1e-6); w = mi.Vector3f(ray.d)
        beta = mi.Color3f(1.); source = mi.Color3f(0.)
        ep = mi.Point3f(si.p); ew = mi.Vector3f(ray.d); eb = mi.Color3f(0.)
        act = mi.Bool(active); depth = U(0)
        scene, sigma, albedo = self.scene, self.sigma, self.albedo

        def cond(rng, p, w, beta, source, ep, ew, eb, act, depth):
            return act & (depth < 128)

        def body(rng, p, w, beta, source, ep, ew, eb, act, depth):
            end = self.sphere_exit(p, w)
            distance = -dr.log(dr.maximum(1. - rng.next_float32(), 1e-12)) / sigma
            coll = act & (distance < end)
            leave = act & ~coll
            ep = mi.Point3f(dr.select(leave, p + w * end, ep)); ew = mi.Vector3f(dr.select(leave, w, ew)); eb = mi.Color3f(dr.select(leave, beta, eb))
            p = mi.Point3f(p + w * dr.minimum(distance, end))
            beta = mi.Color3f(beta * albedo)
            for l in range(lights.n):
                dl = lights.pos[l] - p; d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
                wl = dl / dr.maximum(dist, 1e-12)
                de = self.sphere_exit(p, wl)
                on = coll & (dr.dot(-wl, lights.axis[l]) > lights.cc[l]) & (dist > de + 1e-5)
                shadow = mi.Ray3f(p + wl * (de + 1e-4), wl)
                shadow.maxt = dr.maximum(dist - de - 2e-4, 0.)
                occ = scene.ray_test(shadow, on)
                source = mi.Color3f(source + dr.select(on & ~occ, beta * (1 / (4 * math.pi)) * dr.exp(-sigma * de) * lights.rgb[l] / dr.maximum(d2, 1e-12), 0.))
            u = rng.next_float32(); v = rng.next_float32()
            z = 1 - 2 * u; r = dr.sqrt(dr.maximum(1 - z * z, 0.)); a = 2 * math.pi * v
            w = mi.Vector3f(r * dr.cos(a), r * dr.sin(a), z)
            act = coll; depth = depth + 1
            return rng, p, w, beta, source, ep, ew, eb, act, depth

        rng2, p, w, beta, source, ep, ew, eb, act, depth = dr.while_loop((rng, p, w, beta, source, ep, ew, eb, act, depth), cond, body,
                                                                         labels=('rng', 'p', 'w', 'beta', 'source', 'ep', 'ew', 'eb', 'act', 'depth'),
                                                                         label='WL100_sphere_first_exit')
        if rng2 is not rng:                         # the caller's generator continues after the walk (as with mi.Loop)
            rng.state = rng2.state; rng.inc = rng2.inc
        return source, ep, ew, eb

    # ---- tables
    def set_tables(self, g):
        """Frozen set_tables uploads the constant sample-offset tables (4 x N x K, 2 x N x PSRC) on every edit; here they are
        uploaded once and only the geometry tables are refreshed."""
        if not self._resident:
            return super().set_tables(g)
        self.G = g; self.N = g.N
        self.RT = PT.fl(g.rect_table()); self.PTB = PT.fl(g.patch_table()); self.BX = PT.fl(g.BX)
        dr.eval(self.RT, self.PTB, self.BX)

    def _upload_geo(self):
        g = self.inc.G
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.gs = M.GeoState.__new__(M.GeoState)
        self.gs.N = g.N
        self.gs.cen, self.gs.nrm, self.gs.tan, self.gs.area, self.gs.mat = t(g.PC), t(g.PN), t(g.PT), t(g.PAREA), t(g.PMAT, torch.long)

    def set_light(self, light):
        self.lt = PT.Lights(light)

    def set_geom(self, geom):
        """In-place box update (vertices + normals exactly as pt86.Ctx.set_geom) for the moved boxes only + incremental tables."""
        moved = [b for b in 'AB' if geom[b] != self.geom[b]]
        g_old = {b: dict(self.geom[b]) for b in 'AB'}
        g = self.inc.update(geom, moved)
        for b in moved:
            name = 'box' + b
            Mx = g.box_matrix(b)
            self.params[f'{name}.vertex_positions'] = PT.fl(self.V0[name] @ Mx[:3, :3].T + Mx[:3, 3])
            if name in self.N0:
                R = np.stack(S.box_axes(geom[b]['yaw']), 1)
                self.params[f'{name}.vertex_normals'] = PT.fl(self.N0[name] @ R.T)
        if moved:
            self.params.update()
        self.set_tables(g)
        self._upload_geo()
        self.geom = geom
        return moved, g_old

    # ---- geometry state (relation)
    def _bind(self, tab):
        self.RT, self.PTB, self.BX, self.US, self.UP, self.PMATU = tab['RT'], tab['PTB'], tab['BX'], tab['US'], tab['UP'], tab['PMATU']
        self.ptrs = tab['ptrs']; _CL['flat'] = tab['CLF']

    def tabs(self):
        return {'RT': self.RT, 'PTB': self.PTB, 'BX': self.BX, 'US': self.US, 'UP': self.UP, 'PMATU': self.PMATU,
                'ptrs': list(self.ptrs), 'CLF': self.CLF}

    def _state_prog(self, scene, tab, ri):
        """Same statements as pt86.Ctx.state_rays (ray ids ri), returning device arrays."""
        self._bind(tab)
        pid = ri // K; k = ri % K
        ug = lambda j: dr.gather(F, self.US[j], ri)
        p, nn, (t, bv, n) = self.patch_point(pid, ug(0), ug(1))
        wl = self.dir_in_bin(k // KAPPA, ug(2), ug(3))
        w = t * wl.x + bv * wl.y + n * wl.z
        ok = dr.dot(w, nn) > 1e-3
        o = p + nn * 1e-4
        si = scene.ray_intersect(mi.Ray3f(o, w), ok)
        valid = ok & si.is_valid()
        sid = self.shape_id(si)
        front = si.to_local(-w).z > 1e-6
        ph, bh = self.patch_bin(si, sid, -w)
        hit = dr.select(valid & front, I32(ph), I32(-1))
        tt = dr.select(valid, si.t, 30.0)
        e = o + w * tt
        return hit, I32(bh), o.x, o.y, o.z, e.x, e.y, e.z

    def _trace_rays(self, ri_t):
        """ri_t: int32 torch ray ids -> (hit int32, bin int32, seg float32 [M,6]) torch, all on the GPU."""
        ri = U(I32(ri_t.__dlpack__()))
        out = self._f_state(self.scene, self.tabs(), ri)
        dr.eval(out)
        hit, bh = out[0].torch(), out[1].torch()
        seg = torch.stack([v.torch() for v in out[2:]], 1)
        return hit, bh, seg

    def rebuild_state(self):
        """Full rebuild of every state ray (arm C / initialisation)."""
        ri = torch.arange(self.N * K, dtype=torch.int32, device=dev)
        self.hit, self.dbin, self.seg = self._trace_rays(ri)
        return self.hit.numel()

    def dirty_mask(self, g_old, moved):
        """Frozen rule (ecro86.dirty_rays): origin on a moved box, or segment through the old or new 1 mm-padded volume."""
        o, e = self.seg[:, :3], self.seg[:, 3:]
        d = torch.zeros(self.seg.shape[0], dtype=torch.bool, device=dev)
        for b in moved:
            for gg in (g_old, self.geom):
                c, ax, h = (torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev) for v in S.Geom.obb(type('o', (), {'g': gg})(), b))
                lo = (o - c) @ ax.T; dd = (e - c) @ ax.T - lo
                dd = torch.where(dd.abs() < 1e-12, torch.full_like(dd, 1e-12), dd)
                t0 = (-h - lo) / dd; t1 = (h - lo) / dd
                tmin = torch.minimum(t0, t1).amax(1); tmax = torch.maximum(t0, t1).amin(1)
                d |= (tmax >= tmin.clamp(min=0)) & (tmin <= 1)
            d |= self.ray_on_box[b]
        return d

    def update_state(self, g_old, moved, bucket=True):
        """Selective relation update: re-trace the dirty rays only, scatter into the resident state."""
        if not moved:
            return 0
        idx = torch.nonzero(self.dirty_mask(g_old, moved))[:, 0].to(torch.int32)
        n = int(idx.numel())
        if n == 0:
            return 0
        if bucket and FREEZE:                       # fixed capacities -> a few recordings; padding repeats a dirty ray (same value written)
            cap = 1 << max(10, (n - 1).bit_length())
            idx_p = torch.cat([idx, idx[:1].expand(cap - n)])
        else:
            idx_p = idx
        hit, bh, seg = self._trace_rays(idx_p)
        il = idx_p.long()
        self.hit[il] = hit; self.dbin[il] = bh; self.seg[il] = seg
        return n

    # ---- source encoding
    def _source_prog(self, scene, tab, lp):
        """Same samples / shadow rays as pt86.Ctx.source; returns per-sample slot and RGB weights (reduction outside)."""
        self._bind(tab)
        Np = self.N * PSRC
        lane = dr.arange(U, Np)
        pid = lane // PSRC
        p, nn, (t, bv, n) = self.patch_point(pid, dr.gather(F, self.UP[0], lane), dr.gather(F, self.UP[1], lane))
        outs = []
        for l in range(len(lp['pos'])):
            pos, rgb, axis, cc = lp['pos'][l], lp['rgb'][l], lp['axis'][l], lp['cc'][l]
            dl = pos - p
            d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
            wl = dl * dr.rcp(dist)
            cs = dr.dot(wl, nn)
            on = (cs > 1e-6) & (dr.dot(-wl, axis) > cc)
            r = mi.Ray3f(p + nn * 1e-4, wl)
            r.maxt = dist * (1.0 - 1e-4)
            occ = scene.ray_test(r, on)
            e = dr.select(on & ~occ, cs * dr.rcp(d2), 0.0)
            bn = self.bin_of(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
            outs += [I32(pid * NB + bn), e * rgb.x, e * rgb.y, e * rgb.z]
        return outs

    def light_args(self):
        lt = self.lt
        return {'pos': list(lt.pos), 'rgb': list(lt.rgb), 'axis': list(lt.axis), 'cc': list(lt.cc)}

    def source(self):
        outs = self._f_source(self.scene, self.tabs(), self.light_args())
        dr.eval(outs)
        s = torch.zeros(self.N * NB, 3, dtype=torch.float64, device=dev)
        for l in range(0, len(outs), 4):
            slot = outs[l].torch().long()
            for c in range(3):
                s[:, c].index_add_(0, slot, outs[l + 1 + c].torch().double())
        return (s / PSRC).float().view(1, self.N, NB, 3)

    # ---- learned response (unchanged model86 code on resident tensors)
    @torch.no_grad()
    def respond(self, s, use_single=True, use_global=True):
        self.gs.hit = self.hit.long().view(self.N, K)
        self.gs.dbin = self.dbin.long().view(self.N, K)
        f = self.model.factors(self.gs)
        B = self.model.apply(f, self.gs, s, use_single=use_single, use_global=use_global)[0].contiguous()
        return B, f

    # ---- frame (LOW + lookup), image stays on the GPU
    def _frame_prog(self, scene, tab, lp, Bd, seed, cam_res_spp):
        self._bind(tab)
        cam, res, spp = cam_res_spp
        n = res * res
        N = n * spp
        lt = self.lt
        lt.pos, lt.rgb, lt.axis, lt.cc = list(lp['pos']), list(lp['rgb']), list(lp['axis']), list(lp['cc'])
        sd = mi.UInt64(seed)
        rng = pcg_046(N, sd, mi.UInt64(0))
        ray, pix = PT.gen_rays(cam, res, spp, rng)
        out, _ = self.trace(ray, N, rng, sd, 'ER', 2, lt, 'ecro', B=Bd)
        keys = sorted(out)
        if spp == 1:
            return [out[k] for k in keys]
        bufs = []
        for k in keys:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, out[k], pix)
            bufs.append(buf)
        return bufs

    def frame(self, Bd, cam, res, spp, seed):
        """One frame: per-pixel sums of (e1, e2, hl) x RGB as device arrays (keys sorted: e1*, e2*, hl*)."""
        key = (res, spp)
        if key not in self._frames:
            cam_c = dict(cam)
            self._frames[key] = maybe_freeze(lambda scene, tab, lp, Bd, seed: self._frame_prog(scene, tab, lp, Bd, seed, (cam_c, res, spp)))
        out = self._frames[key](self.scene, self.tabs(), self.light_args(), Bd, dr.opaque(mi.UInt64, seed * 1000003))
        dr.eval(out)
        return out


def render_hq(rt, Bd, cam, res, spp, seed, b=32):
    """Same launch decomposition / seeds as pt86.Ctx.render (launch k: seed * 1000003 + k, b spp each), float64 accumulation."""
    key = ('hq', res, b)
    if key not in rt._frames:
        cam_c = dict(cam)
        rt._frames[key] = maybe_freeze(lambda scene, tab, lp, Bd, sd: rt._frame_prog(scene, tab, lp, Bd, sd, (cam_c, res, b)))
    acc = None
    for k in range(spp // b):
        out = rt._frames[key](rt.scene, rt.tabs(), rt.light_args(), Bd, dr.opaque(mi.UInt64, seed * 1000003 + k))
        dr.eval(out)
        t = torch.stack([o.torch().double() for o in out])
        acc = t if acc is None else acc + t
    return acc.cpu().numpy()          # [9, res*res] keys sorted: e10 e11 e12 e20 e21 e22 hl0 hl1 hl2


def load_model(case):
    ck = ROOT100 / case / 'model.pt'
    m = M.ECRO().to(dev)
    m.mat_states = torch.as_tensor(C100.materials(case), device=dev)
    m.load_state_dict(torch.load(ck, map_location=dev))
    return m.eval()
