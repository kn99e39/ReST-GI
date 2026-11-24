"""WL112 tracer: WL100 SSS physics (pt100.Ctx) with

  (i)   a movable sphere (geometry key 'S': {'c': [x, y, z]}; radius 0.4 m unchanged),
  (ii)  native (N) or contact-valid (C) origin semantics (WL101H rule, h_min 2.124e-7 m, generalised to any sphere position),
  (iii) an analytic pinned first surface for the camera-independent construction (no attachment ray query),
  (iv)  2 x 2 sub-patch addresses (equal-area halves of the existing patch parameters),
  (v)   event-2 query capture (camera readout of any number of tables, camera-conditioned projection diagnostics).

Every physics statement of pt100.Ctx.trace / direct / sss_walk is copied unchanged; the only differences are the marked
[C] origin lines (semantics 'C' only), the [PIN] first-vertex substitution, the [CNT] probe count AFTER the tracer's own front
test (WL101D defect), and the [CAP] capture of the event-2 query. tracer112 never edits a shared production file.
"""
import math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for d in ('ecro86', 'ecro_material100'):
    p = str(EXP / d)
    if p not in sys.path:
        sys.path.insert(0, p)
import common100 as C100  # noqa: E402
import pt100  # noqa: E402

PT = pt100.PT
S = PT.S
mi, dr, F, U = PT.mi, PT.dr, PT.F, PT.U
NB = S.NB
NSUB = 4
INF = 1e30
H_MIN = 2.124e-7          # WL101H Q1 stress result x 4 (frozen there; reused unchanged)
LUM = (0.2126, 0.7152, 0.0722)
SPHERE_R = float(S.FX['sphere']['radius'])
DEFAULT_SC = [float(v) for v in S.FX['sphere']['center']]


def sphere_center(geom):
    return [float(v) for v in geom.get('S', {}).get('c', DEFAULT_SC)]


class Geom112(S.Geom):
    """scene86.Geom with the sphere centre taken from the geometry (patch layout in sphere-local coordinates unchanged)."""

    def __init__(self, geom):
        super().__init__(geom)
        c = np.array(sphere_center(geom), np.float64)
        self.SC = c
        sph = self.PSID == S.SID_SPHERE
        self.PC[sph] = c + self.SR * self.PN[sph]


class Ctx112(pt100.Ctx):
    def __init__(self, geom, semantics='C', h_min=H_MIN):
        assert semantics in ('N', 'C')
        self.case = 'sss'
        self.semantics = semantics
        self.h_min = float(h_min)
        # ---- pt86.Ctx.__init__ (same statements; sphere centre from the geometry)
        self.mats = PT.Mats()
        g = Geom112(geom)
        df = {'type': 'diffuse'}
        d = {'type': 'scene'}
        for i, r in enumerate(S.stationary_rects()):
            d[f'r{i}'] = {'type': 'rectangle', 'to_world': PT.T(S.rect_matrix(*r[:5]).tolist()), 'bsdf': df}
        d['sph'] = {'type': 'sphere', 'center': [float(v) for v in g.SC], 'radius': float(g.SR), 'bsdf': df}
        d['boxA'] = {'type': 'cube', 'to_world': PT.T(g.box_matrix('A').tolist()), 'bsdf': df}
        d['boxB'] = {'type': 'cube', 'to_world': PT.T(g.box_matrix('B').tolist()), 'bsdf': df}
        self.scene = mi.load_dict(d)
        by = {s.id(): mi.ShapePtr(s) for s in self.scene.shapes()}
        self.ptrs = [by[n] for n in PT.SHAPE_NAMES]
        self.params = mi.traverse(self.scene)
        self.SMAT = U(np.array(PT.SHAPE_MAT, np.uint32))
        rs = np.random.default_rng(S.P['representation'].get('table_seed', 86001))
        self.u_state_np = rs.random((4, 1, S.K), np.float32)
        self.u_src_np = rs.random((2, 1, S.PSRC), np.float32)
        # ---- pt100.Ctx.__init__ material adapter (same statements)
        self.mats.states = C100.materials('sss')
        self.mats.cls = U(np.array([2, 0, 1, 2, 2, 2, 2, 2], np.uint32))
        self.sigma = 12.
        self.albedo = mi.Color3f(.92, .80, .65)
        self.geom = geom
        self.set_tables(g)
        self.PSIDU = U(g.PSID.astype(np.uint32))
        self.cvec = mi.Point3f(*[float(v) for v in g.SC])
        # resting boxes (bottom on the floor plane): underside and floor footprint are buried (registered exclusions)
        self.resting = []
        for b in 'AB':
            bb = geom[b]
            if abs(bb['c'][2] - bb['h'][2]) < 1e-9:
                ax = S.box_axes(bb['yaw'])
                self.resting.append((np.array(bb['c'], np.float64), ax[0], ax[1], np.array(bb['h'], np.float64), b))

    # ------------------------------------------------------------------------------------ [C] contact-valid origins
    def gap(self, p, u, on_sphere):
        """Analytic free distance from p along unit u to the other solid of the sphere / floor register entry:
        from a non-sphere point the sphere entry distance; from a sphere point the floor-plane distance (WL101H)."""
        q = p - self.cvec
        b = dr.dot(q, u)
        cc = dr.squared_norm(q) - self.SR * self.SR
        disc = b * b - cc
        approach = (cc > 0) & (b < 0) & (disc > 0)
        t_s = dr.select(approach, cc / (-b + dr.sqrt(dr.maximum(disc, 0.))), F(INF))
        t_f = dr.select(u.z < 0, p.z / dr.select(u.z < 0, -u.z, F(1.)), F(INF))
        return dr.select(on_sphere, t_f, t_s)

    def _fix(self, si, ray, target):
        off = ray.o - si.p
        m = dr.norm(off)
        u = off / dr.maximum(m, 1e-30)
        on_sphere = dr.eq(self.shape_id(si), S.SID_SPHERE)
        g = self.gap(si.p, u, on_sphere)
        need = m >= 0.5 * g
        singular = need & (0.5 * g < self.h_min)
        o = mi.Point3f(dr.select(need, si.p + u * (0.5 * g), ray.o))
        if target is None:
            d, maxt = ray.d, ray.maxt
        else:
            v = target - o
            lv = dr.norm(v)
            d = mi.Vector3f(dr.select(need, v / dr.maximum(lv, 1e-30), ray.d))
            maxt = dr.select(need, lv * (ray.maxt / dr.maximum(dr.norm(target - ray.o), 1e-30)), ray.maxt)
        out = mi.Ray3f(o, d)
        out.maxt = maxt
        return out, singular

    def spawn(self, si, d):
        ray = si.spawn_ray(d)
        if self.semantics == 'N':
            return ray, mi.Bool(False)
        return self._fix(si, ray, None)

    def spawn_to(self, si, target):
        ray = si.spawn_ray_to(target)
        if self.semantics == 'N':
            return ray, mi.Bool(False)
        return self._fix(si, ray, target)

    def exit_origin(self, ep, ew):
        if self.semantics == 'N':
            return mi.Point3f(ep + ew * 1e-4), mi.Bool(False)
        g = dr.select(ew.z < 0, ep.z / dr.select(ew.z < 0, -ew.z, F(1.)), F(INF))
        need = 1e-4 >= 0.5 * g
        singular = need & (0.5 * g < self.h_min)
        step = dr.select(need, 0.5 * g, F(1e-4))
        return mi.Point3f(ep + ew * step), singular

    def origin_inside(self, o):
        """Diagnostic: spawned origin strictly inside the sphere (F6 semantic class) / below the floor plane (junction float band)."""
        return dr.squared_norm(o - self.cvec) < (self.SR * (1 - 1e-6)) ** 2, o.z < 0

    # ------------------------------------------------------------------------------------ direct (pt100.Ctx.direct, case 'sss')
    def direct(self, si, wi, cl, lights, active, not_sss):
        direct = mi.Color3f(0.); through = mi.Color3f(0.); bad_s = mi.Bool(False); bad_f = mi.Bool(False)
        for l in range(lights.n):
            dl = lights.pos[l] - si.p; d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
            wl = dl / dr.maximum(dist, 1e-12); wo = si.to_local(wl)
            on = active & not_sss & (wo.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
            sray, sing = self.spawn_to(si, lights.pos[l])          # [C]
            on = on & ~sing                                         # [C]
            ins, blw = self.origin_inside(sray.o); bad_s |= on & ins; bad_f |= on & blw
            sh = self.scene.ray_intersect(sray, on)
            clear = on & ~sh.is_valid()
            cross = on & sh.is_valid() & dr.eq(self.shape_id(sh), 6)
            de = self.sphere_exit(sh.p + wl * 1e-6, wl)
            sr = mi.Ray3f(sh.p + wl * (de + 1e-4), wl)
            sr.maxt = dr.maximum(dist - sh.t - de - 2e-4, 0.)
            clear2 = cross & ~self.scene.ray_test(sr, cross)
            f = cl.eval(wi, dr.select(on, wo, mi.Vector3f(0, 0, 1)), on)
            direct += dr.select(clear, f * lights.rgb[l] / dr.maximum(d2, 1e-12), 0.)
            through += dr.select(clear2, f * lights.rgb[l] * dr.exp(-self.sigma * de) / dr.maximum(d2, 1e-12), 0.)
        self._bad_direct = (bad_s, bad_f)
        return direct, through

    # ------------------------------------------------------------------------------------ addresses
    def subcell(self, si, pid):
        """2 x 2 sub-patch of a hit inside patch pid: halves of the patch's own (a, b) parameters (rect: e1 / e2, sphere: phi / z)."""
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        o = mi.Point3f(g(0), g(1), g(2)); e1 = mi.Vector3f(g(3), g(4), g(5)); e2 = mi.Vector3f(g(6), g(7), g(8))
        rel = si.p - o
        a = dr.dot(rel, e1) / dr.maximum(dr.squared_norm(e1), 1e-30)
        b = dr.dot(rel, e2) / dr.maximum(dr.squared_norm(e2), 1e-30)
        q = (si.p - self.cvec) * (1.0 / self.SR)
        zf = (q.z + 1.0) * (0.5 * S.SPH_NZ) - g(20)
        pf = (dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / (2 * math.pi)) - g(19)
        sph = g(18) > 0.5
        a = dr.select(sph, pf, a); b = dr.select(sph, zf, b)
        return dr.select(a >= 0.5, U(1), U(0)) + dr.select(b >= 0.5, U(2), U(0))

    def first_si(self, pid, a, b, w):
        """[PIN] analytic first surface interaction at patch parameters (a, b) of patch pid (no ray query)."""
        p, nn, _ = self.patch_point(pid, a, b)
        n = dr.width(pid)
        sid = dr.gather(U, self.PSIDU, pid)
        si = dr.zeros(mi.SurfaceInteraction3f, n)
        si.p = p; si.n = nn; si.sh_frame = mi.Frame3f(nn); si.t = F(2e-3)
        sp = self.ptrs[0]
        for i in range(1, len(self.ptrs)):
            sp = dr.select(dr.eq(sid, U(i)), self.ptrs[i], sp)
        si.shape = sp
        si.wi = si.to_local(w)
        return si, sid, nn

    # ------------------------------------------------------------------------------------ trace (pt100.Ctx.trace + markers)
    def trace112(self, ray, N, rng, sd, kmax, lights, mode, first=None, first_ok=None, capture=False):
        """mode 'probe' (E2..E5 suffix of the pinned first vertex), 'cam' (e1 / e2 / h), 'ecro' (prefix to the event-2 query)."""
        thr = mi.Color3f(1.)
        active = mi.Bool(True) if first_ok is None else mi.Bool(first_ok)
        out = {}; cap = {}
        bad = mi.Float(0.); badf = mi.Float(0.)

        def add(name, color):
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'
                out[key] = out.get(key, F(0.)) + ch

        def contribution(k, color):
            if mode == 'probe':
                if k >= 2:
                    add('f2' if k == 2 else 'fh', color)
            else:
                add('e1' if k == 1 else ('e2' if k == 2 else 'h'), color)
        for k in range(1, kmax + 1):
            si = first if (k == 1 and first is not None) else self.scene.ray_intersect(ray, active)      # [PIN]
            active &= si.is_valid()
            sid = self.shape_id(si); pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            if mode == 'probe' and k == 1:
                cap['count'] = mi.Bool(active)                                                         # [CNT]
            if k == 2 and (mode == 'ecro' or capture):                                                 # [CAP]
                cap.update(pid=pid, bin=bn, sub=self.subcell(si, pid), thr=mi.Color3f(thr), act=mi.Bool(active))
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            is_sss = active & dr.eq(mid, 2)
            qs, ep, ew, eb = self.sss_walk(si, ray, rng, lights, is_sss)
            d, ballistic = self.direct(si, wi, cl, lights, active, ~is_sss)
            bad += dr.select(self._bad_direct[0], 1., 0.); badf += dr.select(self._bad_direct[1], 1., 0.)
            if not (mode == 'probe' and k == 1):
                contribution(k, thr * (d + qs))
            if k < kmax:
                contribution(k + 1, thr * ballistic)
            if k == kmax:
                break
            wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active & ~is_sss)
            newray, hs = self.spawn(si, si.to_world(wo))                                                 # [C]
            eo, es = self.exit_origin(ep, ew)                                                            # [C]
            newray.o = dr.select(is_sss, eo, newray.o)
            newray.d = dr.select(is_sss, ew, newray.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
            valid = valid & ~(hs & ~is_sss) & ~(es & is_sss)                                            # [C]
            ins, blw = self.origin_inside(newray.o)
            bad += dr.select(valid & ins, 1., 0.); badf += dr.select(valid & blw, 1., 0.)
            thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
        out['bad_origin'] = bad
        out['bad_floor'] = badf
        return out, cap

    # ------------------------------------------------------------------------------------ camera rendering
    def cam_pass(self, lights, cam, res, spp, seed, mode, tables=(), b=32, region_masks=None, capture_cells=False):
        """mode 'cam': physical 6-event render (e1, e2, h); mode 'ecro': LOW + event-2 lookup of every table in `tables`
        (list of (name, flat Float array, 'coarse' | 'sub')). Same seeds -> identical camera rays and first two events.
        region_masks: dict name -> bool [res*res]; returns per-cell query weights W[region][channel] (unit-marker readout).
        capture_cells ('cam' only): per-cell sums of h and prefix throughput for camera-conditioned projections."""
        n = res * res
        kmax = 6 if mode == 'cam' else 2
        img = {}
        W = {}
        cells = {}
        rm = {k: mi.Bool(np.ascontiguousarray(v.ravel())) for k, v in (region_masks or {}).items()}
        ncs = self.N * NSUB * NB
        bad_total = 0.; badf_total = 0.
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done)
            Nl = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = PT.gen_rays(cam, res, bb, rng)
            out, cap = self.trace112(ray, Nl, rng, sd, kmax, lights, mode, capture=capture_cells or mode == 'ecro')
            bad_total += float(dr.sum(out.pop('bad_origin'))[0]); badf_total += float(dr.sum(out.pop('bad_floor'))[0])
            if mode == 'ecro':
                act = cap['act']
                ic = cap['pid'] * NB + cap['bin']
                isub = (cap['pid'] * NSUB + cap['sub']) * NB + cap['bin']
                th = cap['thr']
                dr.eval(act, ic, isub, th, pix); dr.sync_thread()
                for name, Tt, kind in tables:                       # one table at a time (memory-safe)
                    idx = ic if kind == 'coarse' else isub
                    vals = {f'{name}|{c}': dr.select(act, tc * dr.gather(F, Tt, idx * 3 + c, act), 0.)
                            for c, tc in enumerate((th.x, th.y, th.z))}
                    a2 = self.accum(vals, pix, n)
                    for k_, v in a2.items():
                        img[k_] = img.get(k_, 0.) + v
                for rname, m in rm.items():
                    inr = act & dr.gather(mi.Bool, m, pix)
                    vals = {f'{rname}|{c}': dr.select(inr, tc, 0.) for c, tc in enumerate((th.x, th.y, th.z))}
                    a2 = self.accum(vals, isub, ncs)
                    for c in range(3):
                        key = (rname, c)
                        W[key] = W.get(key, 0.) + a2[f'{rname}|{c}']
            if capture_cells and mode == 'cam':
                act = cap['act']
                isub = (cap['pid'] * NSUB + cap['sub']) * NB + cap['bin']
                th = cap['thr']
                vals = {}
                for c, tc in enumerate((th.x, th.y, th.z)):
                    vals[f't{c}'] = dr.select(act, tc, 0.)
                    vals[f'h{c}'] = dr.select(act, out[f'h{c}'], 0.)
                vals['n'] = dr.select(act, F(1.), F(0.))
                a = self.accum(vals, isub, ncs)
                for k_, v in a.items():
                    cells[k_] = cells.get(k_, 0.) + v
            a = self.accum(out, pix, n)
            for k_, v in a.items():
                img[k_] = img.get(k_, 0.) + v
        res_img = {}
        names = sorted({k.split('|')[0] if '|' in k else k[:-1] for k in img})
        for nm in names:
            if f'{nm}|0' in img:
                res_img[nm] = np.stack([img[f'{nm}|{c}'] for c in range(3)], -1).reshape(res, res, 3) / spp
            elif f'{nm}0' in img:
                res_img[nm] = np.stack([img[f'{nm}{c}'] for c in range(3)], -1).reshape(res, res, 3) / spp
        Wout = {}
        for (rname, c), v in W.items():
            Wout.setdefault(rname, np.zeros((3, ncs)))[c] = v / spp
        self.last_bad_floor = badf_total
        return res_img, Wout, cells, bad_total

    # ------------------------------------------------------------------------------------ camera-independent construction
    def buried(self, p, sid, nn):
        """Registered exclusions with analytic measure: floor under a resting box, a resting box's underside."""
        out = mi.Bool(False)
        for c, ax0, ax1, h, b in self.resting:
            rp = p - mi.Point3f(*c.tolist())
            u = dr.dot(rp, mi.Vector3f(*ax0.tolist())); v = dr.dot(rp, mi.Vector3f(*ax1.tolist()))
            out |= dr.eq(sid, 0) & (dr.abs(u) < float(h[0])) & (dr.abs(v) < float(h[1]))
            out |= dr.eq(sid, S.SID_BOXA if b == 'A' else S.SID_BOXB) & (nn.z < -0.99)
        out |= dr.eq(sid, S.SID_SPHERE) & (p.z <= 0.)
        return out

    def singular_point(self, p, sid, nn):
        """[C] contact-valid singular set at the first vertex (no representable free-space origin on the native offset side)."""
        if self.semantics == 'N':
            return mi.Bool(False)
        g = self.gap(p, nn, dr.eq(sid, S.SID_SPHERE))
        mag = (1. + dr.max(dr.abs(p))) * float(np.float32(1500) * np.float32(2.0 ** -24))   # Mitsuba spawn offset magnitude
        return (mag >= 0.5 * g) & (0.5 * g < self.h_min)

    def construct(self, lights, n_sub, seed, b=8, legacy=False, synthetic=None, frame='normal'):
        """Camera-independent per-(patch, sub-patch, bin) sums under U (uniform area x solid angle), V (receiver-reaching) and
        G (receiver-reaching x local cos), of E2..E5 outgoing radiance (f2, fh; RGB). n_sub proposals per sub-cell.
        legacy=True: WL100 supervising sampler semantics (positive control; 1 mm proximity acceptance against a native full-scene
        hit, nominal address, back-facing samples in the count). synthetic: callable(p, w) -> Color3f replacing the suffix."""
        ncs = self.N * NSUB * NB
        acc = {}
        for launch, done in enumerate(range(0, n_sub, b)):
            bb = min(b, n_sub - done)
            Nl = ncs * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, Nl) % ncs
            bn = slot % NB; ps = slot // NB; sub = ps % NSUB; pid = ps // NSUB
            u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
            a = (F(sub % 2) + u0) * 0.5; bpar = (F(sub // 2) + u1) * 0.5
            sid = dr.gather(U, self.PSIDU, pid)
            t, bv, n = self.patch_frame(pid)
            local = self.dir_in_bin(bn, u2, u3)
            z = 1.25 * u2 - 1.
            ph = (F(bn % S.NPHI) + u3) * (2 * math.pi / S.NPHI) - math.pi
            rr = dr.sqrt(dr.maximum(1 - z * z, 0.))
            local = dr.select(dr.eq(sid, S.SID_SPHERE) & (bn < S.NPHI), mi.Vector3f(rr * dr.cos(ph), rr * dr.sin(ph), z), local)
            w = t * local.x + bv * local.y + n * local.z
            vals = {}
            if legacy:
                # WL100 field_components: native full-scene hit, 1 mm acceptance, nominal address, count before front test
                p, nn, _ = self.patch_point(pid, a, bpar)
                ok = dr.dot(w, nn) > 1e-3
                ray = mi.Ray3f(p + w * 2e-3, -w)
                si = self.scene.ray_intersect(ray, ok)
                okp = ok & si.is_valid() & (dr.norm(si.p - p) < 1e-3)
                front = okp & (si.to_local(w).z > 1e-6)
                out, cap = self.trace112(ray, Nl, rng, sd, 5, lights, 'probe', first=si, first_ok=okp)
                cnt = okp
                ssid = self.shape_id(si)
                apid, abn = self.patch_bin(si, ssid, w)
                vals['misroute'] = dr.select(okp & (dr.neq(apid, pid) | dr.neq(abn, bn)), F(1.), F(0.))
                vals['backcount'] = dr.select(okp & ~front, F(1.), F(0.))
                wU = dr.select(cnt, F(1.), F(0.))
                for c in range(3):
                    vals[f'U_f2{c}'] = dr.select(cnt, out[f'f2{c}'], 0.); vals[f'U_fh{c}'] = dr.select(cnt, out[f'fh{c}'], 0.)
                    vals[f'U_sq{c}'] = dr.select(cnt, (out[f'f2{c}'] + out[f'fh{c}']) * (out[f'f2{c}'] + out[f'fh{c}']), 0.)
                vals['U_w'] = wU
                vals['bad_origin'] = out['bad_origin']; vals['bad_floor'] = out['bad_floor']
            else:
                si, ssid, nn = self.first_si(pid, a, bpar, w)
                if frame == 'patch':                        # A3 control: tangent from the patch frame (rotation about n only)
                    tp = t - nn * dr.dot(t, nn)
                    tp = dr.select(dr.squared_norm(tp) > 1e-12, dr.normalize(tp), dr.normalize(bv - nn * dr.dot(bv, nn)))
                    si.sh_frame = mi.Frame3f(tp, dr.cross(nn, tp), nn)
                    si.wi = si.to_local(w)
                p = si.p
                bur = self.buried(p, sid, nn)
                sing = ~bur & self.singular_point(p, sid, nn)
                ray = mi.Ray3f(p + w * 2e-3, -w)
                pre = ~bur & ~sing
                if synthetic is None:
                    out, cap = self.trace112(ray, Nl, rng, sd, 5, lights, 'probe', first=si, first_ok=pre)
                    cnt = cap['count']
                else:
                    cnt = pre & (si.to_local(w).z > 1e-6)
                    col = synthetic(p, w, nn)
                    out = {f'f2{c}': dr.select(cnt, cc, 0.) for c, cc in enumerate((col.x, col.y, col.z))}
                    out.update({f'fh{c}': F(0.) for c in range(3)}); out['bad_origin'] = F(0.); out['bad_floor'] = F(0.)
                # address check with the camera's own quantizer applied to the constructed point
                apid, abn = self.patch_bin(si, sid, w)
                asub = self.subcell(si, apid)
                mism = cnt & (dr.neq(apid, pid) | dr.neq(abn, bn) | dr.neq(asub, sub))
                # receiver-reaching forward segment (camera event-2 queries always point at a front face)
                fray, fs = self.spawn(si, w)
                fh = self.scene.ray_intersect(fray, cnt & ~fs)
                recv = cnt & ~fs & fh.is_valid() & (fh.to_local(-w).z > 1e-6)
                if synthetic is not None:
                    recv = mi.Bool(cnt)
                cosl = dr.dot(w, nn)
                for meas, wt in (('U', dr.select(cnt, F(1.), F(0.))), ('V', dr.select(recv, F(1.), F(0.))),
                                 ('G', dr.select(recv, cosl, F(0.)))):
                    vals[f'{meas}_w'] = wt
                    vals[f'{meas}_w2'] = wt * wt
                    for c in range(3):
                        f2c, fhc = out[f'f2{c}'], out[f'fh{c}']
                        vals[f'{meas}_f2{c}'] = wt * f2c
                        vals[f'{meas}_fh{c}'] = wt * fhc
                        vals[f'{meas}_sq{c}'] = wt * (f2c + fhc) * (f2c + fhc)
                vals['buried'] = dr.select(bur, F(1.), F(0.))
                vals['singular'] = dr.select(sing, F(1.), F(0.))
                vals['back'] = dr.select(pre & ~cnt, F(1.), F(0.))
                vals['mismatch'] = dr.select(mism, F(1.), F(0.))
                for c in range(3):
                    vals[f'misL{c}'] = dr.select(mism, out[f'f2{c}'] + out[f'fh{c}'], 0.)
                vals['bad_origin'] = out['bad_origin']; vals['bad_floor'] = out['bad_floor']
                vals['escape'] = dr.select(cnt & ~fs & ~fh.is_valid(), F(1.), F(0.))
                vals['backrecv'] = dr.select(cnt & ~fs & fh.is_valid() & (fh.to_local(-w).z <= 1e-6), F(1.), F(0.))
            # device-side float64 accumulation across launches (no per-launch host copies)
            keys = sorted(vals)
            vv = [vals[k] for k in keys]
            dr.eval(vv, slot); dr.sync_thread()
            if not acc:
                acc = {k: dr.zeros(mi.Float64, ncs) for k in keys}
            for k, v in zip(keys, vv):
                dr.scatter_reduce(dr.ReduceOp.Add, acc[k], mi.Float64(v), slot)
            dr.eval([acc[k] for k in keys]); dr.sync_thread()
        acc = {k: np.asarray(v, np.float64) for k, v in acc.items()}
        acc['proposals'] = np.full(ncs, float(n_sub))
        return acc


# ---------------------------------------------------------------------------------------- tables from construction sums
def tables_from(acc, meas, level):
    """Mean E2..E5 radiance table. level 'sub' -> [N*4*64*3] flat; 'coarse' -> pooled over sub-cells [N*64*3].
    Returns (flat table, weight per cell, f2 table, fh table)."""
    w = acc[f'{meas}_w']
    f2 = np.stack([acc[f'{meas}_f2{c}'] for c in range(3)], -1)
    fh = np.stack([acc[f'{meas}_fh{c}'] for c in range(3)], -1)
    if level == 'coarse':
        N = w.size // (NSUB * NB)
        w = w.reshape(N, NSUB, NB).sum(1).ravel()
        f2 = f2.reshape(N, NSUB, NB, 3).sum(1).reshape(-1, 3)
        fh = fh.reshape(N, NSUB, NB, 3).sum(1).reshape(-1, 3)
    den = np.maximum(w, 1e-30)[:, None]
    t2 = f2 / den; th = fh / den
    T = t2 + th
    T[w <= 0] = 0.
    t2[w <= 0] = 0.; th[w <= 0] = 0.
    return T.astype(np.float32), w, t2.astype(np.float32), th.astype(np.float32)


def lum(x):
    return x[..., 0] * LUM[0] + x[..., 1] * LUM[1] + x[..., 2] * LUM[2]
