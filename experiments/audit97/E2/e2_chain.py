"""WL97 audit, track E2: independent reproduction of the WL95A delivery chain (P0-2 / P0-3) by the relation-sampling route.

Own implementation of: relation drawing (state rays), hit patch / outgoing bin, the flight, the Lambert receiver, exitance, the source
encoding, a physical q table, the 2 x 2 (direction x position) exact / projected per-ray arms and an independent area-form reference.
Primitives taken from the frozen code (read-only copy at <R>/code, commit c152120): scene / geometry tables (scene86.Geom through
mat94.ctx_ck2), Mitsuba ray intersection, ctx.shape_id, the ER closure (closure83_dr.DrClosure83). The object under test is the
M64 matrix (ops95a.npz key A4 = quad87 enc_matrix row 1). a95.py and an94.py are NOT imported.

  python e2_chain.py checks <R>            sanity / identity / analytic Lambert checks        -> results/checks.json
  python e2_chain.py senders <R>           source encodings, tables, q table, own M64         -> results/senders_e2.npz, senders.json
  python e2_chain.py flight <R> <D> <tag> <mode> <seed0>   D relation draws (mode shared|indep) -> results/flight_<tag>.npz
  python e2_chain.py aref <R> <log2 nper> <nbatch> <seed0> <tag> [mat] area-form reference    -> results/aref_<tag>.npz
"""
import hashlib, json, math, os, sys, time
import numpy as np

sys.dont_write_bytecode = True
CMD, R = sys.argv[1], sys.argv[2]
CODE = os.path.join(R, 'code', 'experiments')
sys.path.insert(0, os.path.join(CODE, 'ecro_mat94'))
import mat94 as MT  # noqa: E402  (ctx_ck2 only)

NZ, NPHI, NB, KAPPA = 4, 16, 64, 8
K = NB * KAPPA
DOM = 2 * math.pi / NB
S1 = np.arange(1088, 1130)
BOXA = np.arange(1130, 1184)
R95 = os.path.expanduser('~/rest_x95a')
R94 = os.path.expanduser('~/rest_x94')
OUT = os.path.join(R, 'results')


def sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=float)


class Env:
    def __init__(self):
        PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
        MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
        self.PT, self.ctx, self.lt = PT, ctx, lt
        self.mi, self.dr = PT.mi, PT.dr
        G = ctx.G
        self.G = G
        assert G.N == 1270 and (G.PSID[S1] == 5).all() and (G.PMAT[S1] == 1).all() and (G.PSID[BOXA] == 7).all() and (G.PMAT[BOXA] == 4).all()
        assert S.NZ == NZ and S.NPHI == NPHI and S.KAPPA == KAPPA
        self.Tn, self.Bn, self.Nn = G.PT[S1[0]].copy(), G.PB[S1[0]].copy(), G.PN[S1[0]].copy()
        L = MT.SC['lights']
        assert len(L) == 1 and 'aim' not in L[0]
        self.LP = np.array(L[0]['pos'], np.float64)
        self.LI = np.array(L[0]['rgb'], np.float64)
        self.states = np.asarray(ctx.mats.states, np.float32)
        self.rho = float(np.asarray(ctx.mats.albedo)[4])
        r = 5
        self.pa1, self.pa2, self.ph, self.pg = G.RA1[r].copy(), G.RA2[r].copy(), G.RH[r].copy(), G.RG[r].copy()
        self.po = G.RC[r] - self.pa1 * self.ph[0] - self.pa2 * self.ph[1]
        assert int(G.RBASE[r]) == S1[0] and int(self.pg[0] * self.pg[1]) == len(S1)
        self.parea = 4 * self.ph[0] * self.ph[1]
        self._cl = {}

    def closure(self, mat):
        if mat not in self._cl:
            self._cl[mat] = self.PT.CD.DrClosure83('ER', self.states[mat])
        return self._cl[mat]

    def V3(self, a):
        a = np.ascontiguousarray(np.asarray(a, np.float32))
        return self.mi.Vector3f(self.mi.Float(a[:, 0].copy()), self.mi.Float(a[:, 1].copy()), self.mi.Float(a[:, 2].copy()))

    def P3(self, a):
        a = np.ascontiguousarray(np.asarray(a, np.float32))
        return self.mi.Point3f(self.mi.Float(a[:, 0].copy()), self.mi.Float(a[:, 1].copy()), self.mi.Float(a[:, 2].copy()))

    def const3(self, a, pt=False):
        return (self.mi.Point3f if pt else self.mi.Vector3f)(float(a[0]), float(a[1]), float(a[2]))


# ---------------------------------------------------------------------------------------------- own direction / bin conventions
def bin_of_np(loc):
    iz = np.clip(np.floor(loc[..., 2] * NZ), 0, NZ - 1).astype(np.int64)
    ip = np.clip(np.floor((np.arctan2(loc[..., 1], loc[..., 0]) + math.pi) * (NPHI / (2 * math.pi))), 0, NPHI - 1).astype(np.int64)
    return iz * NPHI + ip


def dir_in_bin_np(b, a, c):
    z = (b // NPHI + a) / NZ
    phi = (b % NPHI + c) * (2 * math.pi / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def to_world(E, loc):
    return loc[..., 0:1] * E.Tn + loc[..., 1:2] * E.Bn + loc[..., 2:3] * E.Nn


def to_local(E, v):
    return np.stack([v @ E.Tn, v @ E.Bn, v @ E.Nn], -1)


# ---------------------------------------------------------------------------------------------- Dr.Jit primitives (own wrappers)
def trace(E, o, w, chunk=4_000_000):
    """closest hit: shape id (99 = miss) and hit point."""
    mi, dr = E.mi, E.dr
    n = len(o)
    sid = np.zeros(n, np.int64); hp = np.zeros((n, 3))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        si = E.ctx.scene.ray_intersect(mi.Ray3f(E.P3(o[a:b]), E.V3(w[a:b])))
        s = dr.select(si.is_valid(), E.ctx.shape_id(si), mi.UInt32(99))
        hx, hy, hz = si.p.x, si.p.y, si.p.z
        dr.eval(s, hx, hy, hz); dr.sync_thread()
        sid[a:b] = np.asarray(s)
        hp[a:b, 0], hp[a:b, 1], hp[a:b, 2] = np.asarray(hx), np.asarray(hy), np.asarray(hz)
    return sid, hp


def er_radiance(E, x, v, mat=1, shadow=True, chunk=4_000_000):
    """Once-scattered radiance leaving panel point x towards the unit world direction v (point light, exact directions):
    L = eval(wi = v, wo = light direction) * I / d^2 * visibility; eval = f * cos(theta_light) (closure83 convention, checked in 'checks')."""
    mi, dr = E.mi, E.dr
    n = len(x)
    out = np.zeros((n, 3))
    if n == 0:
        return out
    T_, B_, N_ = E.const3(E.Tn), E.const3(E.Bn), E.const3(E.Nn)
    cl = E.closure(mat)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        X = E.P3(x[a:b]); Vv = E.V3(v[a:b])
        wi = mi.Vector3f(dr.dot(Vv, T_), dr.dot(Vv, B_), dr.dot(Vv, N_))
        dl = E.const3(E.LP, True) - X
        d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
        wl = dl * dr.rcp(dist)
        wo = mi.Vector3f(dr.dot(wl, T_), dr.dot(wl, B_), dr.dot(wl, N_))
        on = (wi.z > 1e-6) & (wo.z > 1e-6)
        if shadow:
            r = mi.Ray3f(X + N_ * 1e-4, wl)
            r.maxt = dist * (1.0 - 1e-4)
            on &= ~E.ctx.scene.ray_test(r, on)
        up = mi.Vector3f(0, 0, 1)
        f = cl.eval(dr.select(on, wi, up), dr.select(on, wo, up), on)
        Lc = [dr.select(on, f[c] * float(E.LI[c]) * dr.rcp(d2), 0.0) for c in range(3)]
        dr.eval(Lc); dr.sync_thread()
        for c in range(3):
            out[a:b, c] = np.asarray(Lc[c])
    return out


def light_vis(E, x):
    mi, dr = E.mi, E.dr
    X = E.P3(x)
    dl = E.const3(E.LP, True) - X
    dist = dr.norm(dl)
    r = mi.Ray3f(X + E.const3(E.Nn) * 1e-4, dl * dr.rcp(dist))
    r.maxt = dist * (1.0 - 1e-4)
    occ = E.ctx.scene.ray_test(r)
    dr.eval(occ); dr.sync_thread()
    return ~np.asarray(occ)


# ---------------------------------------------------------------------------------------------- senders
def source_enc(E, ua, ub):
    """s [42, NB, 3]: patch-mean direct irradiance by incident bin; ua / ub [42, P] point offsets."""
    G = E.G
    p = G.PO[S1][:, None] + ua[..., None] * G.PE1[S1][:, None] + ub[..., None] * G.PE2[S1][:, None]
    P = p.shape[1]
    pf = p.reshape(-1, 3)
    dl = E.LP - pf
    d2 = (dl * dl).sum(1); wl = dl / np.sqrt(d2)[:, None]
    cs = wl @ E.Nn
    vis = light_vis(E, pf) & (cs > 1e-6)
    e = np.where(vis, cs / d2, 0.0)
    bn = bin_of_np(to_local(E, wl))
    slot = np.repeat(np.arange(len(S1)), P) * NB + bn
    s = np.zeros((len(S1) * NB, 3))
    for c in range(3):
        s[:, c] = np.bincount(slot, weights=e * E.LI[c], minlength=len(S1) * NB)
    return (s / P).reshape(len(S1), NB, 3), float(vis.mean())


def q_table(E, nq, seed, chunk=64):
    """Physical once-scattered radiance averaged over patch area x outgoing bin (uniform in both): [42, NB, 3]."""
    G = E.G
    rng = np.random.default_rng(seed)
    q = np.zeros((len(S1), NB, 3))
    bo = np.repeat(np.arange(NB), nq)
    for j, p in enumerate(S1):
        u = rng.random((4, NB * nq))
        x = G.PO[p] + u[0][:, None] * G.PE1[p] + u[1][:, None] * G.PE2[p]
        v = to_world(E, dir_in_bin_np(bo, u[2], u[3]))
        q[j] = er_radiance(E, x, v).reshape(NB, nq, 3).mean(1)
    return q


def m64_own(E, nm, seed):
    """Own estimate of the bin-averaged operator: M[bi, bo, c] = mean over w in bi, v in bo of f(v, w) (f = eval / cos(w))."""
    mi, dr = E.mi, E.dr
    rng = np.random.default_rng(seed)
    cl = E.closure(1)
    M = np.zeros((NB, NB, 3))
    bo = np.repeat(np.arange(NB), nm)
    for bi in range(NB):
        u = rng.random((4, NB * nm))
        w = dir_in_bin_np(np.full(NB * nm, bi), u[0], u[1])
        v = dir_in_bin_np(bo, u[2], u[3])
        ok = (w[:, 2] > 1e-6) & (v[:, 2] > 1e-6)
        f = cl.eval(E.V3(v), E.V3(w), mi.Bool(np.ascontiguousarray(ok)))
        fc = [f[c] for c in range(3)]
        dr.eval(fc); dr.sync_thread()
        for c in range(3):
            M[bi, :, c] = (np.asarray(fc[c], np.float64) / np.maximum(w[:, 2], 1e-6)).reshape(NB, nm).mean(1)
    return M


def deployed_offsets():
    rs = np.random.default_rng(86001)
    us = rs.random((4, 1, K), np.float32)
    usrc = rs.random((2, 1, 16), np.float32)
    return us, usrc


def senders(E):
    t0 = time.time()
    us, usrc = deployed_offsets()
    info = {}
    s16, v16 = source_enc(E, np.broadcast_to(usrc[0].astype(np.float64), (42, 16)), np.broadcast_to(usrc[1].astype(np.float64), (42, 16)))
    rng = np.random.default_rng(9710001)
    P = 16384
    spool, vp = source_enc(E, rng.random((42, P)), rng.random((42, P)))
    info['light_visibility_fraction_on_panel'] = {'s16': v16, 'pool': vp}
    s_prod = E.ctx.source(E.lt).astype(np.float64)[S1]
    info['s16_vs_production'] = {'max_abs': float(np.abs(s16 - s_prod).max()), 'rel_l1': float(np.abs(s16 - s_prod).sum() / s_prod.sum()),
                                 'sum_mine': float(s16.sum()), 'sum_prod': float(s_prod.sum())}
    fn_ops = os.path.join(R95, 'results', 'ops95a.npz')
    A4 = np.load(fn_ops)['A4'].astype(np.float64)
    info['ops95a_sha256'] = sha(fn_ops)
    t1 = time.time()
    Mreg = MT.mq_ext().astype(np.float64)[1]
    info['mq_ext_regen_s'] = time.time() - t1
    info['A4_vs_mq_ext_regen'] = {'max_abs': float(np.abs(A4 - Mreg).max()), 'rel_l1': float(np.abs(A4 - Mreg).sum() / np.abs(Mreg).sum())}
    t1 = time.time()
    Mo_a = m64_own(E, 4096, 9710002); Mo_b = m64_own(E, 4096, 9710003)
    Mo = 0.5 * (Mo_a + Mo_b)
    info['m64_own_s'] = time.time() - t1
    zb = ((np.arange(NB) // NPHI) + 0.5) / NZ
    zz = (zb[:, None] * zb[None, :])[..., None]
    info['A4_vs_m64_own'] = {'flux_rel_l1': float((np.abs(A4 - Mo) * zz).sum() / (Mo * zz).sum()), 'mass_ratio': float((A4 * zz).sum() / (Mo * zz).sum()),
                             'own_split_half_flux_rel_l1': float(0.5 * (np.abs(Mo_a - Mo_b) * zz).sum() / (Mo * zz).sum()),
                             'ring0_rows_flux_rel_l1': float((np.abs(A4 - Mo)[:16] * zz[:16]).sum() / (Mo[:16] * zz[:16]).sum()),
                             'ring0_rows_mass_ratio': float((A4[:16] * zz[:16]).sum() / (Mo[:16] * zz[:16]).sum())}
    t1 = time.time()
    qa = q_table(E, 8192, 9710004); qb = q_table(E, 8192, 9710005)
    q = 0.5 * (qa + qb)
    info['q_table_s'] = time.time() - t1
    wt = (zb * DOM)[None, :, None]
    info['q_split_half_flux_rel_l1'] = float(0.5 * (np.abs(qa - qb) * wt).sum() / (q * wt).sum())
    T = {'M64_s16': np.einsum('pmc,mdc->pdc', s16, A4), 'M64_spool': np.einsum('pmc,mdc->pdc', spool, A4),
         'M64own_spool': np.einsum('pmc,mdc->pdc', spool, Mo), 'q': q, 'qa': qa, 'qb': qb}
    fm = lambda X, Y: {'flux_rel_l1': float((np.abs(X - Y) * wt).sum() / (Y * wt).sum()), 'energy': float((X * wt).sum() / (Y * wt).sum())}
    info['fields'] = {'M64_s16_vs_q': fm(T['M64_s16'], q), 'M64_spool_vs_q': fm(T['M64_spool'], q), 'M64own_spool_vs_q': fm(T['M64own_spool'], q),
                      'M64_s16_vs_M64_spool': fm(T['M64_s16'], T['M64_spool'])}
    fn_s = os.path.join(R95, 'results', 'senders.npz')
    zs = np.load(fn_s)
    info['vs_wl95a_senders'] = {'sha256': sha(fn_s), 'A4_table_max_abs_diff': float(np.abs(zs['A4'] - T['M64_s16']).max()),
                                'A4_table_rel_l1': float(np.abs(zs['A4'] - T['M64_s16']).sum() / zs['A4'].sum()),
                                'q256_vs_my_q': fm(zs['q256'].astype(np.float64), q), 'M64_spool_vs_mine': fm(zs['M64_spool'].astype(np.float64), T['M64_spool'])}
    # incident geometry on the panel
    info['incident'] = {'s16_energy_by_ring': [float(s16[:, r * 16:(r + 1) * 16].sum() / s16.sum()) for r in range(4)],
                        'spool_energy_by_ring': [float(spool[:, r * 16:(r + 1) * 16].sum() / spool.sum()) for r in range(4)]}
    info['total_s'] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, 'senders_e2.npz'), s16=s16, spool=spool, A4=A4, M64own=Mo, **T)
    jd(info, os.path.join(OUT, 'senders.json'))
    print(json.dumps(info, indent=1, default=float), flush=True)


# ---------------------------------------------------------------------------------------------- relation / flight (own)
def make_rays(E, U4):
    """U4 [D, 4, 54, K] offsets -> origins, directions, cos-weighted ray weights (flattened D * 54 * K)."""
    G = E.G
    k = np.arange(K)
    ring = (k // KAPPA) // NPHI; sect = (k // KAPPA) % NPHI
    z = (ring + U4[:, 2]) / NZ
    phi = (sect + U4[:, 3]) * (2 * math.pi / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    lx, ly = r * np.cos(phi), r * np.sin(phi)
    g = lambda a: a[BOXA][None, :, None, :]
    p = g(G.PO) + U4[:, 0][..., None] * g(G.PE1) + U4[:, 1][..., None] * g(G.PE2)
    w = g(G.PT) * lx[..., None] + g(G.PB) * ly[..., None] + g(G.PN) * z[..., None]
    o = p + g(G.PN) * 1e-4
    return o.reshape(-1, 3), w.reshape(-1, 3), (z * DOM / KAPPA).reshape(-1)


def panel_hits(E, sid, hp, w):
    idx = np.nonzero(sid == 5)[0]
    x = hp[idx]; v = -w[idx]
    geo = (np.abs((x - E.po) @ E.Nn) < 1e-4)
    rel = x - E.po
    a = rel @ E.pa1 / (2 * E.ph[0]); b = rel @ E.pa2 / (2 * E.ph[1])
    geo &= (a > -1e-5) & (a < 1 + 1e-5) & (b > -1e-5) & (b < 1 + 1e-5)
    nu, nv = int(E.pg[0]), int(E.pg[1])
    iu = np.clip(np.floor(a * nu), 0, nu - 1).astype(np.int64); iv = np.clip(np.floor(b * nv), 0, nv - 1).astype(np.int64)
    pid = iu + nu * iv
    bo = bin_of_np(to_local(E, v))
    return idx, x, v, pid, bo, bool(geo.all())


def flight(E, U4, tables, nsub, rng, const=False, mat=1):
    """Per-draw per-patch exitance [D, 54, 3] of every arm (Lambert receiver: rho * sum_r rw_r L_r)."""
    G = E.G
    D = U4.shape[0]
    o, w, rw = make_rays(E, U4.astype(np.float64))
    sid, hp = trace(E, o, w)
    idx, x, v, pid, bo, geo_ok = panel_hits(E, sid, hp, w)
    m = len(idx)
    Lr = {}
    if const:
        Lr['const'] = np.ones((m, 3))
    else:
        for nm, Tb in tables.items():
            Lr[nm] = Tb[pid, bo]
        Lr['X1'] = er_radiance(E, x, v, mat)
        Lr['X1c'] = er_radiance(E, G.PC[S1][pid], v, mat)
        xa = np.zeros((m, 3)); xb = np.zeros((m, 3)); xab = np.zeros((m, 3))
        for _ in range(nsub):
            u = rng.random((4, m))
            xr = G.PO[S1][pid] + u[0][:, None] * G.PE1[S1][pid] + u[1][:, None] * G.PE2[S1][pid]
            vr = to_world(E, dir_in_bin_np(bo, u[2], u[3]))
            xa += er_radiance(E, xr, v, mat)            # exact direction, position uniform in the hit patch
            xb += er_radiance(E, x, vr, mat)            # exact position, direction uniform in the outgoing bin
            xab += er_radiance(E, xr, vr, mat)          # both projected (per-ray Monte Carlo version of the q-table lookup)
        Lr['Xdir_patchavg'], Lr['Xpos_binavg'], Lr['Xproj_both'] = xa / nsub, xb / nsub, xab / nsub
    rec = idx // K                                       # d * 54 + i
    out = {}
    for nm, L in Lr.items():
        e = np.zeros((D * 54, 3))
        for c in range(3):
            e[:, c] = np.bincount(rec, weights=rw[idx] * L[:, c], minlength=D * 54)
        out[nm] = (E.rho * e).reshape(D, 54, 3)
    aux = {'hits': np.bincount(idx // (54 * K), minlength=D), 'geo_ok': geo_ok, 'pid': pid, 'bo': bo, 'idx': idx, 'sid': sid}
    return out, aux


def load_tables():
    z = np.load(os.path.join(OUT, 'senders_e2.npz'))
    return {k: z[k].astype(np.float64) for k in ('M64_s16', 'M64_spool', 'M64own_spool', 'q')}


def run_flight(E, D, tag, mode, seed0, chunk=32):
    tables = load_tables()
    t0 = time.time()
    res = {}; hits = []
    for a in range(0, D, chunk):
        b = min(D, a + chunk)
        U4 = np.zeros((b - a, 4, 54, K), np.float32)
        for d in range(a, b):
            rg = np.random.default_rng(seed0 + d)
            U4[d - a] = rg.random((4, 1, K), np.float32) if mode == 'shared' else rg.random((4, 54, K), np.float32)
        out, aux = flight(E, U4, tables, 4, np.random.default_rng(seed0 + 500000 + a))
        assert aux['geo_ok']
        hits.append(aux['hits'])
        for k_, v_ in out.items():
            res.setdefault(k_, []).append(v_.astype(np.float32))
        print('flight', tag, b, '/', D, round(time.time() - t0, 1), 's', flush=True)
    np.savez_compressed(os.path.join(OUT, f'flight_{tag}.npz'), hits=np.concatenate(hits), seed0=seed0, mode=mode,
                        **{k_: np.concatenate(v_) for k_, v_ in res.items()})
    jd({'tag': tag, 'D': D, 'mode': mode, 'seed0': seed0, 'seconds': time.time() - t0, 'aux_rng': 'default_rng(seed0 + 500000 + chunk_start)',
        'nsub': 4}, os.path.join(OUT, f'flight_{tag}.json'))


# ---------------------------------------------------------------------------------------------- area-form reference (own)
def aref_launch(E, nper, seed, mat=1, const=False):
    """sum over nper samples per box-A patch of L(x -> y) cos_y cos_x / r^2 V(x, y) A_panel  (x uniform on the panel, y uniform on the patch)."""
    mi, dr = E.mi, E.dr
    Fl, U = mi.Float, mi.UInt32
    G = E.G
    n = 54 * nper
    if not hasattr(E, '_rt'):
        E._rt = [Fl(np.ascontiguousarray(a[BOXA].astype(np.float32).ravel())) for a in (G.PO, G.PE1, G.PE2, G.PN)]
    lane = dr.arange(U, n)
    i = lane // dr.opaque(U, nper)
    rng = mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, seed), initseq=mi.UInt64(0))
    u = [rng.next_float32() for _ in range(4)]
    g3 = lambda t: mi.Vector3f(dr.gather(Fl, t, i * 3), dr.gather(Fl, t, i * 3 + 1), dr.gather(Fl, t, i * 3 + 2))
    po, pe1, pe2, ny = (g3(t) for t in E._rt)
    y = mi.Point3f(po + u[0] * pe1 + u[1] * pe2)
    a1, a2, N_ = E.const3(E.pa1), E.const3(E.pa2), E.const3(E.Nn)
    x = E.const3(E.po, True) + (u[2] * float(2 * E.ph[0])) * a1 + (u[3] * float(2 * E.ph[1])) * a2
    d = x - y
    r2 = dr.squared_norm(d); r = dr.sqrt(r2)
    wd = d * dr.rcp(r)
    cy = dr.dot(wd, ny); cx = -dr.dot(wd, N_)
    ok = (cy > 1e-6) & (cx > 1e-6)
    ray = mi.Ray3f(y + ny * 1e-4, wd)
    ray.maxt = r - 1e-3
    ok &= ~E.ctx.scene.ray_test(ray, ok)
    geo = dr.select(ok, cy * cx * dr.rcp(r2) * float(E.parea), 0.0)
    if const:
        Lc = [geo, geo, geo]
    else:
        T_, B_ = E.const3(E.Tn), E.const3(E.Bn)
        wi = mi.Vector3f(-dr.dot(wd, T_), -dr.dot(wd, B_), cx)
        dl = E.const3(E.LP, True) - x
        d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
        wl = dl * dr.rcp(dist)
        wo = mi.Vector3f(dr.dot(wl, T_), dr.dot(wl, B_), dr.dot(wl, N_))
        on = ok & (wo.z > 1e-6)
        sr = mi.Ray3f(x + N_ * 1e-4, wl)
        sr.maxt = dist * (1.0 - 1e-4)
        on &= ~E.ctx.scene.ray_test(sr, on)
        up = mi.Vector3f(0, 0, 1)
        f = E.closure(mat).eval(dr.select(on, wi, up), dr.select(on, wo, up), on)
        Lc = [dr.select(on, f[c] * float(E.LI[c]) * dr.rcp(d2) * geo, 0.0) for c in range(3)]
    dr.eval(Lc, i); dr.sync_thread()                      # evaluate before the scatter
    bufs = []
    for c in range(3):
        bfr = dr.zeros(Fl, 54)
        dr.scatter_reduce(dr.ReduceOp.Add, bfr, Lc[c], i)
        bufs.append(bfr)
    dr.eval(bufs); dr.sync_thread()
    # float32 scatter sums over 2^16 lanes per patch: accumulate in float64 across launches
    return np.stack([np.asarray(bb, np.float64) for bb in bufs], 1)


def area_ref(E, log2n, nbatch, seed0, mat=1, const=False, log2launch=15):
    """[nbatch, 54, 3] irradiance estimates (each batch: 2^log2n samples per patch)."""
    nl = 1 << max(0, log2n - log2launch)
    nper = 1 << min(log2n, log2launch)
    out = np.zeros((nbatch, 54, 3))
    for bt in range(nbatch):
        acc = np.zeros((54, 3))
        for l in range(nl):
            acc += aref_launch(E, nper, seed0 * 1000003 + bt * 4099 + l, mat, const)
        out[bt] = acc / (nl * nper)
    return out


def run_aref(E, log2n, nbatch, seed0, tag, mat=1):
    t0 = time.time()
    est = area_ref(E, log2n, nbatch, seed0, mat)
    np.savez_compressed(os.path.join(OUT, f'aref_{tag}.npz'), E=est, rho=E.rho, log2n=log2n, seed0=seed0, mat=mat)
    m = est.mean(0).sum(1); se = est.sum(2).std(0, ddof=1) / math.sqrt(nbatch)
    print('aref', tag, 'seconds', round(time.time() - t0, 1), 'total irradiance', float(m.sum()), 'rel SE of total',
          float(est.sum((1, 2)).std(ddof=1) / math.sqrt(nbatch) / m.sum()), 'median per-patch rel SE', float(np.median(se[m > 0] / m[m > 0])), flush=True)


# ---------------------------------------------------------------------------------------------- checks
def lambert_polygon_irradiance(y, n, V):
    """Lambert's formula: irradiance at points y [m, 3] (normal n) from a fully visible polygon V [4, 3] of unit radiance."""
    E_ = np.zeros(len(y))
    nv = len(V)
    for k in range(nv):
        a = V[k] - y; b = V[(k + 1) % nv] - y
        a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
        cr = np.cross(a, b)
        s = np.linalg.norm(cr, axis=1)
        th = np.arctan2(s, (a * b).sum(1))
        E_ += th * ((cr / s[:, None]) @ n)
    return 0.5 * np.abs(E_)


def checks(E):
    mi, dr = E.mi, E.dr
    G = E.G
    C = {}
    rng = np.random.default_rng(9700001)
    # (1) closure cosine convention on a Lambert state (material 3, albedo 0.7; material 4 = box A, albedo 0.5)
    n = 4096
    a = rng.normal(size=(n, 3)); a[:, 2] = np.abs(a[:, 2]) + 0.02; a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(n, 3)); b[:, 2] = np.abs(b[:, 2]) + 0.02; b /= np.linalg.norm(b, axis=1, keepdims=True)
    for mat in (3, 4):
        f = E.closure(mat).eval(E.V3(a), E.V3(b), True)
        fv = np.stack([np.asarray(f[c], np.float64) for c in range(3)], 1)
        alb = float(np.asarray(E.ctx.mats.albedo)[mat])
        C[f'cos_convention_lambert_mat{mat}'] = {'albedo': alb, 'max_rel_dev_vs_rho_over_pi_cos_second_arg': float(np.abs(fv / (alb / math.pi * b[:, 2:3]) - 1).max()),
                                                 'max_rel_dev_vs_cos_first_arg': float(np.abs(fv / (alb / math.pi * a[:, 2:3]) - 1).max())}
    # (2) ER reciprocity of f = eval / cos(second) on S1 (so that the argument order cannot matter beyond the cosine)
    f1 = E.closure(1).eval(E.V3(a), E.V3(b), True); f2 = E.closure(1).eval(E.V3(b), E.V3(a), True)
    v1 = np.asarray(f1[1], np.float64) / b[:, 2]; v2 = np.asarray(f2[1], np.float64) / a[:, 2]
    C['S1_reciprocity_max_rel'] = float((np.abs(v1 - v2) / np.maximum(np.abs(v1), 1e-12)).max())
    st = E.states[1]
    C['S1_state'] = {'lam_ch0': st[0:8].tolist(), 'au': st[24:31].tolist(), 'av': st[31:38].tolist(), 'f0': st[38:45].tolist(), 'eta': float(st[68])}
    # (3) own frame vs Mitsuba shading frame on the panel (anisotropy would make this matter)
    x = E.po + rng.random((n, 1)) * 2 * E.ph[0] * E.pa1 + rng.random((n, 1)) * 2 * E.ph[1] * E.pa2
    si = E.ctx.scene.ray_intersect(mi.Ray3f(E.P3(x + E.Nn * 0.01), E.V3(np.tile(-E.Nn, (n, 1)))))
    loc = si.to_local(E.V3(a))
    lm = np.stack([np.asarray(loc[c], np.float64) for c in range(3)], 1)
    C['panel_frame_vs_mitsuba_to_local_max_abs'] = float(np.abs(lm - to_local(E, a.astype(np.float32).astype(np.float64))).max())
    # (4) er_radiance with a Lambert panel state against the analytic rho / pi * I cos / d^2
    v = to_world(E, a)
    Lm = er_radiance(E, x, v, mat=3, shadow=False)
    dl = E.LP - x; d2 = (dl * dl).sum(1); cs = (dl / np.sqrt(d2)[:, None]) @ E.Nn
    La = 0.7 / math.pi * E.LI[0] * cs / d2
    C['lambert_panel_radiance_max_rel_dev'] = float(np.abs(Lm[:, 0] / La - 1).max())
    C['panel_incidence_cos_range'] = [float(cs.min()), float(np.median(cs)), float(cs.max())]
    Ls = er_radiance(E, x, v, mat=3, shadow=True)
    C['panel_points_shadowed_fraction'] = float((Ls[:, 0] == 0).mean())
    # (5) bins: own bin_of vs production Ctx.bin_of
    bp = np.asarray(E.PT.Ctx.bin_of(mi.Float(a[:, 0].astype(np.float32)), mi.Float(a[:, 1].astype(np.float32)), mi.Float(a[:, 2].astype(np.float32))))
    C['bin_agreement_vs_production'] = float((bp == bin_of_np(a)).mean())
    # (6) deployed relation identity (own rays vs production ctx.state_rays) on box A
    us, usrc = deployed_offsets()
    C['deployed_offsets_equal_ctx'] = bool(np.array_equal(us, E.ctx.u_state_np) and np.array_equal(usrc, E.ctx.u_src_np))
    hit_p, bh_p, seg_p = E.ctx.state_rays()
    hit_p = hit_p.reshape(G.N, K)[BOXA].astype(np.int64); bh_p = bh_p.reshape(G.N, K)[BOXA].astype(np.int64)
    U4 = np.broadcast_to(us.astype(np.float64)[None], (1, 4, 54, K)).copy()
    out, aux = flight(E, U4, {}, 1, np.random.default_rng(1), const=True)
    mine_hit = np.full(54 * K, -1, np.int64); mine_bin = np.zeros(54 * K, np.int64)
    mine_hit[aux['idx']] = S1[0] + aux['pid']; mine_bin[aux['idx']] = aux['bo']
    prod_s1 = np.isin(hit_p.ravel(), S1)
    both = prod_s1 & (mine_hit >= 0)
    C['deployed_relation_identity'] = {'prod_rays_to_S1_from_boxA': int(prod_s1.sum()), 'mine': int((mine_hit >= 0).sum()),
                                       'same_set': bool(np.array_equal(prod_s1, mine_hit >= 0)),
                                       'patch_agreement': float((hit_p.ravel()[both] == mine_hit[both]).mean()),
                                       'bin_agreement': float((bh_p.ravel()[both] == mine_bin[both]).mean()), 'geo_ok': aux['geo_ok']}
    # (7) analytic Lambert: unit-radiance panel -> irradiance on the 9 panel-facing patches (Lambert polygon formula, 48 x 48 points)
    V = np.array([E.po, E.po + 2 * E.ph[0] * E.pa1, E.po + 2 * E.ph[0] * E.pa1 + 2 * E.ph[1] * E.pa2, E.po + 2 * E.ph[1] * E.pa2])
    gq = (np.arange(48) + 0.5) / 48
    ana = np.zeros(54)
    for j, p in enumerate(BOXA):
        if abs(G.PN[p] @ np.array([0, 0, 1.0])) > 0.5:
            continue
        yy = G.PO[p] + gq[:, None, None] * G.PE1[p] + gq[None, :, None] * G.PE2[p]
        yy = yy.reshape(-1, 3)
        if ((V[:, None, :] - yy[None]) @ G.PN[p]).min() <= 0:      # polygon not fully above the horizon -> formula not applicable
            continue
        ana[j] = lambert_polygon_irradiance(yy, G.PN[p], V).mean()
    use = ana > 0
    D = 256
    acc = np.zeros(54)
    for a0 in range(0, D, 64):
        U4 = np.stack([np.random.default_rng(9700100 + d).random((4, 54, K), np.float32) for d in range(a0, a0 + 64)])
        o_, _ = flight(E, U4, {}, 1, None, const=True)
        acc += o_['const'][..., 0].sum(0) / E.rho
    hemi = acc / D
    ar = area_ref(E, 20, 4, 9700200, const=True)[..., 0]
    C['analytic_form_factor'] = {'patches_checked': BOXA[use].tolist(), 'analytic_irradiance': ana[use].tolist(),
                                 'hemisphere_flight_over_analytic': (hemi[use] / ana[use]).tolist(),
                                 'area_estimator_over_analytic': (ar.mean(0)[use] / ana[use]).tolist(),
                                 'hemi_total_ratio': float(hemi[use].sum() / ana[use].sum()), 'area_total_ratio': float(ar.mean(0)[use].sum() / ana[use].sum()),
                                 'area_total_rel_se': float(ar[:, use].sum(1).std(ddof=1) / 2 / ana[use].sum()),
                                 'all54_hemi_vs_area_total_ratio': float(hemi.sum() / ar.mean(0).sum()),
                                 'all54_hemi_vs_area_max_rel_dev': float(np.abs(hemi / np.maximum(ar.mean(0), 1e-12) - 1)[ar.mean(0) > 1e-3].max())}
    # (8) Lambert panel lit by the point light: hemisphere flight (exact per-ray) vs area estimator vs numpy quadrature (patch 1134)
    acc = np.zeros((54, 3)); acc_c = np.zeros((54, 3))
    for a0 in range(0, D, 64):
        U4 = np.stack([np.random.default_rng(9700300 + d).random((4, 54, K), np.float32) for d in range(a0, a0 + 64)])
        o_, _ = flight(E, U4, {}, 1, np.random.default_rng(5), mat=3)
        acc += o_['X1'].sum(0) / E.rho
    hemiL = acc / D
    arL = area_ref(E, 20, 4, 9700400, mat=3)
    p = 1134
    gy = (np.arange(12) + 0.5) / 12
    yy = (G.PO[p] + gy[:, None, None] * G.PE1[p] + gy[None, :, None] * G.PE2[p]).reshape(-1, 3)
    gu = (np.arange(240) + 0.5) / 240; gv = (np.arange(280) + 0.5) / 280
    xx = (E.po + gu[:, None, None] * 2 * E.ph[0] * E.pa1 + gv[None, :, None] * 2 * E.ph[1] * E.pa2).reshape(-1, 3)
    dl = E.LP - xx; d2 = (dl * dl).sum(1); cs = (dl / np.sqrt(d2)[:, None]) @ E.Nn
    Lx = 0.7 / math.pi * E.LI[0] * cs / d2
    tot = 0.0
    for y0 in yy:
        dd = xx - y0; r2 = (dd * dd).sum(1); wd = dd / np.sqrt(r2)[:, None]
        tot += (Lx * np.maximum(wd @ G.PN[p], 0) * np.maximum(-(wd @ E.Nn), 0) / r2).sum() * E.parea / len(xx)
    quad = tot / len(yy)
    C['lambert_panel_delivery'] = {'patch': p, 'numpy_quadrature': float(quad), 'hemisphere_flight': float(hemiL[p - BOXA[0], 0]),
                                   'area_estimator': float(arL.mean(0)[p - BOXA[0], 0]),
                                   'all54_total_hemi_over_area': float(hemiL.sum() / arL.mean(0).sum()),
                                   'area_total_rel_se': float(arL.sum((1, 2)).std(ddof=1) / 2 / arL.mean(0).sum()),
                                   'per_patch_max_rel_dev_hemi_vs_area': float(np.abs(hemiL[:, 0] / np.maximum(arL.mean(0)[:, 0], 1e-12) - 1)[arL.mean(0)[:, 0] > 0.01 * arL.mean(0)[:, 0].max()].max())}
    # (9) reference field provenance + box-A rows (saved for the local analysis)
    fn = os.path.join(R94, 'results', 'fields', 'CKF_none.npz')
    z = np.load(fn)
    C['CKF_none'] = {'sha256': sha(fn), 'spp_per_half': int(z['spp']), 'seeds': z['seeds'].tolist(), 'pmat_equal_default': bool(np.array_equal(z['PMAT'], G.PMAT)),
                     'e2_p_on_panel_sum': float(z['e2_p_a'][S1].sum() + z['e2_p_b'][S1].sum())}
    np.savez_compressed(os.path.join(OUT, 'ref_boxA.npz'), **{k: z[k][BOXA] for k in ('e2_p_a', 'e2_p_b', 'e2_o_a', 'e2_o_b')},
                        PN=G.PN[BOXA], PC=G.PC[BOXA], PAREA=G.PAREA[BOXA])
    fnd = os.path.join(R95, 'results', 'deliv.npz')
    zd = np.load(fnd)
    np.savez_compressed(os.path.join(OUT, 'wl95a_deliv_boxA.npz'), **{k: (zd[k][..., BOXA] if zd[k].shape[-1] == 1270 else zd[k]) for k in zd.files})
    C['wl95a_deliv_sha256'] = sha(fnd)
    jd(C, os.path.join(OUT, 'checks.json'))
    print(json.dumps(C, indent=1, default=float), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    E = Env()
    if CMD == 'checks':
        checks(E)
    elif CMD == 'senders':
        senders(E)
    elif CMD == 'flight':
        run_flight(E, int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]))
    elif CMD == 'deployed':
        us, _ = deployed_offsets()
        U4 = np.broadcast_to(us[None], (1, 4, 54, K)).copy()
        out, aux = flight(E, U4, load_tables(), 64, np.random.default_rng(9730001))
        np.savez_compressed(os.path.join(OUT, 'flight_deployed.npz'), hits=aux['hits'], **out)
        print('deployed hits', aux['hits'], flush=True)
    elif CMD == 'aref':
        run_aref(E, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]) if len(sys.argv) > 7 else 1)
    print('DONE', CMD, flush=True)
    os._exit(0)
