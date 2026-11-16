"""WL97 audit, track E1: independent reproduction of P0-1
("The learned S1 sender has large avoidable error, and the deployed learned operator may violate passivity").

Independent verification code. It does NOT import or copy a95.py / an94.py or their helpers. It uses, as physical infrastructure only:
  mat94.ctx_ck2 (CK2 scene, Mitsuba ray intersection), pt86.Ctx.state_rays / pt86.Ctx.source (deployed state and source encoding, each
  cross-checked against an own re-implementation), closure83_dr.DrClosure83 (frozen ER closure; convention checked analytically below),
  model86.ECRO via mat94.load_frozen (frozen checkpoint).

  python e1_audit.py conv     conventions: bins, weights, ER cosine convention (Lambert), S1 isotropy / reciprocity / physical albedo, frames
  python e1_audit.py sender   learned sender reconstruction, independent physical q, own M64, metrics, passivity, box-A delivery, redraws
  python e1_audit.py train    training / validation distribution: S1 incident-bin excitation, in-distribution sender error, out / in energy
Environment: E1_CODE (frozen experiments/ dir, read-only), E1_ROOT (own root), E1_ORIG (WL95A results, read-only), E1_R92A, E1_R94, E1_R86.
"""
import glob, hashlib, json, math, os, sys, time
sys.dont_write_bytecode = True
import numpy as np

CODE = os.environ['E1_CODE']; ROOT = os.environ['E1_ROOT']; ORIG = os.environ['E1_ORIG']
R92A = os.environ['E1_R92A']; R94 = os.environ['E1_R94']; R86 = os.environ['E1_R86']
OUT = os.path.join(ROOT, 'results'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(CODE, 'ecro_mat94'))
import mat94 as MT  # noqa: E402

PT, S, _A, FX, ctx, lt = MT.ctx_ck2(ROOT)
mi, dr, F, U, CD = PT.mi, PT.dr, PT.F, PT.U, PT.CD
G = ctx.G
N = G.N
NZ, NPHI, NB, KAPPA, K, PSRC = S.NZ, S.NPHI, S.NB, S.KAPPA, S.K, S.PSRC
DOM = 2 * math.pi / NB
TWO_PI = 2 * math.pi
assert (NZ, NPHI, NB, KAPPA, K, PSRC) == (4, 16, 64, 8, 512, 16)
assert np.array_equal(MT.default_pmat(G), G.PMAT)
LIGHTS = MT.SC['lights']
assert len(LIGHTS) == 1 and 'aim' not in LIGHTS[0]
LPOS = np.array(LIGHTS[0]['pos'], np.float64); LRGB = np.array(LIGHTS[0]['rgb'], np.float64)
S1 = np.nonzero(G.PSID == 5)[0]
BOXA = np.nonzero(G.PSID == S.SID_BOXA)[0]
assert len(S1) == 42 and (G.PMAT[S1] == 1).all() and len(BOXA) == 54 and (G.PMAT[BOXA] == 4).all()
RHO_A = float(np.asarray(ctx.mats.albedo)[4])
# ---- weights derived from scene86: bin d = iz * NPHI + iphi, ring iz covers z = cos(theta) in [iz, iz + 1] / NZ (equal solid angle
# DOM = 2 pi / NB per bin); the mean of cos(theta) over a bin is its ring centre (iz + 1/2) / NZ, so int_bin cos dOmega = zc * DOM.
ZC = ((np.arange(NB) // NPHI) + 0.5) / NZ
WT = ZC * DOM
W94 = ((np.arange(NB) // 8) + 0.5) / 8 * DOM         # the 8 x 8 weights that an94.W codes (for the sensitivity column only)
WU = np.full(NB, math.pi / NB)                        # unweighted control
ST_S1 = np.asarray(ctx.mats.states)[1]


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=float)


def sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def fl(a):
    return F(np.ascontiguousarray(np.asarray(a, np.float32).ravel()))


def v3(a):
    a = np.asarray(a, np.float64)
    return mi.Vector3f(fl(a[:, 0]), fl(a[:, 1]), fl(a[:, 2]))


def gat(v, idx):
    return mi.Vector3f(dr.gather(F, v.x, idx), dr.gather(F, v.y, idx), dr.gather(F, v.z, idx))


def npc(c):
    dr.eval(c); dr.sync_thread()
    return np.stack([np.asarray(c.x, np.float64), np.asarray(c.y, np.float64), np.asarray(c.z, np.float64)], 1)


def bin_np(x, y, z):
    iz = np.clip(np.floor(z * NZ), 0, NZ - 1)
    ip = np.clip(np.floor((np.arctan2(y, x) + math.pi) * NPHI / TWO_PI), 0, NPHI - 1)
    return (iz * NPHI + ip).astype(np.int64)


def dir_in_bin_dr(bn, uz, up):
    z = (F(bn // NPHI) + uz) * (1.0 / NZ)
    phi = (F(bn % NPHI) + up) * (TWO_PI / NPHI) - math.pi
    r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
    return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)


def closure_s1():
    return CD.DrClosure83('ER', ST_S1)


def light_at(p, n):
    """own direct-light geometry at world points p with normal n (Dr.Jit): unit direction to the light, 1 / d^2, visibility."""
    Lp = mi.Point3f(float(LPOS[0]), float(LPOS[1]), float(LPOS[2]))
    dl = Lp - p
    d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
    wl = dl * dr.rcp(dist)
    on = dr.dot(wl, n) > 1e-6
    ray = mi.Ray3f(p + n * 1e-4, wl); ray.maxt = dist * (1.0 - 1e-4)
    vis = on & ~ctx.scene.ray_test(ray, on)
    return wl, dr.rcp(d2), vis


def fm(X, Y, W=WT):
    """field metrics of X against Y ([n, NB, 3]): cos-dOmega-weighted rel L1, energy ratio, peak retention (X at Y's arg-max bin per patch
    and channel, summed, over Y's maxima) and max-to-max ratio."""
    Wd = W[None, :, None]
    pk = Y.argmax(1)[:, None, :]
    return {'rel_l1': float((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum()), 'energy': float((X * Wd).sum() / (Y * Wd).sum()),
            'peak': float(np.take_along_axis(X, pk, 1).sum() / np.take_along_axis(Y, pk, 1).sum()),
            'max_ratio': float(X.max(1).sum() / Y.max(1).sum())}


def dm(a, r):
    return {'ratio': float(a.sum() / r.sum()), 'abs_rel_l1': float(np.abs(a - r).sum() / r.sum()), 'corr': float(np.corrcoef(a, r)[0, 1])}


# ------------------------------------------------------------------------------------------------ own source encoding
def rect_points(pids, a, b):
    return G.PO[pids][:, None, :] + a[None, :, None] * G.PE1[pids][:, None, :] + b[None, :, None] * G.PE2[pids][:, None, :]


def my_source(usrc, pids):
    """s [len(pids), NB, 3]: mean over the PSRC patch points of I cos / d^2 (shadowed), binned by the incident direction in the patch frame."""
    a, b = usrc[0, 0].astype(np.float64), usrc[1, 0].astype(np.float64)
    p = rect_points(pids, a, b)                                         # [n, P, 3]
    n_, P_ = p.shape[:2]
    nn = np.broadcast_to(G.PN[pids][:, None, :], p.shape); tt = np.broadcast_to(G.PT[pids][:, None, :], p.shape)
    bb = np.broadcast_to(G.PB[pids][:, None, :], p.shape)
    dl = LPOS[None, None, :] - p
    d2 = (dl ** 2).sum(-1); dist = np.sqrt(d2)
    wl = dl / dist[..., None]
    cs = (wl * nn).sum(-1)
    on = cs > 1e-6
    o = (p + nn * 1e-4).reshape(-1, 3); d = wl.reshape(-1, 3)
    ray = mi.Ray3f(mi.Point3f(fl(o[:, 0]), fl(o[:, 1]), fl(o[:, 2])), v3(d)); ray.maxt = fl(dist.ravel() * (1.0 - 1e-4))
    occ = np.asarray(ctx.scene.ray_test(ray)).reshape(n_, P_)
    e = np.where(on & ~occ, cs / d2, 0.0)
    bn = bin_np((wl * tt).sum(-1), (wl * bb).sum(-1), cs)
    s = np.zeros((n_, NB, 3))
    np.add.at(s, (np.repeat(np.arange(n_), P_), bn.ravel()), e.ravel()[:, None] * LRGB[None, :])
    return s / P_


# ------------------------------------------------------------------------------------------------ own flight
def ray_w(us):
    k = np.arange(K)
    z = ((k // KAPPA) // NPHI + us[2, 0].astype(np.float64)) / NZ
    return z * DOM / KAPPA


def flight_np(hit, dbin, rw, B):
    """E1 [N, NB, 3]: irradiance by incident bin gathered through the state rays from a sender radiance table B [N, NB, 3]."""
    h = hit.astype(np.int64).reshape(N, K); db = dbin.astype(np.int64).reshape(N, K)
    val = np.where((h >= 0)[..., None], B[np.clip(h, 0, None), db], 0.0)
    return (val * rw[None, :, None]).reshape(N, NB, KAPPA, 3).sum(2)


# ------------------------------------------------------------------------------------------------ independent physical tables
def phys_q(seed, launches, SZ=64):
    """Own Monte Carlo of the bin-averaged once-scattered outgoing radiance of the S1 patches: per (patch, outgoing bin) and launch
    SZ x SZ jittered direction strata uniform in solid angle inside the bin, a uniform random patch point per sample, the exact light
    direction, a shadow ray and one ER evaluation f(v, w_l) cos(w_l) I / d^2. Returns q [42, NB, 3] and qc = bin mean of L cos(theta_o)."""
    spp = SZ * SZ
    nslot = len(S1) * NB
    lanes = nslot * spp
    PO, E1_, E2_, T_, B_, N_ = (v3(x[S1]) for x in (G.PO, G.PE1, G.PE2, G.PT, G.PB, G.PN))
    cl = closure_s1()
    acc = np.zeros((nslot, 3)); accc = np.zeros((nslot, 3))
    for it in range(launches):
        li = dr.arange(U, lanes)
        slot = li // spp; j = li % spp
        pj = slot // NB; bn = slot % NB
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(97))
        a = rng.next_float32(); b = rng.next_float32()
        uz = (F(j // SZ) + rng.next_float32()) * (1.0 / SZ); up = (F(j % SZ) + rng.next_float32()) * (1.0 / SZ)
        v = dir_in_bin_dr(bn, uz, up)
        t, bv, n = gat(T_, pj), gat(B_, pj), gat(N_, pj)
        p = mi.Point3f(gat(PO, pj) + a * gat(E1_, pj) + b * gat(E2_, pj))
        wl, id2, vis = light_at(p, n)
        wloc = mi.Vector3f(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
        f = cl.eval(v, dr.select(vis, wloc, mi.Vector3f(0, 0, 1)), vis)
        Lr = dr.select(vis, f * id2, 0.0) * mi.Color3f(float(LRGB[0]), float(LRGB[1]), float(LRGB[2]))
        x = npc(Lr)
        vz = np.asarray(v.z, np.float64)
        acc += x.reshape(nslot, spp, 3).sum(1); accc += (x * vz[:, None]).reshape(nslot, spp, 3).sum(1)
    d = launches * spp
    return (acc / d).reshape(len(S1), NB, 3), (accc / d).reshape(len(S1), NB, 3)


def own_m64(seed, launches, SB=8):
    """Own bin-to-bin S1 operator: M[m, d, c] = mean over w in incident bin m and v in outgoing bin d (both uniform in solid angle,
    SB x SB jittered strata each) of the BRDF f(v, w) = eval(v, w) / cos(w); Mc = the same with f cos(theta_v) (exact albedo)."""
    spp = SB ** 4
    lanes = NB * NB * spp
    cl = closure_s1()
    acc = np.zeros((NB * NB, 3)); accc = np.zeros((NB * NB, 3))
    for it in range(launches):
        li = dr.arange(U, lanes)
        pair = li // spp; j = li % spp
        m = pair // NB; d = pair % NB
        jw = j // (SB * SB); jv = j % (SB * SB)
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(98))
        w = dir_in_bin_dr(m, (F(jw // SB) + rng.next_float32()) * (1.0 / SB), (F(jw % SB) + rng.next_float32()) * (1.0 / SB))
        v = dir_in_bin_dr(d, (F(jv // SB) + rng.next_float32()) * (1.0 / SB), (F(jv % SB) + rng.next_float32()) * (1.0 / SB))
        ok = (w.z > 1e-4) & (v.z > 1e-6)
        f = cl.eval(v, dr.select(ok, w, mi.Vector3f(0, 0, 1)), ok)
        fr = dr.select(ok, f * dr.rcp(dr.maximum(w.z, 1e-4)), 0.0)
        x = npc(fr)
        vz = np.asarray(v.z, np.float64)
        acc += x.reshape(NB * NB, spp, 3).sum(1); accc += (x * vz[:, None]).reshape(NB * NB, spp, 3).sum(1)
    dnm = launches * spp
    return (acc / dnm).reshape(NB, NB, 3), (accc / dnm).reshape(NB, NB, 3)


def boxa_ref(seed, launches, SG=512, state=None, const=False, hist=None):
    """Own reference for the exactly-2-event, S1-mediated exitance of the box-A patches (channel sum): area sampling of the panel.
    B_x = rho * mean over x in the patch of int_panel L(y -> x) cos_x cos_y / r^2 V(x, y) dA_y, with L(y -> x) the once-scattered panel
    radiance (ER eval at the exact outgoing / light directions, shadowed). SG x SG jittered strata over the panel per launch."""
    spp = SG * SG
    nb = len(BOXA)
    lanes = nb * spp
    PO, E1_, E2_, NA = (v3(x[BOXA]) for x in (G.PO, G.PE1, G.PE2, G.PN))
    r5 = 5                                                         # panel rectangle
    pc, a1, a2, h1, h2 = G.RC[r5], G.RA1[r5], G.RA2[r5], G.RH[r5, 0], G.RH[r5, 1]
    pn = np.cross(a1, a2)
    area = 4 * h1 * h2
    cl = closure_s1() if state is None else CD.DrClosure83('ER', state)
    V = lambda x: mi.Vector3f(float(x[0]), float(x[1]), float(x[2]))
    acc = np.zeros(nb)
    for it in range(launches):
        li = dr.arange(U, lanes)
        pj = li // spp; j = li % spp
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(99))
        x = mi.Point3f(gat(PO, pj) + rng.next_float32() * gat(E1_, pj) + rng.next_float32() * gat(E2_, pj))
        nx = gat(NA, pj)
        su = ((F(j // SG) + rng.next_float32()) * (2.0 / SG) - 1.0) * h1; sv = ((F(j % SG) + rng.next_float32()) * (2.0 / SG) - 1.0) * h2
        y = mi.Point3f(V(pc) + su * V(a1) + sv * V(a2))
        dxy = y - x
        r2 = dr.squared_norm(dxy); r = dr.sqrt(r2)
        om = dxy * dr.rcp(r)
        cx = dr.dot(om, nx); cy = -dr.dot(om, V(pn))
        ok = (cx > 1e-6) & (cy > 1e-6)
        # visibility segment between the two OFFSET end points (x + 1e-4 n_x, y + 1e-4 n_y). A first version shot the ray from the offset
        # origin along the un-offset direction with maxt = r - 2e-4; for cos_y < 0.5 that ray reaches the panel plane before maxt and
        # every grazing sample was wrongly rejected (own defect, found by the constant-radiance form-factor test; see debug stage).
        xo = x + nx * 1e-4; yo = y + V(pn) * 1e-4
        dv = yo - xo
        dn = dr.norm(dv)
        ray = mi.Ray3f(xo, dv * dr.rcp(dn)); ray.maxt = dn * (1.0 - 1e-5)
        ok &= ~ctx.scene.ray_test(ray, ok)
        wl, id2, vis = light_at(y, V(pn))
        vis &= ok
        wi = mi.Vector3f(-dr.dot(om, V(a1)), -dr.dot(om, V(a2)), cy)                 # panel frame (t, b, n) = (a1, a2, a1 x a2)
        wo = mi.Vector3f(dr.dot(wl, V(a1)), dr.dot(wl, V(a2)), dr.dot(wl, V(pn)))
        f = cl.eval(dr.select(vis, wi, mi.Vector3f(0, 0, 1)), dr.select(vis, wo, mi.Vector3f(0, 0, 1)), vis)
        Lsum = (f.x * float(LRGB[0]) + f.y * float(LRGB[1]) + f.z * float(LRGB[2])) * id2
        if const:
            c = dr.select(ok, cx * cy * dr.rcp(r2) * float(area), 0.0)
        else:
            c = dr.select(vis, Lsum * cx * cy * dr.rcp(r2) * float(area), 0.0)
        dr.eval(c, su, sv); dr.sync_thread()
        cn = np.asarray(c, np.float64)
        acc += cn.reshape(nb, spp).sum(1)
        if hist is not None:                                        # contribution map over the panel (16 x 16 cells), all box-A patches
            iu = np.clip(((np.asarray(su) / h1 + 1) * 8).astype(np.int64), 0, 15); iv = np.clip(((np.asarray(sv) / h2 + 1) * 8).astype(np.int64), 0, 15)
            hist += RHO_A * np.bincount(iu * 16 + iv, weights=cn, minlength=256).reshape(16, 16) / (launches * spp)
    return RHO_A * acc / (launches * spp)


def boxa_ref_hemi(seed, launches, spp=1 << 18, state=None, const=False, hist=None):
    """Second own estimator (different sampling): cosine-weighted hemisphere rays from the box-A patch points, ray-traced to the panel."""
    nb = len(BOXA)
    lanes = nb * spp
    PO, E1_, E2_, TA, BA, NA = (v3(x[BOXA]) for x in (G.PO, G.PE1, G.PE2, G.PT, G.PB, G.PN))
    cl = closure_s1() if state is None else CD.DrClosure83('ER', state)
    acc = np.zeros(nb)
    r5 = 5
    pc, a1, a2, h1, h2 = G.RC[r5], G.RA1[r5], G.RA2[r5], G.RH[r5, 0], G.RH[r5, 1]
    Lp = mi.Point3f(float(LPOS[0]), float(LPOS[1]), float(LPOS[2]))
    for it in range(launches):
        li = dr.arange(U, lanes)
        pj = li // spp
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(100))
        x = mi.Point3f(gat(PO, pj) + rng.next_float32() * gat(E1_, pj) + rng.next_float32() * gat(E2_, pj))
        nx = gat(NA, pj)
        u1 = rng.next_float32(); ph = rng.next_float32() * TWO_PI
        rr = dr.sqrt(u1); zz = dr.sqrt(dr.maximum(1.0 - u1, 0.0))
        w = gat(TA, pj) * (rr * dr.cos(ph)) + gat(BA, pj) * (rr * dr.sin(ph)) + nx * zz
        si = ctx.scene.ray_intersect(mi.Ray3f(x + nx * 1e-4, w))
        sid = ctx.shape_id(si)
        wi = si.to_local(-w)
        ok = si.is_valid() & dr.eq(sid, U(5)) & (wi.z > 1e-6)
        dl = Lp - si.p
        d2 = dr.squared_norm(dl)
        wo = si.to_local(dl * dr.rsqrt(d2))
        on = ok & (wo.z > 1e-6)
        vis = on & ~ctx.scene.ray_test(si.spawn_ray_to(Lp), on)
        f = cl.eval(dr.select(vis, wi, mi.Vector3f(0, 0, 1)), dr.select(vis, wo, mi.Vector3f(0, 0, 1)), vis)
        c = dr.select(vis, (f.x * float(LRGB[0]) + f.y * float(LRGB[1]) + f.z * float(LRGB[2])) * dr.rcp(d2), 0.0)
        if const:
            c = dr.select(ok, F(1.0), F(0.0))
        dr.eval(c, si.p); dr.sync_thread()
        cn = np.asarray(c, np.float64)
        acc += cn.reshape(nb, spp).sum(1)
        if hist is not None:
            hp = np.stack([np.asarray(si.p.x, np.float64), np.asarray(si.p.y, np.float64), np.asarray(si.p.z, np.float64)], 1) - pc[None]
            iu = np.clip((((hp * a1[None]).sum(1) / h1 + 1) * 8).astype(np.int64), 0, 15); iv = np.clip((((hp * a2[None]).sum(1) / h2 + 1) * 8).astype(np.int64), 0, 15)
            hist += RHO_A * math.pi * np.bincount((iu * 16 + iv)[cn > 0], weights=cn[cn > 0], minlength=256).reshape(16, 16) / (launches * spp)
    return RHO_A * math.pi * acc / (launches * spp)


# ------------------------------------------------------------------------------------------------ stage: conventions
def conv():
    o = {}
    rng = np.random.default_rng(97001)
    # bins: own formula vs the production bin function, weights
    v = rng.normal(size=(8192, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); v[:, 2] = np.abs(v[:, 2])
    bp = np.asarray(PT.Ctx.bin_of(fl(v[:, 0]), fl(v[:, 1]), fl(v[:, 2])))
    o['bins'] = {'agreement_own_vs_production': float((bp == bin_np(v[:, 0], v[:, 1], v[:, 2])).mean()), 'sum_WT': float(WT.sum()), 'pi': math.pi,
                 'WT_ring_values': [float(WT[16 * r]) for r in range(4)], 'W94_first16': W94[:16].tolist(), 'W94_over_WT_bin0_bin8': [float(W94[0] / WT[0]), float(W94[8] / WT[8])]}
    # MC check that int_bin cos dOmega = zc * DOM (ring 0, sector 0)
    u = rng.random((1 << 20, 2)); z0 = u[:, 0] / NZ
    o['bins']['mc_mean_cos_ring0'] = float(z0.mean()); o['bins']['zc_ring0'] = float(ZC[0])
    # ER cosine convention on a Lambert material (id 3, albedo 0.7): which argument carries the cosine
    n_ = 4096
    a_ = rng.normal(size=(n_, 3)); a_[:, 2] = np.abs(a_[:, 2]) + 0.02; a_ /= np.linalg.norm(a_, axis=1, keepdims=True)
    b_ = rng.normal(size=(n_, 3)); b_[:, 2] = np.abs(b_[:, 2]) + 0.02; b_ /= np.linalg.norm(b_, axis=1, keepdims=True)
    cl3 = CD.DrClosure83('ER', np.asarray(ctx.mats.states)[3])
    f = npc(cl3.eval(v3(a_), v3(b_), True))
    alb = float(np.asarray(ctx.mats.albedo)[3])
    o['lambert'] = {'albedo': alb, 'max_rel_dev_vs_rho_over_pi_cos_second_arg': float(np.abs(f / (alb / math.pi * b_[:, 2:3]) - 1).max()),
                    'max_rel_dev_vs_rho_over_pi_cos_first_arg': float(np.abs(f / (alb / math.pi * a_[:, 2:3]) - 1).max())}
    # S1: reciprocity of f = eval / cos(second), azimuthal isotropy
    cl = closure_s1()
    fab = npc(cl.eval(v3(a_), v3(b_), True)) / b_[:, 2:3]; fba = npc(cl.eval(v3(b_), v3(a_), True)) / a_[:, 2:3]
    o['s1_reciprocity'] = {'max_rel': float((np.abs(fab - fba) / np.maximum(np.maximum(fab, fba), 1e-12)).max()),
                           'flux_weighted_rel': float(np.abs(fab - fba).sum() / fab.sum())}
    th = 0.7
    rot = lambda x: np.stack([x[:, 0] * math.cos(th) - x[:, 1] * math.sin(th), x[:, 0] * math.sin(th) + x[:, 1] * math.cos(th), x[:, 2]], 1)
    f0 = npc(cl.eval(v3(a_), v3(b_), True)); f1 = npc(cl.eval(v3(rot(a_)), v3(rot(b_)), True))
    o['s1_isotropy'] = {'flux_weighted_rel_change_under_0.7rad_rotation': float(np.abs(f0 - f1).sum() / f0.sum())}
    o['s1_state'] = {'au': ST_S1[24:31].tolist(), 'av': ST_S1[31:38].tolist(), 'eta': float(ST_S1[68])}
    # S1 physical directional albedo: (i) BSDF importance sampling, alpha(w) = E[weight] = int eval(w, v) dv = int f(w, v) cos(v) dv;
    # (ii) uniform hemisphere with the light as SECOND argument: int f(v, w) cos(v) dv = 2 pi mean(eval(v, w) / cos(w) * cos(v))
    al = []
    ns = 1 << 22
    for mu in (0.02, 0.035, 0.05, 0.08, 0.125, 0.2, 0.375, 0.625, 0.875, 0.99):
        w = np.tile(np.array([[math.sqrt(1 - mu * mu) * math.cos(0.3), math.sqrt(1 - mu * mu) * math.sin(0.3), mu]]), (1, 1))
        wv = mi.Vector3f(float(w[0, 0]), float(w[0, 1]), float(w[0, 2])) + dr.zeros(mi.Vector3f, ns)
        r = mi.PCG32(size=ns, initstate=dr.opaque(mi.UInt64, 97002), initseq=mi.UInt64(int(mu * 1000)))
        wo, pdf, wgt, valid = cl.sample(wv, r.next_float32(), mi.Point2f(r.next_float32(), r.next_float32()), True)
        a1 = npc(dr.select(valid, wgt, 0.0)).mean(0)
        zu = r.next_float32(); pu = r.next_float32() * TWO_PI
        ru = dr.sqrt(dr.maximum(1 - zu * zu, 0.0))
        vu = mi.Vector3f(ru * dr.cos(pu), ru * dr.sin(pu), zu)
        fu = cl.eval(vu, wv, vu.z > 1e-6)
        a2 = npc(fu * (vu.z / mu)).mean(0) * TWO_PI
        al.append({'mu': mu, 'albedo_importance': a1.tolist(), 'albedo_uniform_light_second': a2.tolist()})
    o['s1_physical_albedo'] = al
    # frame: Mitsuba shading frame of the panel vs the patch frame (t, b, n) of scene86
    pj = S1[rng.integers(0, 42, 512)]
    p = G.PC[pj]; d = -G.PN[pj] + 0.3 * rng.normal(size=(512, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    o_ = p - d * 0.5
    si = ctx.scene.ray_intersect(mi.Ray3f(mi.Point3f(fl(o_[:, 0]), fl(o_[:, 1]), fl(o_[:, 2])), v3(d)))
    tl = npc(si.to_local(v3(-d)))
    pl = np.stack([(-d * G.PT[pj]).sum(1), (-d * G.PB[pj]).sum(1), (-d * G.PN[pj]).sum(1)], 1)
    onp = np.asarray(dr.eq(ctx.shape_id(si), U(5)) & si.is_valid())
    o['frame'] = {'panel_hits': int(onp.sum()), 'max_abs_diff_to_local_vs_patch_frame': float(np.abs(tl - pl)[onp].max())}
    # geometry facts of the workload
    cs = ((LPOS[None] - G.PC[S1]) * G.PN[S1]).sum(1) / np.linalg.norm(LPOS[None] - G.PC[S1], axis=1)
    o['workload'] = {'light_pos': LPOS.tolist(), 'light_rgb': LRGB.tolist(), 'panel_plane_x': float(G.RC[5][0]), 'light_height_above_panel_plane_m': float(LPOS[0] - G.RC[5][0]),
                     'cos_incidence_at_S1_patch_centres_min_med_max': [float(cs.min()), float(np.median(cs)), float(cs.max())],
                     'S1_ids': [int(S1[0]), int(S1[-1])], 'BOXA_ids': [int(BOXA[0]), int(BOXA[-1])], 'rho_boxA': RHO_A}
    jd(o, os.path.join(OUT, 'conv.json'))
    print(json.dumps(o, indent=1, default=float), flush=True)


# ------------------------------------------------------------------------------------------------ stage: sender
def load_models():
    import torch
    import model86 as M86
    mods = {}
    for sd in (8601, 8602):
        m = MT.load_frozen(R92A, sd)
        mods[sd] = m
    return torch, M86, mods


def factors_np(torch, M86, model, hit, dbin):
    gs = M86.GeoState(G, hit, dbin, torch.device('cuda'))
    with torch.no_grad():
        Fz = model.factors(gs)
    return gs, Fz, Fz['c'].double().cpu().numpy(), Fz['a'].double().cpu().numpy()


def sender():
    o = {}
    t0 = time.time()
    # --- deployed random tables: own regeneration from the protocol seed
    rs = np.random.default_rng(S.P['representation'].get('table_seed', 86001))
    us = rs.random((4, 1, K), np.float32); usrc = rs.random((2, 1, PSRC), np.float32)
    o['tables'] = {'u_state_equal': bool(np.array_equal(us, ctx.u_state_np)), 'u_src_equal': bool(np.array_equal(usrc, ctx.u_src_np))}
    rw = ray_w(us)
    # --- source encoding: production vs own
    s_prod = ctx.source(lt).astype(np.float64)
    rect = np.arange(G.SBASE)
    s_own = my_source(usrc, rect)
    o['source'] = {'max_abs_diff_rect_patches': float(np.abs(s_own - s_prod[rect]).max()), 'max_prod': float(s_prod.max()),
                   'rel_l1_S1': float(np.abs(s_own[S1] - s_prod[S1]).sum() / s_prod[S1].sum()),
                   'S1_irradiance_by_ring': (s_prod[S1].sum((0, 2)).reshape(4, 16).sum(1) / s_prod[S1].sum()).tolist(),
                   'S1_nonzero_incident_bins': np.nonzero(s_prod[S1].sum((0, 2)) > 0)[0].tolist(),
                   'S1_irradiance_by_bin': {int(m): float(v) for m, v in enumerate(s_prod[S1].sum((0, 2)) / s_prod[S1].sum()) if v > 0},
                   'S1_patch_irradiance_min_med_max': [float(x) for x in np.quantile(s_prod[S1].sum((1, 2)) / 3, [0, 0.5, 1])]}
    s16 = s_prod
    # --- deployed state
    hit, dbin, seg = ctx.state_rays()
    o['state'] = {'digest': hashlib.sha256(hit.tobytes() + dbin.tobytes()).hexdigest()[:16], 'rays_hitting_S1': int(np.isin(hit, S1).sum()),
                  'sum_rw': float(rw.sum())}
    # --- learned factors and sender
    torch, M86, mods = load_models()
    orig = dict(np.load(os.path.join(ORIG, 'senders.npz')))
    ops = dict(np.load(os.path.join(ORIG, 'ops95a.npz')))
    M1 = ops['A4'].astype(np.float64)
    o['inputs'] = {'senders_npz_sha256': sha(os.path.join(ORIG, 'senders.npz'))[:16], 'ops95a_npz_sha256': sha(os.path.join(ORIG, 'ops95a.npz'))[:16],
                   'ckpt_sha256': {sd: sha(os.path.join(R92A, 'results', 'models', f'own_s{sd}.pt'))[:16] for sd in (8601, 8602)},
                   'CKF_none_sha256': sha(os.path.join(R94, 'results', 'fields', 'CKF_none.npz'))[:16]}
    Bq = {}; C = {}; Afac = {}
    o['learned'] = {}
    for sd, model in mods.items():
        gs, Fz, c, a = factors_np(torch, M86, model, hit, dbin)
        C[sd], Afac[sd] = c, a
        u = np.einsum('nmcr,nmc->ncr', c, s16)
        Bq[sd] = np.einsum('ndcr,ncr->ndc', a, u)                          # learned once-scattered outgoing radiance, all patches
        # the deployed single-flight term from the production apply() vs own chain (own flight of own Bq, own receiver contraction)
        with torch.no_grad():
            Bs_prod = model.apply(Fz, gs, torch.as_tensor(s16.astype(np.float32), device='cuda').view(1, N, NB, 3), use_global=False)[0].double().cpu().numpy()
        E1 = flight_np(hit, dbin, rw, Bq[sd])
        Bs_own = np.einsum('ndcr,ncr->ndc', a, np.einsum('nmcr,nmc->ncr', c, E1))
        st = orig[f'A0_s{sd}'].astype(np.float64)
        o['learned'][sd] = {'stored_sender_max_abs_diff': float(np.abs(Bq[sd][S1] - st).max()), 'stored_sender_max': float(st.max()),
                            'stored_sender_rel_l1': float(np.abs(Bq[sd][S1] - st).sum() / st.sum()),
                            'single_flight_term_own_vs_production_apply_rel_l1': float(np.abs(Bs_own - Bs_prod).sum() / Bs_prod.sum()),
                            'rw_buffer_max_abs_diff': float(np.abs(model.rw.double().cpu().numpy() - rw).max()),
                            'factor_min': [float(c.min()), float(a.min())]}
    # --- independent physical q (two halves, different seeds), own M64 (two halves)
    t1 = time.time()
    qa, qca = phys_q(97011, 4); qb, qcb = phys_q(97012, 4)
    q = 0.5 * (qa + qb); qc = 0.5 * (qca + qcb)
    o['timing_q_s'] = time.time() - t1
    t1 = time.time()
    Ma, Mca = own_m64(97021, 2); Mb, Mcb = own_m64(97022, 2)
    Mo = 0.5 * (Ma + Mb); Mco = 0.5 * (Mca + Mcb)
    o['timing_m64_s'] = time.time() - t1
    snd = {'own_q': q, 'stored_q256': orig['q256'].astype(np.float64), 'stored_A4_M64s16': orig['A4'].astype(np.float64),
           'stored_M64_spool': orig['M64_spool'].astype(np.float64), 'ownM64_s16': np.einsum('nmc,mdc->ndc', s16[S1], Mo),
           'A4op_s16': np.einsum('nmc,mdc->ndc', s16[S1], M1), 'A0_s8601': Bq[8601][S1], 'A0_s8602': Bq[8602][S1]}
    for k_ in ('A1', 'A2', 'A3'):
        snd[k_] = np.einsum('nmc,mdc->ndc', s16[S1], ops[k_].astype(np.float64))
    o['q'] = {'own_split_half_rel_l1_half': 0.5 * fm(qa, qb)['rel_l1'], 'own_qa_vs_qb': fm(qa, qb),
              'stored_q256_vs_own_q': fm(snd['stored_q256'], q), 'stored_qA_vs_qB': fm(orig['qA'].astype(np.float64), orig['qB'].astype(np.float64)),
              'A4op_s16_vs_stored_A4': fm(snd['A4op_s16'], snd['stored_A4_M64s16']),
              'exitance_exact_cos_over_WT_exitance_own_q': float((qc.sum(1) * DOM).sum() / (q * WT[None, :, None]).sum()),
              'own_q_energy_by_ring': ((q * WT[None, :, None]).sum((0, 2)).reshape(4, 16).sum(1) / (q * WT[None, :, None]).sum()).tolist()}
    o['m64'] = {'own_split_half_flux_l1_half': 0.5 * float((np.abs(Ma - Mb) * (ZC[:, None, None] * ZC[None, :, None])).sum() / (Mo * (ZC[:, None, None] * ZC[None, :, None])).sum()),
                'own_vs_stored_A4_flux_l1': float((np.abs(Mo - M1) * (ZC[:, None, None] * ZC[None, :, None])).sum() / (M1 * (ZC[:, None, None] * ZC[None, :, None])).sum()),
                'own_vs_stored_A4_flux_l1_by_incident_ring': [float((np.abs(Mo - M1)[16 * r:16 * r + 16] * ZC[None, :, None]).sum() / (M1[16 * r:16 * r + 16] * ZC[None, :, None]).sum()) for r in range(4)],
                'ownM64_s16_vs_stored_A4': fm(snd['ownM64_s16'], snd['stored_A4_M64s16'])}
    # --- metrics of every sender against every reference, three weightings
    refs = ('own_q', 'stored_q256', 'stored_A4_M64s16', 'ownM64_s16')
    met = {}
    for k_, X in snd.items():
        met[k_] = {r_: {wn: fm(X, snd[r_], W_) for wn, W_ in (('WT', WT), ('W94_as_coded', W94), ('unweighted', WU))} for r_ in refs if r_ != k_}
    o['sender_metrics'] = met
    o['sender_energy_by_ring_WT'] = {k_: ((X * WT[None, :, None]).sum((0, 2)).reshape(4, 16).sum(1) / (q * WT[None, :, None]).sum()).tolist() for k_, X in snd.items()}
    # per-channel breakdown for the learned senders against own q
    o['sender_metrics_per_channel_vs_own_q'] = {k_: [fm(snd[k_][..., c:c + 1], q[..., c:c + 1]) for c in range(3)] for k_ in ('A0_s8601', 'A0_s8602', 'stored_A4_M64s16')}
    # --- passivity: directional albedo of a patch-local operator A [in bin m, out bin d, ch]
    # A maps irradiance by incident bin (W m^-2, s) to bin-averaged outgoing radiance; the exitance it produces from unit irradiance
    # arriving in bin m is sum_d A[m, d] int_bin cos dOmega = sum_d A[m, d] WT[d]. Passive <=> this is <= 1 for every m (then, by
    # non-negativity and linearity, exitance <= irradiance for every non-negative input).
    def alb_op(Aop):
        return (Aop * WT[None, :, None]).sum(1)                             # [in bin, ch]
    pas = {}
    pas['stored_A4_M64'] = {'albedo_max': float(alb_op(M1).max()), 'albedo_by_ring_mean': alb_op(M1).reshape(4, 16, 3).mean((1, 2)).tolist(),
                            'albedo_by_ring_max': alb_op(M1).reshape(4, 16, 3).max((1, 2)).tolist()}
    pas['own_M64'] = {'albedo_max_WT': float(alb_op(Mo).max()), 'albedo_by_ring_mean_WT': alb_op(Mo).reshape(4, 16, 3).mean((1, 2)).tolist(),
                      'albedo_by_ring_max_WT': alb_op(Mo).reshape(4, 16, 3).max((1, 2)).tolist(),
                      'albedo_exact_cos_max': float((Mco.sum(1) * DOM).max()), 'albedo_exact_cos_by_ring_mean': (Mco.sum(1) * DOM).reshape(4, 16, 3).mean((1, 2)).tolist()}
    for k_ in ('A1', 'A2', 'A3'):
        pas[k_] = {'albedo_max': float(alb_op(ops[k_].astype(np.float64)).max())}
    s1s = s16[S1]
    alb_all = {}
    for sd in (8601, 8602):
        c, a = C[sd], Afac[sd]
        aw = np.einsum('ndcr,d->ncr', a, WT)                                 # exitance of each channel activation
        al = np.einsum('nmcr,ncr->nmc', c, aw)                               # [N, in bin, ch] directional albedo, every patch
        alb_all[sd] = al
        x = al[S1]
        viol = x > 1.0
        per_bin_max = x.max((0, 2)); per_bin_frac = viol.mean((0, 2))
        ein = s1s.sum(1); eout = (Bq[sd][S1] * WT[None, :, None]).sum(1)     # [42, 3]
        exc = s1s > 0
        d = {'albedo_max': float(x.max()), 'albedo_median': float(np.median(x)), 'albedo_mean': float(x.mean()), 'albedo_p90': float(np.quantile(x, 0.9)),
             'fraction_entries_gt1': float(viol.mean()), 'fraction_incident_bins_with_any_violation': float((per_bin_max > 1).mean()),
             'patches_with_any_violation': int(viol.any((1, 2)).sum()), 'argmax_patch_bin_ch': [int(v) for v in np.unravel_index(x.argmax(), x.shape)],
             'max_by_incident_ring': x.reshape(42, 4, 16, 3).max((0, 2, 3)).tolist(), 'mean_by_incident_ring': x.reshape(42, 4, 16, 3).mean((0, 2, 3)).tolist(),
             'fraction_gt1_by_incident_ring': viol.reshape(42, 4, 16, 3).mean((0, 2, 3)).tolist(),
             'max_by_channel': x.max((0, 1)).tolist(), 'fraction_gt1_by_channel': viol.mean((0, 1)).tolist(),
             'per_incident_bin_max': per_bin_max.tolist(), 'per_incident_bin_fraction_gt1': per_bin_frac.tolist(),
             # the CK2 input actually applied
             'ck2_excited_entries': int(exc.sum()), 'ck2_excited_entries_gt1': int((viol & exc).sum()),
             'ck2_albedo_on_excited_entries_min_med_max': [float(v) for v in np.quantile(x[exc], [0, 0.5, 1])],
             'ck2_energy_out_over_in_total': float(eout.sum() / ein.sum()), 'ck2_energy_out_over_in_per_channel': (eout.sum(0) / ein.sum(0)).tolist(),
             'ck2_energy_out_over_in_per_patch_min_med_max': [float(v) for v in np.quantile(eout.sum(1) / ein.sum(1), [0, 0.5, 1])],
             'ck2_patches_out_gt_in': int((eout.sum(1) > ein.sum(1)).sum()),
             'ck2_irradiance_share_on_entries_gt1': float(s1s[viol].sum() / s1s.sum())}
        # every patch of the scene, by material
        d['scene_by_material'] = {int(m): {'n_patches': int((G.PMAT == m).sum()), 'albedo_max': float(al[G.PMAT == m].max()), 'albedo_mean': float(al[G.PMAT == m].mean()),
                                           'fraction_entries_gt1': float((al[G.PMAT == m] > 1).mean()), 'patches_with_any_violation': int((al[G.PMAT == m] > 1).any((1, 2)).sum())}
                                  for m in np.unique(G.PMAT)}
        pas[f'A0_s{sd}'] = d
    eq = (q * WT[None, :, None]).sum(1); em = (snd['stored_A4_M64s16'] * WT[None, :, None]).sum(1)
    pas['ck2_energy_out_over_in'] = {'own_q_WT': float(eq.sum() / s1s.sum()), 'own_q_exact_cos': float((qc.sum(1) * DOM).sum() / s1s.sum()),
                                     'M64_s16': float(em.sum() / s1s.sum())}
    o['passivity'] = pas
    # --- delivery to box A through the deployed relation (S1 rows only as sender), exact Lambert receiver: exitance = rho * sum_m E1
    t1 = time.time()
    ra = boxa_ref(97031, 4); rb = boxa_ref(97032, 4)
    rh_a = boxa_ref_hemi(97041, 8); rh_b = boxa_ref_hemi(97042, 8)
    o['timing_boxa_ref_s'] = time.time() - t1
    ref_own = 0.5 * (ra + rb); ref_h = 0.5 * (rh_a + rh_b)
    z = np.load(os.path.join(R94, 'results', 'fields', 'CKF_none.npz'))
    ex = lambda T: (T.astype(np.float64) * WT[None, :, None]).sum((1, 2))
    sa, sb = ex(z['e2_p_a'])[BOXA], ex(z['e2_p_b'])[BOXA]
    ref_st = 0.5 * (sa + sb)
    o['boxa_reference'] = {'own_area_split_half_noise': float(0.5 * np.abs(ra - rb).sum() / ref_own.sum()),
                           'own_hemi_split_half_noise': float(0.5 * np.abs(rh_a - rh_b).sum() / ref_h.sum()),
                           'own_hemi_vs_own_area': dm(ref_h, ref_own), 'stored_split_half_noise': float(0.5 * np.abs(sa - sb).sum() / ref_st.sum()),
                           'stored_e2p_vs_own_area': dm(ref_st, ref_own), 'e2_p_on_S1_sum': float(z['e2_p_a'][S1].sum()),
                           'total_exitance_own_area': float(ref_own.sum()), 'total_exitance_stored': float(ref_st.sum())}

    def deliver(Bs1):
        Bf = np.zeros((N, NB, 3)); Bf[S1] = Bs1
        return RHO_A * flight_np(hit, dbin, rw, Bf).sum((1, 2))[BOXA]
    dl = {}
    for k_ in ('A0_s8601', 'A0_s8602', 'stored_A4_M64s16', 'ownM64_s16', 'own_q', 'stored_q256', 'A1', 'A2', 'A3'):
        dv = deliver(snd[k_])
        dl[k_] = {'vs_stored_e2p': dm(dv, ref_st), 'vs_own_reference': dm(dv, ref_own), 'value_at_ref_peak_ratio': float(dv[ref_own.argmax()] / ref_own.max()),
                  'arm_peak_patch': int(dv.argmax()), 'ref_peak_patch': int(ref_own.argmax())}
    o['delivery_boxA_deployed'] = dl
    oz = np.load(os.path.join(ORIG, 'deliv.npz'))
    o['delivery_stored_check'] = {k_: float(np.abs(deliver(snd[k_]) - oz[f'dep|{k2}'][BOXA]).max() / oz[f'dep|{k2}'][BOXA].max())
                                  for k_, k2 in (('A0_s8601', 'A0_s8601'), ('A0_s8602', 'A0_s8602'), ('stored_A4_M64s16', 'A4'))}
    # --- sensitivity to the 16 fixed source points: redraw the source points (own source code), fixed learned factors / deployed relation
    red = {k_: [] for k_ in ('A0_s8601', 'A0_s8602', 'M64')}
    for j in range(32):
        uj = np.random.default_rng(97100 + j).random((2, 1, PSRC)).astype(np.float32)
        sj = my_source(uj, S1)
        for k_, Bj in (('A0_s8601', np.einsum('ndcr,ncr->ndc', Afac[8601][S1], np.einsum('nmcr,nmc->ncr', C[8601][S1], sj))),
                       ('A0_s8602', np.einsum('ndcr,ncr->ndc', Afac[8602][S1], np.einsum('nmcr,nmc->ncr', C[8602][S1], sj))),
                       ('M64', np.einsum('nmc,mdc->ndc', sj, M1))):
            r_ = fm(Bj, q); dv = dm(deliver(Bj), ref_own)
            red[k_].append([r_['rel_l1'], r_['energy'], r_['peak'], dv['ratio'], dv['abs_rel_l1'], float((Bj * WT[None, :, None]).sum() / sj.sum())])
    o['source_point_redraws'] = {k_: {'columns': ['rel_l1_vs_own_q', 'energy_vs_own_q', 'peak_vs_own_q', 'boxA_ratio_vs_own_ref', 'boxA_abs_rel_l1_vs_own_ref', 'energy_out_over_in'],
                                      'mean': np.mean(v, 0).tolist(), 'min': np.min(v, 0).tolist(), 'max': np.max(v, 0).tolist(), 'sd': np.std(v, 0).tolist()} for k_, v in red.items()}
    # --- sensitivity of the learned sender to the state realization (the factors read the state rays through the form factors)
    stt = {sd: [] for sd in (8601, 8602)}
    u0 = ctx.u_state_np.copy()
    for j in range(6):
        ctx.u_state_np = np.random.default_rng(97200 + j).random(u0.shape).astype(np.float32); ctx.set_tables(G)
        hj, dj, _ = ctx.state_rays()
        for sd, model in mods.items():
            _, _, cj, aj = factors_np(torch, M86, model, hj, dj)
            Bj = np.einsum('ndcr,ncr->ndc', aj[S1], np.einsum('nmcr,nmc->ncr', cj[S1], s16[S1]))
            alj = np.einsum('nmcr,ncr->nmc', cj[S1], np.einsum('ndcr,d->ncr', aj[S1], WT))
            r_ = fm(Bj, q)
            stt[sd].append([fm(Bj, Bq[sd][S1])['rel_l1'], r_['rel_l1'], r_['energy'], r_['peak'], float(alj.max())])
    ctx.u_state_np = u0; ctx.set_tables(G)
    o['state_redraws'] = {sd: {'columns': ['rel_l1_vs_deployed_learned_sender', 'rel_l1_vs_own_q', 'energy_vs_own_q', 'peak_vs_own_q', 'albedo_max'],
                               'rows': v} for sd, v in stt.items()}
    o['seconds'] = time.time() - t0
    o['gpu_mem_peak_mib_torch'] = torch.cuda.max_memory_allocated() / 2 ** 20
    np.savez_compressed(os.path.join(OUT, 'sender_e1.npz'), qa=qa, qb=qb, qca=qca, qcb=qcb, Ma=Ma, Mb=Mb, Mca=Mca, Mcb=Mcb, s16_S1=s16[S1], s_own_S1=s_own[S1],
                        Bq_s8601=Bq[8601][S1], Bq_s8602=Bq[8602][S1], alb_s8601=alb_all[8601], alb_s8602=alb_all[8602], pmat=G.PMAT,
                        boxa_ref_a=ra, boxa_ref_b=rb, boxa_hemi_a=rh_a, boxa_hemi_b=rh_b, boxa_stored_a=sa, boxa_stored_b=sb,
                        **{f'deliv_{k_}': deliver(snd[k_]) for k_ in ('A0_s8601', 'A0_s8602', 'stored_A4_M64s16', 'own_q')})
    jd(o, os.path.join(OUT, 'sender.json'))
    print(json.dumps(o, indent=1, default=float), flush=True)


# ------------------------------------------------------------------------------------------------ stage: training distribution
def train():
    import torch
    import model86 as M86
    dev = torch.device('cuda')
    mods = {sd: MT.load_frozen(R92A, sd) for sd in (8601, 8602)}
    M1 = np.load(os.path.join(ORIG, 'ops95a.npz'))['A4'].astype(np.float64)
    M1t = torch.as_tensor(M1, device=dev)
    WTt = torch.as_tensor(WT, device=dev)
    exd = json.load(open(os.path.join(R92A, 'results', 'exclusion.json')))['excluded']
    S1t = torch.as_tensor(S1, device=dev)
    o = {}
    for sp, step in (('train', 4), ('val', 1)):
        ex = {e[0] for e in exd.get(sp, [])}
        fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in ex][::step]
        acc_s = np.zeros(NB); cnt_s = np.zeros(NB); n_pl = 0
        acc_e1 = {sd: np.zeros(NB) for sd in mods}
        rows = {sd: [] for sd in mods}
        num = {sd: 0.0 for sd in mods}; den = 0.0; eo = {sd: 0.0 for sd in mods}; em = 0.0; ei = 0.0
        albmax = {sd: [] for sd in mods}; viol_irr = {sd: 0.0 for sd in mods}
        for fn in fns:
            z = np.load(fn)
            g = type('G', (), {})()
            g.N = z['pc'].shape[0]
            g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
            assert (g.PMAT[S1] == 1).all()
            hit, dbin = z['hit'], z['dbin']
            s = torch.as_tensor(z['s'].astype(np.float64), device=dev)                    # [L, N, NB, 3]
            Ls = s.shape[0]
            ss = s[:, S1t]                                                               # [L, 42, NB, 3]
            acc_s += ss.sum((0, 1, 3)).cpu().numpy(); cnt_s += (ss.sum(3) > 0).sum((0, 1)).cpu().numpy(); n_pl += Ls * 42
            Bm = torch.einsum('lnmc,mdc->lndc', ss, M1t)
            gs = M86.GeoState(g, hit, dbin, dev)
            hh = gs.hit; ok = (hh >= 0).view(1, g.N * K, 1); idx = (hh.clamp(min=0) * NB + gs.dbin).view(-1)
            rwt = None
            for sd, model in mods.items():
                with torch.no_grad():
                    Fz = model.factors(gs)
                c, a = Fz['c'].double(), Fz['a'].double()
                Bq = torch.einsum('ndcr,lncr->lndc', a, torch.einsum('nmcr,lnmc->lncr', c, s))
                Bl = Bq[:, S1t]
                al = torch.einsum('nmcr,ncr->nmc', c[S1t], torch.einsum('ndcr,d->ncr', a[S1t], WTt))     # [42, NB, 3]
                albmax[sd].append(float(al.max()))
                viol_irr[sd] += float((ss * (al > 1).unsqueeze(0)).sum())
                e_abs = ((Bl - Bm).abs() * WTt.view(1, 1, NB, 1)).sum((1, 2, 3)); e_m = (Bm * WTt.view(1, 1, NB, 1)).sum((1, 2, 3))
                e_l = (Bl * WTt.view(1, 1, NB, 1)).sum((1, 2, 3)); e_in = ss.sum((1, 2, 3))
                r0 = ss[:, :, :16].sum((1, 2, 3))
                for l in range(Ls):
                    if float(e_in[l]) > 0:
                        rows[sd].append([float(e_abs[l] / e_m[l]), float(e_l[l] / e_m[l]), float(e_l[l] / e_in[l]), float(e_m[l] / e_in[l]), float(r0[l] / e_in[l]), float(e_in[l])])
                num[sd] += float(e_abs.sum()); eo[sd] += float(e_l.sum())
                # receiver-role excitation of the S1 patches: single learned flight of the model's own once-scattered table
                rwt = model.rw.double()
                Lk = Bq.reshape(Ls, g.N * NB, 3)[:, idx] * ok
                E1 = (Lk.view(Ls, g.N, NB, KAPPA, 3) * rwt.view(1, 1, NB, KAPPA, 1)).sum(3)
                acc_e1[sd] += E1[:, S1t].sum((0, 1, 3)).cpu().numpy()
            den += float(e_m.sum()); em += float(e_m.sum()); ei += float(e_in.sum())
        d = {'files': len(fns), 'direct_irradiance_share_by_incident_ring': (acc_s.reshape(4, 16).sum(1) / acc_s.sum()).tolist(),
             'direct_irradiance_share_by_incident_bin': (acc_s / acc_s.sum()).tolist(),
             'fraction_of_patch_lightings_exciting_bin': (cnt_s / n_pl).tolist(), 'incident_bins_never_excited_directly': np.nonzero(cnt_s == 0)[0].tolist(),
             'M64_energy_out_over_in_pooled': em / ei}
        for sd in mods:
            r = np.array(rows[sd])
            grp = {}
            for nm, lo, hi in (('ring0_share_lt_0.05', -1, 0.05), ('0.05_to_0.5', 0.05, 0.5), ('0.5_to_0.95', 0.5, 0.95), ('ge_0.95', 0.95, 2)):
                mk = (r[:, 4] >= lo) & (r[:, 4] < hi)
                if mk.sum():
                    w = r[mk, 5] * r[mk, 3]                                              # weight = M64 exitance of the lighting
                    grp[nm] = {'n': int(mk.sum()), 'rel_l1_vs_M64_median': float(np.median(r[mk, 0])), 'rel_l1_vs_M64_flux_weighted': float((r[mk, 0] * w).sum() / w.sum()),
                               'energy_vs_M64_median': float(np.median(r[mk, 1])), 'out_over_in_learned_median': float(np.median(r[mk, 2])),
                               'out_over_in_learned_max': float(r[mk, 2].max()), 'out_over_in_M64_median': float(np.median(r[mk, 3]))}
            d[f'A0_s{sd}'] = {'n_lightings': len(r), 'S1_sender_rel_l1_vs_M64_pooled': num[sd] / den, 'S1_sender_energy_vs_M64_pooled': eo[sd] / em,
                              'rel_l1_vs_M64_per_lighting_quantiles_10_50_90': [float(v) for v in np.quantile(r[:, 0], [0.1, 0.5, 0.9])],
                              'energy_vs_M64_per_lighting_quantiles_10_50_90': [float(v) for v in np.quantile(r[:, 1], [0.1, 0.5, 0.9])],
                              'out_over_in_learned_pooled': eo[sd] / ei, 'out_over_in_learned_per_lighting_max': float(r[:, 2].max()),
                              'lightings_with_out_gt_in': int((r[:, 2] > 1).sum()),
                              'albedo_max_over_geometries_min_med_max': [float(v) for v in np.quantile(albmax[sd], [0, 0.5, 1])],
                              'direct_irradiance_share_on_entries_with_albedo_gt1': viol_irr[sd] / ei,
                              'by_ring0_irradiance_share': grp,
                              'receiver_E1_share_by_incident_ring': (acc_e1[sd].reshape(4, 16).sum(1) / acc_e1[sd].sum()).tolist(),
                              'receiver_E1_share_by_incident_bin': (acc_e1[sd] / acc_e1[sd].sum()).tolist(),
                              'receiver_E1_total_over_direct_total': float(acc_e1[sd].sum() / acc_s.sum())}
        o[sp] = d
    jd(o, os.path.join(OUT, 'train.json'))
    print(json.dumps(o, indent=1, default=float), flush=True)


def debug():
    """Cross-check of the two own box-A estimators: constant panel radiance (form factor), Lambert panel (state 3), S1."""
    o = {}
    for nm, kw in (('const', {'const': True}), ('lambert07', {'state': np.asarray(ctx.mats.states)[3]}), ('s1', {})):
        ha = np.zeros((16, 16)); hh = np.zeros((16, 16))
        a = boxa_ref(97051, 2, hist=ha, **kw); h = boxa_ref_hemi(97052, 4, hist=hh, **kw)
        o[nm] = {'area_total': float(a.sum()), 'hemi_total': float(h.sum()), 'hemi_over_area': float(h.sum() / a.sum()), 'hemi_vs_area': dm(h, a),
                 'area_first9': a[:9].tolist(), 'hemi_first9': h[:9].tolist(), 'hist_area_sum': float(ha.sum()), 'hist_hemi_sum': float(hh.sum()),
                 'hist_area': ha.tolist(), 'hist_hemi': hh.tolist()}
    jd(o, os.path.join(OUT, 'debug.json'))
    # analytic-style control in plain numpy (no renderer): form factor of box-A patch 0 to the panel, hemisphere and area forms
    p0 = BOXA[0]; rg = np.random.default_rng(5); m = 4_000_000
    x = G.PO[p0] + rg.random((m, 1)) * G.PE1[p0] + rg.random((m, 1)) * G.PE2[p0]
    u1 = rg.random(m); ph = rg.random(m) * TWO_PI
    w = G.PT[p0] * (np.sqrt(u1) * np.cos(ph))[:, None] + G.PB[p0] * (np.sqrt(u1) * np.sin(ph))[:, None] + G.PN[p0] * np.sqrt(1 - u1)[:, None]
    t = (G.RC[5][0] - x[:, 0]) / w[:, 0]
    hp = x + w * t[:, None]
    hit = (t > 0) & (np.abs(hp[:, 1] - G.RC[5][1]) < G.RH[5, 0]) & (np.abs(hp[:, 2] - G.RC[5][2]) < G.RH[5, 1])
    y = np.stack([np.full(m, G.RC[5][0]), G.RC[5][1] + (2 * rg.random(m) - 1) * G.RH[5, 0], G.RC[5][2] + (2 * rg.random(m) - 1) * G.RH[5, 1]], 1)
    dxy = y - x; r2 = (dxy ** 2).sum(1); om = dxy / np.sqrt(r2)[:, None]
    cx = (om * G.PN[p0]).sum(1); cy = -om[:, 0]
    cc = np.where((cx > 0) & (cy > 0), cx * cy / r2 * 4 * G.RH[5, 0] * G.RH[5, 1], 0)
    o['numpy_form_factor_patch0'] = {'hemisphere_rho_pi_F': float(RHO_A * math.pi * hit.mean()), 'area_rho_pi_F': float(RHO_A * cc.mean()),
                                     'drjit_area_const': o['const']['area_first9'][0], 'drjit_hemi_const': o['const']['hemi_first9'][0]}
    jd(o, os.path.join(OUT, 'debug.json'))
    for k_, v in o.items():
        print(k_, {x: v[x] for x in v if not x.startswith('hist_a') and not x.startswith('hist_h') or x.endswith('sum')}, flush=True)


if __name__ == '__main__':
    {'conv': conv, 'sender': sender, 'train': train, 'debug': debug}[sys.argv[1]]()
    print('E1_DONE', sys.argv[1], flush=True)
    os._exit(0)
