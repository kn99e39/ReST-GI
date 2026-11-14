"""WL95A (protocol ecro_attr95a): sharp sender representability and flight attribution smoke on CK2 'none' (learning-free diagnostics).
Production code (model86 / ecro86 / pt86 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python a95.py stage0 <R> <R92A>     contract tests, WL94D old-convention reproduction, bound operators (results/s0.json, ops95a.npz)
  python a95.py main   <R> <R92A>     senders, propagation draws, deliveries, render tables (results/main.json, deliv.npz, tables/T95A.npz)
  python a95.py an     <R> <out>      delivery / render metrics, verdicts, figures (results/an.json)
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import an94 as AN  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_attr95a.json')))
DEV = MT.DEV
D_PILOT = 2 if DEV else 16
D_EXT = 4 if DEV else 64
S1 = np.arange(1088, 1130)
BOXA = np.arange(1130, 1184)
PFACE = np.arange(1130, 1139)
SEEDS = [8601, 8602]
NB = 64
DOM = 2 * math.pi / NB
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * DOM                                         # corrected cos dOmega (4 rings x 16 sectors)
W94 = ((np.arange(NB) // 8) + 0.5) / 8 * DOM          # an94.W as coded (8 x 8)
ZZ = ZB[:, None] * ZB[None, :]


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1, default=float)


def jl(fn):
    return json.load(open(fn))


def setup(R):
    PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    return PT, S, A, FX, ctx, lt


def exitance(T, W):
    """per-patch exitance (bins and channels summed) of a table [..., N, NB, 3]."""
    return (np.asarray(T, np.float64) * W[None, :, None]).sum((-1, -2))


def ref_fields(R):
    z = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    g = lambda k: z[k].astype(np.float64)
    o = {k: 0.5 * (g(f'{k}_a') + g(f'{k}_b')) for k in ('e2_p', 'e2_o', 'e3_p', 'e3_o')}
    o['e2_p_a'], o['e2_p_b'] = g('e2_p_a'), g('e2_p_b')
    return o


# ------------------------------------------------------------------------------------------------ propagation primitives
def rw_of(u, S):
    k = np.arange(S.K)
    z = ((k // S.KAPPA) // S.NPHI + u[2, 0]) / S.NZ
    return (z * S.DOMEGA / S.KAPPA).astype(np.float64)


def relation(ctx, u):
    ctx.u_state_np = u
    ctx.set_tables(ctx.G)
    return ctx.state_rays()


def flight(hit, dbin, rw, Bs1, N, S):
    """E1 [N, NB, 3] from a sender table restricted to the S1 rows (Bs1 [42, NB, 3]); every other sender is zero."""
    Bfull = np.zeros((N, NB, 3)); Bfull[S1] = Bs1
    h = hit.astype(np.int64)
    ok = h >= 0
    val = np.where(ok[:, None], Bfull.reshape(N * NB, 3)[np.clip(h, 0, None) * NB + dbin.astype(np.int64)], 0.0)
    return (val.reshape(N, NB, S.KAPPA, 3) * rw.reshape(1, NB, S.KAPPA, 1)).sum(2)


def flight_ray(Lray, rw, N, S):
    return (Lray.reshape(N, NB, S.KAPPA, 3) * rw.reshape(1, NB, S.KAPPA, 1)).sum(2)


def receive(E1, Mrcv):
    return np.einsum('nmc,nmdc->ndc', E1, Mrcv, optimize=True)


def x1_radiance(PT, ctx, lt, o_np, w_np, centre_np=None):
    """Exact once-scattered S1 radiance towards -w (ER eval at the exact light / outgoing directions, shadowed), at the ray hit point or,
    with centre_np, at the centre of the hit panel patch (same outgoing direction -w)."""
    mi, dr, F, U = PT.mi, PT.dr, PT.F, PT.U
    n = len(o_np)
    out = np.zeros((n, 3))
    if n == 0:
        return out
    w = mi.Vector3f(F(w_np[:, 0].astype(np.float32)), F(w_np[:, 1].astype(np.float32)), F(w_np[:, 2].astype(np.float32)))
    if centre_np is None:
        o = mi.Point3f(F(o_np[:, 0].astype(np.float32)), F(o_np[:, 1].astype(np.float32)), F(o_np[:, 2].astype(np.float32)))
        si = ctx.scene.ray_intersect(mi.Ray3f(o, w))
    else:
        nrm = ctx.G.PN[S1[0]]
        c = centre_np + nrm[None, :] * 1e-3
        oc = mi.Point3f(F(c[:, 0].astype(np.float32)), F(c[:, 1].astype(np.float32)), F(c[:, 2].astype(np.float32)))
        si = ctx.scene.ray_intersect(mi.Ray3f(oc, mi.Vector3f(float(-nrm[0]), float(-nrm[1]), float(-nrm[2]))))
    sid = ctx.shape_id(si)
    valid = si.is_valid() & dr.eq(sid, U(5))
    wi = si.to_local(-w)
    valid &= wi.z > 1e-6
    mid = dr.gather(U, ctx.SMAT, sid)
    cl = PT.CD.DrClosure83('ER', ctx.mats.states, mid)
    L = mi.Color3f(0.0)
    for l in range(lt.n):
        dl = lt.pos[l] - si.p
        d2 = dr.squared_norm(dl)
        wl = dl * dr.rsqrt(d2)
        wo = si.to_local(wl)
        on = valid & (wo.z > 1e-6) & (dr.dot(-wl, lt.axis[l]) > lt.cc[l])
        vis = on & ~ctx.scene.ray_test(si.spawn_ray_to(lt.pos[l]), on)
        f = cl.eval(dr.select(vis, wi, mi.Vector3f(0, 0, 1)), dr.select(vis, wo, mi.Vector3f(0, 0, 1)), vis)
        L += dr.select(vis, f * lt.rgb[l] * dr.rcp(d2), 0.0)
    dr.eval(L, valid); dr.sync_thread()
    out[:, 0], out[:, 1], out[:, 2] = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    return out


def x1_rays(PT, ctx, lt, hit, seg, centre):
    """per-ray exact S1 radiance [N*K, 3] for the rays whose recorded hit is an S1 patch (0 elsewhere)."""
    idx = np.nonzero(np.isin(hit, S1))[0]
    o = seg[idx, :3].astype(np.float64); e = seg[idx, 3:].astype(np.float64)
    w = e - o; w /= np.linalg.norm(w, axis=1, keepdims=True)
    L = np.zeros((len(hit), 3))
    L[idx] = x1_radiance(PT, ctx, lt, o, w, ctx.G.PC[hit[idx].astype(np.int64)] if centre else None)
    return L, len(idx)


# ------------------------------------------------------------------------------------------------ sender metrics
def field_metrics(X, Y, W=WT):
    """S1 field [42, NB, 3]: flux rel L1 / energy / peak of X against reference Y."""
    Wd = W[None, :, None]
    pk = Y.argmax(1)[:, None, :]
    return {'rel_l1': float((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum()), 'energy': float((X * Wd).sum() / (Y * Wd).sum()),
            'peak': float(np.take_along_axis(X, pk, 1).sum() / np.take_along_axis(Y, pk, 1).sum())}


def op_metrics(A, M):
    """operator [NB_in, NB_out, 3] vs M64: flux-L1 (zb_in zb_out), mass, directional albedo max (W_T), min entry."""
    alb = (A * WT[None, :, None]).sum(1)
    return {'op_flux_l1': float((np.abs(A - M) * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()),
            'mass': float((A * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()), 'albedo_max': float(alb.max()),
            'albedo_mean': float(alb.mean()), 'min_entry': float(A.min()), 'passive': bool(alb.max() <= 1 + 1e-6), 'nonneg': bool(A.min() >= 0)}


def learned(R92A, seed, ctx, st, S):
    import torch
    import model86 as M86
    model = MT.load_frozen(R92A, seed)
    dev = torch.device('cuda')
    gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
    with torch.no_grad():
        F = model.factors(gs)
        c, a = F['c'][S1].double().cpu().numpy(), F['a'][S1].double().cpu().numpy()    # [42, NB, 3, R]
    return c, a


# ------------------------------------------------------------------------------------------------ Stage 0
def stage0(R, R92A):
    import ecro86 as EC
    import model86 as M86
    PT, S, A, FX, ctx, lt = setup(R)
    mi, dr, F, U = PT.mi, PT.dr, PT.F, PT.U
    T = {}
    rng = np.random.default_rng(95000)
    v = rng.normal(size=(4096, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); v[:, 2] = np.abs(v[:, 2])
    b = np.asarray(PT.Ctx.bin_of(F(v[:, 0].astype(np.float32)), F(v[:, 1].astype(np.float32)), F(v[:, 2].astype(np.float32))))
    iz = np.clip(np.floor(v[:, 2] * 4), 0, 3); ip = np.clip(np.floor((np.arctan2(v[:, 1], v[:, 0]) + math.pi) * 16 / (2 * math.pi)), 0, 15)
    agree = float((b == (iz * 16 + ip).astype(np.int64)).mean())
    T['T_BINS'] = {'pass': S.NZ == 4 and S.NPHI == 16 and S.NB == 64 and agree >= 0.999 and abs(WT.sum() - math.pi) < 1e-9,
                   'bin_formula_agreement': agree, 'sum_WT': float(WT.sum()), 'sum_W94': float(W94.sum()),
                   'W94_vs_WT_ring0_sector_ratio': [float(W94[0] / WT[0]), float(W94[8] / WT[8])], 'an94_W_equals_W94': bool(np.allclose(AN.W, W94))}
    Mk = AN.masks(R); G = ctx.G
    T['T_MASKS'] = {'pass': bool(np.array_equal(np.nonzero(Mk['S1'])[0], S1) and (G.PSID[S1] == 5).all() and (G.PMAT[S1] == 1).all()
                                 and np.array_equal(np.nonzero(Mk['T'])[0], BOXA) and np.array_equal(np.nonzero(Mk['T_panel_face'])[0], PFACE)
                                 and np.array_equal(MT.rect_patches(G, 6), PFACE) and int((G.PMAT[BOXA] == 4).sum()) == 54)}
    rf = ref_fields(R)
    T['T_EVENT'] = {'pass': bool(rf['e2_p'][S1].sum() == 0 and rf['e2_p'][BOXA].sum() > 0), 'e2_p_on_panel': float(rf['e2_p'][S1].sum()),
                    'note': 'field_src split: k = 2 vertex material 1 -> _p (pt86a.py:175-180)'}
    # ER cosine convention on a Lambert material (id 3, albedo 0.7)
    wi = mi.Vector3f(0.3, 0.1, math.sqrt(1 - 0.1)); n_ = 64
    wo_np = rng.normal(size=(n_, 3)); wo_np[:, 2] = np.abs(wo_np[:, 2]) + 0.05; wo_np /= np.linalg.norm(wo_np, axis=1, keepdims=True)
    wo = mi.Vector3f(F(wo_np[:, 0].astype(np.float32)), F(wo_np[:, 1].astype(np.float32)), F(wo_np[:, 2].astype(np.float32)))
    cl = PT.CD.DrClosure83('ER', ctx.mats.states, dr.full(U, 3, n_))
    f = cl.eval(mi.Vector3f(dr.full(F, 0.3, n_), dr.full(F, 0.1, n_), dr.full(F, math.sqrt(0.9), n_)), wo, True)
    fv = np.asarray(f.y)
    alb = float(np.asarray(ctx.mats.albedo)[3])
    T['T_COS'] = {'pass': bool(np.allclose(fv, alb / math.pi * wo_np[:, 2], rtol=1e-4)), 'max_rel': float(np.abs(fv / (alb / math.pi * wo_np[:, 2]) - 1).max())}
    # WL94D old-convention reproduction (W94, deployed state, M64(s16) on S1, exact receiver)
    st = EC.build_state(ctx)
    s16 = ctx.source(lt).astype(np.float64)
    Mq = MT.mq_ext().astype(np.float64)
    Mrcv = Mq[G.PMAT]
    rw0 = M86.ray_weights().astype(np.float64)
    Bm = np.einsum('nmc,mdc->ndc', s16[S1], Mq[1])
    Sx = receive(flight(st.hit, st.dbin, rw0, Bm, G.N, S), Mrcv)
    exm, ea, eb = exitance(Sx, W94), exitance(rf['e2_p_a'], W94), exitance(rf['e2_p_b'], W94)
    er = 0.5 * (ea + eb)
    rc = np.ones(G.N, bool); rc[S1] = False
    old = jl(os.path.join(R, 'results', 'wl94d', 'audit.json'))['gates']['G_PHYS']['ck2_flight_floor_diagnostic']
    rep = {}; dmax = 0.0
    for nm, mm in (('receivers', rc), ('boxA', np.isin(np.arange(G.N), BOXA)), ('panel_face', np.isin(np.arange(G.N), PFACE))):
        r_ = {'ratio': float(exm[mm].sum() / er[mm].sum()), 'corr': float(np.corrcoef(exm[mm], er[mm])[0, 1]),
              'rel_l1': float(np.abs(exm[mm] - er[mm]).sum() / er[mm].sum())}
        rep[nm] = r_
        dmax = max(dmax, *(abs(r_[k] - old[nm][k]) for k in r_))
    c94 = {}
    for sd in SEEDS:
        c, a = learned(R92A, sd, ctx, st, S)
        Bl = np.einsum('ndcr,ncr->ndc', a, np.einsum('nmcr,nmc->ncr', c, s16[S1]))
        old_s = jl(os.path.join(R, 'results', 'wl94d', f'SN_own_s{sd}.json'))['ck2']['rel_l1_flux']
        new_s = field_metrics(Bl, Bm, W94)['rel_l1']
        c94[sd] = {'rel_l1_flux_W94': new_s, 'wl94d': old_s}
        dmax = max(dmax, abs(new_s - old_s))
    T['T_REPRO94D'] = {'pass': dmax <= 1e-3, 'max_abs_dev': dmax, 'floor_W94': rep, 'wl94d_floor': old, 'sender_ck2_W94': c94}
    # bound operators
    ops = dict(np.load(os.path.join(R, 'results', 'wl95_calc', 'S1_ops.npz')))
    sys.path.insert(0, os.path.join(HERE, 'wl95calc'))
    import wl95_calc as C
    M = C.Mq[1]
    tr, va = C.load_fields('train', 1), C.load_fields('val', 1)
    E = sum(r.sum(0) for r in tr)
    k = 32
    A3 = np.zeros_like(M)
    for ch in range(3):                                   # verbatim copy of the wl95_calc2.py ladder (rank k)
        V = M[..., ch]; Om0 = E[:, ch][:, None] * C.W94[None, :]
        rg = np.random.default_rng(9500 + k * 10 + ch)
        sc = np.sqrt(V.mean() / k)
        Wf = rg.random((64, k)) * sc + 1e-12; Hf = rg.random((k, 64)) * sc + 1e-12
        for _ in range(300):
            Hf *= (Wf.T @ (Om0 * V)) / (Wf.T @ (Om0 * (Wf @ Hf)) + 1e-30)
            Wf *= ((Om0 * V) @ Hf.T) / ((Om0 * (Wf @ Hf)) @ Hf.T + 1e-30)
        d = 1e-3 * V.mean()
        for _ in range(60):
            Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(100):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + 1e-30)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + 1e-30)
        A3[..., ch] = Wf @ Hf
    lad = jl(os.path.join(R, 'results', 'wl95_calc', 'wl95_calc2.json'))['fa_nmf_l1_ladder_randinit'][str(k)]
    regen = {'op_l1': C.op_metric(A3, M), 'val_field_rel_l1': C.field_metrics(va, A3, M, C.W94)['field_rel_l1']}
    T['T_OPS'] = {'pass': bool(abs(regen['op_l1'] - lad['op_l1']) <= 1e-6 and abs(regen['val_field_rel_l1'] - lad['val_W94']['field_rel_l1']) <= 1e-6
                               and ops['nmf16_wl92'].min() >= 0 and ops['nmf16_l1irls'].min() >= 0 and A3.min() >= 0
                               and np.allclose(M, Mq[1], rtol=1e-5, atol=1e-7)),
                  'regenerated_rank': k, 'regen': regen, 'wl95': {'op_l1': lad['op_l1'], 'val': lad['val_W94']['field_rel_l1']},
                  'M64_nmf_npz_equals_mq_ext': bool(np.allclose(M, Mq[1], rtol=1e-5, atol=1e-7))}
    np.savez_compressed(os.path.join(R, 'results', 'ops95a.npz'), A1=ops['nmf16_wl92'], A2=ops['nmf16_l1irls'], A3=A3, A4=Mq[1])
    st_ = 'PASS' if all(t['pass'] for t in T.values()) else 'STOP'
    jd({'status': st_, 'tests': T, 'ops_sha256': MT.sha_file(os.path.join(R, 'results', 'ops95a.npz')),
        'protocol_sha256': MT.sha_file(os.path.join(HERE, 'protocol', 'ecro_attr95a.json'))}, os.path.join(R, 'results', 's0.json'))
    print('STAGE0', st_, json.dumps({k_: bool(v_['pass']) for k_, v_ in T.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ main
def main(R, R92A):
    import ecro86 as EC
    import model86 as M86
    PT, S, A, FX, ctx, lt = setup(R)
    G = ctx.G; N = G.N
    t0 = time.time(); TM = {}
    ops = dict(np.load(os.path.join(R, 'results', 'ops95a.npz')))
    Mq = MT.mq_ext().astype(np.float64); Mrcv = Mq[G.PMAT]; M1 = Mq[1]
    st = EC.build_state(ctx)
    u0 = ctx.u_state_np.copy(); us0 = ctx.u_src_np.copy()
    s16 = ctx.source(lt).astype(np.float64)
    rf = ref_fields(R)
    out = {'timing': TM, 'counts': {}}
    # senders (S1 rows)
    snd = {f'A{i}': np.einsum('nmc,mdc->ndc', s16[S1], ops[f'A{i}']) for i in (1, 2, 3, 4)}
    lop = {}
    for sd in SEEDS:
        c, a = learned(R92A, sd, ctx, st, S)
        snd[f'A0_s{sd}'] = np.einsum('ndcr,ncr->ndc', a, np.einsum('nmcr,nmc->ncr', c, s16[S1]))
        lop[sd] = np.einsum('nmcr,ndcr->nmdc', c, a)                                        # per-patch learned operator [42, in, out, 3]
    TM['learned_s'] = time.time() - t0
    # source-point draws (separate factor)
    t1 = time.time(); sd_list = []
    for d in range(D_EXT if not DEV else D_PILOT):
        ctx.u_src_np = np.random.default_rng(951000 + d).random(us0.shape).astype(np.float32); ctx.set_tables(G)
        sd_list.append(ctx.source(lt).astype(np.float64)[S1])
    ctx.u_src_np = us0; ctx.set_tables(G)
    s_pool = np.mean(sd_list, 0)
    snd['M64_spool'] = np.einsum('nmc,mdc->ndc', s_pool, M1)
    src_var = [field_metrics(np.einsum('nmc,mdc->ndc', x, M1), snd['M64_spool']) for x in sd_list]
    TM['source_draws_s'] = time.time() - t1
    # physical q (two MIS seeds), restricted to S1
    t1 = time.time()
    import quad87 as Q
    qs = Q._q_static(ctx, 256)
    pts = (S1[:, None] * 256 + np.arange(256)[None]).ravel().astype(np.uint32)
    qs['er_pts'] = PT.U(pts); qs['n_er'] = len(pts)
    MQ = 64 if DEV else 1024
    qA = Q.qtable(ctx, lt, 256, MQ, 95001)[S1]; qB = Q.qtable(ctx, lt, 256, MQ, 95002)[S1]
    snd['q256'] = 0.5 * (qA + qB)
    TM['q_s'] = time.time() - t1
    out['counts']['q_lanes_per_seed'] = int(len(pts) * MQ)
    # sender metrics (CK2 S1 field, W_T)
    sm = {}
    for k_, X in snd.items():
        sm[k_] = {'vs_M64_s16': field_metrics(X, snd['A4']), 'vs_q256': field_metrics(X, snd['q256']),
                  'vs_M64_s16_W94': field_metrics(X, snd['A4'], W94)}
    sm['q_split_half'] = {'rel_l1_half': 0.5 * field_metrics(qA, qB)['rel_l1'], 'qA_vs_qB': field_metrics(qA, qB)}
    sm['source_draw_var'] = {'rel_l1_single_draw_vs_pool_mean': float(np.mean([v['rel_l1'] for v in src_var])),
                             'max': float(np.max([v['rel_l1'] for v in src_var]))}
    om = {f'A{i}': op_metrics(ops[f'A{i}'], M1) for i in (1, 2, 3, 4)}
    for sd in SEEDS:
        per = [op_metrics(lop[sd][j], M1) for j in range(len(S1))]
        om[f'A0_s{sd}'] = {'op_flux_l1_mean': float(np.mean([p['op_flux_l1'] for p in per])), 'albedo_max': float(np.max([p['albedo_max'] for p in per])),
                           'passive_all': bool(all(p['passive'] for p in per)), 'nonneg': bool(min(p['min_entry'] for p in per) >= 0),
                           'mass_mean': float(np.mean([p['mass'] for p in per]))}
    out['sender'] = sm; out['operator'] = om
    # propagation: deployed + pooled draws
    rw0 = rw_of(u0, S)
    TAB = [k_ for k_ in snd]                                   # table senders
    RAY = ['X1c', 'X1']
    ex = {}; tabs_mean = {}; tabs_dep = {}

    def deliver(hit, dbin, seg, rw, store):
        res = {}
        for k_ in TAB:
            Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
            res[k_] = Dt
        for k_, cen in (('X1c', True), ('X1', False)):
            L, nr = x1_rays(PT, ctx, lt, hit, seg, cen)
            res[k_] = receive(flight_ray(L, rw, N, S), Mrcv)
            store['rays_S1'] = nr
        return res
    t1 = time.time()
    hit0, dbin0, seg0 = relation(ctx, u0)
    T_DEP = bool(np.array_equal(hit0, st.hit) and np.array_equal(dbin0, st.dbin))
    cnt = {}
    dep = deliver(hit0, dbin0, seg0, rw0, cnt)
    out['counts']['deployed_rays_to_S1'] = cnt.get('rays_S1')
    for k_, Dt in dep.items():
        tabs_dep[k_] = Dt; ex[f'dep|{k_}'] = exitance(Dt, WT)
    TM['deployed_s'] = time.time() - t1

    def run_draws(d0, d1):
        for d in range(d0, d1):
            u = np.random.default_rng(950000 + d).random(u0.shape).astype(np.float32)
            hit, dbin, seg = relation(ctx, u)
            c_ = {}
            res = deliver(hit, dbin, seg, rw_of(u, S), c_)
            for k_, Dt in res.items():
                tabs_mean[k_] = tabs_mean.get(k_, 0) + Dt
                ex.setdefault(f'draws|{k_}', []).append(exitance(Dt, WT))
            out['counts'].setdefault('draw_rays_to_S1', []).append(c_.get('rays_S1'))
    t1 = time.time()
    run_draws(0, D_PILOT)
    TM['pilot_s'] = time.time() - t1
    ctx.u_state_np = u0; ctx.set_tables(G)
    out['D_pilot'] = D_PILOT
    # extension rule is applied in 'an' via an extension request file; main can be re-run with EXTEND=1
    if os.environ.get('A95_EXTEND') == '1':
        t1 = time.time()
        run_draws(D_PILOT, D_EXT)
        ctx.u_state_np = u0; ctx.set_tables(G)
        TM['extension_s'] = time.time() - t1
        out['D_used'] = D_EXT
    else:
        out['D_used'] = D_PILOT
    D = out['D_used']
    for k_ in tabs_mean:
        tabs_mean[k_] = tabs_mean[k_] / D
    out['T_DEPLOYED_ID'] = T_DEP
    # render tables: arm delivered S1-mediated e2 + reference e2_o + e3 (identical across arms); oracle-table floor
    base = rf['e2_o'] + rf['e3_p'] + rf['e3_o']
    tab = {'floor': rf['e2_p'] + base}
    for k_ in ('A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4'):
        tab[f'dep_{k_}'] = tabs_dep[k_] + base
    for k_ in ('A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4', 'M64_spool', 'q256', 'X1c', 'X1'):
        tab[f'pool_{k_}'] = tabs_mean[k_] + base
    for sd in SEEDS:
        g = np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{sd}.npz'))['none|G'].astype(np.float64)
        tab[f'G_s{sd}'] = rf['e2_p'] + rf['e2_o'] + g
    os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
    np.savez_compressed(os.path.join(R, 'results', 'tables', 'T95A.npz'), **{k_: v.astype(np.float32) for k_, v in tab.items()})
    np.savez_compressed(os.path.join(R, 'results', 'deliv.npz'), **{k_: np.asarray(v) for k_, v in ex.items()},
                        ref_a=exitance(rf['e2_p_a'], WT), ref_b=exitance(rf['e2_p_b'], WT))
    np.savez_compressed(os.path.join(R, 'results', 'senders.npz'), **snd, qA=qA, qB=qB)
    TM['total_s'] = time.time() - t0
    import torch
    out['gpu_mem_peak_mib_torch'] = torch.cuda.max_memory_allocated() / 2 ** 20
    out['tables'] = sorted(tab)
    jd(out, os.path.join(R, 'results', 'main.json'))
    lst = os.path.join(R, 'results', 'lists', 'r95a.txt'); os.makedirs(os.path.dirname(lst), exist_ok=True)
    spp = 32 if DEV else 4096
    with open(lst, 'w') as fh:
        for k_ in sorted(tab):
            for r in (0, 1):
                fh.write(f'none results/tables/T95A.npz:{k_} {86600 + r} {spp} CK_none_{k_}_X_CK2_T1_r{r}\n')
    print('MAIN DONE', 'deployed_id', T_DEP, flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def dmetrics(arm, ref, ra, rb, mask):
    a, r = arm[mask], ref[mask]
    o = {'ratio': float(a.sum() / r.sum()), 'signed': float((a - r).sum() / r.sum()), 'abs_rel_l1': float(np.abs(a - r).sum() / r.sum()),
         'ref_half_noise': float(0.5 * np.abs(ra[mask] - rb[mask]).sum() / r.sum()),
         'peak_patch_arm': int(np.argmax(a)), 'peak_patch_ref': int(np.argmax(r)), 'peak_amp_ratio': float(a.max() / r.max()),
         'value_at_ref_peak_ratio': float(a[np.argmax(r)] / r[np.argmax(r)])}
    if len(a) > 2:
        o['corr'] = float(np.corrcoef(a, r)[0, 1])
    return o


def jk(draws, ref, mask, fn):
    """jackknife over draws of fn(mean exitance) -> (value on all, SE)."""
    X = np.asarray(draws)
    D = len(X)
    full = fn(X.mean(0))
    if D < 2:
        return full, float('nan')
    loo = np.array([fn((X.sum(0) - X[i]) / (D - 1)) for i in range(D)])
    return full, float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


def an(R, outd):
    z = dict(np.load(os.path.join(R, 'results', 'deliv.npz')))
    mj = jl(os.path.join(R, 'results', 'main.json'))
    ra, rb = z['ref_a'], z['ref_b']; ref = 0.5 * (ra + rb)
    N = len(ref)
    mA = np.isin(np.arange(N), BOXA); mP = np.isin(np.arange(N), PFACE)
    arms = sorted({k.split('|')[1] for k in z if '|' in k})
    D = {}
    for k in arms:
        for mn, mm in (('boxA', mA), ('panel_face', mP)):
            if f'dep|{k}' in z:
                D[f'dep|{k}|{mn}'] = dmetrics(z[f'dep|{k}'], ref, ra, rb, mm)
            if f'draws|{k}' in z:
                dr_ = z[f'draws|{k}']
                D[f'pool|{k}|{mn}'] = dmetrics(dr_.mean(0), ref, ra, rb, mm)
                per = [dmetrics(x, ref, ra, rb, mm) for x in dr_]
                D[f'pool|{k}|{mn}']['per_draw'] = {q: {'mean': float(np.mean([p[q] for p in per])), 'median': float(np.median([p[q] for p in per])),
                                                       'p90': float(np.quantile([p[q] for p in per], 0.9)),
                                                       'worst': float(np.max([abs(p[q] - (1 if q == 'ratio' else 0)) for p in per]))}
                                                   for q in ('ratio', 'abs_rel_l1')}
                D[f'pool|{k}|{mn}']['se_abs_rel_l1'] = jk(dr_, ref, mm, lambda m: float(np.abs(m[mm] - ref[mm]).sum() / ref[mm].sum()))[1]
    n0 = D['dep|A4|boxA']['ref_half_noise']
    # paired arm-vs-arm comparisons on common draws (box A)
    def E_pool(k):
        return D[f'pool|{k}|boxA']['abs_rel_l1'] - n0

    def diff_se(k1, k2):
        X1_, X2_ = np.asarray(z[f'draws|{k1}']), np.asarray(z[f'draws|{k2}'])
        f = lambda m1, m2: float(np.abs(m1[mA] - ref[mA]).sum() / ref[mA].sum() - np.abs(m2[mA] - ref[mA]).sum() / ref[mA].sum())
        Dn = len(X1_)
        loo = np.array([f((X1_.sum(0) - X1_[i]) / (Dn - 1), (X2_.sum(0) - X2_[i]) / (Dn - 1)) for i in range(Dn)])
        return f(X1_.mean(0), X2_.mean(0)), float(np.sqrt((Dn - 1) / Dn * ((loo - loo.mean()) ** 2).sum()))
    E0 = D['dep|A4|boxA']['abs_rel_l1'] - n0
    Rs, se_s = E0 - E_pool('M64_spool'), D['pool|M64_spool|boxA']['se_abs_rel_l1']
    Ra, se_a = diff_se('M64_spool', 'X1c')
    Rp, se_p = diff_se('X1c', 'X1')
    Rq_inc, se_qi = diff_se('M64_spool', 'q256')
    Rq_out, se_qo = diff_se('q256', 'X1c')
    cnt = lambda r, se: bool(r >= 0.25 * E0 and r > 2 * (se if se == se else 0))
    s_ok, a_ok = cnt(Rs, se_s), cnt(Ra, se_a)
    EX1 = E_pool('X1')
    if EX1 > 0.5 * E0:
        fa = 'INCONCLUSIVE'
    elif s_ok and a_ok:
        fa = 'MIXED'
    elif s_ok and not a_ok and Rs >= 0.5 * E0:
        fa = 'SAMPLING_DOMINANT'
    elif a_ok and not s_ok:
        fa = 'ANGULAR_PROJECTION_EVIDENCED'
    else:
        fa = 'INCONCLUSIVE'
    near = any(abs(r - 0.25 * E0) < 2 * se for r, se in ((Rs, se_s), (Ra, se_a)) if se == se)
    flight = {'n0': n0, 'E0': E0, 'Rs': Rs, 'Rs_se': se_s, 'Ra': Ra, 'Ra_se': se_a, 'Rp': Rp, 'Rp_se': se_p, 'Rq_incident': Rq_inc, 'Rq_incident_se': se_qi,
              'Rq_outgoing': Rq_out, 'Rq_outgoing_se': se_qo, 'E_X1_pool': EX1, 'E_M64_spool_pool': E_pool('M64_spool'), 'E_X1c_pool': E_pool('X1c'),
              'E_q256_pool': E_pool('q256'), 'E_A4_pool': E_pool('A4'), 'verdict': fa, 'near_boundary_extend': bool(near and mj['D_used'] < D_EXT)}
    # renders
    M = AN.masks(R)
    refimg = AN.load_ref(R, 'none'); tot = refimg['tot'].mean(0)
    tabs = mj['tables']; img = {}
    for k in tabs:
        x = AN.load_hl(R, 'none', k, 'X')
        if x is not None:
            img[k] = AN.eh(refimg['h'], x, tot, M['ROI_T'])
    floor = img['floor']['E_H']

    def tol(a, b):
        return max(2 * max(img[a]['noise_bound'], img[b]['noise_bound']), 0.005)
    # RANK16_DELIVERY (pooled, vs A4)
    rk = {}
    for X in ('A1', 'A2', 'A3'):
        pX, p4 = D[f'pool|{X}|boxA'], D['pool|A4|boxA']
        Xm, A4m = np.asarray(z[f'draws|{X}']).mean(0), np.asarray(z['draws|A4']).mean(0)
        mapdiff = float(np.abs(Xm[mA] - A4m[mA]).sum() / A4m[mA].sum())
        c1 = mj['operator'][X]['passive']
        c2 = 0.9 <= pX['ratio'] / p4['ratio'] <= 1.1
        c3 = mapdiff <= 0.10
        c4 = img[f'pool_{X}']['E_H'] <= img['pool_A4']['E_H'] + tol(f'pool_{X}', 'pool_A4')
        rk[X] = {'passive': c1, 'flux_ratio_vs_A4': pX['ratio'] / p4['ratio'], 'map_rel_l1_vs_A4': mapdiff, 'E_H': img[f'pool_{X}']['E_H'],
                 'E_H_A4': img['pool_A4']['E_H'], 'tol': tol(f'pool_{X}', 'pool_A4'), 'criteria': [bool(c1), bool(c2), bool(c3), bool(c4)], 'all': bool(c1 and c2 and c3 and c4)}
    if rk['A1']['all'] or rk['A2']['all']:
        r16 = 'ADEQUATE_ON_TESTED_WORKLOAD'
    else:
        def clear_fail(X):
            r = rk[X]
            return ((not r['passive']) or abs(r['flux_ratio_vs_A4'] - 1) > 0.1 + 2 * 0.02 or r['map_rel_l1_vs_A4'] > 0.10 + 0.02
                    or r['E_H'] > r['E_H_A4'] + r['tol'])
        r16 = 'INADEQUATE_ON_TESTED_WORKLOAD' if clear_fail('A1') and clear_fail('A2') else 'INCONCLUSIVE'
    # NEXT_DESIGN_TARGET
    comp = {}
    for sd in SEEDS:
        dS = img[f'dep_A0_s{sd}']['E_H'] - floor; dF = img['dep_A4']['E_H'] - floor; dG = img[f'G_s{sd}']['E_H'] - floor
        t_ = max(tol(f'dep_A0_s{sd}', 'floor'), tol('dep_A4', 'floor'), tol(f'G_s{sd}', 'floor'))
        comp[sd] = {'dS': dS, 'dF': dF, 'dG': dG, 'tol': t_}
    allc = lambda f: all(f(c) for c in comp.values())
    if allc(lambda c: c['dG'] > max(c['dS'], c['dF']) + c['tol']):
        ndt = 'HIGHER_ORDER_TRANSPORT'
    elif allc(lambda c: c['dS'] > max(c['dF'], c['dG']) + c['tol']):
        ndt = 'LEARNED_SENDER'
    elif allc(lambda c: c['dF'] >= max(c['dS'], c['dG']) - c['tol'] and c['dF'] > c['tol']) and fa in ('ANGULAR_PROJECTION_EVIDENCED', 'MIXED'):
        ndt = 'DIRECTIONAL_CARRIER'
    else:
        ndt = 'UNDETERMINED'
    res = {'delivery': D, 'flight': flight, 'rank16': rk, 'images': img, 'floor_E_H': floor, 'components': comp,
           'verdicts': {'RANK16_DELIVERY': r16, 'FLIGHT_ATTRIBUTION': fa, 'NEXT_DESIGN_TARGET': ndt}, 'D_used': mj['D_used']}
    jd(res, os.path.join(R, 'results', 'an.json'))
    files = figures(R, outd, res, z, mj)
    jd({'worklog': '95A', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
       os.path.join(outd, 'manifest.json'))
    print('AN', 'EXTEND' if flight['near_boundary_extend'] else 'DONE', flush=True)


# ------------------------------------------------------------------------------------------------ figures
def figures(R, outd, res, z, mj):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viz94 as VZ
    os.makedirs(outd, exist_ok=True)
    files = []
    D = res['delivery']; img = res['images']
    # F1 senders
    sm = mj['sender']; keys = ['A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4', 'M64_spool', 'q256']
    fig, ax = plt.subplots(1, 4, figsize=(24, 4.5))
    x = np.arange(len(keys))
    for j, (ref, lab) in enumerate((('vs_M64_s16', 'vs M64(s16)'), ('vs_q256', 'vs physical q256'))):
        ax[0].bar(x + 0.4 * j, [sm[k][ref]['rel_l1'] for k in keys], 0.4, label=lab)
        ax[1].bar(x + 0.4 * j, [sm[k][ref]['energy'] for k in keys], 0.4, label=lab)
        ax[2].bar(x + 0.4 * j, [sm[k][ref]['peak'] for k in keys], 0.4, label=lab)
    for a_, t in zip(ax[:3], ('S1 sender field flux rel L1 (CK2, W_T)', 'energy ratio', 'peak retention')):
        a_.set_title(t, fontsize=8); a_.set_xticks(x + 0.2); a_.set_xticklabels(keys, rotation=30, fontsize=7); a_.legend(fontsize=7)
    ax[1].axhline(1, color='k', lw=0.7); ax[2].axhline(1, color='k', lw=0.7)
    om = mj['operator']; ok_ = ['A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4']
    ax[3].bar(ok_, [om[k]['albedo_max'] for k in ok_]); ax[3].axhline(1, color='r', ls='--', label='passivity limit')
    ax[3].set_title('operator max directional albedo (W_T)', fontsize=8); ax[3].legend(fontsize=7)
    fig.suptitle(f"WL95A senders on CK2 'none' S1 field; q split-half rel L1 {sm['q_split_half']['rel_l1_half']:.3f}", fontsize=10)
    fn = os.path.join(outd, 'figA1_senders.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # F2 delivery
    arms = ['A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4', 'M64_spool', 'q256', 'X1c', 'X1']
    fig, ax = plt.subplots(2, 2, figsize=(20, 9))
    for i, mn in enumerate(('boxA', 'panel_face')):
        dep = [D.get(f'dep|{k}|{mn}', {}).get('ratio', np.nan) for k in arms]
        pol = [D[f'pool|{k}|{mn}']['ratio'] for k in arms]
        ax[i, 0].bar(np.arange(len(arms)) - 0.2, dep, 0.4, label='deployed relation'); ax[i, 0].bar(np.arange(len(arms)) + 0.2, pol, 0.4, label=f"pooled ({res['D_used']} draws)")
        ax[i, 0].axhline(1, color='k', lw=0.7); ax[i, 0].set_title(f'{mn}: delivered S1-mediated e2 flux / reference e2_p', fontsize=8)
        depl = [D.get(f'dep|{k}|{mn}', {}).get('abs_rel_l1', np.nan) for k in arms]
        poll = [D[f'pool|{k}|{mn}']['abs_rel_l1'] for k in arms]
        se = [D[f'pool|{k}|{mn}'].get('se_abs_rel_l1', 0) for k in arms]
        ax[i, 1].bar(np.arange(len(arms)) - 0.2, depl, 0.4, label='deployed'); ax[i, 1].bar(np.arange(len(arms)) + 0.2, poll, 0.4, yerr=[2 * s for s in se], label='pooled (2 SE)')
        ax[i, 1].axhline(D[f'dep|A4|{mn}']['ref_half_noise'], color='r', ls=':', label='reference half-split noise')
        ax[i, 1].set_title(f'{mn}: abs rel L1 vs reference e2_p (W_T exitance)', fontsize=8)
        for a_ in ax[i]:
            a_.set_xticks(np.arange(len(arms))); a_.set_xticklabels(arms, rotation=30, fontsize=7); a_.legend(fontsize=7)
    fl = res['flight']
    fig.suptitle(f"WL95A delivery (CK2 none, one light, exact Lambert receivers): FLIGHT_ATTRIBUTION {fl['verdict']}; E0 {fl['E0']:.3f}, Rs {fl['Rs']:.3f}, Ra {fl['Ra']:.3f}, Rp {fl['Rp']:.3f}", fontsize=10)
    fn = os.path.join(outd, 'figA2_delivery.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # F3 footprint maps: box A faces (6 x 3x3) for reference and key arms (shared scale)
    ref = 0.5 * (z['ref_a'] + z['ref_b'])
    cols = [('reference e2_p', ref), ('A0_s8601 dep', z['dep|A0_s8601']), ('A4 M64 dep', z['dep|A4']), ('A1 nmf16 pooled', np.asarray(z['draws|A1']).mean(0)),
            ('A4 M64 pooled', np.asarray(z['draws|A4']).mean(0)), ('q256 pooled', np.asarray(z['draws|q256']).mean(0)), ('X1 pooled', np.asarray(z['draws|X1']).mean(0))]
    vmax = max(c[1][BOXA].max() for c in cols)
    fig, ax = plt.subplots(len(cols), 6, figsize=(12, 2.0 * len(cols)))
    for i, (t, v) in enumerate(cols):
        for f_ in range(6):
            ax[i, f_].imshow(v[1130 + 9 * f_:1139 + 9 * f_].reshape(3, 3), vmin=0, vmax=vmax, cmap='magma')
            ax[i, f_].set_xticks([]); ax[i, f_].set_yticks([])
            if f_ == 0:
                ax[i, f_].set_ylabel(t, fontsize=7)
            if i == 0:
                ax[i, f_].set_title(f'box A face {f_}' + (' (panel-facing)' if f_ == 0 else ''), fontsize=7)
    fig.suptitle(f'WL95A box-A delivered S1-mediated exitance per patch (shared absolute scale 0..{vmax:.3g})', fontsize=9)
    fn = os.path.join(outd, 'figA3_footprint.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # F4 renders
    M = AN.masks(R)
    refimg = AN.load_ref(R, 'none'); h = refimg['h'].mean(0)
    l0 = AN.lum(refimg['tot'].mean(0)); key = 0.5 / float(np.median(l0[l0 > 0]))
    cy, cx = VZ.crop_of(M)
    show = ['floor', 'dep_A0_s8601', 'dep_A0_s8602', 'dep_A1', 'dep_A4', 'pool_A1', 'pool_A4', 'pool_q256', 'pool_X1', 'G_s8601', 'G_s8602']
    hl = {k: AN.load_hl(R, 'none', k, 'X').mean(0) for k in show if AN.load_hl(R, 'none', k, 'X') is not None}
    E_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(v - h))[cy, cx].ravel() for v in hl.values()]), 0.999))
    fig, ax = plt.subplots(2, len(hl) + 1, figsize=(2.6 * (len(hl) + 1), 6))
    ax[0, 0].imshow(VZ.tm(h[cy, cx], key)); ax[0, 0].set_title('REF h', fontsize=8); ax[1, 0].axis('off')
    for j, (k, v) in enumerate(hl.items(), 1):
        ax[0, j].imshow(VZ.tm(v[cy, cx], key)); ax[0, j].set_title(f"{k}\nE_H {img[k]['E_H']:.4f}", fontsize=7)
        VZ.sq(ax[1, j], AN.lum(v - h)[cy, cx], E_MAX, plt, 'signed hl - h')
    for a_ in ax[0]:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'WL95A CK2 none higher-order close-ups: arm S1-mediated e2 + reference e2_o + e3 (identical across arms); floor = reference table; '
                 f'G_s = reference e2 + own learned global (diagnostic); shared error scale {E_MAX:.3g}, one exposure', fontsize=9)
    fn = os.path.join(outd, 'figA4_renders.png'); fig.savefig(fn, dpi=70, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'stage0':
        stage0(a[1], a[2])
    elif cmd == 'main':
        main(a[1], a[2])
    elif cmd == 'an':
        an(a[1], a[2])
    os._exit(0)
