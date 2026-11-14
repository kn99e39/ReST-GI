"""WL93D POST-HOC QUALITATIVE ADDENDUM (non-numbered), revision 2: patch-aligned checkerboard albedo on a movable box that the sharp
panel reflects into the camera.

Revision 1 (back-wall checker, E2 T1) is superseded: the S1 panel lobe at ~1.8 m does not resolve 25 cm cells, so the checker was not
visible in the reflection. Revision 2 changes the scene composition (owner instruction): box A carries the checker on its faces and is
placed close to the panel inside the training box distribution; the placement was adjusted by inspecting actual renders (probe).

Frozen checkpoints (WL92A own s8601 / s8602) and the frozen WL93D arm evaluation (ops93d.arm_eval); nothing is trained or refitted.
The texture is a per-PATCH material-id map consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors);
  (ii)  the frozen-ER M64 matrices: Mn = Mq[PMAT[patch]];
  (iii) the physical reference / LOW / ECRO-lookup renderer: at every surface hit the material id is PMAT[patch of the hit] instead of the
        per-shape id (one lookup inserted into pt86.Ctx.trace by a run-time source patch; pt86.py is not edited).
Scene: scene_ck2.json (geometry, light set, checker target faces). Variants: none (untextured, the box keeps its own material),
bx38 = Lambert 0.3 / 0.8 (new rows from the WL86 lambert_state constructor), bx75 = existing ids 3 (0.7) / 4 (0.5).
  python checker93d.py probe  <R> <variant> <spp> <out_prefix>       (scene iteration: textured + untextured render, PNGs, stats)
  python checker93d.py pre    <R> <R92A> <R93A> <R93D>
  python checker93d.py tables <R> <R92A> <R93A> <variant>
  python checker93d.py ref    <R> <variant> <seed> <spp> <otag>
  python checker93d.py render <R> <variant> <listfile>
  python checker93d.py viz    <R> <out>
"""
import glob, hashlib, inspect, json, math, os, re, sys, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import ops93d as OP  # noqa: E402  (puts the frozen WL86 / WL92 / WL93 module directories on sys.path)

DEV = '--dev' in sys.argv
SCENE_FN = os.environ.get('CK_SCENE', os.path.join(HERE, 'scene_ck2.json'))
SC = json.load(open(SCENE_FN))
TAG = SC['tag'] + '_T1'
E2 = 'E2_T1'                                  # code-path reproduction tests (WL93D tables) use the untextured E2 T1 geometry
TARGET_RECTS = [r for b in SC['targets'] for r in range(6 + 6 * 'AB'.index(b), 12 + 6 * 'AB'.index(b))]
TARGET_SIDS = [7 + 'AB'.index(b) for b in SC['targets']]
VAR = {'none': None, 'bx38': (6, 7), 'bx75': (3, 4)}
NEW_ALB = {6: 0.3, 7: 0.8}
ALB = {3: 0.7, 4: 0.5, 5: 0.6, 6: 0.3, 7: 0.8}
SEEDS = [8601, 8602]
ARMS = ['C0', 'C2_src', 'C2_rcv', 'C2_both']
_INST = {}


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1)


def lambert_state(alb):
    """Copy of the WL86 constructor (mats86.lambert_state; mats86.py writes files on import, so it is not imported). T_state checks it."""
    st = np.zeros(69, np.float32)
    for c in range(3):
        st[c * 8 + 0] = alb
    st[24:31] = 0.3; st[31:38] = 0.3; st[38:45] = 0.5
    st[45] = 1.0
    st[53:68] = 1.0; st[68] = 1.0
    return st


def new_rows():
    return np.stack([lambert_state(NEW_ALB[i]) for i in (6, 7)]).astype(np.float32)


def default_pmat(G):
    pm = np.zeros(G.N, np.int64)
    for r in range(len(G.RMAT)):
        nu, nv = G.RG[r]
        pm[G.RBASE[r]:G.RBASE[r] + nu * nv] = G.RMAT[r]
    pm[G.SBASE:] = 2
    return pm


def checker_map(G, variant):
    """Per-patch material ids: the default map with the checker on every target face ((iu + iv) even -> a, odd -> b)."""
    pm = default_pmat(G)
    if VAR[variant] is not None:
        a, b = VAR[variant]
        for r in TARGET_RECTS:
            nu, nv = G.RG[r]
            for iv in range(nv):
                for iu in range(nu):
                    pm[G.RBASE[r] + iu + nu * iv] = a if (iu + iv) % 2 == 0 else b
    return pm


def install(variant):
    """Material table + per-patch material map + patched material lookup in the renderer. Idempotent; one variant per process."""
    import fx92
    PT, S = fx92.PT, fx92.S
    if _INST:
        assert _INST['variant'] == variant
        return PT, S
    _INST['variant'] = variant
    _INST['orig_trace'] = PT.Ctx.trace
    om = PT.Mats.__init__

    def mats_init(self):
        om(self)
        self.states = np.concatenate([self.states, new_rows()]).astype(np.float32)
        self.desc = np.concatenate([self.desc, np.tile(self.desc[3:4], (2, 1))]).astype(np.float32)
        self.lambert = np.concatenate([self.lambert, np.array([1, 1], self.lambert.dtype)])
        self.albedo = np.concatenate([self.albedo, np.array([NEW_ALB[6], NEW_ALB[7]], self.albedo.dtype)])
        self.desc_cols = [PT.fl(self.desc[:, j]) for j in range(10)]
        self.lam_f = PT.fl(self.lambert); self.alb_f = PT.fl(self.albedo)
        self.cls = PT.U(np.array(list(S.MAT_CLASS) + [2, 2], np.uint32))
    PT.Mats.__init__ = mats_init
    og = S.Geom.__init__

    def geom_init(self, g):
        og(self, g)
        self.PMAT[:] = checker_map(self, variant)
    S.Geom.__init__ = geom_init
    ost = PT.Ctx.set_tables

    def set_tables(self, g):
        ost(self, g)
        self.PMATU = PT.U(np.asarray(g.PMAT, np.uint32))
    PT.Ctx.set_tables = set_tables
    src = textwrap.dedent(inspect.getsource(_INST['orig_trace']))
    pat = re.compile(r'^(\s*)mid = dr\.gather\(U, self\.SMAT, sid\)\n', re.M)
    m = list(pat.finditer(src))
    assert len(m) == 1, 'material lookup line not found exactly once'
    ind = m[0].group(1)
    ins = (f'{ind}mid = dr.gather(U, self.SMAT, sid)\n'
           f'{ind}_pid, _ = self.patch_bin(si, sid, -ray.d)\n'
           f'{ind}mid = dr.select(active, dr.gather(U, self.PMATU, _pid, active), mid)\n')
    src2 = src[:m[0].start()] + ins + src[m[0].end():]
    g = dict(PT.__dict__)
    exec(compile(src2, 'pt86.Ctx.trace[checker93d per-patch material lookup]', 'exec'), g)
    PT.Ctx.trace = g['trace']
    _INST['patched_trace'] = g['trace']
    _INST['patch_sha'] = hashlib.sha256(src2.encode()).hexdigest()
    return PT, S


def set_pmat(PT, ctx, pm):
    ctx.G.PMAT[:] = pm
    ctx.PMATU = PT.U(np.asarray(pm, np.uint32))


def extend_model(own):
    import torch
    own.mat_states = torch.cat([own.mat_states, torch.as_tensor(new_rows(), device=own.mat_states.device)], 0)


def mq_ext(Mq):
    lam = Mq[3].astype(np.float64) / 0.7
    return np.concatenate([Mq, (lam * 0.3)[None].astype(Mq.dtype), (lam * 0.8)[None].astype(Mq.dtype)])


def ctx_of(variant, R, tag=None):
    import fx92
    fx92.inject(R)
    PT, S = install(variant)
    FX = fx92.FX
    FX[SC['tag']] = {'T0': SC['geom'], 'T1': SC['geom'], 'moved': [], 'light_set': SC['light_set']}
    if 'lights' in SC:
        FX['light_sets'][SC['light_set']] = SC['lights']
    if 'camera' in SC:
        FX['camera'] = SC['camera']
    geom, ls, e, cfg = fx92.cfg_ls(tag or TAG)
    ctx = PT.Ctx(geom)
    return fx92, PT, S, FX, ctx, PT.Lights(FX['light_sets'][ls])


def tables_of(R92A, R93A, ctx, lt, S, Mq):
    import torch
    import ecro86 as EC
    import model86 as M86
    dev = torch.device('cuda')
    st = EC.build_state(ctx)
    gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
    out, inp = {}, {}
    with torch.no_grad():
        for sd in SEEDS:
            M = OP.load_family(R92A, R93A, 'own', sd)
            extend_model(M['own'])
            own = M['own']
            inp[sd] = own.mat_mlp(own.mat_states[gs.mat]).cpu().numpy()
            f = OP.local_factors(M, gs); G = OP.global_term(M, gs, s); rw = own.rw
            Eq = torch.zeros_like(s)
            for arm in ARMS:
                S_, _, _, _ = OP.arm_eval(M, f, f, Mn, gs, s, Eq, arm, rw)
                out[f'B_own_s{sd}_{arm}'] = (S_ + G)[0].cpu().numpy().astype(np.float32)
    return out, inp


# ------------------------------------------------------------------------------------------------ geometry queries
def gbuf(PT, ctx, cam):
    """Centre-ray G-buffer + ideal-mirror bounce off the panel: per pixel sid, pid, hit point; mirror hit sid / pid / distance."""
    dr, mi, U = PT.dr, PT.mi, PT.U
    res = cam['res']
    ray, pix = PT.gen_rays(cam, res, 1, None)
    si = ctx.scene.ray_intersect(ray)
    sid = ctx.shape_id(si)
    front = si.is_valid() & (si.to_local(-ray.d).z > 1e-6)
    pid, _ = ctx.patch_bin(si, sid, -ray.d)
    w5 = front & dr.eq(sid, 5)
    nrm = si.n
    r2 = mi.Ray3f(si.p + nrm * 1e-4, dr.normalize(ray.d - 2.0 * dr.dot(ray.d, nrm) * nrm))
    s2 = ctx.scene.ray_intersect(r2, w5)
    sid2 = ctx.shape_id(s2)
    pid2, _ = ctx.patch_bin(s2, sid2, -r2.d)
    ok2 = w5 & s2.is_valid() & (s2.to_local(-r2.d).z > 1e-6)
    dr.eval(front, sid, pid, si.p, w5, ok2, sid2, pid2, s2.t); dr.sync_thread()
    a = lambda v: np.asarray(v).reshape(res, res)
    return {'front': a(front), 'sid': np.where(a(front), a(sid), -1).astype(np.int64), 'pid': a(pid).astype(np.int64),
            'p': np.stack([a(si.p.x), a(si.p.y), a(si.p.z)], -1).astype(np.float64), 'panel': a(w5),
            'msid': np.where(a(ok2), a(sid2), -1).astype(np.int64), 'mpid': np.where(a(ok2), a(pid2), -1).astype(np.int64),
            'mt': np.where(a(ok2), a(s2.t), np.nan)}


def target_masks(G, gb):
    tgt_p = np.zeros(G.N, bool)
    for r in TARGET_RECTS:
        nu, nv = G.RG[r]
        tgt_p[G.RBASE[r]:G.RBASE[r] + nu * nv] = True
    roi = gb['panel'] & np.isin(gb['msid'], TARGET_SIDS) & (gb['mpid'] >= 0) & tgt_p[np.clip(gb['mpid'], 0, G.N - 1)]
    direct = gb['front'] & np.isin(gb['sid'], TARGET_SIDS)
    return roi, direct, tgt_p


def tm(x, key):
    y = x * key
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


# ------------------------------------------------------------------------------------------------ scene iteration (probe)
def probe(R, variant, spp, outp):
    """Render the textured scene and the untextured scene (same seed, common random numbers) and write PNGs + statistics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fx92, PT, S, FX, ctx, lt = ctx_of(variant, R)
    cam = FX['camera']
    gb = gbuf(PT, ctx, cam)
    roi, direct, tgt_p = target_masks(ctx.G, gb)
    pm_t = ctx.G.PMAT.copy()
    acc_t, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), 86100, 'cam', b=min(int(spp), 32))
    set_pmat(PT, ctx, default_pmat(ctx.G))
    acc_u, _ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), 86100, 'cam', b=min(int(spp), 32))
    set_pmat(PT, ctx, pm_t)
    ot = PT.pack(acc_t, cam['res'], int(spp)); ou = PT.pack(acc_u, cam['res'], int(spp))
    tot_t, tot_u = (ot['e1'] + ot['e2'] + ot['h']).astype(np.float64), (ou['e1'] + ou['e2'] + ou['h']).astype(np.float64)
    low_t, low_u = (ot['e1'] + ot['e2']).astype(np.float64), (ou['e1'] + ou['e2']).astype(np.float64)
    key = 0.5 / float(np.median(tot_u.mean(-1)[tot_u.mean(-1) > 0]))
    mp = gb['mpid']
    ids_hit = np.where(roi, pm_t[np.clip(mp, 0, ctx.N - 1)], -1)
    st = {'variant': variant, 'spp': int(spp), 'panel_pixels': int(gb['panel'].sum()), 'roi_pixels': int(roi.sum()),
          'roi_share_of_panel': float(roi.sum() / max(gb['panel'].sum(), 1)), 'target_direct_pixels': int(direct.sum()),
          'mirror_patches_hit': int(len(np.unique(mp[roi]))), 'mirror_distance_m_median': float(np.nanmedian(gb['mt'][roi])) if roi.any() else None,
          'wall_s': T_['wall_s']}
    if VAR[variant] is not None and roi.any():
        a, b = VAR[variant]
        for comp, xt, xu in (('LOW', low_t, low_u), ('tot', tot_t, tot_u)):
            r = xt.mean(-1) / np.maximum(xu.mean(-1), 1e-12)
            st[f'roi_ratio_{comp}'] = {f'id{a}': float(np.median(r[roi & (ids_hit == a)])) if (roi & (ids_hit == a)).any() else None,
                                       f'id{b}': float(np.median(r[roi & (ids_hit == b)])) if (roi & (ids_hit == b)).any() else None,
                                       'ideal': {f'id{a}': ALB[a] / ALB[4], f'id{b}': ALB[b] / ALB[4]}}
    ys, xs = np.nonzero(gb['panel'] | direct); pad = 15
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad)) if len(ys) else (slice(None), slice(None))
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(tm(tot_u, key)); ax[0, 0].set_title('untextured, full frame')
    ax[0, 1].imshow(tm(tot_t, key)); ax[0, 1].set_title(f'{variant}, full frame')
    ax[0, 2].imshow(np.where(roi[..., None], [0, 1, 1], np.where(direct[..., None], [1, 1, 0], tm(tot_t, key) * 0.5)))
    ax[0, 2].set_title('cyan: panel pixels whose mirror ray hits the checker box; yellow: box seen directly')
    ax[1, 0].imshow(tm(tot_u[sl], key)); ax[1, 0].set_title('untextured, close-up')
    ax[1, 1].imshow(tm(tot_t[sl], key)); ax[1, 1].set_title(f'{variant}, close-up (full radiance)')
    ax[1, 2].imshow(tm(low_t[sl], key)); ax[1, 2].set_title(f'{variant}, close-up, events 1-2 only (LOW)')
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'probe {SC["tag"]} {json.dumps(SC["geom"])}  light {SC["light_set"]}  {spp} spp\n' + json.dumps({k: v for k, v in st.items() if k.startswith('roi')}), fontsize=8)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    fig.savefig(outp + '.png', dpi=90, bbox_inches='tight'); plt.close(fig)
    jd(st, outp + '.json')
    print('PROBE', json.dumps(st), flush=True)


# ------------------------------------------------------------------------------------------------ preflight
def pre(R, R92A, R93A, R93D):
    import torch  # noqa: F401
    T = {}
    fx92, PT, S, FX, ctx, lt = ctx_of('bx38', R)
    dr, mi, CD = PT.dr, PT.mi, PT.CD
    U, F = PT.U, PT.F
    z = np.load(os.path.join(os.path.dirname(PT.__file__), 'data', 'mats86.npz'))
    T['T_state'] = {'pass': all(np.array_equal(lambert_state(a), z['states'][i]) for i, a in ((3, 0.7), (4, 0.5), (5, 0.6)))}
    ok = True
    for sd in SEEDS:
        M = OP.load_family(R92A, R93A, 'own', sd)
        b0 = M['own'].mat_states.cpu().numpy().copy()
        ok &= bool(np.array_equal(b0, z['states']))
        extend_model(M['own'])
        b1 = M['own'].mat_states.cpu().numpy()
        ok &= bool(np.array_equal(b1[:6], b0)) and bool(np.array_equal(b1[6:], new_rows())) and b1.shape[0] == 8
    T['T_buffer'] = {'pass': ok}
    import diag92 as DG
    _, Mq = DG.local_matrices()
    l3 = Mq[3].astype(np.float64) / 0.7
    dm = max(float(np.abs(Mq[4] / 0.5 - l3).max()), float(np.abs(Mq[5] / 0.6 - l3).max())) / float(np.abs(l3).max())
    T['T_mq'] = {'pass': dm <= 1e-5, 'rel_max': dm}
    n = 4096
    rs = np.random.default_rng(93900)
    idx_np = np.repeat(np.array([3, 6, 7], np.uint32), n)

    def hemi(k):
        v = rs.normal(size=(k, 3)); v[:, 2] = np.abs(v[:, 2]) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    wi_np = np.tile(hemi(n), (3, 1)); wo_np = np.tile(hemi(n), (3, 1)); u_np = np.tile(rs.random((n, 3)), (3, 1))
    v3 = lambda a: mi.Vector3f(*[F(np.ascontiguousarray(a[:, k].astype(np.float32))) for k in range(3)])
    cl = CD.DrClosure83('ER', ctx.mats.states, U(idx_np))
    act = mi.Bool(np.ones(3 * n, bool))
    f = cl.eval(v3(wi_np), v3(wo_np), act)
    wo, pdf, w, valid = cl.sample(v3(wi_np), F(u_np[:, 0].astype(np.float32)), mi.Point2f(F(u_np[:, 1].astype(np.float32)), F(u_np[:, 2].astype(np.float32))), act)
    g3 = lambda c: np.stack([np.asarray(c.x), np.asarray(c.y), np.asarray(c.z)], 1).astype(np.float64)
    fv, wv, wov = g3(f), g3(w), g3(wo)
    vv = np.asarray(valid)
    fr = [fv[k * n:(k + 1) * n] / a for k, a in enumerate((0.7, 0.3, 0.8))]
    wr = [wv[k * n:(k + 1) * n] / a for k, a in enumerate((0.7, 0.3, 0.8))]
    ev = max(float(np.abs(fr[k] - fr[0]).max()) for k in (1, 2)) / max(float(np.abs(fr[0]).max()), 1e-30)
    ew = max(float(np.abs(wr[k] - wr[0]).max()) for k in (1, 2)) if vv.all() else 1.0
    dwo = max(float(np.abs(wov[k * n:(k + 1) * n] - wov[:n]).max()) for k in (1, 2))
    T['T_closure'] = {'pass': ev <= 1e-5 and ew <= 1e-5 and dwo == 0.0, 'eval_rel': ev, 'weight_abs': ew, 'dir_abs': dwo}
    # code-path reproduction on the untextured E2 T1 geometry (default per-shape materials)
    _, _, _, _, cn, lt2 = ctx_of('bx38', R, E2)
    set_pmat(PT, cn, default_pmat(cn.G))
    I = np.load(os.path.join(R93D, 'results', 'tables', f'I_{E2}.npz'))
    Bt = PT.fl(I['B_own_s8601_C0'])
    camE = dict(FX['camera']); camE['res'] = 128
    res_t = {}
    rel = lambda x, y: max(float(np.abs(x[k] - y[k]).max() / max(np.abs(x[k]).max(), 1e-30)) for k in x)
    for mode in ('cam', 'ecro'):
        outs = []
        for tr in (_INST['orig_trace'], _INST['orig_trace'], _INST['patched_trace']):
            PT.Ctx.trace = tr
            acc, _ = cn.render(lt2, 'ER', camE, 128, 8, 93901, mode, B=Bt if mode == 'ecro' else None, b=8)
            outs.append(acc)
        PT.Ctx.trace = _INST['patched_trace']
        same = sorted(outs[0]) == sorted(outs[2])
        res_t[mode] = {'orig_vs_orig_rel': rel(outs[0], outs[1]), 'orig_vs_patched_rel': rel(outs[0], outs[2]) if same else None}
        res_t[mode]['pass'] = bool(same and res_t[mode]['orig_vs_patched_rel'] <= 1e-5)
    T['T_identity'] = {'pass': all(v['pass'] for v in res_t.values()), **res_t, 'note': 'float-atomic accumulation: tolerance rel 1e-5 (rev. 1 A1)'}
    tn, _ = tables_of(R92A, R93A, cn, lt2, S, mq_ext(Mq))
    I = np.load(os.path.join(R93D, 'results', 'tables', f'I_{E2}.npz'))
    dt = {k: float(np.abs(tn[k] - I[k]).max() / max(np.abs(I[k]).max(), 1e-30)) for k in tn}
    T['T_tables'] = {'pass': max(dt.values()) <= 1e-6, 'rel_max': dt}
    # correspondence on the checker scene: patched lookup at target hits vs an independent numpy face / cell computation
    cam = FX['camera']; res = cam['res']
    gb = gbuf(PT, ctx, cam)
    roi, direct, tgt_p = target_masks(ctx.G, gb)
    G = ctx.G
    p = gb['p'].reshape(-1, 3); sidf = gb['sid'].ravel()
    lookup = ctx.G.PMAT[np.clip(gb['pid'].ravel(), 0, G.N - 1)]
    exp = np.full(p.shape[0], -1); dist = np.full(p.shape[0], np.inf); face = np.full(p.shape[0], -1)
    pm_t = G.PMAT.copy()
    for r in TARGET_RECTS:
        nrm = np.cross(G.RA1[r], G.RA2[r])
        rel_ = p - G.RC[r]
        on = (np.abs(rel_ @ nrm) < 1e-3) & np.isin(sidf, TARGET_SIDS)
        u = (rel_ @ G.RA1[r] / G.RH[r, 0] + 1) * 0.5; v = (rel_ @ G.RA2[r] / G.RH[r, 1] + 1) * 0.5
        on &= (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        nu, nv = G.RG[r]
        iu = np.clip(np.floor(u * nu).astype(int), 0, nu - 1); iv = np.clip(np.floor(v * nv).astype(int), 0, nv - 1)
        dcell = np.minimum(np.minimum(np.abs(u * nu - np.round(u * nu)) * 2 * G.RH[r, 0] / nu, np.abs(v * nv - np.round(v * nv)) * 2 * G.RH[r, 1] / nv), 1.0)
        exp = np.where(on, pm_t[G.RBASE[r] + iu + nu * iv], exp); dist = np.where(on, dcell, dist); face = np.where(on, r, face)
    sel = (exp >= 0) & (dist > 1e-3)
    mism = int((lookup[sel] != exp[sel]).sum())
    T['T_corr'] = {'pass': mism == 0 and int(sel.sum()) > 300, 'pixels': int(sel.sum()), 'mismatch': mism}
    # end-to-end albedo ratio (kmax = 1, common random numbers): original renderer on the untextured map vs patched on the checker map
    PT.Ctx.trace = _INST['orig_trace']
    set_pmat(PT, ctx, default_pmat(G))
    a0, _ = ctx.render(lt, 'ER', cam, res, 4, 93902, 'cam', kmax=1, b=4)
    PT.Ctx.trace = _INST['patched_trace']
    set_pmat(PT, ctx, pm_t)
    a1, _ = ctx.render(lt, 'ER', cam, res, 4, 93902, 'cam', kmax=1, b=4)
    e0 = sum(a0[f'e1{c}'] for c in range(3)); e1 = sum(a1[f'e1{c}'] for c in range(3))
    fm = face.reshape(res, res)
    er_ = fm >= 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            er_ &= np.roll(np.roll(fm, dy, 0), dx, 1) == fm
    lit = (exp >= 0) & (dist > 0.02) & er_.ravel() & (e0 > 1e-6 * max(e0.max(), 1e-30))
    base = {b: ALB[4 + 'AB'.index(b)] for b in SC['targets']}
    bmat = np.array([ALB[G.RMAT[r]] if r >= 0 else 1.0 for r in face])
    expr = np.array([ALB[int(m)] for m in np.where(exp >= 0, exp, 3)]) / bmat
    er = float(np.abs(e1[lit] / e0[lit] - expr[lit]).max()) if lit.any() else 1.0
    T['T_e2e'] = {'pass': er <= 1e-4 and int(lit.sum()) > 100, 'lit_pixels': int(lit.sum()), 'ratio_abs_err_max': er, 'base_albedo': base}
    # model awareness: per-patch inputs change exactly on the target patches
    _, inp_c = tables_of(R92A, R93A, ctx, lt, S, mq_ext(Mq))
    set_pmat(PT, ctx, default_pmat(G))
    _, inp_n = tables_of(R92A, R93A, ctx, lt, S, mq_ext(Mq))
    set_pmat(PT, ctx, pm_t)
    aw = {}
    for sd in SEEDS:
        d = np.abs(inp_c[sd] - inp_n[sd]).max(1)
        aw[sd] = {'target_patches_changed': bool((d[tgt_p] > 0).all()), 'other_patches_unchanged': bool((d[~tgt_p] == 0).all())}
    T['T_aware'] = {'pass': all(v['target_patches_changed'] and v['other_patches_unchanged'] for v in aw.values()), 'bx38': aw,
                    'target_patches': int(tgt_p.sum())}
    V = {'panel_pixels': int(gb['panel'].sum()), 'roi_pixels': int(roi.sum()), 'target_direct_pixels': int(direct.sum()),
         'mirror_patches_hit': int(len(np.unique(gb['mpid'][roi]))), 'mirror_distance_m_median': float(np.nanmedian(gb['mt'][roi])) if roi.any() else None}
    out = os.path.join(R, 'results', 'pre'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'masks_ck2.npz'), ROI_T=roi, panel=gb['panel'], target_direct=direct, frame=gb['front'],
                        mirror_patch=np.where(roi, gb['mpid'], -1), sid=gb['sid'], pid=gb['pid'], PMAT_bx38=pm_t, PMAT_bx75=checker_map(G, 'bx75'),
                        PMAT_none=default_pmat(G))
    okk = all(v['pass'] for v in T.values())
    status = 'PASS' if okk and V['roi_pixels'] >= 300 else ('STOP_TESTS' if not okk else 'STOP_NOT_VISIBLE')
    jd({'status': status, 'scene': SC, 'tests': T, 'visibility': V, 'trace_patch_sha256': _INST['patch_sha']}, os.path.join(out, 'pre.json'))
    print('PRE', status, json.dumps({k: v['pass'] for k, v in T.items()}), json.dumps(V), flush=True)


# ------------------------------------------------------------------------------------------------ tables / renders
def tables(R, R92A, R93A, variant):
    import diag92 as DG
    fx92, PT, S, FX, ctx, lt = ctx_of(variant, R)
    _, Mq = DG.local_matrices()
    out, _ = tables_of(R92A, R93A, ctx, lt, S, mq_ext(Mq))
    fn = os.path.join(R, 'results', 'tables', f'CK_{variant}_{TAG}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **out, PMAT=ctx.G.PMAT)
    print('tables', variant, len(out), flush=True)


def ref(R, variant, seed, spp, otag):
    fx92, PT, S, FX, ctx, lt = ctx_of(variant, R)
    cam = FX['camera']
    acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'cam', b=32)
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{otag}.npz'), **PT.pack(acc, cam['res'], int(spp)))
    jd({'tag': TAG, 'variant': variant, 'seed': int(seed), 'spp': int(spp), 'timing': T_, 'trace_patch_sha256': _INST['patch_sha'], 'scene': SC},
       os.path.join(out, f'{otag}.json'))
    print(otag, json.dumps(T_), flush=True)


def render(R, variant, listfn):
    fx92, PT, S, FX, ctx, lt = ctx_of(variant, R)
    cam = FX['camera']
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    cache = {}
    for line in open(listfn):
        if not line.strip():
            continue
        src, seed, spp, otag = line.split()
        if os.path.exists(os.path.join(out, f'{otag}.npz')):
            continue
        fn, key = src.split(':')
        if fn not in cache:
            cache = {fn: np.load(os.path.join(R, fn))}
        B = cache[fn][key].astype(np.float32)
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'ecro', B=PT.fl(B), b=32)
        np.savez_compressed(os.path.join(out, f'{otag}.npz'), hl=PT.pack(acc, cam['res'], int(spp))['hl'])
        jd({'variant': variant, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{otag}.json'))
        print(otag, round(T_['wall_s'], 2), flush=True)
    print('RENDER_DONE', variant, flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'probe':
        probe(a[1], a[2], a[3], a[4])
    elif cmd == 'pre':
        pre(*a[1:5])
    elif cmd == 'tables':
        tables(*a[1:5])
    elif cmd == 'ref':
        ref(*a[1:6])
    elif cmd == 'render':
        render(*a[1:4])
    elif cmd == 'viz':
        import viz_checker93d as VZ
        VZ.main(*a[1:3])
    else:
        raise SystemExit(cmd)
