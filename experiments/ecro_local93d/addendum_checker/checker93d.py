"""WL93D POST-HOC QUALITATIVE ADDENDUM (non-numbered): patch-aligned checkerboard albedo on the back wall, E2 T1.

Frozen checkpoints (WL92A own s8601 / s8602) and the frozen WL93D arm evaluation (ops93d.arm_eval); nothing is trained or refitted.
The texture is a per-PATCH material-id map consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors);
  (ii)  the frozen-ER M64 matrices: Mn = Mq[PMAT[patch]];
  (iii) the physical reference / LOW / ECRO-lookup renderer: at a back-wall hit the material id is PMAT[patch of the hit] instead of the
        per-shape id (one lookup inserted into pt86.Ctx.trace by a run-time source patch; pt86.py is not edited).
Variants (protocol_checker93d.json): ck38 = Lambert 0.3 / 0.8 (new rows from the WL86 lambert_state constructor), ck75 = existing ids
3 (0.7) / 4 (0.5), none = patched code path with the checker off (identity control).
  python checker93d.py pre    <R> <R86> <R92A> <R93A> <R93D>
  python checker93d.py tables <R> <R92A> <R93A> <variant>
  python checker93d.py ref    <R> <variant> <seed> <spp> <otag>
  python checker93d.py render <R> <variant> <listfile>
  python checker93d.py viz    <R> <R86> <R93D> <out>
"""
import glob, hashlib, inspect, json, math, os, re, sys, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import ops93d as OP  # noqa: E402  (puts the frozen WL86 / WL92 / WL93 module directories on sys.path)

DEV = '--dev' in sys.argv
TAG = 'E2_T1'
WALL = 2                                     # rectangle index of the back wall (y = 2, 16 x 12 patches of 25 cm)
VAR = {'none': None, 'ck38': (6, 7), 'ck75': (3, 4)}
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


def checker_ids(variant, nu=16, nv=12):
    a, b = VAR[variant]
    return np.array([[a if (iu + iv) % 2 == 0 else b for iu in range(nu)] for iv in range(nv)], np.int64)   # [iv, iu]


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
        if VAR[variant] is not None:
            nu, nv = self.RG[WALL]
            ids = checker_ids(variant, nu, nv)
            for iv in range(nv):
                for iu in range(nu):
                    self.PMAT[self.RBASE[WALL] + iu + nu * iv] = ids[iv, iu]
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
           f'{ind}_pw = active & dr.eq(sid, {WALL})\n'
           f'{ind}_pid, _ = self.patch_bin(si, sid, -ray.d)\n'
           f'{ind}mid = dr.select(_pw, dr.gather(U, self.PMATU, _pid, _pw), mid)\n')
    src2 = src[:m[0].start()] + ins + src[m[0].end():]
    g = dict(PT.__dict__)
    exec(compile(src2, 'pt86.Ctx.trace[checker93d per-patch material lookup]', 'exec'), g)
    PT.Ctx.trace = g['trace']
    _INST['patched_trace'] = g['trace']
    _INST['patch_sha'] = hashlib.sha256(src2.encode()).hexdigest()
    return PT, S


def extend_model(own):
    import torch
    own.mat_states = torch.cat([own.mat_states, torch.as_tensor(new_rows(), device=own.mat_states.device)], 0)


def mq_ext(Mq):
    lam = Mq[3].astype(np.float64) / 0.7
    return np.concatenate([Mq, (lam * 0.3)[None].astype(Mq.dtype), (lam * 0.8)[None].astype(Mq.dtype)])


def ctx_of(variant, R):
    import fx92
    fx92.inject(R)
    PT, S = install(variant)
    FX = fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(TAG)
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


# ------------------------------------------------------------------------------------------------ preflight
def pre(R, R86, R92A, R93A, R93D):
    import torch
    T, V = {}, {}
    fx92, PT, S, FX, ctx, lt = ctx_of('ck38', R)
    dr, mi, CD = PT.dr, PT.mi, PT.CD
    U, F = PT.U, PT.F
    z = np.load(os.path.join(os.path.dirname(PT.__file__), 'data', 'mats86.npz'))
    T['T_state'] = {'pass': all(np.array_equal(lambert_state(a), z['states'][i]) for i, a in ((3, 0.7), (4, 0.5), (5, 0.6)))}
    # T_buffer
    ok = True
    for sd in SEEDS:
        M = OP.load_family(R92A, R93A, 'own', sd)
        b0 = M['own'].mat_states.cpu().numpy().copy()
        ok &= bool(np.array_equal(b0, z['states']))
        extend_model(M['own'])
        b1 = M['own'].mat_states.cpu().numpy()
        ok &= bool(np.array_equal(b1[:6], b0)) and bool(np.array_equal(b1[6:], new_rows())) and b1.shape[0] == 8
    T['T_buffer'] = {'pass': ok}
    # T_mq
    import diag92 as DG
    _, Mq = DG.local_matrices()
    l3 = Mq[3].astype(np.float64) / 0.7
    dm = max(float(np.abs(Mq[4] / 0.5 - l3).max()), float(np.abs(Mq[5] / 0.6 - l3).max())) / float(np.abs(l3).max())
    T['T_mq'] = {'pass': dm <= 1e-5, 'rel_max': dm, 'Mq_shape': list(Mq.shape)}
    # T_closure
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
    fr = [fv[k * n:(k + 1) * n] / a for k, a in enumerate((0.7, 0.3, 0.8))]
    wr = [wv[k * n:(k + 1) * n] / a for k, a in enumerate((0.7, 0.3, 0.8))]
    ev = max(float(np.abs(fr[k] - fr[0]).max()) for k in (1, 2)) / max(float(np.abs(fr[0]).max()), 1e-30)
    ew = max(float(np.abs(wr[k][np.asarray(valid)[k * n:(k + 1) * n]] - wr[0][np.asarray(valid)[:n]]).max()) for k in (1, 2)) if np.asarray(valid).all() else 1.0
    dwo = max(float(np.abs(wov[k * n:(k + 1) * n] - wov[:n]).max()) for k in (1, 2))
    lam_ref = float(np.abs(fv[:n] - 0.7 / math.pi * np.clip(wo_np[:n, 2:3], 0, None)).max())
    T['T_closure'] = {'pass': ev <= 1e-5 and ew <= 1e-5 and dwo == 0.0, 'eval_rel': ev, 'weight_abs': ew, 'dir_abs': dwo,
                      'id3_vs_lambert_0.7cos/pi_abs': lam_ref, 'valid_all': bool(np.asarray(valid).all())}
    # T_identity (variant none emulated: the checker is a property of ctx.G -> use a ctx with PMAT reset to the per-shape map)
    import ecro86 as EC  # noqa: F401
    geom = fx92.cfg_ls(TAG)[0]
    cn = PT.Ctx(geom)                                   # this process has ck38 installed: reset the wall to its per-shape id 3
    nu, nv = cn.G.RG[WALL]
    for iv in range(nv):
        for iu in range(nu):
            cn.G.PMAT[cn.G.RBASE[WALL] + iu + nu * iv] = 3
    cn.PMATU = U(np.asarray(cn.G.PMAT, np.uint32))
    I = np.load(os.path.join(R93D, 'results', 'tables', f'I_{TAG}.npz'))
    Bt = PT.fl(I['B_own_s8601_C0'])
    cam = FX['camera']
    res_t = {}
    for mode in ('cam', 'ecro'):
        outs = []
        for tr in (_INST['orig_trace'], _INST['orig_trace'], _INST['patched_trace']):      # orig twice: float-atomic accumulation floor
            PT.Ctx.trace = tr
            acc, _ = cn.render(lt, 'ER', cam, 128, 8, 93901, mode, B=Bt if mode == 'ecro' else None, b=8)
            outs.append(acc)
        PT.Ctx.trace = _INST['patched_trace']
        rel = lambda x, y: max(float(np.abs(x[k] - y[k]).max() / max(np.abs(x[k]).max(), 1e-30)) for k in x)
        same_keys = sorted(outs[0]) == sorted(outs[2])
        res_t[mode] = {'keys_equal': same_keys, 'orig_vs_orig_rel': rel(outs[0], outs[1]), 'orig_vs_patched_rel': rel(outs[0], outs[2]) if same_keys else None}
        res_t[mode]['pass'] = bool(same_keys and res_t[mode]['orig_vs_patched_rel'] <= 1e-5)
    T['T_identity'] = {'pass': all(v['pass'] for v in res_t.values()), **res_t,
                       'note': 'bit-exact comparison is not possible: pt86.accum sums samples with float atomics (scatter_reduce), whose order varies between runs; tolerance rel 1e-5 (amended after the dev smoke, before any full-budget render)'}
    # T_tables (variant none: per-shape materials, original renderer path is irrelevant for tables)
    tn, inp_n = tables_of(R92A, R93A, cn, lt, S, mq_ext(Mq))
    dt = {k: float(np.abs(tn[k] - I[k]).max() / max(np.abs(I[k]).max(), 1e-30)) for k in tn}
    T['T_tables'] = {'pass': max(dt.values()) <= 1e-6, 'rel_max': dt}
    # T_corr: material id at back-wall pixels from the patched lookup == analytic checker from the hit position
    res = cam['res']
    ray, pix = PT.gen_rays(cam, res, 1, None)
    si = ctx.scene.ray_intersect(ray)
    sid = ctx.shape_id(si)
    pid, _ = ctx.patch_bin(si, sid, -ray.d)
    w = si.is_valid() & dr.eq(sid, WALL) & (si.to_local(-ray.d).z > 1e-6)
    mid = dr.select(w, dr.gather(U, ctx.PMATU, pid, w), dr.gather(U, ctx.SMAT, sid))
    dr.eval(mid, w, si.p); dr.sync_thread()
    wn = np.asarray(w); px = np.asarray(si.p.x).astype(np.float64); pz = np.asarray(si.p.z).astype(np.float64); md = np.asarray(mid)
    iu = np.floor((px + 2.0) / 0.25).astype(int); iv = np.floor(pz / 0.25).astype(int)
    dist = np.minimum(np.abs((px + 2.0) / 0.25 - np.round((px + 2.0) / 0.25)), np.abs(pz / 0.25 - np.round(pz / 0.25))) * 0.25
    edge = dist < 1e-3
    inside = (iu >= 0) & (iu < 16) & (iv >= 0) & (iv < 12)
    sel = wn & ~edge & inside
    sel_j = wn & ~(dist < 0.03) & inside           # jittered renders: stay >= 3 cm (about 3 pixel footprints) away from cell boundaries
    ids = checker_ids('ck38')
    exp = ids[np.clip(iv, 0, 11), np.clip(iu, 0, 15)]
    mism = int((md[sel] != exp[sel]).sum())
    T['T_corr'] = {'pass': mism == 0 and int(sel.sum()) > 1000, 'pixels': int(sel.sum()), 'mismatch': mism, 'edge_excluded': int((wn & edge).sum())}
    wall_px = wn.reshape(res, res)
    # T_e2e: kmax = 1, common random numbers, original renderer on the untextured ctx vs patched on ck38
    PT.Ctx.trace = _INST['orig_trace']
    cu = PT.Ctx(geom)
    nu_, nv_ = cu.G.RG[WALL]
    for iv_ in range(nv_):
        for iu_ in range(nu_):
            cu.G.PMAT[cu.G.RBASE[WALL] + iu_ + nu_ * iv_] = 3
    cu.PMATU = U(np.asarray(cu.G.PMAT, np.uint32))
    a0, _ = cu.render(lt, 'ER', cam, res, 4, 93902, 'cam', kmax=1, b=4)
    PT.Ctx.trace = _INST['patched_trace']
    a1, _ = ctx.render(lt, 'ER', cam, res, 4, 93902, 'cam', kmax=1, b=4)
    e0 = sum(a0[f'e1{c}'] for c in range(3)); e1 = sum(a1[f'e1{c}'] for c in range(3))
    wg = wn.reshape(res, res)
    er_ = wg.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            er_ &= np.roll(np.roll(wg, dy, 0), dx, 1)
    er_[0, :] = er_[-1, :] = er_[:, 0] = er_[:, -1] = False             # 3 x 3 neighbourhood all back wall: no silhouette pixels
    lit = sel_j & er_.ravel() & (e0 > 1e-6 * max(e0.max(), 1e-30))
    ratio = e1[lit] / e0[lit]
    expr = np.vectorize(lambda m: ALB[int(m)] / 0.7)(exp[lit])
    er = float(np.abs(ratio - expr).max()) if lit.any() else 1.0
    T['T_e2e'] = {'pass': er <= 1e-4 and int(lit.sum()) > 500, 'lit_pixels': int(lit.sum()), 'ratio_abs_err_max': er,
                  'ratio_abs_err_p50': float(np.median(np.abs(ratio - expr))) if lit.any() else None,
                  'note': 'pixels whose 3 x 3 neighbourhood is not entirely back wall are excluded (jittered samples of silhouette pixels hit other objects; amended after the dev smoke, before any full-budget render)'}
    # T_aware: model inputs change exactly on the back-wall patches whose id changed
    _, inp_c = tables_of(R92A, R93A, ctx, lt, S, mq_ext(Mq))     # ck38 tables (also used below as a smoke of the table path)
    chg = np.zeros(ctx.N, bool)
    chg[ctx.G.RBASE[WALL]:ctx.G.RBASE[WALL] + nu * nv] = True
    aw = {}
    for sd in SEEDS:
        d = np.abs(inp_c[sd] - inp_n[sd]).max(1)
        aw[sd] = {'changed_patches_changed': bool((d[chg] > 0).all()), 'other_patches_unchanged': bool((d[~chg] == 0).all())}
    ids75 = checker_ids('ck75')
    exp75 = cn.G.PMAT.copy()
    for iv_ in range(nv):
        for iu_ in range(nu):
            exp75[cn.G.RBASE[WALL] + iu_ + nu * iv_] = ids75[iv_, iu_]
    T['T_aware'] = {'pass': all(v['changed_patches_changed'] and v['other_patches_unchanged'] for v in aw.values()), 'ck38': aw,
                    'ck75_changes_patches': int((exp75 != cn.G.PMAT).sum()), 'ck38_changes_patches': int((ctx.G.PMAT != cn.G.PMAT).sum()),
                    'note': 'ck75 keeps id 3 on half of the wall, so only the id-4 cells change (checked via the PMAT map; same lookup path as ck38)'}
    # ---------------------------------------------------------------- visibility through the panel mirror (geometry only)
    pn = wn.copy()
    w5 = si.is_valid() & dr.eq(sid, 5) & (si.to_local(-ray.d).z > 1e-6)
    nrm = si.n
    dd = ray.d - 2.0 * dr.dot(ray.d, nrm) * nrm
    r2 = mi.Ray3f(si.p + nrm * 1e-4, dr.normalize(dd))
    s2 = ctx.scene.ray_intersect(r2, w5)
    sid2 = ctx.shape_id(s2)
    pid2, _ = ctx.patch_bin(s2, sid2, -r2.d)
    hw = w5 & s2.is_valid() & dr.eq(sid2, WALL) & (s2.to_local(-r2.d).z > 1e-6)
    dr.eval(w5, hw, pid2, s2.t); dr.sync_thread()
    w5n = np.asarray(w5).reshape(res, res); hwn = np.asarray(hw).reshape(res, res); p2 = np.asarray(pid2).reshape(res, res)
    cells = p2[hwn] - ctx.G.RBASE[WALL]
    par = (cells % nu + cells // nu) % 2
    uc = np.unique(cells)
    upar = (uc % nu + uc // nu) % 2
    Mz = np.load(os.path.join(R86, 'results', 'masks', 'E2.npz'))
    Cm = Mz['C']; Pm = Mz['obj_panel']
    V = {'panel_pixels_camera': int(w5n.sum()), 'panel_pixels_reflect_back_wall': int(hwn.sum()), 'share': float(hwn.sum() / max(w5n.sum(), 1)),
         'cells_hit': int(len(uc)), 'cells_even': int((upar == 0).sum()), 'cells_odd': int((upar == 1).sum()),
         'pixels_even': int((par == 0).sum()), 'pixels_odd': int((par == 1).sum()),
         'C_pixels': int(Cm.sum()), 'C_on_panel': int((Cm & w5n).sum()), 'C_reflect_back_wall': int((Cm & hwn).sum()),
         'obj_panel_pixels': int(Pm.sum()), 'mean_mirror_distance_m': float(np.asarray(s2.t).reshape(res, res)[hwn].mean()) if hwn.any() else None}
    V['visible'] = bool(V['share'] >= 0.25 and V['cells_even'] >= 2 and V['cells_odd'] >= 2)
    out = os.path.join(R, 'results', 'pre'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'masks_ck.npz'), ROI_W=hwn, panel_cam=w5n, wall_direct=wall_px, mirror_cell=np.where(hwn, p2 - ctx.G.RBASE[WALL], -1))
    ok = all(v['pass'] for v in T.values())
    status = 'PASS' if ok and V['visible'] else ('STOP_TESTS' if not ok else 'STOP_NOT_VISIBLE')
    jd({'status': status, 'tests': T, 'visibility': V, 'trace_patch_sha256': _INST['patch_sha']}, os.path.join(out, 'pre.json'))
    print('PRE', status, json.dumps({k: v['pass'] for k, v in T.items()}), json.dumps(V), flush=True)


def preview(R, variant, spp, otag):
    """64-spp reference preview (presentation only)."""
    fx92, PT, S, FX, ctx, lt = ctx_of(variant, R)
    cam = FX['camera']
    acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), 86100, 'cam', b=32)
    out = os.path.join(R, 'results', 'pre'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{otag}.npz'), **PT.pack(acc, cam['res'], int(spp)))
    print('preview', otag, round(T_['wall_s'], 2), flush=True)


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
    jd({'tag': TAG, 'variant': variant, 'seed': int(seed), 'spp': int(spp), 'timing': T_, 'trace_patch_sha256': _INST['patch_sha']}, os.path.join(out, f'{otag}.json'))
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
    if cmd == 'pre':
        pre(*a[1:6])
    elif cmd == 'preview':
        preview(a[1], a[2], a[3], a[4])
    elif cmd == 'tables':
        tables(*a[1:5])
    elif cmd == 'ref':
        ref(*a[1:6])
    elif cmd == 'render':
        render(*a[1:4])
    elif cmd == 'viz':
        import viz_checker93d as VZ
        VZ.main(*a[1:5])
    else:
        raise SystemExit(cmd)
