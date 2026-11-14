"""WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are appended to
        the BUFFER (WL86 constructor), no parameter changes;
  (ii)  frozen-ER M64: Mn = Mq[PMAT];
  (iii) the physical references: pt86.Ctx.trace (camera / ECRO lookup renders) AND pt86a.field_src (patch x bin field references) get the
        same one-line run-time patch: at every surface hit mid = PMAT[patch of the hit] instead of the per-shape id. Source files unedited.

  python mat94.py pre    <R> <R92A> <R92B> <R93D> <RCK>          Stage A tests + masks (results/pre/)
  python mat94.py fref   <R> <variant>                            CK2 field reference, two halves (results/fields/CKF_<v>.npz)
  python mat94.py iref   <R> <variant> <r>                        CK2 REF-ER image, 8,192 spp, seed 86100 + r
  python mat94.py tables <R> <R92A> <spec> <otag>                 CK2 tables of every variant; spec own:<seed> | cand:<tag>
  python mat94.py render <R> <listfile>                           ECRO lookup renders: lines '<variant> <tables npz>:<key> <seed> <spp> <otag>'
"""
import hashlib, inspect, json, math, os, re, sys, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_local93d'))
import ops93d as OP  # noqa: E402  (puts the frozen WL86 / WL86A / WL92 / WL93 module directories on sys.path)

DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_mat94.json')))
SCENE_FN = os.path.join(HERE, '..', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')
SC = json.load(open(SCENE_FN))
TAG = SC['tag'] + '_T1'
ALB = {3: 0.7, 4: 0.5, 5: 0.6, 6: 0.3, 7: 0.8, 8: 0.2, 9: 0.4, 10: 0.9}
NEW_IDS = [6, 7, 8, 9, 10]
TARGET_RECTS = list(range(6, 12))            # box A faces
PANEL_FACE = 6                               # outward normal -x (towards the panel)
LAMBERT_RECTS = [1, 2, 3, 4] + list(range(6, 18))
VARIANTS = ['none', 'U07', 'U03', 'bx75', 'bx38', 'bx38i']
VDEF = {'none': None, 'U07': ('u', 3), 'U03': ('u', 6), 'bx75': ('c', 3, 4), 'bx38': ('c', 6, 7), 'bx38i': ('c', 7, 6)}
SEEDS = [8601, 8602]
ARMS_FROZEN = ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1', 'C2_srcNonS1']
FREF_SPP, FREF_SEEDS = 4096, (94301, 94302)
IREF_SPP, ASPP = 8192, 4096
_INST = {}


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def sha_file(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def lambert_state(alb):
    """Copy of the WL86 constructor (mats86.lambert_state writes files on import, so it is not imported; T_STATE checks the copy)."""
    st = np.zeros(69, np.float32)
    for c in range(3):
        st[c * 8 + 0] = alb
    st[24:31] = 0.3; st[31:38] = 0.3; st[38:45] = 0.5
    st[45] = 1.0
    st[53:68] = 1.0; st[68] = 1.0
    return st


def new_rows():
    return np.stack([lambert_state(ALB[i]) for i in NEW_IDS]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ material maps
def default_pmat(G):
    pm = np.zeros(G.N, np.int64)
    for r in range(len(G.RMAT)):
        nu, nv = G.RG[r]
        pm[G.RBASE[r]:G.RBASE[r] + nu * nv] = G.RMAT[r]
    pm[G.SBASE:] = 2
    return pm


def rect_patches(G, r):
    nu, nv = G.RG[r]
    return np.arange(G.RBASE[r], G.RBASE[r] + nu * nv)


def variant_pmat(G, v):
    pm = default_pmat(G)
    d = VDEF[v]
    if d is None:
        return pm
    for r in TARGET_RECTS:
        nu, nv = G.RG[r]
        for iv in range(nv):
            for iu in range(nu):
                p = G.RBASE[r] + iu + nu * iv
                pm[p] = d[1] if d[0] == 'u' else (d[1] if (iu + iv) % 2 == 0 else d[2])
    return pm


def train_pmat_B(G, idx):
    """Candidate B: one albedo per Lambert rectangle, uniform over V_train (ids 8, 9, 4, 5, 3, 10)."""
    rng = np.random.default_rng(94100 + int(idx))
    pm = default_pmat(G)
    ids = np.array([8, 9, 4, 5, 3, 10])
    for r in LAMBERT_RECTS:
        pm[rect_patches(G, r)] = ids[rng.integers(len(ids))]
    return pm


def val_pmat_ck(G, idx):
    """Held-out layout: per-patch 0.3 / 0.8 checker on every Lambert rectangle, phase from rng(94200 + idx)."""
    rng = np.random.default_rng(94200 + int(idx))
    pm = default_pmat(G)
    for r in LAMBERT_RECTS:
        ph = int(rng.integers(2)); nu, nv = G.RG[r]
        for iv in range(nv):
            for iu in range(nu):
                pm[G.RBASE[r] + iu + nu * iv] = 6 if (iu + iv + ph) % 2 == 0 else 7
    return pm


def pm_digest(pm):
    return hashlib.sha256(np.asarray(pm, np.int64).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ renderer / reference patch
def _patch_src(fn, obj, label, probe_k1=False):
    """probe_k1 (field_src only): at k = 1, when the hit lies on the probe patch's own shape it IS the probe point of patch pid0, so its
    material is PMAT[pid0] (patch_bin's cell assignment rounded a few probe points to a neighbouring cell at cell edges; dev preflight,
    fixed before the freeze). Hits on another shape within 1 mm (contacts) keep the original lookup."""
    src = textwrap.dedent(inspect.getsource(fn))
    pat = re.compile(r'^(\s*)mid = dr\.gather\(U, ' + obj + r'\.SMAT, sid\)\n', re.M)
    m = list(pat.finditer(src))
    assert len(m) == 1, f'material lookup line not found exactly once in {label}'
    ind = m[0].group(1)
    ins = (f'{ind}mid = dr.gather(U, {obj}.SMAT, sid)\n'
           f'{ind}_pid, _ = {obj}.patch_bin(si, sid, -ray.d)\n'
           + (f'{ind}if k == 1:\n{ind}    _pid = dr.select(dr.eq(sid, dr.gather(U, {obj}.PSIDU, pid0)), pid0, _pid)\n' if probe_k1 else '')
           + f'{ind}mid = dr.select(active, dr.gather(U, {obj}.PMATU, _pid, active), mid)\n')
    return src[:m[0].start()] + ins + src[m[0].end():]


def install():
    """Material table (+ ids 6..10), PMATU tables and the per-patch material lookup in Ctx.trace and field_src. Idempotent."""
    import fx92
    import pt86a as A
    PT, S = fx92.PT, fx92.S
    if _INST:
        return PT, S, A
    _INST['orig_trace'] = PT.Ctx.trace
    _INST['orig_field'] = A.field_src
    om = PT.Mats.__init__

    def mats_init(self):
        om(self)
        k = len(NEW_IDS)
        self.states = np.concatenate([self.states, new_rows()]).astype(np.float32)
        self.desc = np.concatenate([self.desc, np.tile(self.desc[3:4], (k, 1))]).astype(np.float32)
        self.lambert = np.concatenate([self.lambert, np.ones(k, self.lambert.dtype)])
        self.albedo = np.concatenate([self.albedo, np.array([ALB[i] for i in NEW_IDS], self.albedo.dtype)])
        self.desc_cols = [PT.fl(self.desc[:, j]) for j in range(10)]
        self.lam_f = PT.fl(self.lambert); self.alb_f = PT.fl(self.albedo)
        self.cls = PT.U(np.array(list(S.MAT_CLASS) + [2] * k, np.uint32))
    PT.Mats.__init__ = mats_init
    ost = PT.Ctx.set_tables

    def set_tables(self, g):
        ost(self, g)
        self.PMATU = PT.U(np.asarray(g.PMAT, np.uint32))
        self.PSIDU = PT.U(np.asarray(g.PSID, np.uint32))
    PT.Ctx.set_tables = set_tables
    s1 = _patch_src(_INST['orig_trace'], 'self', 'pt86.Ctx.trace')
    g = dict(PT.__dict__)
    exec(compile(s1, 'pt86.Ctx.trace[mat94 per-patch material lookup]', 'exec'), g)
    PT.Ctx.trace = g['trace']; _INST['patched_trace'] = g['trace']
    s2 = _patch_src(_INST['orig_field'], 'ctx', 'pt86a.field_src', probe_k1=True)
    g2 = dict(A.__dict__)
    exec(compile(s2, 'pt86a.field_src[mat94 per-patch material lookup]', 'exec'), g2)
    A.field_src = g2['field_src']; _INST['patched_field'] = g2['field_src']
    _INST['patch_sha'] = {'trace': hashlib.sha256(s1.encode()).hexdigest(), 'field_src': hashlib.sha256(s2.encode()).hexdigest()}
    return PT, S, A


def set_pmat(PT, ctx, pm):
    ctx.G.PMAT[:] = pm
    ctx.PMATU = PT.U(np.asarray(pm, np.uint32))


def ctx_ck2(R):
    """Ctx of the CK2 scene (geometry, light set, camera of the WL93D addendum rev. 2), default material map."""
    import fx92
    fx92.inject(R)
    PT, S, A = install()
    FX = fx92.FX
    FX[SC['tag']] = {'T0': SC['geom'], 'T1': SC['geom'], 'moved': [], 'light_set': SC['light_set']}
    FX['light_sets'][SC['light_set']] = SC['lights']
    FX['camera'] = SC['camera']
    ctx = PT.Ctx(SC['geom'])
    return PT, S, A, FX, ctx, PT.Lights(SC['lights'])


# ------------------------------------------------------------------------------------------------ models / operators
def extend_buffer(model):
    import torch
    if model.mat_states.shape[0] == 6:
        model.mat_states = torch.cat([model.mat_states, torch.as_tensor(new_rows(), device=model.mat_states.device)], 0)
    return model


def load_frozen(R92A, seed):
    import torch
    import train93 as TR
    return extend_buffer(TR.load_own(R92A, int(seed), torch.device('cuda')))


def load_cand(R, tag):
    import torch
    import model86 as M86
    dev = torch.device('cuda')
    m = extend_buffer(M86.ECRO().to(dev))
    m.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{tag}.pt'), map_location=dev))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def load_spec(R, R92A, spec):
    k, v = spec.split(':')
    return load_frozen(R92A, v) if k == 'own' else load_cand(R, v)


def mq_ext():
    import diag92 as DG
    _, Mq = DG.local_matrices()
    lam = Mq[3].astype(np.float64) / 0.7
    return np.concatenate([Mq] + [(lam * ALB[i])[None].astype(Mq.dtype) for i in NEW_IDS])


def arm_tables(model, gs, s, Mn, s1, arms):
    """Single flight per arm (WL93D arm definitions + C2_srcS1 / C2_srcNonS1), the deployed global term, and sender-split C0 / C2_rcv.
    s1: bool tensor [N] (S1 panel patches)."""
    import torch
    import eval93b as EB
    F = model.factors(gs)
    G = model.apply(F, gs, s, use_single=False)
    ap = lambda x: torch.einsum('ndcr,lncr->lndc', F['a'], torch.einsum('nmcr,lnmc->lncr', F['c'], x))
    ex = lambda x: torch.einsum('lnmc,nmdc->lndc', x, Mn)
    m = s1.view(1, -1, 1, 1)
    Bl, Bm = ap(s), ex(s)
    src = {'C0': Bl, 'C2_src': Bm, 'C2_rcv': Bl, 'C2_both': Bm, 'C2_srcS1': torch.where(m, Bm, Bl), 'C2_srcNonS1': torch.where(m, Bl, Bm)}
    rcv = {'C0': ap, 'C2_src': ap, 'C2_rcv': ex, 'C2_both': ex, 'C2_srcS1': ap, 'C2_srcNonS1': ap}
    out = {'G': G}
    for a in arms:
        out[f'S_{a}'] = rcv[a](EB.flight(model.rw, gs, src[a]))
    E1s, E1o = EB.flight(model.rw, gs, Bl * m), EB.flight(model.rw, gs, Bl * ~m)
    out['S_C0_fromS1'], out['S_C0_fromOther'] = ap(E1s), ap(E1o)
    out['S_C2_rcv_fromS1'], out['S_C2_rcv_fromOther'] = ex(E1s), ex(E1o)
    return out


def ck_tables(R, R92A, spec, otag, arms=None):
    """Tables of one model on CK2 for every variant -> results/tables/<otag>.npz keys '<variant>|<name>'."""
    import torch
    import ecro86 as EC
    import model86 as M86
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    dev = torch.device('cuda')
    model = load_spec(R, R92A, spec)
    Mq = mq_ext()
    st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    s1 = torch.as_tensor(ctx.G.PSID == 5, device=dev)
    out = {}
    with torch.no_grad():
        for v in VARIANTS:
            set_pmat(PT, ctx, variant_pmat(ctx.G, v))
            gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
            Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
            t = arm_tables(model, gs, s, Mn, s1, arms or ARMS_FROZEN)
            for k, x in t.items():
                out[f'{v}|{k}'] = x[0].cpu().numpy().astype(np.float32)
            for a in (arms or ARMS_FROZEN):
                out[f'{v}|B_{a}'] = out[f'{v}|S_{a}'] + out[f'{v}|G']
    fn = os.path.join(R, 'results', 'tables', f'{otag}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **out)
    print('TABLES', otag, len(out), flush=True)


# ------------------------------------------------------------------------------------------------ references / renders
def fref(R, v):
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    set_pmat(PT, ctx, variant_pmat(ctx.G, v))
    spp = 32 if DEV else FREF_SPP
    t0 = time.time(); out = {}
    for h, sd in zip('ab', FREF_SEEDS):
        t = A.field_src(ctx, A.Src(lt, 'true', ()), spp, sd, b=128, split=True)
        for k, x in t.items():
            out[f'{k}_{h}'] = x
    fn = os.path.join(R, 'results', 'fields', f'CKF_{v}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, PMAT=ctx.G.PMAT, spp=spp, seeds=np.array(FREF_SEEDS), **out)
    jd({'variant': v, 'spp_per_half': spp, 'seeds': FREF_SEEDS, 'seconds': time.time() - t0, 'patch_sha256': _INST['patch_sha'],
        'pmat_digest': pm_digest(ctx.G.PMAT)}, fn.replace('.npz', '.json'))
    print('FREF', v, round(time.time() - t0, 1), 's', flush=True)


def iref(R, v, r):
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    set_pmat(PT, ctx, variant_pmat(ctx.G, v))
    cam = FX['camera']; spp = 32 if DEV else IREF_SPP; seed = 86100 + int(r)
    acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], spp, seed, 'cam', b=32)
    otag = f'REF_ER_CK_{v}_{TAG}_r{r}'
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{otag}.npz'), **PT.pack(acc, cam['res'], spp))
    jd({'variant': v, 'seed': seed, 'spp': spp, 'timing': T_, 'patch_sha256': _INST['patch_sha'], 'pmat_digest': pm_digest(ctx.G.PMAT)},
       os.path.join(out, f'{otag}.json'))
    print(otag, round(T_['wall_s'], 1), flush=True)


def render(R, listfn):
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    cam = FX['camera']
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    cache = {}
    for line in open(listfn):
        if not line.strip():
            continue
        v, src, seed, spp, otag = line.split()
        if os.path.exists(os.path.join(out, f'{otag}.npz')):
            continue
        fn, key = src.split(':')
        if fn not in cache:
            cache = {fn: np.load(os.path.join(R, fn))}
        set_pmat(PT, ctx, variant_pmat(ctx.G, v))
        B = cache[fn][key].astype(np.float32)
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'ecro', B=PT.fl(B), b=32)
        np.savez_compressed(os.path.join(out, f'{otag}.npz'), hl=PT.pack(acc, cam['res'], int(spp))['hl'])
        jd({'variant': v, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{otag}.json'))
        print(otag, round(T_['wall_s'], 2), flush=True)
    print('RENDER_DONE', listfn, flush=True)


# ------------------------------------------------------------------------------------------------ masks (geometry only)
def masks(PT, S, ctx, cam):
    """Image regions ROI_T / DIFF, patch sets and bin groups (geometry only, before any radiance)."""
    import ecro86 as EC
    dr, mi, U, F = PT.dr, PT.mi, PT.U, PT.F
    G = ctx.G; res = cam['res']
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
    dr.eval(front, sid, pid, si.p, si.n, w5, ok2, sid2, pid2); dr.sync_thread()
    a = lambda x: np.asarray(x).reshape(res, res)
    front_, sid_, pid_ = a(front), a(sid).astype(np.int64), a(pid).astype(np.int64)
    sid_ = np.where(front_, sid_, -1)
    tgt_p = np.zeros(G.N, bool)
    for r in TARGET_RECTS:
        tgt_p[rect_patches(G, r)] = True
    msid, mpid = np.where(a(ok2), a(sid2), -1).astype(np.int64), np.where(a(ok2), a(pid2), -1).astype(np.int64)
    roi = a(w5) & (msid == 7) & (mpid >= 0) & tgt_p[np.clip(mpid, 0, G.N - 1)]
    # DIFF: camera-visible floor / wall / ceiling pixels; cosine-weighted view factor to box A from the centre-ray hit point
    P_ = np.stack([a(si.p.x), a(si.p.y), a(si.p.z)], -1).reshape(-1, 3)
    N_ = np.stack([a(si.n.x), a(si.n.y), a(si.n.z)], -1).reshape(-1, 3)
    cand = np.nonzero((np.isin(sid_, [0, 1, 2, 3, 4])).ravel())[0]
    nr = 256
    vf = np.zeros(res * res)
    rs = np.random.default_rng(94400)
    for c0 in range(0, len(cand), 16384):
        ci = cand[c0:c0 + 16384]
        u1 = rs.random((len(ci), nr)); u2 = rs.random((len(ci), nr))
        rr = np.sqrt(u1); ph = 2 * np.pi * u2
        loc = np.stack([rr * np.cos(ph), rr * np.sin(ph), np.sqrt(np.maximum(1 - u1, 0))], -1)
        n = N_[ci]
        t = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]]))
        t /= np.linalg.norm(t, axis=1, keepdims=True); b = np.cross(n, t)
        d = loc[..., :1] * t[:, None] + loc[..., 1:2] * b[:, None] + loc[..., 2:] * n[:, None]
        o = np.repeat(P_[ci] + n * 1e-4, nr, 0); d = d.reshape(-1, 3)
        rays = mi.Ray3f(mi.Point3f(*[F(np.ascontiguousarray(o[:, k].astype(np.float32))) for k in range(3)]),
                        mi.Vector3f(*[F(np.ascontiguousarray(d[:, k].astype(np.float32))) for k in range(3)]))
        sh = ctx.scene.ray_intersect(rays)
        hs = ctx.shape_id(sh)
        hitA = sh.is_valid() & dr.eq(hs, 7)
        dr.eval(hitA); dr.sync_thread()
        vf[ci] = np.asarray(hitA).reshape(len(ci), nr).mean(1)
    vf = vf.reshape(res, res)
    diff = vf >= 0.10
    rule = 'view factor >= 0.10'
    if diff.sum() < 2000:
        thr = np.sort(vf.ravel())[-5000]
        diff = vf >= thr; rule = f'top 5000 view factor (threshold {thr:.4f})'
    # bin groups at box-A patches: outgoing bins whose state rays hit S1 in >= half of their kappa rays
    st = EC.build_state(ctx)
    hit = st.hit.reshape(G.N, S.K).astype(np.int64)
    hs1 = (hit >= 0) & (G.PSID[np.clip(hit, 0, G.N - 1)] == 5)
    frac = hs1.reshape(G.N, S.NB, S.KAPPA).mean(2)
    pfb = (frac >= 0.5) & tgt_p[:, None]
    face = np.full(G.N, -1)
    for r in TARGET_RECTS:
        face[rect_patches(G, r)] = r
    return {'ROI_T': roi, 'DIFF': diff, 'DIFF_rule': rule, 'vf_boxA': vf.astype(np.float32), 'panel': a(w5), 'frame': front_, 'sid': sid_,
            'pid': pid_, 'mirror_patch': np.where(roi, mpid, -1), 'T': tgt_p, 'T_panel_face': face == PANEL_FACE,
            'T_other_faces': tgt_p & (face != PANEL_FACE), 'panel_facing_bins': pfb, 'S1': G.PSID == 5, 'PSID': G.PSID.copy(),
            'PMAT_default': default_pmat(G)}


# ------------------------------------------------------------------------------------------------ Stage A tests
def pre(R, R92A, R92B, R93D, RCK):
    import torch
    import ecro86 as EC
    import model86 as M86
    T = {}
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    dr, mi, CD, U, F = PT.dr, PT.mi, PT.CD, PT.U, PT.F
    dev = torch.device('cuda')
    rep = jl(os.path.join(R92B, 'results', 'repro.json'))['checkpoints']
    ck = {f'own_s{s}': sha_file(os.path.join(R92A, 'results', 'models', f'own_s{s}.pt')) for s in SEEDS}
    T['T_CKPT'] = {'pass': all(ck[k] == rep[k] for k in ck), 'loaded': ck}
    z = np.load(os.path.join(os.path.dirname(PT.__file__), 'data', 'mats86.npz'))
    T['T_STATE'] = {'pass': all(np.array_equal(lambert_state(a_), z['states'][i]) for i, a_ in ((3, 0.7), (4, 0.5), (5, 0.6)))}
    ok = True; npar = set()
    for sd in SEEDS:
        m = load_frozen(R92A, sd)
        b1 = m.mat_states.cpu().numpy()
        ok &= bool(np.array_equal(b1[:6], z['states'])) and bool(np.array_equal(b1[6:], new_rows())) and b1.shape[0] == 11
        npar.add(sum(p.numel() for p in m.parameters()))
    T['T_BUFFER'] = {'pass': ok and npar == {735664}, 'params': sorted(npar)}
    Mq = mq_ext()
    l3 = Mq[3].astype(np.float64) / 0.7
    dm = max(float(np.abs(Mq[i] / ALB[i] - l3).max()) for i in (4, 5)) / float(np.abs(l3).max())
    T['T_MQ'] = {'pass': dm <= 1e-5, 'rel_max': dm}
    n = 4096; rs = np.random.default_rng(94900); ids = [3] + NEW_IDS
    idx_np = np.repeat(np.array(ids, np.uint32), n)

    def hemi(k):
        v = rs.normal(size=(k, 3)); v[:, 2] = np.abs(v[:, 2]) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    q = len(ids)
    wi_np = np.tile(hemi(n), (q, 1)); wo_np = np.tile(hemi(n), (q, 1)); u_np = np.tile(rs.random((n, 3)), (q, 1))
    v3 = lambda x: mi.Vector3f(*[F(np.ascontiguousarray(x[:, k].astype(np.float32))) for k in range(3)])
    cl = CD.DrClosure83('ER', ctx.mats.states, U(idx_np))
    act = mi.Bool(np.ones(q * n, bool))
    f = cl.eval(v3(wi_np), v3(wo_np), act)
    wo, pdf, w, valid = cl.sample(v3(wi_np), F(u_np[:, 0].astype(np.float32)), mi.Point2f(F(u_np[:, 1].astype(np.float32)), F(u_np[:, 2].astype(np.float32))), act)
    g3 = lambda c: np.stack([np.asarray(c.x), np.asarray(c.y), np.asarray(c.z)], 1).astype(np.float64)
    fv, wv, wov = g3(f), g3(w), g3(wo)
    fr = [fv[k * n:(k + 1) * n] / ALB[i] for k, i in enumerate(ids)]; wr = [wv[k * n:(k + 1) * n] / ALB[i] for k, i in enumerate(ids)]
    ev = max(float(np.abs(fr[k] - fr[0]).max()) for k in range(1, q)) / max(float(np.abs(fr[0]).max()), 1e-30)
    ew = max(float(np.abs(wr[k] - wr[0]).max()) for k in range(1, q)) if np.asarray(valid).all() else 1.0
    dwo = max(float(np.abs(wov[k * n:(k + 1) * n] - wov[:n]).max()) for k in range(1, q))
    T['T_CLOSURE'] = {'pass': ev <= 1e-5 and ew <= 1e-5 and dwo == 0.0, 'eval_rel': ev, 'weight_abs': ew, 'dir_abs': dwo}
    # identity of the patched transport with the default map
    rel_img = lambda x, y: max(float(np.abs(x[k] - y[k]).max() / max(np.abs(x[k]).mean(), 1e-30)) for k in x)
    camE = dict(FX['camera']); camE['res'] = 128
    idt = {}
    for nm, geom, lts in (('CK2', None, lt), ('E2_T1', FX['E2']['T1'], PT.Lights(FX['light_sets'][FX['E2']['light_set']]))):
        c_ = ctx if geom is None else PT.Ctx(geom)
        set_pmat(PT, c_, default_pmat(c_.G))
        outs = []
        for tr in (_INST['orig_trace'], _INST['patched_trace']):
            PT.Ctx.trace = tr
            acc, _ = c_.render(lts, 'ER', camE, 128, 64, 94901, 'cam', b=16)
            outs.append(acc)
        PT.Ctx.trace = _INST['patched_trace']
        idt[nm] = rel_img(outs[0], outs[1]) if sorted(outs[0]) == sorted(outs[1]) else 1.0
    T['T_ID_TRACE'] = {'pass': max(idt.values()) <= 1e-5, 'rel_max_over_mean': idt}
    set_pmat(PT, ctx, default_pmat(ctx.G))
    fo = _INST['orig_field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
    fo2 = _INST['orig_field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
    fp = _INST['patched_field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
    l1 = lambda x, y: sum(float(np.abs(x[k] - y[k]).sum()) for k in x) / max(sum(float(np.abs(x[k]).sum()) for k in x), 1e-30)
    T['T_ID_FIELD'] = {'pass': l1(fo, fp) <= 1e-6, 'rel_l1_orig_vs_patched': l1(fo, fp), 'rel_l1_orig_vs_orig_floor': l1(fo, fo2)}
    # the reference consumes the per-patch material: exactly-2-event part at box-A patches / albedo = untextured
    G = ctx.G
    tg = np.concatenate([rect_patches(G, r) for r in TARGET_RECTS])
    pm38 = variant_pmat(G, 'bx38')
    set_pmat(PT, ctx, default_pmat(G))
    f0 = A.field_src(ctx, A.Src(lt, 'true', ()), 256, 94903, b=128, split=True)
    set_pmat(PT, ctx, pm38)
    f1 = A.field_src(ctx, A.Src(lt, 'true', ()), 256, 94903, b=128, split=True)
    e0 = (f0['e2_p'] + f0['e2_o'])[tg].astype(np.float64).sum((1, 2)); e1 = (f1['e2_p'] + f1['e2_o'])[tg].astype(np.float64).sum((1, 2))
    rr = e1 / np.maximum(e0, 1e-30) / (np.array([ALB[int(m)] for m in pm38[tg]]) / 0.5)
    lit = e0 > 1e-6 * e0.max()
    T['T_FIELD_E2'] = {'pass': bool(lit.sum() >= 40 and np.abs(rr[lit] - 1).max() <= 1e-4), 'patches': int(lit.sum()),
                       'max_abs_dev': float(np.abs(rr[lit] - 1).max()) if lit.any() else None}
    # model input awareness
    st = EC.build_state(ctx)
    aw = {}
    for sd in SEEDS:
        m = load_frozen(R92A, sd)
        with torch.no_grad():
            set_pmat(PT, ctx, pm38)
            gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
            i1 = m.mat_mlp(m.mat_states[gs.mat]).cpu().numpy()
            exp = m.mat_mlp(torch.as_tensor(np.stack([lambert_state(ALB[int(k)]) for k in pm38[tg]]), device=dev)).cpu().numpy()
            set_pmat(PT, ctx, default_pmat(G))
            gs0 = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
            i0 = m.mat_mlp(m.mat_states[gs0.mat]).cpu().numpy()
        oth = np.ones(G.N, bool); oth[tg] = False
        aw[sd] = bool(np.allclose(i1[tg], exp, rtol=0, atol=1e-6) and np.array_equal(i1[oth], i0[oth]) and (np.abs(i1[tg] - i0[tg]).max(1) > 0).all())
    T['T_MODEL_INPUT'] = {'pass': all(aw.values()), 'per_seed': aw}
    # frozen reproduction: WL93D E2 T1 tables (default map) and addendum CK2 tables
    import fx92
    geomE, lsE, _, _ = fx92.cfg_ls('E2_T1')
    cE = PT.Ctx(geomE); ltE = PT.Lights(FX['light_sets'][lsE])
    stE = EC.build_state(cE)
    gsE = M86.GeoState(cE.G, stE.hit, stE.dbin, dev)
    sE = torch.as_tensor(cE.source(ltE), device=dev).view(1, cE.N, S.NB, 3)
    MnE = torch.as_tensor(Mq[cE.G.PMAT].astype(np.float32), device=dev)
    I = np.load(os.path.join(R93D, 'results', 'tables', 'I_E2_T1.npz'))
    dt = {}; dl = {}
    with torch.no_grad():
        for sd in SEEDS:
            m = load_frozen(R92A, sd)
            t = arm_tables(m, gsE, sE, MnE, torch.as_tensor(cE.G.PSID == 5, device=dev), ARMS_FROZEN)
            for a_ in ('C0', 'C2_src', 'C2_rcv', 'C2_both'):
                k = f'B_own_s{sd}_{a_}'
                x = (t[f'S_{a_}'] + t['G'])[0].cpu().numpy()
                dt[k] = float(np.abs(x - I[k]).max() / max(np.abs(I[k]).max(), 1e-30))
            lhs = t['S_C2_srcS1'] + t['S_C2_srcNonS1']; rhs = t['S_C2_src'] + t['S_C0']
            dl[sd] = float((lhs - rhs).abs().max() / rhs.abs().max())
    T['T_TABLES_REPRO'] = {'pass': max(dt.values()) <= 1e-5, 'rel_max': dt}
    T['T_SRCS1'] = {'pass': max(dl.values()) <= 1e-5, 'rel_max': dl}
    ckr = {}
    tmp = os.path.join(R, 'results', 'tables', '_pre_frozen_s8601.npz')
    ck_tables(R, R92A, 'own:8601', '_pre_frozen_s8601', ['C0', 'C2_src', 'C2_rcv', 'C2_both'])
    mine = np.load(tmp)
    for v in ('none', 'bx38', 'bx75'):
        ref = np.load(os.path.join(RCK, 'results', 'tables', f'CK_{v}_{TAG}.npz'))
        for a_ in ('C0', 'C2_src', 'C2_rcv', 'C2_both'):
            k = f'B_own_s8601_{a_}'
            ckr[f'{v}:{a_}'] = float(np.abs(mine[f'{v}|B_{a_}'] - ref[k]).max() / max(np.abs(ref[k]).max(), 1e-30))
    T['T_CK_REPRO'] = {'pass': max(ckr.values()) <= 1e-5, 'rel_max': ckr}
    # addendum reference reproduction (reuse condition)
    fnr = os.path.join(RCK, 'results', 'renders', f'REF_ER_CK_bx38_{TAG}_r0.npz')
    set_pmat(PT, ctx, pm38)
    cam = FX['camera']
    if DEV:
        T['T_REF_REPRO'] = {'pass': True, 'note': 'dev: skipped (8,192 spp)'}
    else:
        acc, _ = ctx.render(lt, 'ER', cam, cam['res'], IREF_SPP, 86100, 'cam', b=32)
        o = PT.pack(acc, cam['res'], IREF_SPP); zr = np.load(fnr)
        rr_ = {k: float(np.abs(o[k].astype(np.float64) - zr[k]).max() / max(np.abs(zr[k]).mean(), 1e-30)) for k in ('e1', 'e2', 'h')}
        T['T_REF_REPRO'] = {'pass': max(rr_.values()) <= 1e-4, 'rel_max_over_mean': rr_, 'file_sha256': sha_file(fnr)}
    # masks (geometry only)
    set_pmat(PT, ctx, default_pmat(G))
    M = masks(PT, S, ctx, cam)
    np.savez_compressed(os.path.join(R, 'results', 'pre', 'masks94.npz'), **{k: v for k, v in M.items() if k != 'DIFF_rule'})
    V = {'roi_pixels': int(M['ROI_T'].sum()), 'diff_pixels': int(M['DIFF'].sum()), 'diff_rule': M['DIFF_rule'],
         'panel_facing_bins_per_target_patch_mean': float(M['panel_facing_bins'][M['T']].sum(1).mean()),
         'panel_face_patches': int(M['T_panel_face'].sum())}
    okk = all(v['pass'] for v in T.values())
    status = 'PASS' if okk else ('STOP_REPRO' if not all(T[k]['pass'] for k in ('T_CKPT', 'T_TABLES_REPRO', 'T_CK_REPRO', 'T_REF_REPRO')) else 'STOP_CONTRACT')
    jd({'status': status, 'tests': T, 'geometry': V, 'patch_sha256': _INST['patch_sha'], 'scene_sha256': sha_file(SCENE_FN),
        'protocol_sha256': sha_file(os.path.join(HERE, 'protocol', 'ecro_mat94.json'))}, os.path.join(R, 'results', 'pre', 'pre.json'))
    print('PRE', status, json.dumps({k: v['pass'] for k, v in T.items()}), json.dumps(V), flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'pre':
        pre(*a[1:6])
    elif cmd == 'fref':
        fref(a[1], a[2])
    elif cmd == 'iref':
        iref(a[1], a[2], a[3])
    elif cmd == 'tables':
        ck_tables(a[1], a[2], a[3], a[4])
    elif cmd == 'render':
        render(a[1], a[2])
    os._exit(0)
