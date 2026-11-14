"""WL94D (protocol ecro_s94d): sharp sender oracle-supervised smoke. WL94 / WL94A / WL94B / WL94C modules imported UNCHANGED.

The learned once-scattered sender of the ECRO single term is Bl = a (c^T s) per patch (model86.apply: Bq). The exact counterpart in the
same discretisation is Bm = s M64[material] (mat94.arm_tables: the C2_srcS1 arm uses Bm at the S1 panel patches in place of Bl). Both map
the same incident field s [L, N, NB_in, 3] to outgoing bin radiance [L, N, NB_out, 3] -> directly comparable; no new head.

Continuations from WL94C D_s<seed> (same item stream, D query weights, OWN objective), 1,000 steps:
  T = + lam_T x teacher(Bl vs Bm on S1 panel patches);  R = + lam_R x teacher(Bl vs Bm on all non-S1 patches);  P = plain continuation.

  python s94.py audit <R> <R86> <R92A>          Stage A contract / oracle audit, teacher weights (results/audit94d/)
  python s94.py train <R> <R86> <R92A> <seed> <T|R|P>
  python s94.py sender <R> <R86> <R92A> <spec> <otag>  standalone sender evaluation of one model (train / val items, CK2, operator matrix)
  python s94.py an <R> <seed>  /  final <R> <out>
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_q94c'))
import q94 as Q  # noqa: E402

B, MT, C94, AN, R94 = Q.B, Q.MT, Q.C94, Q.AN, Q.R94
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_s94d.json')))
DEV = MT.DEV
SEEDS = [8601, 8602]
STEPS = 60 if DEV else 1000
WARM, LR0, LR1 = 100, 1e-4, 1e-5
NLAM = 4 if DEV else 16
VAR, TEX = AN.VARIANTS, AN.TEX
PANEL = list(range(1088, 1130))


def jd(o, fn):
    AN.jd(o, fn)


def dev_():
    import torch
    return torch.device('cuda')


def mq_t():
    import torch
    return torch.as_tensor(MT.mq_ext().astype(np.float32), device=dev_())          # [11, NB_in, NB_out, 3]


def sender_pair(F, s, mat, Mq, mask):
    """Bl (learned) and Bm (exact) on the patches in mask: [L, n, NB, 3] each."""
    import torch
    c, a = F['c'][mask], F['a'][mask]
    u = torch.einsum('nmcr,lnmc->lncr', c, s[:, mask])
    Bl = torch.einsum('ndcr,lncr->lndc', a, u)
    Bm = torch.einsum('lnmc,nmdc->lndc', s[:, mask], Mq[mat[mask]])
    return Bl, Bm


def teacher_loss(Bl, Bm):
    """per-entry normalised squared error, den = per-(lighting, patch, channel) bin mean of the target + 0.05 x its mean (OWN form)."""
    den = Bm.mean(2, keepdim=True) + 0.05 * Bm.mean(dim=(1, 2, 3), keepdim=True) + 1e-12
    return (((Bl - Bm) / den) ** 2).mean()


def load_D(R, seed, train=True):
    import torch
    import model86 as M86
    m = MT.extend_buffer(M86.ECRO().to(dev_()))
    m.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'D_s{int(seed)}.pt'), map_location=dev_()))
    if not train:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    return m


def items_D(R, R86, seed):
    import torch
    W = np.load(os.path.join(R, 'results', 'manifest', f'W_s{int(seed)}.npz'))
    items = R94.mix_items(R, R86, 'train', 'C')
    for it in items:
        it.qw = torch.as_tensor(W[f'D|{it.name}'], device=dev_()).view(*W[f'D|{it.name}'].shape, 1, 1)
    return items


def masks_of(it):
    import torch
    s1 = torch.zeros(it.pm.shape[0], dtype=torch.bool, device=dev_()); s1[PANEL] = True
    return {'T': s1, 'R': ~s1}


# ------------------------------------------------------------------------------------------------ Stage A
def audit(R, R86, R92A):
    import torch
    import scene86 as S
    import eval93b as EB
    dev = dev_(); Mq = mq_t()
    T, G = {}, {}
    files = C94.sel_files(R, 'train')
    z0 = S.Geom(json.loads(str(np.load(os.path.join(R86, 'data', 'train', files[0]['file']))['geom'])))
    pan = MT.rect_patches(z0, 5)
    T['T_INDEX'] = {'pass': bool(np.array_equal(pan, np.array(PANEL)) and (z0.PSID[pan] == 5).all() and (z0.PMAT[pan] == 1).all()
                                 and int((z0.PSID == 5).sum()) == len(PANEL))}
    # T_SEM: my Bm path at S1 reproduces the stored C2_srcS1 single flight of WL94C D on CK2 (exact positive-control identity)
    import ecro86 as EC
    import model86 as M86
    PT, S_, A, FX, ctx, lt = MT.ctx_ck2(R)
    st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
    sem = {}
    for sd in SEEDS:
        m = load_D(R, sd, train=False)
        with torch.no_grad():
            F = m.factors(gs)
            mask = torch.zeros(ctx.N, dtype=torch.bool, device=dev); mask[PANEL] = True
            Bl, Bm = sender_pair(F, s, gs.mat, Mq, torch.ones(ctx.N, dtype=torch.bool, device=dev))
            src = torch.where(mask.view(1, -1, 1, 1), Bm, Bl)
            ap = lambda x: torch.einsum('ndcr,lncr->lndc', F['a'], torch.einsum('nmcr,lnmc->lncr', F['c'], x))
            Smine = ap(EB.flight(m.rw, gs, src))[0].cpu().numpy()
            Bl_apply = m.apply(F, gs, s, use_global=False)                             # deployed single term uses Bl internally
            Sdep = Bl_apply[0].cpu().numpy(); Sl = ap(EB.flight(m.rw, gs, Bl))[0].cpu().numpy()
        tab = np.load(os.path.join(R, 'results', 'tables', f'CD_D_s{sd}.npz'))
        sem[sd] = {'C2_srcS1_vs_stored': float(np.abs(Smine - tab['none|S_C2_srcS1']).max() / np.abs(tab['none|S_C2_srcS1']).max()),
                   'Bl_path_vs_model_apply': float(np.abs(Sl - Sdep).max() / np.abs(Sdep).max())}
    T['T_SEM'] = {'pass': all(v['C2_srcS1_vs_stored'] <= 1e-5 and v['Bl_path_vs_model_apply'] <= 1e-5 for v in sem.values()), **{str(k): v for k, v in sem.items()}}
    # G_PHYS: exact-sender (Bm at S1) single flight through the exact receiver vs the e2_p path label on the 96 training items (exitance per
    # (lighting, receiver patch); receivers = all patches except the panel; positive physical control of the teacher quantity)
    items = R94.mix_items(R, R86, 'train', 'C')
    W = torch.as_tensor(AN.W, device=dev, dtype=torch.float32)
    num = den = 0.0; xs, ys = [], []
    bins_tr = np.zeros((S.NB, 3), bool)
    for it in items:
        bins_tr |= (it.it.s[:, PANEL].sum((0, 1)) > 0)
    # G_PHYS (revised before the freeze; the first draft's corr >= 0.9 rule is reported, not gated): the exact S1 sender + exact receiver single-
    # flight exitance per (lighting, receiver patch) vs the WL94B e2_p path label on the 96 training items. The 256-spp e2_p label of a sharp
    # reflector path is itself noisy, so agreement is required at least at the label's own reliability (corr with the independent 64-spp
    # realization); energy within [0.8, 1.25]. CK2 (2 x 4,096 spp) is reported as the precise diagnostic of the 64-bin flight floor.
    m0 = load_D(R, SEEDS[0], train=False)
    X, Yt, Yi = [], [], []
    for it in items:
        gs_ = it.gs(); s_ = torch.as_tensor(it.it.s, device=dev)
        mk = torch.zeros(gs_.N, dtype=torch.bool, device=dev); mk[PANEL] = True
        with torch.no_grad():
            Mn = Mq[gs_.mat]
            Bm = torch.einsum('lnmc,nmdc->lndc', s_, Mn) * mk.view(1, -1, 1, 1)
            ex_m = (torch.einsum('lnmc,nmdc->lndc', EB.flight(m0.rw, gs_, Bm), Mn) * W.view(1, 1, -1, 1)).sum((2, 3)).cpu().numpy()
        a = np.load(os.path.join(R, 'results', 'audit', 'tgt', it.name)); b = np.load(os.path.join(R, 'results', 'audit', 'ind', it.name))
        sel = np.ones(gs_.N, bool); sel[PANEL] = False
        X.append(ex_m[:, sel].ravel()); Yt.append(a['ex'][0][:, sel].ravel()); Yi.append(b['ex'][0][:, sel].ravel())
    X, Yt, Yi = map(np.concatenate, (X, Yt, Yi))
    cr = lambda u, v: float(np.corrcoef(u, v)[0, 1])
    with torch.no_grad():
        Mn = Mq[gs.mat]
        mk = torch.zeros(ctx.N, dtype=torch.bool, device=dev); mk[PANEL] = True
        Sx = torch.einsum('lnmc,nmdc->lndc', EB.flight(m0.rw, gs, torch.einsum('lnmc,nmdc->lndc', s, Mn) * mk.view(1, -1, 1, 1)), Mn)[0].cpu().numpy()
    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    exm = (Sx * AN.W[None, :, None]).sum((1, 2)); ea = (zf['e2_p_a'] * AN.W[None, :, None]).sum((1, 2)); eb = (zf['e2_p_b'] * AN.W[None, :, None]).sum((1, 2))
    er = 0.5 * (ea + eb); Mk = AN.masks(R)
    rc = np.ones(ctx.N, bool); rc[PANEL] = False
    ck = {nm: {'ratio': float(exm[mm].sum() / er[mm].sum()), 'corr': cr(exm[mm], er[mm]), 'corr_halves': cr(ea[mm], eb[mm]),
               'rel_l1': float(np.abs(exm[mm] - er[mm]).sum() / er[mm].sum()), 'half_noise_rel_l1': float(0.5 * np.abs(ea[mm] - eb[mm]).sum() / er[mm].sum())}
          for nm, mm in (('receivers', rc), ('boxA', Mk['T']), ('panel_face', Mk['T_panel_face']))}
    G['G_PHYS'] = {'pass': bool(0.8 <= X.sum() / Yt.sum() <= 1.25 and cr(X, Yt) >= cr(Yt, Yi)),
                   'pooled_ratio': float(X.sum() / Yt.sum()), 'corr_exact_vs_label': cr(X, Yt), 'label_reliability_corr_256_vs_64': cr(Yt, Yi),
                   'corr_exact_vs_independent_label': cr(X, Yi), 'rel_l1': float(np.abs(X - Yt).sum() / Yt.sum()),
                   'first_draft_rule': 'ratio in [0.8, 1.25] and corr >= 0.9 -> failed (corr 0.641) because the label reliability is 0.525',
                   'ck2_flight_floor_diagnostic': ck}
    # incident-bin coverage at the panel: training vs CK2 (held-out directions)
    ck_bins = s[0, PANEL].sum(0).cpu().numpy() > 0                                          # [NB, 3]
    val = C94.load_items(R, R86, 'val', 'orig')
    val_bins = np.zeros((S.NB, 3), bool)
    for it in val:
        val_bins |= (it.it.s[:, PANEL].sum((0, 1)) > 0)
    cov = {'train_bins': int(bins_tr[:, 1].sum()), 'val_bins': int(val_bins[:, 1].sum()), 'val_bins_not_in_train': int((val_bins & ~bins_tr)[:, 1].sum()),
           'ck2_bins': int(ck_bins[:, 1].sum()), 'ck2_bins_not_in_train': int((ck_bins & ~bins_tr)[:, 1].sum()),
           'ck2_irradiance_share_in_train_bins': float(s[0, PANEL][:, torch.as_tensor(bins_tr[:, 1], device=dev)].sum().item() / max(s[0, PANEL].sum().item(), 1e-30))}
    np.save(os.path.join(R, 'results', 'audit94d', 'train_bins.npy'), bins_tr)
    # energy accounting of the teacher (M64 S1): directional albedo per incident bin (flux-transfer rows)
    zb = (np.arange(S.NB) // S.NPHI + 0.5) / S.NZ
    Mq1 = MT.mq_ext()[1].astype(np.float64)
    alb = (Mq1 * zb[None, :, None]).sum(1) * (2 * math.pi / S.NB)
    T['T_ENERGY'] = {'pass': bool(alb.max() <= 1.0 + 1e-3), 'albedo_max': float(alb.max()), 'albedo_mean': float(alb.mean())}
    # teacher weights: equal initial gradient norm with the WL94C D objective (first NLAM training items, D_s<seed> init)
    lam = {}
    for sd in SEEDS:
        m = load_D(R, sd, train=True)
        its = items_D(R, R86, sd)[:NLAM]
        g = {'own': [], 'T': [], 'R': []}; l0 = {'T': [], 'R': []}
        for it in its:
            gs_ = it.gs(); s_ = torch.as_tensor(it.it.s, device=dev)
            for key in ('own', 'T', 'R'):
                m.zero_grad(set_to_none=True)
                F = m.factors(gs_)
                if key == 'own':
                    t = it.tgt(dev, 'f'); t2 = it.tgt(dev, 'f2')
                    Q.CUR['w'] = it.qw
                    loss = Q.wobjective(m.apply(F, gs_, s_, use_global=False), m.apply(F, gs_, s_, use_single=False), t, t2)[0]
                else:
                    Bl, Bm = sender_pair(F, s_, gs_.mat, Mq, masks_of(it)[key])
                    loss = teacher_loss(Bl, Bm); l0[key].append(float(loss))
                loss.backward()
                g[key].append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None))))
        lam[str(sd)] = {'T': float(np.mean(g['own']) / np.mean(g['T'])), 'R': float(np.mean(g['own']) / np.mean(g['R'])),
                        'grad_own': float(np.mean(g['own'])), 'grad_T': float(np.mean(g['T'])), 'grad_R': float(np.mean(g['R'])),
                        'teacher_loss_T0': float(np.mean(l0['T'])), 'teacher_loss_R0': float(np.mean(l0['R']))}
        del m
    G['G_LAM'] = {'pass': all(math.isfinite(v['T']) and math.isfinite(v['R']) and v['T'] > 0 and v['R'] > 0 for v in lam.values()), 'lam': lam}
    st_ = 'PASS' if all(t['pass'] for t in T.values()) and all(g_['pass'] for g_ in G.values()) else ('SOURCE_CONTRACT_UNAVAILABLE' if not T['T_SEM']['pass'] or not G['G_PHYS']['pass'] else 'STOP_TEST')
    out = {'status': st_, 'tests': T, 'gates': G, 'coverage': cov, 'protocol_sha256': MT.sha_file(os.path.join(HERE, 'protocol', 'ecro_s94d.json'))}
    jd(out, os.path.join(R, 'results', 'audit94d', 'audit.json'))
    jd(lam, os.path.join(R, 'results', 'audit94d', 'lam.json'))
    print('AUDIT94D', st_, json.dumps({k: v['pass'] for k, v in {**T, **G}.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ training
def train(R, R86, R92A, seed, cand):
    import torch
    seed = int(seed); tag = f'{cand}_s{seed}'; dev = dev_()
    torch.manual_seed(seed); np.random.seed(seed)
    lam = MT.jl(os.path.join(R, 'results', 'audit94d', 'lam.json'))[str(seed)].get(cand, 0.0) if cand in 'TR' else 0.0
    Mq = mq_t()
    items = items_D(R, R86, seed); val = C94.load_items(R, R86, 'val', 'orig')
    model = load_D(R, seed, train=True)
    init_hash = C94.sd_hash(model)
    opt_ = torch.optim.Adam(model.parameters(), lr=LR0)

    def lr(i):
        if i < WARM:
            return LR0 * (i + 1) / WARM
        return LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * (i - WARM) / max(STEPS - WARM, 1)))
    rng = np.random.default_rng(seed + 9400)
    order = []; log = []; run = []; nonfinite = 0
    v0 = C94.val_whole(model, val, dev); t0 = time.time()
    for i in range(STEPS):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt_.param_groups:
            g['lr'] = lr(i)
        gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev); t = it.tgt(dev, 'f'); t2 = it.tgt(dev, 'f2')
        F = model.factors(gs)
        Bs = model.apply(F, gs, s, use_global=False); Bg = model.apply(F, gs, s, use_single=False)
        Q.CUR['w'] = it.qw
        lo, ls, lc2, lc3 = Q.wobjective(Bs, Bg, t, t2)
        mk = masks_of(it)
        BlT, BmT = sender_pair(F, s, gs.mat, Mq, mk['T'])
        lT = teacher_loss(BlT, BmT)
        if cand == 'R':
            BlR, BmR = sender_pair(F, s, gs.mat, Mq, mk['R']); lR = teacher_loss(BlR, BmR)
            loss = lo + lam * lR
        elif cand == 'T':
            lR = None; loss = lo + lam * lT
        else:
            lR = None; loss = lo
        opt_.zero_grad(set_to_none=True); loss.backward()
        g_ = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not (math.isfinite(float(loss)) and math.isfinite(g_)):
            nonfinite += 1
        opt_.step()
        run.append((float(loss), float(ls), float(lc2), float(lc3), float(lT), float(lR) if lR is not None else float('nan'), g_))
        if (i + 1) % 50 == 0 or i + 1 == STEPS:
            m_ = np.nanmean(np.array(run), 0)
            r = {'step': i + 1, 'objective': m_[0], 'L_sum': m_[1], 'L_single': m_[2], 'L_global': m_[3], 'teacher_S1': m_[4], 'teacher_nonS1': m_[5],
                 'grad_norm_mean': m_[6], 's': round(time.time() - t0, 1)}
            if i + 1 == STEPS:
                r['val_orig_rel_l1'] = C94.val_whole(model, val, dev)
            log.append({k: float(v) if not isinstance(v, int) else v for k, v in r.items()}); run = []
    out = os.path.join(R, 'results', 'models')
    res = {'tag': tag, 'seed': seed, 'candidate': cand, 'steps': STEPS, 'lam': lam, 'params': sum(p.numel() for p in model.parameters()),
           'init_sha256': init_hash, 'final_sha256': C94.sd_hash(model), 'train_s': time.time() - t0, 'nonfinite_steps': nonfinite,
           'val_orig_rel_l1_D': v0, 'val_orig_rel_l1_final': log[-1]['val_orig_rel_l1'], 'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    jd(res, os.path.join(out, f'{tag}.json'))
    print('FINAL', tag, round(res['train_s'], 1), 's nonfinite', nonfinite, flush=True)


# ------------------------------------------------------------------------------------------------ standalone sender evaluation
def sender(R, R86, R92A, spec, otag):
    """S1 sender fields Bl vs Bm (train items, val items, CK2) and the S1 local operator matrix vs M64 (trained vs never-excited incident bins)."""
    import torch
    import scene86 as S
    import eval93b as EB
    import ecro86 as EC
    import model86 as M86
    dev = dev_(); Mq = mq_t()
    model = MT.load_spec(R, R92A, spec)
    W = torch.as_tensor(AN.W, device=dev, dtype=torch.float32).view(1, 1, -1, 1)
    zb = torch.as_tensor((np.arange(S.NB) // S.NPHI + 0.5) / S.NZ, device=dev, dtype=torch.float32)
    btr = torch.as_tensor(np.load(os.path.join(R, 'results', 'audit94d', 'train_bins.npy')), device=dev)       # [NB_in, 3]
    M1 = Mq[1]
    out = {}

    def field_metrics(acc):
        return {'rel_l1_flux': acc['l1'] / acc['m'], 'energy_ratio': acc['l'] / acc['m'], 'peak_ratio': acc['pl'] / acc['pm'],
                'delivery_rel_l1': acc['dl1'] / acc['dm'], 'delivery_ratio': acc['dl'] / acc['dm'], 'delivery_box_rel_l1': acc['bl1'] / acc['bm']}

    def accumulate(acc, F, gs, s):
        mk = torch.zeros(gs.N, dtype=torch.bool, device=dev); mk[PANEL] = True
        Bl, Bm = sender_pair(F, s, gs.mat, Mq, mk)
        acc['l1'] += float(((Bl - Bm).abs() * W).sum()); acc['m'] += float((Bm * W).sum()); acc['l'] += float((Bl * W).sum())
        pk = Bm.argmax(2, keepdim=True)
        acc['pl'] += float(Bl.gather(2, pk).sum()); acc['pm'] += float(Bm.gather(2, pk).sum())
        full_l = torch.zeros(s.shape[0], gs.N, S.NB, 3, device=dev); full_m = torch.zeros_like(full_l)
        full_l[:, PANEL] = Bl; full_m[:, PANEL] = Bm
        Mn = Mq[gs.mat]
        Dl = torch.einsum('lnmc,nmdc->lndc', EB.flight(model.rw, gs, full_l), Mn)            # learned S1 sender, exact receiver
        Dm = torch.einsum('lnmc,nmdc->lndc', EB.flight(model.rw, gs, full_m), Mn)            # exact S1 sender, exact receiver
        rc = torch.ones(gs.N, dtype=torch.bool, device=dev); rc[PANEL] = False
        bx = torch.zeros(gs.N, dtype=torch.bool, device=dev); bx[1130:1238] = True
        el, em = (Dl * W).sum((2, 3)), (Dm * W).sum((2, 3))
        acc['dl1'] += float((el - em)[:, rc].abs().sum()); acc['dm'] += float(em[:, rc].sum()); acc['dl'] += float(el[:, rc].sum())
        acc['bl1'] += float((el - em)[:, bx].abs().sum()); acc['bm'] += float(em[:, bx].sum())

    def operator(F):
        """S1 local operator A[n, m, d, c] = sum_r a[n, d, c, r] c[n, m, c, r] vs M64 S1, flux-transfer weighted (zb_in zb_out), split by incident bins."""
        A = torch.einsum('nmcr,ndcr->nmdc', F['c'][PANEL], F['a'][PANEL])
        X = (A - M1.unsqueeze(0)).abs() * zb.view(1, -1, 1, 1) * zb.view(1, 1, -1, 1)
        T_ = M1.unsqueeze(0).expand_as(A) * zb.view(1, -1, 1, 1) * zb.view(1, 1, -1, 1)
        o = {}
        for nm, sel in (('train_bins', btr), ('heldout_bins', ~btr)):
            w = sel.float().view(1, -1, 1, 3)
            o[nm] = [float((X * w).sum()), float((T_ * w).sum())]
        return o
    with torch.no_grad():
        for split, assign in (('train', 'C'), ('val', 'orig')):
            its = R94.mix_items(R, R86, 'train', 'C') if split == 'train' else C94.load_items(R, R86, 'val', 'orig')
            acc = {k: 0.0 for k in ('l1', 'm', 'l', 'pl', 'pm', 'dl1', 'dm', 'dl', 'bl1', 'bm')}
            op = {'train_bins': [0.0, 0.0], 'heldout_bins': [0.0, 0.0]}
            for it in its:
                gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
                F = model.factors(gs)
                accumulate(acc, F, gs, s)
                for k, v in operator(F).items():
                    op[k][0] += v[0]; op[k][1] += v[1]
            out[split] = {**field_metrics(acc), 'operator_rel_l1': {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in op.items()}}
        PT, S_, A, FX, ctx, lt = MT.ctx_ck2(R)
        st = EC.build_state(ctx)
        s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
        MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
        gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
        F = model.factors(gs)
        acc = {k: 0.0 for k in ('l1', 'm', 'l', 'pl', 'pm', 'dl1', 'dm', 'dl', 'bl1', 'bm')}
        accumulate(acc, F, gs, s)
        op = operator(F)
        out['ck2'] = {**field_metrics(acc), 'operator_rel_l1': {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in op.items()}}
    jd({'spec': spec, **out}, os.path.join(R, 'results', 'sender', f'{otag}.json'))
    print('SENDER', otag, flush=True)


# ------------------------------------------------------------------------------------------------ analysis
MODELS = ['own', 'D', 'T', 'R', 'P']


def an(R, seed):
    seed = int(seed)
    F = AN.fref(R); M = AN.masks(R)
    out = {'seed': seed, 'models': {}}
    for c in MODELS:
        tg = f'{c}_s{seed}'
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{seed}.npz' if c == 'own' else f'CD_{tg}.npz')))
        m = {'table': AN.strip(AN.table_metrics(tab, F, M, ['C0', 'C2_srcS1'])), 'image': AN.image_metrics(R, M, tg, ['C0', 'C2_srcS1']),
             'pop': AN.pop_metrics(R, f'P_{tg}'), 'face': B.face_ext(tab, F, M),
             'sender': MT.jl(os.path.join(R, 'results', 'sender', f'SN_{tg}.json'))}
        if c in ('T', 'R', 'P'):
            j = MT.jl(os.path.join(R, 'results', 'models', f'{tg}.json'))
            m['fit'] = {k: j[k] for k in ('val_orig_rel_l1_D', 'val_orig_rel_l1_final', 'nonfinite_steps', 'init_sha256', 'final_sha256', 'train_s', 'lam', 'steps')}
            m['fit']['teacher_S1_first50'] = j['log'][0]['teacher_S1']; m['fit']['teacher_S1_last50'] = j['log'][-1]['teacher_S1']
        out['models'][c] = m
    ms = out['models']
    own_v = MT.jl(os.path.join(R, 'results', 'models', f'D_s{seed}.json'))['val_orig_rel_l1_init']
    eo = ms['own']['image']['C0']['none|ROI_T|E_H']
    crit = {}
    for c in ('D', 'T', 'R', 'P'):
        m = ms[c]
        vfin = m['fit']['val_orig_rel_l1_final'] if 'fit' in m else MT.jl(os.path.join(R, 'results', 'models', f'D_s{seed}.json'))['val_orig_rel_l1_final']
        nf = m['fit']['nonfinite_steps'] if 'fit' in m else 0
        mr_t = AN.mr_ok({'C0': m['table']['C0']}); pl = m['pop']['B|lambert']
        mr_p = 0.85 <= pl['amplitude'] <= 1.15 and (pl['corr'] or 0) >= 0.90
        e = m['image']['C0']['none|ROI_T|E_H']
        tol = max(2 * max(e['noise_bound'], eo['noise_bound']), 0.005)
        crit[c] = {'F2': bool(vfin <= 1.10 * own_v and nf == 0), 'V_orig': vfin, 'F2_limit': 1.10 * own_v, 'MR': bool(mr_t and mr_p),
                   'E_H': e['E_H'], 'signed': e['signed'], 'noise_bound': e['noise_bound'], 'SHARP_OK_vs_own': bool(e['E_H'] <= eo['E_H'] + tol),
                   'E_H_exact_sender': m['image']['C2_srcS1']['none|ROI_T|E_H']['E_H'],
                   'sender_val': m['sender']['val']['rel_l1_flux'], 'sender_ck2': m['sender']['ck2']['rel_l1_flux']}
    T_, R_, P_ = crit['T'], crit['R'], crit['P']
    sender_imp = bool(T_['sender_val'] <= 0.8 * min(R_['sender_val'], P_['sender_val']) and T_['sender_ck2'] <= 0.8 * min(R_['sender_ck2'], P_['sender_ck2']))

    def better(x, y):
        a, b = ms[x]['image']['C0']['none|ROI_T|E_H'], ms[y]['image']['C0']['none|ROI_T|E_H']
        tol = max(2 * max(a['noise_bound'], b['noise_bound']), 0.005)
        return {'diff': b['E_H'] - a['E_H'], 'tol': tol, 'better': bool(b['E_H'] - a['E_H'] > tol), 'worse': bool(a['E_H'] - b['E_H'] > tol)}
    pairs = {'T_vs_P': better('T', 'P'), 'T_vs_R': better('T', 'R'), 'T_vs_D': better('T', 'D'), 'P_vs_D': better('P', 'D'), 'R_vs_P': better('R', 'P')}
    render_imp = pairs['T_vs_P']['better'] and pairs['T_vs_R']['better']
    out.update({'criteria': crit, 'pairs': pairs, 'SENDER_IMP': sender_imp, 'RENDER_IMP': bool(render_imp), 'KEEP_T': bool(T_['F2'] and T_['MR']),
                'init_ok': ms['T']['fit']['init_sha256'] == ms['R']['fit']['init_sha256'] == ms['P']['fit']['init_sha256']})
    jd(out, os.path.join(R, 'results', 'an', f's_s{seed}.json'))
    print('SAN', seed, flush=True)


def outcome(rs):
    if len(rs) < 2 or not all(r['init_ok'] for r in rs) or any(r['models'][c]['fit']['nonfinite_steps'] for r in rs for c in 'TRP'):
        return 'INCONCLUSIVE'
    si = [r['SENDER_IMP'] for r in rs]; ri = [r['RENDER_IMP'] for r in rs]; kp = [r['KEEP_T'] for r in rs]
    if all(si) and all(ri) and all(kp):
        return 'SENDER_AND_RENDER_IMPROVED'
    if all(si) and not any(ri):
        return 'SENDER_FIT_ONLY'
    if not any(si):
        return 'NO_RELIABLE_IMPROVEMENT'
    return 'INCONCLUSIVE'


def final(R, outd):
    rs = [MT.jl(os.path.join(R, 'results', 'an', f's_s{s}.json')) for s in SEEDS if os.path.exists(os.path.join(R, 'results', 'an', f's_s{s}.json'))]
    oc = outcome(rs)
    res = {'outcome': oc, 'per_seed': {r['seed']: {k: r[k] for k in ('SENDER_IMP', 'RENDER_IMP', 'KEEP_T')} for r in rs}}
    jd(res, os.path.join(R, 'results', 'an', 'final_s.json'))
    files = figures(R, outd, rs)
    jd({'worklog': '94D', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R}, os.path.join(outd, 'manifest.json'))
    print('FINAL', oc, flush=True)


def stop_final(R, status):
    jd({'outcome': 'SOURCE_CONTRACT_UNAVAILABLE' if status == 'SOURCE_CONTRACT_UNAVAILABLE' else 'INCONCLUSIVE', 'stop': status},
       os.path.join(R, 'results', 'an', 'final_s.json'))
    print('FINAL STOP', status, flush=True)


def figures(R, outd, rs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viz94 as VZ
    os.makedirs(outd, exist_ok=True)
    files = []
    nm = MODELS
    fig, ax = plt.subplots(2, 4, figsize=(26, 10))
    for si, r in enumerate(rs):
        ms = r['models']; s = r['seed']; off = si * 0.4; x = np.arange(len(nm))
        for k, (sp, key, ttl) in enumerate((('train', 'rel_l1_flux', 'S1 sender field Bl vs Bm, TRAINING items (flux-weighted rel L1)'),
                                            ('val', 'rel_l1_flux', 'S1 sender field, VALIDATION items (held-out geometries / lights)'),
                                            ('ck2', 'rel_l1_flux', 'S1 sender field, CK2 (evaluation scene)'),
                                            ('ck2', 'delivery_box_rel_l1', 'CK2 S1 single flight to box (exact receiver): learned vs exact sender'))):
            ax[0, k].bar(x + off, [ms[c]['sender'][sp][key] for c in nm], 0.38, label=f's{s}'); ax[0, k].set_title(ttl, fontsize=8)
        ax[1, 0].bar(x + off, [ms[c]['sender']['val']['operator_rel_l1']['train_bins'] for c in nm], 0.38, label=f's{s} val geometries')
        ax[1, 0].plot(x + off, [ms[c]['sender']['ck2']['operator_rel_l1']['train_bins'] for c in nm], 'k_', ms=14, label=f's{s} CK2 geometry')
        ax[1, 0].set_title('S1 local operator matrix vs M64, all 64 incident bins (flux-transfer weighted rel L1)', fontsize=8)
        e = [ms[c]['image']['C0']['none|ROI_T|E_H'] for c in nm]
        ax[1, 1].bar(x + off, [q['E_H'] for q in e], 0.38, yerr=[2 * q['noise_bound'] for q in e], label=f's{s}')
        ax[1, 1].set_title('untextured ROI_T E_H (C0), bars 2 x noise', fontsize=8)
        ax[1, 2].bar(x + off, [ms[c]['image']['C2_srcS1']['none|ROI_T|E_H']['E_H'] for c in nm], 0.38, label=f's{s}')
        ax[1, 2].set_title('untextured ROI_T E_H with the exact panel sender (C2_srcS1)', fontsize=8)
        ax[1, 3].bar(x + off, [ms[c]['face']['G|panel_face']['signed'] for c in nm], 0.38, label=f's{s}')
        ax[1, 3].set_title('global term signed bias, panel face', fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks(np.arange(len(nm)) + 0.2); a_.set_xticklabels(nm); a_.legend(fontsize=6); a_.axhline(0, color='k', lw=0.6)
    fig.suptitle('WL94D: own (frozen) / D (WL94C) / T (+ S1 sender oracle) / R (+ non-S1 sender oracle) / P (plain continuation), 1,000-step continuations from D; '
                 + '; '.join(f"s{r['seed']} V_orig " + ', '.join(f"{c} {r['criteria'][c]['V_orig']:.4f}" for c in 'DTRP') + f" (F2 limit {r['criteria']['T']['F2_limit']:.4f})" for r in rs), fontsize=9)
    fn = os.path.join(outd, 'figS1_summary.png'); fig.savefig(fn, dpi=75, bbox_inches='tight'); plt.close(fig); files.append(fn)
    fig, ax = plt.subplots(1, 4, figsize=(24, 4.5))
    for r in rs:
        s = r['seed']
        for c, ls in (('T', '-'), ('R', '--'), ('P', ':')):
            j = MT.jl(os.path.join(R, 'results', 'models', f'{c}_s{s}.json'))
            st = [q['step'] for q in j['log']]
            for k, q in enumerate(('L_sum', 'L_single', 'L_global', 'teacher_S1')):
                ax[k].plot(st, [x[q] for x in j['log']], ls, label=f'{c} s{s}'); ax[k].set_yscale('log'); ax[k].set_title(f'{q} (unweighted; per 50 steps)', fontsize=8)
    for a_ in ax:
        a_.legend(fontsize=6)
    fn = os.path.join(outd, 'figS2_training.png'); fig.savefig(fn, dpi=75, bbox_inches='tight'); plt.close(fig); files.append(fn)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
    for si, r in enumerate(rs):
        ms = r['models']
        for c, mk in zip(nm, ('k*', 'o', 'D', 's', '^')):
            ax[0].plot(VAR, [ms[c]['image']['C0'][f'{v}|ROI_T|E_H']['E_H'] for v in VAR], mk, ls='-' if si == 0 else '--', label=f"{c} s{r['seed']}")
            ax[1].plot(TEX, [ms[c]['table']['C0'][f'{v}|B|T']['amplitude'] for v in TEX], mk, ls='-' if si == 0 else '--', label=f"{c} s{r['seed']}")
    ax[0].set_title('ROI_T E_H per variant (C0)', fontsize=8); ax[1].set_title('CK2 box-A B amplitude (C0); band = MR', fontsize=8); ax[1].axhspan(0.85, 1.15, color='g', alpha=0.1)
    for a_ in ax:
        a_.legend(fontsize=6, ncol=2)
    fn = os.path.join(outd, 'figS3_textured.png'); fig.savefig(fn, dpi=75, bbox_inches='tight'); plt.close(fig); files.append(fn)
    M = AN.masks(R)
    refs = {v: AN.load_ref(R, v) for v in VAR}
    href = {v: refs[v]['h'].mean(0) for v in VAR}
    l0 = AN.lum(refs['none']['tot'].mean(0)); key = 0.5 / float(np.median(l0[l0 > 0]))
    cy, cx = VZ.crop_of(M)
    cols = []
    for r in rs:
        s = r['seed']
        cols += [(f'own_s{s}', 'C0'), (f'D_s{s}', 'C0'), (f'T_s{s}', 'C0'), (f'R_s{s}', 'C0'), (f'P_s{s}', 'C0'), (f'D_s{s}', 'C2_srcS1')]
    hl = {}
    for mt, a in cols:
        for v in VAR:
            x = AN.load_hl(R, v, mt, a)
            if x is not None:
                hl[(mt, a, v)] = x.mean(0)
    E_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(x - href[v]))[cy, cx].ravel() for (mt, a, v), x in hl.items()]), 0.999))
    D_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(href[v] - href['none']))[cy, cx].ravel() for v in TEX]), 0.999))
    for v in VAR:
        cc_ = [(mt, a) for mt, a in cols if (mt, a, v) in hl]
        fig, ax = plt.subplots(3, len(cc_) + 1, figsize=(2.8 * (len(cc_) + 1), 9))
        ax[0, 0].imshow(VZ.tm(href[v][cy, cx], key)); ax[0, 0].set_title(f'REF h [{v}]', fontsize=8); ax[1, 0].axis('off')
        if v != 'none':
            VZ.sq(ax[2, 0], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF texture response')
        else:
            ax[2, 0].axis('off')
        for j, (mt, a) in enumerate(cc_, 1):
            x = hl[(mt, a, v)]
            ax[0, j].imshow(VZ.tm(x[cy, cx], key)); ax[0, j].set_title(f'{mt} {a}' + (' (exact S1 sender, control)' if a != 'C0' else ''), fontsize=7)
            VZ.sq(ax[1, j], AN.lum(x - href[v])[cy, cx], E_MAX, plt, 'signed error hl - h')
            if v != 'none' and (mt, a, 'none') in hl:
                VZ.sq(ax[2, j], AN.lum(x - hl[(mt, a, 'none')])[cy, cx], D_MAX, plt, 'texture response')
            else:
                ax[2, j].axis('off')
        for a_ in ax[0]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f'WL94D CK2 {v} ({VZ.VLAB[v]}): higher-order-only sharp-panel close-up; REF / own / D / T / R / P (C0) / D with exact S1 sender, both seeds - '
                     f'shared absolute scales (error max {E_MAX:.3g}, response max {D_MAX:.3g}); one exposure', fontsize=9)
        fn = os.path.join(outd, f'figS4_images_{v}.png'); fig.savefig(fn, dpi=65, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'audit':
        audit(a[1], a[2], a[3])
    elif cmd == 'train':
        train(*a[1:6])
    elif cmd == 'sender':
        sender(*a[1:6])
    elif cmd == 'an':
        an(a[1], a[2])
    elif cmd == 'final':
        final(a[1], a[2])
    elif cmd == 'stop':
        stop_final(a[1], a[2])
    os._exit(0)
