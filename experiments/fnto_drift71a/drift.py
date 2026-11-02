"""Non-evidence smoke (drift_protocol.json): does sparse-source physical fine-tuning progressively destroy cross-source transfer,
and does raw J vs J/J0 matter? WL68/WL71 code unchanged (snapshot ./repo); WL68/WL71/smoke files are read-only inputs.
    python drift.py tests | stageA | train R | train N | eval | label
"""
import sys, os, json, time, math, hashlib
from pathlib import Path
HOME = Path.home(); D = Path(__file__).resolve().parent
os.environ.setdefault('R71_ROOT', str(HOME / 'rest_cover71'))  # fit71 reads fit_targets71.pt / ports.json there (read-only)
for d in ('fnto_cover71', 'fnto_repr68'):
    sys.path.insert(0, str(D / 'repo/experiments' / d))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import common71 as C71  # noqa: E402  (sets R68_ROOT -> ~/rest_repr68)
import common68 as C68  # noqa: E402
RG = C68.environment(D / 'repo')
import family68 as FM  # noqa: E402
import family71 as FW  # noqa: E402
import fit68  # noqa: E402
import fit71 as F71  # noqa: E402
import eval68 as E  # noqa: E402

R68 = C71.R68; INP = R68 / 'inputs'; OUT = D / 'out'; CK = (0, 500, 2000, 6000, 10000, 20000); STEPS = 20000
SPEC = dict(pop='S4', replicas=[0, 1, 2, 3], target='phys'); RADII = (0.005, 0.01, 0.02, 0.04)
P = {0: 2085, 1: 2441, 2: 1123, 3: 2069, 4: 2148, 5: 1025, 6: 2707, 7: 2422}


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def tsha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def geoms():
    return {g: RG.device(load(INP / 'wl59' / f'geometry_{g}.pt')) for g in ('rest', 'pt0', 'cf3')}


def model(spec):
    """'init' = P_5902 + fixed net_all_5902 A/O (fit71.model_from); 'R@k'/'N@k' checkpoints; historical candidates."""
    from runtime import Transport
    if spec == 'init':
        return F71.model_from('P_5902').eval()
    m = Transport().cuda()
    src = {'smokeA': D / 'historical/student_A.pt', 'C_5902': C71.ROOT / 'runs/C_5902.pt', 'net_all_5902': R68 / 'runs/net_all_5902.pt',
           'frozen_P_5902': INP / 'wl59/P_5902.pt'}.get(spec)
    if src is None:
        arm, k = spec.split('@'); src = OUT / f'ckpt_{arm}_{int(k):05d}.pt'
    m.load_state_dict(load(src)['model']); return m.eval()


# ------------------------------------------------------------------ tests
def tests():
    torch.backends.cuda.matmul.allow_tf32 = False; res = {}; G = geoms()
    m0 = model('init'); m1 = model('init'); res['same_init'] = tsha(m0.state_dict()) == tsha(m1.state_dict()); res['init_tsha'] = tsha(m0.state_dict())
    res['init_equals_smoke_init'] = res['init_tsha'] == 'a5d1c087e8ac259efe44c12ddb162f933681bdc91cdc59f4f4fbdf131dd0655c'
    a, o = F71.angular()
    res['AO_frozen_group'] = all(not p.requires_grad for n, p in m0.named_parameters() if n in ('in_logits', 'out_logits')) if fit68.group_params(m0, 'net') else False
    res['AO_equal_net_all'] = bool(torch.equal(m0.in_logits.cpu(), a) and torch.equal(m0.out_logits.cpu(), o))
    for name in ('smokeA', 'C_5902'):
        mm = model(name); res[f'AO_equal_in_{name}'] = bool(torch.equal(mm.in_logits.cpu(), a) and torch.equal(mm.out_logits.cpu(), o))
    pb = F71.Problem(RG, SPEC); m = model('init').train(); params = fit68.group_params(m, 'net')
    S4 = load(INP / 'fit_targets.pt')['S']
    res['target_identity'] = {g: C68.tensor_sha(S4[g]) for g in ('rest', 'pt0')}
    res['target_equals_smoke'] = res['target_identity'] == {'rest': '73b496562b7a423f6a70a686e5fa51e40dd3dfe6d17d4b066be46304bba27b4d', 'pt0': 'd9971065bb47ef377e4bbd0c521c33c937f775625c1fac6c0889d8d7dc65e4a6'}
    ft71 = load(C71.ROOT / 'fit_targets71.pt')['S']['r0123']; res['wl71_S4_targets_equal_wl68'] = all(torch.equal(ft71[g][:4], S4[g]) for g in ('rest', 'pt0')); del ft71
    # objective identity: fit71 path vs family68 objective (smoke) at init
    with torch.no_grad():
        J71 = float(pb.loss(m)); bm = FM.bin_map('cuda'); Js = []
        for g in ('rest', 'pt0'):
            pts = [P[s] for s in range(4)]; tg = FM.Target(S4[g].cuda(), (1 - pb.gs[g]['beta'][pts]).float(), pb.gs[g]['parent'], bm)
            T, A, O = FW.network_columns(m, pb.gs[g], pts); Js.append(float(FM.objective(T, A, O, tg))); del tg
        Jsm = sum(Js) / 2
    res['J0_fit71'] = J71; res['J0_family68'] = Jsm; res['objective_identity'] = abs(J71 - Jsm) <= 1e-6 * abs(J71)
    # scaling equivalence and clipping accounting
    for p_ in params: p_.grad = None
    J = pb.loss(m); J.backward(); g1 = [p_.grad.clone() for p_ in params]
    for p_ in params: p_.grad = None
    (pb.loss(m) / J71).backward(); g2 = [p_.grad.clone() for p_ in params]
    res['scaling_equivalence'] = max(float((x / J71 - y).norm() / (x / J71).norm()) for x, y in zip(g1, g2)) <= 1e-5
    pre = float(torch.sqrt(sum((x.double() ** 2).sum() for x in g2))); gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    post = float(torch.sqrt(sum((p_.grad.double() ** 2).sum() for p_ in params)))
    res['clip_accounting'] = dict(pre=pre, returned=gn, post=post, ok=abs(pre - gn) <= 1e-4 * pre and (post <= 1.0 + 1e-5 if pre > 1 else abs(post - pre) <= 1e-4 * pre))
    # prediction equivalence and passivity
    with torch.no_grad():
        pts = [P[s] for s in range(8)]
        res['columns_equivalence'] = max(float((x - y).abs().max() / y.abs().max()) for g in G for x, y in zip(FW.network_columns(m0, G[g], pts), FM.network_columns(m0, G[g], pts)))
        torch.manual_seed(71071); mp = model('init')
        for p_ in list(mp.parameters())[::3]: p_.add_(torch.randn_like(p_) * .3)
        res['passivity_perturbed_max_colsum'] = max(float(mp.column_sums(G[g], mp.coefficients(G[g])).max()) for g in G)
    res['held_out_isolation'] = sorted(str(p_) for p_ in F71.ALLOWED.values())
    ok = all(res[k] for k in ('same_init', 'init_equals_smoke_init', 'AO_frozen_group', 'AO_equal_net_all', 'AO_equal_in_smokeA', 'AO_equal_in_C_5902', 'target_equals_smoke', 'wl71_S4_targets_equal_wl68', 'objective_identity', 'scaling_equivalence')) \
        and res['clip_accounting']['ok'] and res['columns_equivalence'] <= 1e-6 and res['passivity_perturbed_max_colsum'] <= 1 + 1e-6
    res['passed'] = bool(ok); OUT.mkdir(exist_ok=True); C68.write_new(OUT / 'tests.json', json.loads(json.dumps(res, default=str)))
    print('TESTS', ok, json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))}, default=str), flush=True)
    if not ok: raise SystemExit(1)


# ------------------------------------------------------------------ training
def train(arm):
    assert arm in ('R', 'N'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4); torch.manual_seed(0); out = OUT / f'train_{arm}.pt'
    if out.exists(): raise FileExistsError(out)
    seen = []; orig = torch.load
    def spy(f, *a, **k): seen.append(str(f)); return orig(f, *a, **k)
    torch.load = spy
    try:
        pb = F71.Problem(RG, SPEC); m = model('init').train()
    finally:
        torch.load = orig
    allowed = {str(p_) for p_ in F71.ALLOWED.values()} | {str(INP / 'wl59/P_5902.pt')}
    assert all(s in allowed for s in seen), ('held-out isolation', seen)
    params = fit68.group_params(m, 'net'); a0, o0 = m.in_logits.detach().clone(), m.out_logits.detach().clone(); sha0 = tsha(m.state_dict())
    with torch.no_grad(): J0 = float(pb.loss(m))
    scale = 1.0 / J0 if arm == 'N' else 1.0
    opt = torch.optim.Adam(params, lr=3e-3); gns = torch.zeros(STEPS, device='cuda'); Js = torch.zeros(STEPS, device='cuda')
    def save(k):
        assert torch.equal(m.in_logits, a0) and torch.equal(m.out_logits, o0), 'A/O changed'
        C68.save_new(OUT / f'ckpt_{arm}_{k:05d}.pt', dict(model={n: v.detach().cpu() for n, v in m.state_dict().items()}, step=k, arm=arm, tsha=tsha(m.state_dict()),
                     optimizer=opt.state_dict() if k else None, J0=J0, scale=scale))
    save(0); torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for step in range(STEPS):
        for q in opt.param_groups: q['lr'] = F71.cosine(step, STEPS, 3e-3, 3e-5)
        opt.zero_grad(set_to_none=True); J = pb.loss(m); (J * scale).backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        gns[step] = gn.detach(); Js[step] = J.detach()
        if step % 25 == 0:
            if not math.isfinite(float(J)): raise RuntimeError('nonfinite objective')
            if step % 1000 == 0: print('FIT', arm, step, float(J), float(gn), flush=True)
        if step + 1 in CK: save(step + 1)
    torch.cuda.synchronize(); sec = time.perf_counter() - t0
    with torch.no_grad(): Jf = float(pb.loss(m))
    g = gns.cpu(); j = Js.cpu(); raw = g / scale
    win = lambda a, b: dict(clipped_fraction=float((g[a:b] > 1).float().mean()), grad_norm_median=float(g[a:b].median()), raw_J_grad_norm_median=float(raw[a:b].median()), J_end=float(j[b - 1]))
    C68.save_new(out, dict(arm=arm, J0=J0, scale=scale, J_final=Jf, J_per_step=j, grad_norm_per_step=g, init_tsha=sha0, seconds=sec, peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        windows={f'{a}-{b}': win(a, b) for a, b in zip((0,) + CK[1:-1], CK[1:])}, clipped_fraction_total=float((g > 1).float().mean()), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), opened=seen))
    print('TRAIN_DONE', arm, J0, Jf, round(sec), flush=True)


# ------------------------------------------------------------------ evaluation
def refs_load(G):
    bm = FM.bin_map('cuda'); R = {}
    def ref(g, s, f, reps=None):
        b = load(f); S = b['S'] if reps is None else b['S'][reps]; B = b['B16'] if reps is None else b['B16'][reps]; i = P[s]
        return E.Ref((g, s), S, B, (1 - G[g]['beta'][i]).float(), G[g]['parent'], bm, E.ball16(G[g], i, bm))
    for g in ('rest', 'pt0'):
        for s in range(4): R[(g, s)] = ref(g, s, INP / 'fit' / f'{g}_s{s}.pt', [4, 5, 6, 7])
        for s in range(4, 8): R[(g, s)] = ref(g, s, R68 / 'heldout/bundles' / f'{g}_s{s}.pt')
    for s in range(8): R[('cf3', s)] = ref('cf3', s, R68 / 'heldout/bundles' / f'cf3_s{s}.pt')
    Dl = {}
    for name, g, slots in (('DEF_cf3_h', 'cf3', range(4, 8)), ('DEF_pt0_h', 'pt0', range(4, 8)), ('DEF_pt0_fit', 'pt0', range(4))):
        for s in slots:
            a, b = R[(g, s)], R[('rest', s)]; K = min(a.K, b.K)
            Dl[(name, s)] = E.Ref((name, s), a.S[:K] - b.S[:K], a.B16[:K] - b.B16[:K], a.d, a.parent, bm, a.ball16 - b.ball16)
    return R, Dl


def sigSE(r):
    h = torch.zeros(16, r.S.shape[2], 64, 3, device='cuda'); Sg, Bg = r.gpu(); cg = E.cross(r, Sg, Bg, h); del Sg, Bg
    return [c['signal_over_SE'] for c in E.pooled([r], [cg], [('abs_coarse', 'abs', 'coarse')])['abs_coarse']]


GROUPS = {'FIT_rest': [('rest', s) for s in range(4)], 'FIT_pt0': [('pt0', s) for s in range(4)], 'H1_rest': [('rest', s) for s in range(4, 8)],
          'H1_pt0': [('pt0', s) for s in range(4, 8)], 'H3_cf3': [('cf3', s) for s in range(4, 8)], 'H2_cf3': [('cf3', s) for s in range(4)]}


def evaluate(name, G, R, Dl, qual, dq, lights, init_T):
    m = model(name); hs = {}; cg = {}; row = dict(groups={}, per_matrix={}, deformation={}, flux={}, T_change={})
    with torch.no_grad():
        st = {g: m.state(G[g]) for g in G}
        for (g, s), r in R.items():
            T, A, O = FW.network_columns(m, G[g], [P[s]], st[g]); h = FM.predict(T, (1 - G[g]['beta'][[P[s]]]).float(), A, O, 'native')[0]; hs[(g, s)] = h
            Sg, Bg = r.gpu(); cg[(g, s)] = E.cross(r, Sg, Bg, h); del Sg, Bg
            row['per_matrix'][f'{g}_s{s}'] = {k: [c['hi'] for c in v] for k, v in E.pooled([r], [cg[(g, s)]], [E.STD[0], E.STD[1], E.STD[2], E.STD[5]]).items()}
        for gn, keys in GROUPS.items():
            keys = [k for k in keys if qual[k]]
            if keys:
                p = E.pooled([R[k] for k in keys], [cg[k] for k in keys], E.STD)
                row['groups'][gn] = dict(pooled=p, Q=E.Q(p), matrices=[f'{a}_s{b}' for a, b in keys], corr_slope=E.corr_slope([R[k] for k in keys], [cg[k] for k in keys]))
        for dn in ('DEF_cf3_h', 'DEF_pt0_h', 'DEF_pt0_fit'):
            keys = [k for k in Dl if k[0] == dn and dq[k]]; dcg = []
            for k in keys:
                g = 'cf3' if dn == 'DEF_cf3_h' else 'pt0'; r = Dl[k]; Sg, Bg = r.gpu(); dcg.append(E.cross(r, Sg, Bg, hs[(g, k[1])] - hs[('rest', k[1])])); del Sg, Bg
            if keys:
                p = E.pooled([Dl[k] for k in keys], dcg, E.STD[:3] + E.STD[6:])
                row['deformation'][dn] = dict(pooled=p, corr_slope=E.corr_slope([Dl[k] for k in keys], dcg), matrices=[f's{k[1]}' for k in keys],
                    per_matrix={f's{k[1]}': dict(err=[c['hi'] for c in E.pooled([Dl[k]], [c_], E.STD[2:3])['abs_coarse']], corr=[c['corr'] for c in E.corr_slope([Dl[k]], [c_])]) for k, c_ in zip(keys, dcg)})
        for g in G:
            x0 = G[g]['rest_xyz']; T, A, O = FW.network_columns(m, G[g], [P[s] for s in range(8)], st[g]); d = (1 - G[g]['beta'][[P[s] for s in range(8)]]).float()
            f = torch.einsum('na,rac,njrc->nj', d, A, T)
            for k, s in enumerate(range(8)):
                r_ = (x0 - x0[P[s]]).norm(dim=1); row['flux'][f'{g}_s{s}'] = [float(f[k][r_ <= rad].sum() / f[k].sum()) for rad in RADII]
            row['T_change'][g] = (((T - init_T[g]).flatten(1).norm(dim=1)) / init_T[g].flatten(1).norm(dim=1)).tolist()
        G3 = {g: G[g] for g in ('rest', 'pt0', 'cf3')}
        row['passivity'] = E.passivity(m, G3, lights); row['extended_fields'] = E.extended(m, G3, lights)
    return row


def evaluate_all():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_grad_enabled(False); t0 = time.perf_counter(); out = OUT / 'eval.json'
    if out.exists(): raise FileExistsError(out)
    G = geoms(); lights = load(INP / 'wl59/lights.pt')['test']; R, Dl = refs_load(G)
    sig = {f'{k[0]}_s{k[1]}': sigSE(r) for k, r in R.items()}; dsig = {f'{k[0]}_s{k[1]}': sigSE(r) for k, r in Dl.items()}
    qual = {k: min(sig[f'{k[0]}_s{k[1]}']) >= 10 for k in R}; dq = {k: min(dsig[f'{k[0]}_s{k[1]}']) >= 3 for k in Dl}
    phys = {}
    for (g, s), r in R.items():
        f = r.S.double().mean(0).sum((0, 2, 3)); x0 = G[g]['rest_xyz'].cpu().double(); rr = (x0 - x0[P[s]]).norm(dim=1)
        phys[f'{g}_s{s}'] = [float(f[rr <= rad].sum() / f.sum()) for rad in RADII]
    m0 = model('init'); init_T = {g: FW.network_columns(m0, G[g], [P[s] for s in range(8)])[0] for g in G}; del m0
    cands = ['init'] + [f'{a}@{k}' for a in ('R', 'N') for k in CK[1:]] + ['smokeA', 'C_5902', 'net_all_5902', 'frozen_P_5902']
    res = {}
    for c in cands:
        res[c] = evaluate(c, G, R, Dl, qual, dq, lights, init_T); torch.cuda.empty_cache(); print('EVAL', c, round(time.perf_counter() - t0), flush=True)
    C68.write_new(out, json.loads(json.dumps(dict(results=res, precision=dict(abs=sig, deformation=dsig, qualified={f'{a}_s{b}': v for (a, b), v in qual.items()},
        deformation_qualified={f'{a}_s{b}': v for (a, b), v in dq.items()}), physical_flux=phys, radii=RADII, seconds=time.perf_counter() - t0,
        peak_GPU_allocated=torch.cuda.max_memory_allocated()), default=float)))
    print('EVAL_DONE', round(time.perf_counter() - t0), flush=True)


# ------------------------------------------------------------------ stage A + label
def worst(row, g, m):
    return max(c['hi'] for c in row['groups'][g]['pooled'][m])


def stageA():
    """Re-score smoke A and WL71 C_5902 with this evaluator (part of eval.json) and compare with their reported H3 values."""
    ev = json.loads((OUT / 'eval.json').read_text())['results']
    w71 = json.loads((D / 'historical/wl71_analysis.json').read_text())['results']['C_5902']['groups']['DEV_cf3_wl68ports']['pooled']
    rep = {'smokeA': dict(abs_coarse=.070, abs_fine16=.154, con_coarse=.268), 'C_5902': {m: max(c['hi'] for c in w71[m]) for m in ('abs_coarse', 'abs_fine16', 'con_coarse')}}
    out = {}
    for n, r in rep.items():
        now = {m: worst(ev[n], 'H3_cf3', m) for m in r}; out[n] = dict(reported=r, reproduced=now, max_abs_diff=max(abs(now[m] - r[m]) for m in r))
    out['passed'] = all(v['max_abs_diff'] <= 0.01 for v in out.values() if isinstance(v, dict)); return out


def label():
    ev = json.loads((OUT / 'eval.json').read_text())['results']; tr = {a: load(OUT / f'train_{a}.pt') for a in ('R', 'N')}; tests_ = json.loads((OUT / 'tests.json').read_text())
    A = stageA(); h1 = lambda c, m: max(worst(ev[c], g, m) for g in ('H1_rest', 'H1_pt0'))
    fit = lambda c: max(worst(ev[c], g, m) for g in ('FIT_rest', 'FIT_pt0') for m in ('abs_coarse', 'abs_fine16', 'con_coarse'))
    Jat = lambda a, k: float(tr[a]['J_final']) if k == STEPS else float(tr[a]['J_per_step'][k])  # J_per_step[k] = J of the model after k updates
    drift = {}
    for a in ('R', 'N'):
        e2, e20 = {m: h1(f'{a}@2000', m) for m in ('abs_coarse', 'con_coarse')}, {m: h1(f'{a}@20000', m) for m in ('abs_coarse', 'con_coarse')}
        drift[a] = dict(H1_2000=e2, H1_20000=e20, ratio={m: e20[m] / e2[m] for m in e2}, J_2000=Jat(a, 2000), J_20000=Jat(a, 20000), FIT_2000=fit(f'{a}@2000'), FIT_20000=fit(f'{a}@20000'))
        drift[a]['drift'] = any(e20[m] >= 1.25 * e2[m] for m in e2) and drift[a]['J_20000'] < drift[a]['J_2000'] and drift[a]['FIT_20000'] <= drift[a]['FIT_2000']
    scal = {k: {m: max(h1(f'R@{k}', m) / h1(f'N@{k}', m), h1(f'N@{k}', m) / h1(f'R@{k}', m)) for m in ('abs_coarse', 'con_coarse')} for k in (2000, 20000)}
    passiv = all(v['passed'] for c in ev.values() for v in c['passivity'].values())
    valid = A['passed'] and tests_['passed'] and passiv and all(math.isfinite(float(tr[a]['J_final'])) for a in tr)
    labels = []
    if not valid: labels = ['INCONCLUSIVE']
    else:
        if any(drift[a]['drift'] for a in drift): labels.append('TRAINING DRIFT SUPPORTED (' + ','.join(a for a in drift if drift[a]['drift']) + ')')
        else: labels.append('NO DRIFT OBSERVED')
        if any(v >= 1.25 for k in scal for v in scal[k].values()): labels.append('SCALING SENSITIVITY SUPPORTED')
    C68.write_new(OUT / 'label.json', dict(labels=labels, stageA=A, drift=drift, scaling=scal, passivity_all=passiv, tests_passed=tests_['passed']))
    print('LABEL', labels, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    {'tests': tests, 'train': lambda: train(a[1]), 'eval': evaluate_all, 'label': label}[a[0]]()
    sys.stdout.flush(); os._exit(0)
