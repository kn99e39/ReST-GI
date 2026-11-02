"""WL71B read-only attribution of the WL71A drift (attrib_protocol.json): exact W/UV/V/mass decomposition of the WL59 spatial T,
per-source component statistics, and a 2^4 coefficient counterfactual (W, U, V, mass) between R@2000 and R@20000 (N as control).
No training. WL71A checkpoints and WL68 references are read-only.   python attrib.py tests | stageA | stageC | label
"""
import sys, os, json, time, math, hashlib, itertools
from pathlib import Path
HOME = Path.home(); D = Path(__file__).resolve().parent
A71 = Path(os.environ.get('A71_OUT', HOME / 'rest_drift_smoke/out'))
os.environ.setdefault('R71_ROOT', str(HOME / 'rest_cover71'))
for d in ('fnto_cover71', 'fnto_repr68'):
    sys.path.insert(0, str(D / 'repo/experiments' / d))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import common71 as C71  # noqa: E402
import common68 as C68  # noqa: E402
RG = C68.environment(D / 'repo')
import family68 as FM  # noqa: E402
import family71 as FW  # noqa: E402
import fit71 as F71  # noqa: E402
import eval68 as E  # noqa: E402
from family68 import M, TR, C  # noqa: E402

R68 = C71.R68; INP = R68 / 'inputs'; OUT = D / 'out'; CK = (0, 500, 2000, 6000, 10000, 20000); RADII = (0.01, 0.02)
P = {0: 2085, 1: 2441, 2: 1123, 3: 2069, 4: 2148, 5: 1025, 6: 2707, 7: 2422}; PORTS = [P[s] for s in range(8)]; GEO = ('rest', 'pt0', 'cf3')
FACT = ('W', 'U', 'V', 'M')


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def geoms():
    return {g: RG.device(load(INP / 'wl59' / f'geometry_{g}.pt')) for g in GEO}


def net(arm, k):
    from runtime import Transport
    m = Transport().cuda(); m.load_state_dict(load(A71 / f'ckpt_{arm}_{k:05d}.pt')['model']); return m.eval()


# ------------------------------------------------------------------ exact factorisation
def factors(m, g, st=None):
    """Raw factors of family71.network_columns for PORTS: W_raw (n,128,M,C) on edges dst (n,128), U (P,TR,M,C), V_raw (n,TR,M,C), mass (n,M,C)."""
    st = m.state(g) if st is None else st; z = st['z']; Pn = g['P']; nh = m.node_head(z)
    U = torch.softmax(nh[:, :TR * M * C].view(Pn, TR, M, C), dim=0)
    V = F.softplus(nh[:, TR * M * C:2 * TR * M * C].view(Pn, TR, M, C)); mass = torch.sigmoid(nh[:, 2 * TR * M * C:].view(Pn, M, C))
    e = FW.edges(g, PORTS); src = g['src'][e]; dst = g['dst'][e]; n = e.shape[0]
    h = st['s'][src] + st['d'][dst] + m.pair_layer(g['pair'][e])
    W = F.softplus(m.edge(h)).view(n, -1, M, C) * g['prior'][e][:, :, None, :]
    pi = torch.as_tensor(PORTS, device=z.device)
    return dict(W=W, dst=dst, U=U, V=V[pi], M=mass[pi])


def assemble(f, Pn):
    """Original normalisation algebra: scale = mass / (sum W + sum V + 1e-12); T_W = scatter(W*scale), T_UV = U (V*scale)."""
    scale = f['M'] / (f['W'].sum(1) + f['V'].sum(1) + 1e-12); n = f['W'].shape[0]
    TW = torch.zeros(n, Pn, M, C, device=f['W'].device, dtype=f['W'].dtype).scatter_add(1, f['dst'][:, :, None, None].expand(-1, -1, M, C), f['W'] * scale[:, None])
    TUV = torch.einsum('jkrc,nkrc->njrc', f['U'], f['V'] * scale[:, None]); return TW, TUV, scale


def AO(m):
    return torch.softmax(m.in_logits, 0), torch.softmax(m.out_logits, 1)


def flux(T, A, d):
    """Collided flux per receiver, RGB and bins summed (WL71 Stage-1 definition): sum_c sum_a d[a] sum_r A[r,a,c] T[j,r,c]."""
    return torch.einsum('na,rac,njrc->nj', d, A, T)


# ------------------------------------------------------------------ tests
def tests():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_grad_enabled(False); res = {}; G = geoms()
    rec = {l.split()[1].split('/')[-1]: l.split()[0] for l in (D / 'checkpoints71a.sha256').read_text().split('\n') if l.strip()}
    res['checkpoint_identity'] = {f: C68.sha(A71 / f) == h for f, h in rec.items() if f.startswith('ckpt_')}
    a, o = F71.angular(); ao = []
    for arm in 'RN':
        for k in CK:
            m = net(arm, k); ao.append(bool(torch.equal(m.in_logits.cpu(), a) and torch.equal(m.out_logits.cpu(), o)))
    res['AO_identical'] = all(ao)
    clo = []; cf = []; pas = []
    for arm in 'RN':
        for k in CK:
            m = net(arm, k)
            for gn, g in G.items():
                f = factors(m, g); TW, TUV, _ = assemble(f, g['P']); T = FW.network_columns(m, g, PORTS)[0]
                clo.append(float((TW + TUV - T).abs().max() / T.abs().max()))
    res['decomposition_closure_max'] = max(clo)
    for arm in 'RN':
        e, l = net(arm, 2000), net(arm, 20000)
        for gn, g in G.items():
            fe, fl = factors(e, g), factors(l, g)
            for combo in itertools.product('EL', repeat=4):
                f = dict(dst=fe['dst'], **{x: (fe if c == 'E' else fl)[x] for x, c in zip(FACT, combo)})
                TW, TUV, _ = assemble(f, g['P']); T = TW + TUV; A, O = AO(e); d = (1 - g['beta'][PORTS]).float()
                if combo in (('E',) * 4, ('L',) * 4):
                    ref = FW.network_columns(e if combo[0] == 'E' else l, g, PORTS)[0]; cf.append(float((T - ref).abs().max() / ref.abs().max()))
                colsum = T.sum(1); out = d[:, :, None] + 0 * d[:, :, None]
                resp = (1 - d)[:, :, None] + d[:, :, None] * torch.einsum('rac,nrc->nac', A, colsum)  # beta + (1-beta) sum_r A mass
                pas.append(dict(min=float(T.min()), colsum_minus_mass=float((T.double().sum(1) - f['M'].double()).abs().max() / f['M'].double().abs().max()), max_colsum=float(colsum.max()), max_response=float(resp.max())))
    res['counterfactual_closure_max'] = max(cf); res['passivity'] = dict(min_T=min(p['min'] for p in pas), colsum_err=max(p['colsum_minus_mass'] for p in pas),
        max_colsum=max(p['max_colsum'] for p in pas), max_response=max(p['max_response'] for p in pas))
    res['ports_correspond'] = P == {s['slot']: s['port'] for s in json.loads((INP / 'wl66_ports.json').read_text())['ports']} | {s['slot']: s['port'] for s in json.loads((R68 / 'heldout/ports.json').read_text())['ports']}
    ok = all(res['checkpoint_identity'].values()) and len(res['checkpoint_identity']) == 12 and res['AO_identical'] and res['decomposition_closure_max'] <= 1e-6 \
        and res['counterfactual_closure_max'] <= 1e-6 and res['passivity']['min_T'] >= 0 and res['passivity']['colsum_err'] <= 1e-5 and res['passivity']['max_colsum'] < 1 \
        and res['passivity']['max_response'] <= 1 + 1e-6 and res['ports_correspond']
    res['passed'] = bool(ok); OUT.mkdir(exist_ok=True); C68.write_new(OUT / 'tests.json', res); print('TESTS', ok, json.dumps({k: v for k, v in res.items() if k != 'checkpoint_identity'}), flush=True)
    if not ok: raise SystemExit(1)


# ------------------------------------------------------------------ stage A
def stageA():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_grad_enabled(False); G = geoms(); t0 = time.perf_counter(); out = {}
    fitp = PORTS[:4]
    for gn, g in G.items():
        x0 = g['rest_xyz']; dist = (x0[None] - x0[PORTS][:, None]).norm(dim=2)  # (8,P)
        nearfit = torch.stack([(x0 - x0[p]).norm(dim=1) <= .01 for p in fitp]).any(0)  # receivers within 1 cm of a fit port
        base = {}
        for arm in 'RN':
            for k in CK:
                m = net(arm, k); f = factors(m, g); TW, TUV, scale = assemble(f, g['P']); A, O = AO(m); d = (1 - g['beta'][PORTS]).float()
                fw, fu = flux(TW, A, d), flux(TUV, A, d); ft = fw + fu
                row = {}
                for s in range(8):
                    own = (x0 - x0[P[s]]).norm(dim=1) <= .01; nf = nearfit & ~own
                    row[f's{s}'] = dict(W_raw_sum=float(f['W'][s].sum(0).mean()), V_raw_sum=float(f['V'][s].sum(0).mean()), mass=float(f['M'][s].mean()), scale=float(scale[s].mean()),
                        flux_W=float(fw[s].sum()), flux_UV=float(fu[s].sum()), W_share=float(fw[s].sum() / ft[s].sum()),
                        within={f'{int(r*100)}cm': dict(total=float(ft[s][dist[s] <= r].sum() / ft[s].sum()), W=float(fw[s][dist[s] <= r].sum() / max(float(fw[s].sum()), 1e-30)),
                                                       UV=float(fu[s][dist[s] <= r].sum() / max(float(fu[s].sum()), 1e-30))) for r in RADII},
                        UV_near_fit_ports=float(fu[s][nf].sum() / max(float(fu[s].sum()), 1e-30)), total_near_fit_ports=float(ft[s][nf].sum() / ft[s].sum()))
                if k == 0: base[arm] = (TW, TUV, f)
                if k == 2000: mid = (TW, TUV, f)
                b = base[arm]
                for s in range(8):
                    rel = lambda x, y: float((x - y).norm() / y.norm().clamp(min=1e-30))
                    row[f's{s}']['change_from_init'] = dict(T_W=rel(TW[s], b[0][s]), T_UV=rel(TUV[s], b[1][s]), V=rel(f['V'][s], b[2]['V'][s]), mass=rel(f['M'][s], b[2]['M'][s]), W_raw=rel(f['W'][s], b[2]['W'][s]))
                    if k == 20000:
                        row[f's{s}']['change_2000_to_20000'] = dict(T_W=rel(TW[s], mid[0][s]), T_UV=rel(TUV[s], mid[1][s]), V=rel(f['V'][s], mid[2]['V'][s]), mass=rel(f['M'][s], mid[2]['M'][s]), W_raw=rel(f['W'][s], mid[2]['W'][s]))
                Ucol = f['U'].permute(1, 2, 3, 0).reshape(-1, g['P'])  # (TR*M*C, P) distributions over destinations
                row['U'] = dict(change_from_init=float((f['U'] - b[2]['U']).norm() / b[2]['U'].norm()), mass_near_fit_ports=float(Ucol[:, nearfit].sum(1).mean()),
                                max_column_entropy_bits=float((-(Ucol.clamp(min=1e-30) * Ucol.clamp(min=1e-30).log2()).sum(1)).mean()))
                out[f'{gn}|{arm}@{k}'] = row
        print('STAGEA', gn, round(time.perf_counter() - t0), flush=True)
    C68.write_new(OUT / 'stageA.json', out)


# ------------------------------------------------------------------ stage C (evaluation identical to WL71A drift.py)
GROUPS = {'FIT_rest': [('rest', s) for s in range(4)], 'FIT_pt0': [('pt0', s) for s in range(4)], 'H1_rest': [('rest', s) for s in range(4, 8)],
          'H1_pt0': [('pt0', s) for s in range(4, 8)], 'H3_cf3': [('cf3', s) for s in range(4, 8)], 'H2_cf3': [('cf3', s) for s in range(4)]}
DEFQ = {'DEF_cf3_h': ('cf3', (4, 5)), 'DEF_pt0_h': ('pt0', (4, 7)), 'DEF_pt0_fit': ('pt0', (0, 1, 3))}


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
    for name, (g, slots) in DEFQ.items():
        for s in slots:
            a, b = R[(g, s)], R[('rest', s)]; K = min(a.K, b.K)
            Dl[(name, s)] = E.Ref((name, s), a.S[:K] - b.S[:K], a.B16[:K] - b.B16[:K], a.d, a.parent, bm, a.ball16 - b.ball16)
    return R, Dl


def score(Tg, A, O, G, R, Dl):
    """Tg[g] = T (8,P,M,C) for PORTS at geometry g -> WL71A-style group/per-matrix/deformation/flux record."""
    row = dict(groups={}, per_matrix={}, deformation={}, flux={}); hs = {}; cg = {}
    for (g, s), r in R.items():
        h = FM.predict(Tg[g][[s]], (1 - G[g]['beta'][[P[s]]]).float(), A, O, 'native')[0]; hs[(g, s)] = h
        Sg, Bg = r.gpu(); cg[(g, s)] = E.cross(r, Sg, Bg, h); del Sg, Bg
        row['per_matrix'][f'{g}_s{s}'] = {k: [c['hi'] for c in v] for k, v in E.pooled([r], [cg[(g, s)]], [E.STD[0], E.STD[1], E.STD[2], E.STD[5]]).items()}
    for gn, keys in GROUPS.items():
        p = E.pooled([R[k] for k in keys], [cg[k] for k in keys], E.STD); row['groups'][gn] = dict(pooled=p, Q=E.Q(p))
    for dn, (g, slots) in DEFQ.items():
        dcg = []
        for s in slots:
            r = Dl[(dn, s)]; Sg, Bg = r.gpu(); dcg.append(E.cross(r, Sg, Bg, hs[(g, s)] - hs[('rest', s)])); del Sg, Bg
        row['deformation'][dn] = dict(pooled=E.pooled([Dl[(dn, s)] for s in slots], dcg, E.STD[:3] + E.STD[6:]), corr_slope=E.corr_slope([Dl[(dn, s)] for s in slots], dcg))
    for g in G:
        x0 = G[g]['rest_xyz']; d = (1 - G[g]['beta'][PORTS]).float(); f = flux(Tg[g], A, d)
        for s in range(8):
            r_ = (x0 - x0[P[s]]).norm(dim=1); row['flux'][f'{g}_s{s}'] = float(f[s][r_ <= .01].sum() / f[s].sum())
    return row


def stageC():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_grad_enabled(False); t0 = time.perf_counter(); G = geoms(); R, Dl = refs_load(G); out = {}
    for arm in 'RN':
        e, l = net(arm, 2000), net(arm, 20000); A, O = AO(e); fe = {g: factors(e, G[g]) for g in G}; fl = {g: factors(l, G[g]) for g in G}
        for combo in itertools.product('EL', repeat=4):
            Tg = {}
            for g in G:
                f = dict(dst=fe[g]['dst'], **{x: (fe[g] if c == 'E' else fl[g])[x] for x, c in zip(FACT, combo)}); TW, TUV, _ = assemble(f, G[g]['P']); Tg[g] = TW + TUV
            out[f'{arm}|{"".join(combo)}'] = score(Tg, A, O, G, R, Dl); torch.cuda.empty_cache()
            print('STAGEC', arm, ''.join(combo), round(time.perf_counter() - t0), flush=True)
    C68.write_new(OUT / 'stageC.json', out)


# ------------------------------------------------------------------ label
def label():
    C_ = json.loads((OUT / 'stageC.json').read_text()); ev = json.loads((D / 'eval71a.json').read_text())['results']; tests_ = json.loads((OUT / 'tests.json').read_text())
    w = lambda r, g, m: max(c['hi'] for c in r['groups'][g]['pooled'][m])
    metrics = {'m1_H1_coarse': lambda r: max(w(r, g, 'abs_coarse') for g in ('H1_rest', 'H1_pt0')), 'm2_H1_contrast': lambda r: max(w(r, g, 'con_coarse') for g in ('H1_rest', 'H1_pt0')),
               'H3_coarse': lambda r: w(r, 'H3_cf3', 'abs_coarse'), 's6_coarse_rest': lambda r: max(r['per_matrix']['rest_s6']['abs_coarse']),
               's5_coarse_rest': lambda r: max(r['per_matrix']['rest_s5']['abs_coarse']), 's6_flux_rest': lambda r: r['flux']['rest_s6'], 'FIT_worst': lambda r: max(w(r, g, mm) for g in ('FIT_rest', 'FIT_pt0') for mm in ('abs_coarse', 'abs_fine16', 'con_coarse'))}
    # evaluation identity: EEEE / LLLL vs WL71A eval of R@2000 / R@20000
    ident = {}
    for arm in 'RN':
        for tag, ck in (('EEEE', 2000), ('LLLL', 20000)):
            a, b = C_[f'{arm}|{tag}'], ev[f'{arm}@{ck}']
            ident[f'{arm}|{tag}'] = max(abs(w(a, g, m) - w(b, g, m)) for g in ('H1_rest', 'H1_pt0', 'H3_cf3', 'FIT_rest') for m in ('abs_coarse', 'abs_fine16', 'con_coarse'))
    groups = {'W': 'W', 'GLOBAL': 'UV', 'U': 'U', 'V': 'V', 'M': 'M'}
    def combo(base, sub, letters):
        return ''.join((sub if x in letters else base) for x in FACT)
    res = {}
    for arm in 'RN':
        res[arm] = {}
        for mn, fn in metrics.items():
            val = {c: fn(C_[f'{arm}|{c}']) for c in (''.join(t) for t in itertools.product('EL', repeat=4))}
            Dm = val['LLLL'] - val['EEEE']; sh = {}
            for gname, letters in groups.items():
                deg = val[combo('E', 'L', letters)] - val['EEEE']; rec = val['LLLL'] - val[combo('L', 'E', letters)]
                sh[gname] = dict(degrade=deg, recover=rec, share=(deg + rec) / (2 * Dm) if Dm else float('nan'))
            # factorial effects (+1 = L): main effects and interactions
            eff = {}
            for r in range(1, 5):
                for sub in itertools.combinations(range(4), r):
                    eff[''.join(FACT[i] for i in sub)] = sum(v * np.prod([1 if c[i] == 'L' else -1 for i in sub]) for c, v in val.items()) / 8
            res[arm][mn] = dict(values=val, drift=Dm, shares=sh, effects=eff)
    def lab(r):
        Dm = r['drift']; sh = r['shares']
        for g, name in (('W', 'LOCAL-MECHANISM DOMINANT'), ('GLOBAL', 'GLOBAL-FACTOR DOMINANT'), ('M', 'NORMALIZATION DOMINANT')):
            others = [o for o in ('W', 'GLOBAL', 'M') if o != g]
            if sh[g]['share'] >= .5 and sh[g]['degrade'] >= .5 * Dm and sh[g]['recover'] >= .5 * Dm and all(sh[o]['share'] < .25 for o in others): return name
        return 'INTERACTING / MULTIPLE MECHANISMS'
    labels = {arm: {mn: lab(res[arm][mn]) for mn in ('m1_H1_coarse', 'm2_H1_contrast')} for arm in 'RN'}
    valid = tests_['passed'] and all(v <= 1e-5 for v in ident.values())
    final = 'INCONCLUSIVE' if not valid else (labels['R']['m1_H1_coarse'] if labels['R']['m1_H1_coarse'] == labels['R']['m2_H1_contrast'] else 'INTERACTING / MULTIPLE MECHANISMS')
    C68.write_new(OUT / 'label.json', json.loads(json.dumps(dict(label=final, per_metric=labels, evaluation_identity=ident, attribution=res, tests_passed=tests_['passed']), default=float)))
    print('LABEL', final, labels, ident, flush=True)


if __name__ == '__main__':
    {'tests': tests, 'stageA': stageA, 'stageC': stageC, 'label': label}[sys.argv[1]](); sys.stdout.flush(); os._exit(0)
