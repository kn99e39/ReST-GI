"""WL85 Stage 0: ER material response census and fixture material selection (torch, closure82 reference implementation, frozen closures).

  python census85.py <out>
Writes census_<pop>.npz (per material x lobe x incidence: albedo share, r50, r90), census.json (summaries), fixture.json / fixture.npz
(selected materials, 69-float states incl. Lambert surfaces, lobe classes, descriptors).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ['fnto_compiled83', 'fnto_compiled82', 'fnto_compiled81']:
    sys.path.insert(0, os.path.join(HERE, '..', d))
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch)
import closure83_dr as CD  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'char85.json')))
C0 = P['stage0_census']
W83 = os.path.expanduser('~/rest_compiled83/results')
dev = torch.device('cuda')
DEV = '--dev' in sys.argv
TH = C0['incidence_deg']; AZ = C0['azimuth_deg']; NS = C0['samples'] if not DEV else 2048


def load_p(name):
    z = torch.from_numpy(np.load(os.path.join(W83, 'fit', f'fit_{name}_L_ER_s8301.npz'))['z']).to(dev)
    if DEV:
        z = z[:16]
    with torch.no_grad():
        return CL.decode('ER', 'cert', z)


def level_sets(p, wi):
    """p: params for R rows; wi [R,3]. Returns albedo a [R], Omega50, Omega90 [R] of the closure's outgoing flux density (channel mean)."""
    R = wi.shape[0]
    idx = torch.arange(R, device=dev).repeat_interleave(NS)
    pl = CL.gather(p, idx)
    WI = wi.repeat_interleave(NS, 0)
    u = torch.rand(R * NS, 3, device=dev)
    wo, ok, _ = CL.sample_closure('ER', pl, WI, u[:, 0], u[:, 1], u[:, 2])
    pdf = CL.pdf_closure('ER', pl, WI, wo)
    e = CL.eval_closure('ER', pl, WI, wo).mean(1)
    good = ok & (pdf > 0)
    w = torch.where(good, e / pdf.clamp_min(1e-30), torch.zeros_like(e)).view(R, NS).double()
    ip = torch.where(good, 1.0 / pdf.clamp_min(1e-30), torch.zeros_like(e)).view(R, NS).double()
    ev = torch.where(good, e, torch.full_like(e, -1.0)).view(R, NS)
    a = w.mean(1)
    order = torch.argsort(ev, dim=1, descending=True)
    cw = torch.cumsum(torch.gather(w, 1, order), 1) / NS
    co = torch.cumsum(torch.gather(ip, 1, order), 1) / NS
    out = []
    for q in (0.5, 0.9):
        k = (cw < (q * a)[:, None]).sum(1).clamp(max=NS - 1)
        out.append(torch.gather(co, 1, k[:, None])[:, 0])
    return a, out[0], out[1]


def census(p):
    """Returns dict of arrays [M, 9 (8 lobes + whole), nTH, nAZ]: a, r50, r90."""
    M = p['lam'].shape[0]
    res = {k: np.zeros((M, 9, len(TH), len(AZ))) for k in ['a', 'r50', 'r90']}
    for li in range(9):
        q = dict(p)
        if li < 8:
            lam = torch.zeros_like(p['lam']); lam[:, :, li] = p['lam'][:, :, li]
            q['lam'] = lam
        for ti, th in enumerate(TH):
            for ai, az in enumerate(AZ):
                t = math.radians(th); f = math.radians(az)
                wi1 = torch.tensor([math.sin(t) * math.cos(f), math.sin(t) * math.sin(f), math.cos(t)], device=dev)
                for m0 in range(0, M, 128):
                    sl = slice(m0, min(M, m0 + 128))
                    qs = {k: v[sl] for k, v in q.items()}
                    with torch.no_grad():
                        a, o50, o90 = level_sets(qs, wi1[None].repeat(sl.stop - sl.start, 1))
                    res['a'][sl, li, ti, ai] = a.cpu().numpy()
                    res['r50'][sl, li, ti, ai] = np.sqrt(o50.cpu().numpy() / math.pi)
                    res['r90'][sl, li, ti, ai] = np.sqrt(o90.cpu().numpy() / math.pi)
    return res


def classes_of(r50_45):
    c = C0['classes']
    return np.where(r50_45 < 0.15, 0, np.where(r50_45 < 0.45, 1, 2))      # 0 S, 1 M, 2 B


def summarise(res):
    a = res['a'].mean(3); r50 = np.exp(np.log(np.maximum(res['r50'], 1e-6)).mean(3))        # mean over azimuth
    M = a.shape[0]
    lobe_a = a[:, :8]; tot = np.maximum(lobe_a.sum(1), 1e-30)                               # [M, nTH]
    share = lobe_a / tot[:, None]
    cls = classes_of(r50[:, :8])                                                            # class at each incidence
    out = {'n_materials': M, 'incidence_deg': TH}
    for ti, th in enumerate(TH):
        s = {['S', 'M', 'B'][c]: (share[:, :, ti] * (cls[:, :, ti] == c)).sum(1) for c in range(3)}
        qv = lambda v: [float(np.quantile(v, x)) for x in (0.1, 0.5, 0.9)]
        out[f'theta_{th}'] = {
            'class_energy_share_q10_q50_q90': {k: qv(v) for k, v in s.items()},
            'lambert_slot_share_q10_q50_q90': qv(share[:, 0, ti]),
            'frac_materials_lambert_share_ge_10pct': float((share[:, 0, ti] >= 0.10).mean()),
            'frac_materials_sharp_share_ge_25pct': float((s['S'] >= 0.25).mean()),
            'frac_materials_broad_share_ge_75pct': float((s['B'] >= 0.75).mean()),
            'n_lobes_ge_1pct_median': float(np.median((share[:, :, ti] >= 0.01).sum(1))),
            'n_lobes_ge_10pct_median': float(np.median((share[:, :, ti] >= 0.10).sum(1))),
            'closure_r50_q10_q50_q90': qv(r50[:, 8, ti]),
            'albedo_weighted_r50_hist': {'log10_edges': np.linspace(-2.5, 0.0, 26).tolist(),
                                         'mass': (np.histogram(np.log10(np.clip(r50[:, :8, ti], 10 ** -2.5, 0.999)).ravel(), bins=np.linspace(-2.5, 0.0, 26),
                                                               weights=share[:, :, ti].ravel())[0] / M).tolist()}}
    return out


def lambert_state(alb):
    st = np.zeros(69, np.float32)
    for c in range(3):
        st[c * 8 + 0] = alb
    st[24:31] = 0.3; st[31:38] = 0.3; st[38:45] = 0.5
    st[45] = 1.0
    st[53:68] = 1.0; st[68] = 1.0
    return st


def fixture(out, cen):
    m83 = np.load(os.path.join(W83, 'data', 'materials83.npz'))
    pool = [(s, i, m83[s][i]) for s in ['test83_iid', 'test83_hard'] for i in range(len(m83[s]) if not DEV else 16)]
    pool = [x for x in pool if x[2][6:9].mean() >= 0.6 and x[2][2] * (1 - x[2][3:6].mean()) <= 0.15]      # reflective conductor under a weakly absorbing slab
    ai = np.array([x[2][0] for x in pool]); ab = np.array([x[2][9] for x in pool])
    k1 = int(np.argmin(np.maximum(ai, ab)))
    s2 = np.abs(np.log(ai / 0.12)) + np.abs(np.log(ab / 0.12)); s2[k1] = np.inf
    k2 = int(np.argmin(s2))
    sb = np.minimum(ai, ab).copy(); sb[[k1, k2]] = -np.inf
    kb = int(np.argmax(sb))
    FX = P['stage1_fixture']
    sel = {'B': pool[kb], 'S1': pool[k1], 'S2': pool[k2]}
    order = ['B', 'S1', 'S2']                      # material ids 0, 1, 2; 3 = wall Lambert, 4 = baffle Lambert
    states, cls, desc, r50 = [], [], [], []
    info = {}
    for name in order:
        s, i, d = sel[name]
        z = torch.from_numpy(np.load(os.path.join(W83, 'fit', f'fit_{s}_L_ER_s8301.npz'))['z'][i:i + 1]).to(dev)
        with torch.no_grad():
            p = CL.decode('ER', 'cert', z)
        states.append(CD.compiled_state('ER', p)[0])
        c = cen[s]
        ti = TH.index(45)
        r = np.exp(np.log(np.maximum(c['r50'][i, :8, ti], 1e-6)).mean(1))
        a = c['a'][i, :8].mean(2)                                                             # [8, nTH]
        cls.append(classes_of(r)); desc.append(d); r50.append(r)
        info[name] = {'set': s, 'index': int(i), 'descriptor': d.tolist(), 'lobe_r50_at_45deg': r.tolist(), 'lobe_class': ['SMB'[k] for k in classes_of(r)],
                      'lobe_albedo_by_incidence': a.tolist(), 'closure_r50_by_incidence': np.exp(np.log(np.maximum(c['r50'][i, 8], 1e-6)).mean(1)).tolist(),
                      'closure_albedo_by_incidence': c['a'][i, 8].mean(1).tolist()}
    for alb in (FX['shapes']['wall']['albedo'], FX['shapes']['baffle']['albedo']):
        states.append(lambert_state(alb)); cls.append(np.full(8, 2)); desc.append(np.array([0.3, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3])); r50.append(np.full(8, 0.77))
    np.savez(os.path.join(out, 'fixture.npz'), states=np.stack(states).astype(np.float32), cls=np.stack(cls).astype(np.int32), desc=np.stack(desc).astype(np.float32),
             lambert=np.array([0, 0, 0, 1, 1], np.int32), albedo=np.array([0, 0, 0, FX['shapes']['wall']['albedo'], FX['shapes']['baffle']['albedo']], np.float32))
    json.dump({'material_ids': {'B': 0, 'S1': 1, 'S2': 2, 'wall_lambert': 3, 'baffle_lambert': 4}, 'pool_size': len(pool), 'materials': info},
              open(os.path.join(out, 'fixture.json'), 'w'), indent=1)
    print(json.dumps({k: (v['set'], v['index'], [round(x, 3) for x in v['descriptor']], v['lobe_class']) for k, v in info.items()}), flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(85010)
    t0 = time.time()
    cen, summ = {}, {}
    for name in ['test83_iid', 'test83_hard', 'test83_block', 'real83']:
        cen[name] = census(load_p(name))
        np.savez_compressed(os.path.join(out, f'census_{name}.npz'), **cen[name])
        summ[name] = summarise(cen[name])
        print(name, 'done', round(time.time() - t0), 's', json.dumps(summ[name]['theta_45']['class_energy_share_q10_q50_q90']), flush=True)
    fixture(out, cen)
    summ['seconds'] = time.time() - t0
    summ['classes'] = C0['classes']
    json.dump(summ, open(os.path.join(out, 'census.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
    os._exit(0)
