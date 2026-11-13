"""WL93 post hoc descriptive diagnostic of the failed identifiability gate (labelled post hoc; does not change the gate, the label or any rule).

  python posthoc93.py <R> <R86> <R92A>   -> results/posthoc_ident.json

On the same 55 validation geometries x 4 lights as the gate:
  - per-geometry panel-patch rel L1 of enc_q(s16) vs q (the gate's irreducible term) and of WL92A OWN Bq vs q;
  - q Monte Carlo noise: a second independent q realisation (seed + 500,000) vs q, and enc_q vs the mean of both realisations
    (how much of the 'irreducible' term is target noise rather than encoding loss);
  - flux ratio enc_q / q on panel patches (level vs placement).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval93  # noqa: F401  (path setup)


def main():
    R, R86, R92A = sys.argv[1:4]
    import torch
    import fx92
    import diag92 as DG
    import xport87 as X
    import quad87 as Q
    import train93 as TR
    PT, E87 = fx92.PT, fx92.E87
    P = eval93.P
    _, Mq = DG.local_matrices()
    dev = torch.device('cuda')
    import glob
    ex = {n for n, _, _ in json.load(open(os.path.join(R92A, 'results', 'exclusion.json')))['excluded'].get('val', [])}
    fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', 'val', 'g*.npz'))) if os.path.basename(f) not in ex
           and os.path.exists(os.path.join(R, 'data', 'val_q', os.path.basename(f))) and os.path.exists(os.path.join(R92A, 'data', 'val_f2', os.path.basename(f)))]
    owns = {s: TR.load_own(R92A, s, dev) for s in eval93.SEEDS}
    rows = []; ctx = None
    tot = {k: 0.0 for k in ('enc', 'own8601', 'own8602', 'qq', 'enc_qm', 'd', 'dm', 'enc_flux', 'q_flux')}
    for fn in fns:
        it = TR.Item(R, R92A, 'val', fn)
        idx = int(os.path.basename(fn)[1:6])
        zz = np.load(fn); lights = json.loads(str(zz['lights'])); geom = json.loads(str(zz['geom']))
        gs = it.it.gs(); s, t, t2, q = it.tens(dev)
        G = type('G', (), {'PMAT': it.it.tab['pmat'], 'N': len(it.it.tab['pmat'])})()
        pm = it.it.tab['pmat'] == 1
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        qn = q.double().cpu().numpy(); sn = s.double().cpu().numpy()
        qa = np.stack([Q.qtable(ctx, PT.Lights(l), E87['P'], E87['Mq'], P['source_stage']['q_seed'] + 4 * idx + li + 500000) for li, l in enumerate(lights)])
        with torch.no_grad():
            Bq = {sd: torch.einsum('ndcr,lncr->lndc', f['a'], torch.einsum('nmcr,lnmc->lncr', f['c'], s)).double().cpu().numpy()
                  for sd, f in ((sd, m.factors(gs)) for sd, m in owns.items())}
        r = {'idx': idx, 'enc': 0.0, 'own8601': 0.0, 'own8602': 0.0, 'qq': 0.0, 'enc_qm': 0.0, 'd': 0.0}
        for l in range(sn.shape[0]):
            eq = X.enc_q(G, sn[l], Mq)
            qm = 0.5 * (qn[l] + qa[l])
            d = qn[l][pm].sum(); dm = qm[pm].sum()
            r['enc'] += np.abs(eq[pm] - qn[l][pm]).sum(); r['d'] += d
            r['own8601'] += np.abs(Bq[eval93.SEEDS[0]][l][pm] - qn[l][pm]).sum(); r['own8602'] += np.abs(Bq[eval93.SEEDS[1]][l][pm] - qn[l][pm]).sum()
            r['qq'] += np.abs(qa[l][pm] - qn[l][pm]).sum()
            tot['enc_qm'] += np.abs(eq[pm] - qm[pm]).sum(); tot['dm'] += dm
            tot['enc_flux'] += eq[pm].sum(); tot['q_flux'] += d
        for k in ('enc', 'own8601', 'own8602', 'qq', 'd'):
            tot[k] += r[k]
        rows.append({'idx': idx, 'enc': r['enc'] / r['d'], 'own8601': r['own8601'] / r['d'], 'own8602': r['own8602'] / r['d'], 'q_vs_q2': r['qq'] / r['d'], 'panel_q_sum': r['d']})
        print('geom', idx, json.dumps({k: round(v, 3) for k, v in rows[-1].items() if k != 'idx'}), flush=True)
    out = {'label': 'POST HOC DESCRIPTIVE (gate, label and rules unchanged)',
           'pooled': {'irreducible_enc_vs_q': tot['enc'] / tot['d'], 'own_s8601_vs_q': tot['own8601'] / tot['d'], 'own_s8602_vs_q': tot['own8602'] / tot['d'],
                      'q_vs_independent_q': tot['qq'] / tot['d'], 'implied_single_q_noise': 0.5 * tot['qq'] / tot['d'],
                      'enc_vs_mean_of_two_q': tot['enc_qm'] / tot['dm'], 'enc_over_q_panel_flux': tot['enc_flux'] / tot['q_flux']},
           'per_geometry': rows}
    json.dump(out, open(os.path.join(R, 'results', 'posthoc_ident.json'), 'w'), indent=1)
    print('POSTHOC', json.dumps(out['pooled']), flush=True)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
