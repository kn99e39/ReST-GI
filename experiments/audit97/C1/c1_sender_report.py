"""WL97 C1: print the WL94D sender recomputation tables (coded vs true weights, box A vs box A+B, matched operator norms).

  python experiments/audit97/C1/c1_sender_report.py > results/audit97/C1/sender_report.txt
"""
import json

J = json.load(open('results/audit97/C1/sender_metrics.json'))
M = J['models']
tags = list(M.keys())
print('REFERENCE OPERATORS vs M64 S1 (matched norms)')
for k, v in J['reference_operators'].items():
    if k == 'M64_S1_albedo':
        print('  M64 S1 directional albedo  coded max/mean %.3f / %.3f   true max/mean %.3f / %.3f' % (v['coded']['max'], v['coded']['mean'], v['true']['max'], v['true']['mean']))
    else:
        print('  %-14s flux-L1 %.3f  mass %.3f  flux-Fro(R,G,B) %s  op-peak %.3f  min %.2e' % (k, v['flux_l1'], v['mass'], ' / '.join('%.3f' % x for x in v['flux_fro_c']), v['op_peak_retention'], v['min']))
print()
print('LEARNED S1 OPERATOR vs M64 (same norms)')
for sp in ('val', 'ck2'):
    for t in tags:
        o = M[t][sp]['operator']
        print('  %-4s %-10s flux-L1 %.3f  mass %.3f  flux-Fro pooled %s  patch-mean %s  op-peak %.3f' % (sp, t, o['flux_l1'], o['mass'], ' / '.join('%.3f' % x for x in o['flux_fro_pooled_c']),
              ' / '.join('%.3f' % x for x in o['flux_fro_patchmean_c']), o['op_peak_retention']))
print()
print('FIELD METRICS Bl vs Bm on the S1 panel: rel L1 (coded / true / none), energy ratio (coded / true / none), peak ratio')
for sp in ('val', 'ck2'):
    for t in tags:
        f = M[t][sp]['field']
        print('  %-4s %-10s L1 %.3f / %.3f / %.3f   energy %.3f / %.3f / %.3f   peak %.3f' % (sp, t, f['coded']['rel_l1'], f['true']['rel_l1'], f['none']['rel_l1'],
              f['coded']['energy_ratio'], f['true']['energy_ratio'], f['none']['energy_ratio'], f['peak_ratio']))
print()
print('DELIVERY (learned vs M64 S1 sender through the exact receiver): rel L1 [ratio]; coded -> true')
for sp in ('val', 'ck2'):
    for t in tags:
        d = M[t][sp]['delivery']
        s = '  %-4s %-10s' % (sp, t)
        for rn in ('receivers', 'boxAB', 'boxA', 'boxB', 'panel_face'):
            a, b = d[f'coded|{rn}'], d[f'true|{rn}']
            s += '  %s %.3f[%.2f]->%.3f[%.2f]' % (rn, a['rel_l1'], a['ratio'], b['rel_l1'], b['ratio'])
        print(s)
print()
print('SENDER_IMP rule  T <= 0.8 x min(R, P) on val AND ck2 (field rel L1):  coded | true | none')
for sd in (8601, 8602):
    for wn in ('coded', 'true', 'none'):
        r = {}
        for sp in ('val', 'ck2'):
            T_, R_, P_ = (M[f'{k}_s{sd}'][sp]['field'][wn]['rel_l1'] for k in 'TRP')
            r[sp] = T_ / min(R_, P_)
        print('  s%d %-5s val T/min(R,P) %.3f  ck2 %.3f  -> SENDER_IMP %s' % (sd, wn, r['val'], r['ck2'], r['val'] <= 0.8 and r['ck2'] <= 0.8))
print()
print('CK2 SINGLE FLIGHT vs physical e2_p (exact receiver): ratio / corr / rel L1 (half-noise); coded then true')
for t in tags:
    fl = M[t]['ck2_flight_vs_e2p']
    for sn in ('M64_sender', 'learned_sender'):
        if sn == 'M64_sender' and not t.startswith('own_s8601'):
            continue
        for rn in ('receivers', 'boxA', 'boxAB', 'panel_face'):
            a, b = fl[f'coded|{sn}|{rn}'], fl[f'true|{sn}|{rn}']
            print('  %-10s %-14s %-10s coded %.3f / %+.3f / %.3f (%.3f)   true %.3f / %+.3f / %.3f (%.3f)  n=%d' % (t, sn, rn, a['ratio'], a['corr'], a['rel_l1'], a['half_noise_rel_l1'],
                  b['ratio'], b['corr'], b['rel_l1'], b['half_noise_rel_l1'], a['n']))
