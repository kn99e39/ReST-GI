"""WL97 C1: compare my coded-weight recomputation with the stored WL94 Stage B JSON (identity check of the reimplementation), then list
every coded -> true weight change for the headline table metrics of WL94..WL94D. numpy-free; run locally from the repo root.

  python experiments/audit97/C1/c1_compare_tables.py > results/audit97/C1/tables_compare.txt
"""
import json

R = json.load(open('results/audit97/C1/tables_recompute.json'))['jobs']


def cmp_stored(stored, mine, tag):
    mx = 0.0; n = 0; bad = []
    for arm, st in stored.items():
        if arm not in mine:
            continue
        if arm == 'sender_split':
            for nm, d in st.items():
                for k, v in d.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            m = mine[arm][nm][k].get(kk)
                            if vv is not None and m is not None:
                                dd = abs(vv - m); mx = max(mx, dd); n += 1
                                if dd > 1e-6:
                                    bad.append((arm, nm, k, kk, vv, m))
                    elif k in mine[arm][nm] and v is not None:
                        dd = abs(v - mine[arm][nm][k]); mx = max(mx, dd); n += 1
                        if dd > 1e-6:
                            bad.append((arm, nm, k, v, mine[arm][nm][k]))
            continue
        for key, v in st.items():
            if key.endswith('binerr') or key not in mine[arm]:
                continue
            for kk, vv in v.items():
                m = mine[arm][key].get(kk)
                if isinstance(vv, (int, float)) and m is not None:
                    dd = abs(vv - m) / (max(1.0, abs(vv)) if kk == 'ref_snr_median' else 1.0)
                    mx = max(mx, dd); n += 1
                    if dd > 1e-6:
                        bad.append((arm, key, kk, vv, m))
    print(f'IDENTITY {tag}: {n} values compared, max abs diff {mx:.3e}, mismatches > 1e-6: {len(bad)}')
    for b in bad[:10]:
        print('   ', b)


B = json.load(open('results/ecro_mat94/stageB.json'))
for sd in ('s8601', 's8602'):
    cmp_stored(B['table'][sd], R[f'WL94|own_{sd}']['coded'], f'WL94 stageB {sd}')
C = json.load(open('results/ecro_mat94/stageC_s8601.json'))
for c in 'AB':
    cmp_stored(C['models'][c]['table'], R[f'WL94|{c}_s8601']['coded'], f'WL94 stageC {c}')
for sd in (8601, 8602):
    r = json.load(open(f'results/ecro_mat94r/r_s{sd}.json'))
    for c, lab in (('C', f'WL94A|C_s{sd}'), ('A', 'WL94|A_s8601' if sd == 8601 else 'WL94A|A_s8602')):
        if c in r['models']:
            cmp_stored(r['models'][c]['table'], R[lab]['coded'], f'WL94A r_s{sd} {c}')
    q = json.load(open(f'results/ecro_q94c/q_s{sd}.json'))
    for c in ('D', 'E'):
        if c in q.get('models', {}):
            cmp_stored(q['models'][c]['table'], R[f'WL94C|CD_{c}_s{sd}']['coded'], f'WL94C q_s{sd} {c}')
    s = json.load(open(f'results/ecro_s94d/s_s{sd}.json'))
    for c in ('T', 'R', 'P'):
        if c in s.get('models', {}):
            cmp_stored(s['models'][c]['table'], R[f'WL94D|CD_{c}_s{sd}']['coded'], f'WL94D s_s{sd} {c}')

print()
print('CODED -> TRUE weight changes (C0 arm unless stated). amplitude / corr of B over box-A patches (MR gate inputs)')
print(f'{"job":22s} {"variant":6s} {"amp_coded":>9s} {"amp_true":>9s} {"d":>7s} {"corr_c":>7s} {"corr_t":>7s} {"MRc":>4s} {"MRt":>4s}')
TEX = ['U07', 'U03', 'bx75', 'bx38', 'bx38i']
big = []
for job, o in R.items():
    if 'coded' not in o:
        continue
    for v in TEX:
        a, b = o['coded']['C0'][f'{v}|B|T'], o['true']['C0'][f'{v}|B|T']
        f = lambda x: 'None' if x is None else f'{x:.3f}'
        print(f'{job:22s} {v:6s} {a["amplitude"]:9.3f} {b["amplitude"]:9.3f} {b["amplitude"] - a["amplitude"]:+7.3f} {f(a["corr"]):>7s} {f(b["corr"]):>7s} '
              f'{str(o["coded_MR_ok_C0"])[0]:>4s} {str(o["true_MR_ok_C0"])[0]:>4s}')

print()
print('Largest |coded - true| over ALL amplitude / corr / err metrics per job')
for job, o in R.items():
    if 'coded' not in o:
        continue
    rows = []
    for arm in o['coded']:
        if arm == 'sender_split':
            for nm, d in o['coded'][arm].items():
                for k, v in d.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            t = o['true'][arm][nm][k][kk]
                            if vv is not None and t is not None:
                                rows.append((abs(vv - t), f'{arm}|{nm}|{k}|{kk}', vv, t))
                    elif v is not None:
                        rows.append((abs(v - o['true'][arm][nm][k]), f'{arm}|{nm}|{k}', v, o['true'][arm][nm][k]))
            continue
        for key, v in o['coded'][arm].items():
            for kk, vv in v.items():
                t = o['true'][arm][key].get(kk)
                if kk in ('patches', 'ref_snr_median') or vv is None or t is None:
                    continue
                rows.append((abs(vv - t), f'{arm}|{key}|{kk}', vv, t))
    rows.sort(reverse=True)
    print(job)
    for r in rows[:6]:
        print(f'    {r[1]:60s} coded {r[2]:+.4f} true {r[3]:+.4f} diff {r[3] - r[2]:+.4f}')

print()
print('Sender split, untextured (C0_fromS1 = learned panel sender, learned receiver; C2_rcv_fromS1 = learned panel sender, exact receiver) vs e2_p')
for job, o in R.items():
    if 'coded' not in o:
        continue
    for nm in ('C0_fromS1', 'C2_rcv_fromS1', 'C0_fromOther'):
        if nm in o['coded']['sender_split']:
            a, b = o['coded']['sender_split'][nm], o['true']['sender_split'][nm]
            print(f'{job:22s} {nm:16s} err_T coded {a["baseline_err_T"]:.3f} true {b["baseline_err_T"]:.3f} (signed true {b["baseline_signed_T"]:+.3f}, half-noise true {b["baseline_half_noise_T"]:.3f}) '
                  f'| panel-face err true {b["baseline_err_Tpf"]:.3f} (noise {b["baseline_half_noise_Tpf"]:.3f}) | share coded {a["ref_share_of_S_untextured_T"]:.3f} true {b["ref_share_of_S_untextured_T"]:.3f} '
                  f'pf share true {b["ref_share_of_S_untextured_Tpf"]:.3f}')
