"""WL115 Stage B frozen analysis: independent qualification of the CAR-16 oracle against the dense 4 x 4 oracle on HELD_SEEN + FRESH
contact-valid views (full-depth references; metrics113.gate3 unchanged; R5 RMSE control = PC4_O).

  python analyzeB115.py ROOT  -> analysis/stageB_metrics.json, analysis/stageB_decision.json
Frozen adequacy rules (protocol sr115.json stage_B_frozen.adequacy):
  G1 total R1L failing windows reduced >= 70 % vs PC4_O (all classes) AND >= 70 % within FRESH
  G2 median over views of the max |window error| reduced >= 25 %
  G3 innermost ring (0-10 cm) mean |bias| reduced >= 25 %
  G4 lifted (gap >= 30 mm) mean near |bias| reduced >= 25 % (all classes) AND within FRESH if FRESH has lifted views
  G5 no regression: R5a / R5b bias pass counts >= PC4_O's, R5a / R5b RMSE <= 1.10 x PC4_O at every view, R1 and R2 pass counts >= PC4_O's
QUALIFIED iff G1-G5 hold. The approved 3 % gate is reported separately (expected to fail from the Stage A2 angular component).
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_fd114')); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common115 as C  # noqa: E402
import metrics113 as MX  # noqa: E402
import stageB115 as SB  # noqa: E402

LIFT = 0.03


def load_refs(root, tag, v):
    st = []
    for j in range(2):
        src = (C.WL114_ROOT / 'stageB' / f'stream_{tag}_{j}.npz') if v['cls'] == 'HELD_SEEN' else (root / 'refs' / f'ref_{tag}_{j}.npz')
        with np.load(src) as z:
            s = {k: z[k] for k in z.files if k.startswith('phys_')}
        st.append(s)
    gsrc = (C.WL114_ROOT / 'stageB' / f'gbuffer_{tag}.npz') if v['cls'] == 'HELD_SEEN' else (root / 'refs' / f'gbuffer_{tag}.npz')
    with np.load(gsrc) as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def view_gate(st, gb, v, arms, extra_streams):
    """arms: list of names present in extra_streams[j]['hl|name']; returns (metrics dict, kappa, reg, Rf, Mc)."""
    g3, reg, bands = MX.gate_masks(gb, v['g'], v['l'][0]['pos'])
    h = [s['phys_h'].astype(np.float64) for s in st]
    tail = [st[j]['phys_c7_12'].astype(np.float64) + st[j]['phys_c13_24'] for j in range(2)]
    kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
    Rf = [st[j]['phys_e1'].astype(np.float64) + st[j]['phys_e2'] + h[j] + tail[j] for j in range(2)]
    low = [extra_streams[j]['ro_e1'].astype(np.float64) + extra_streams[j]['ro_e2'] for j in range(2)]
    pair = max(float((np.abs(extra_streams[j]['ro_e1'] - st[j]['phys_e1']).sum() + np.abs(extra_streams[j]['ro_e2'] - st[j]['phys_e2']).sum()) /
                     (np.abs(st[j]['phys_e1']).sum() + np.abs(st[j]['phys_e2']).sum())) for j in range(2))
    Mc = {a: [low[j] + extra_streams[j][f'hl|{a}'] for j in range(2)] for a in arms}
    gap = float(v['g']['S']['c'][2] - C.R_SPHERE)
    out = {'gap_m': gap, 'lifted': bool(gap >= LIFT), 'contact_valid': C.contact_valid(reg), 'kappa': kap, 'low_pairing_rel_l1': pair, 'arms': {}}
    return out, g3, reg, Rf, Mc, kap


def arm_metrics(Mc_a, Rf, g3, kap, ctrl, cv):
    o = MX.gate3(Mc_a, Rf, g3, kap, ctrl=ctrl)
    p = dict(o['pass'])
    if not cv:
        p = {k: x for k, x in p.items() if k.startswith('R5')}
    o['all_pass'] = bool(all(p.values()))
    o['b_near'] = o['R1_near']['b'] if not math.isnan(o['R1_near'].get('b', float('nan'))) else None
    return o


def summary(rows, arm):
    cv = [r for r in rows if r['contact_valid']]
    P = lambda req: int(sum(r['arms'][arm]['pass'][req] for r in cv))
    r0 = [abs(r['arms'][arm]['R4_rings'][0]['b']) for r in cv if not math.isnan(r['arms'][arm]['R4_rings'][0]['b'])]
    lif = [abs(r['arms'][arm]['b_near']) for r in cv if r['lifted']]
    return {'n': len(cv), 'all_pass': int(sum(r['arms'][arm]['all_pass'] for r in cv)),
            'pass_counts': {k: P(k) for k in ('R1', 'R1L', 'R2', 'R3_bias', 'R4', 'R5a_bias', 'R5b_bias', 'R5a_rmse', 'R5b_rmse')},
            'R1L_failing_total': int(sum(r['arms'][arm]['R1L']['failing'] for r in cv)),
            'R1L_max_abs_e_median': float(np.median([r['arms'][arm]['R1L']['max_abs_e'] for r in cv])),
            'ring0_abs_mean': float(np.mean(r0)) if r0 else None, 'lifted_near_abs_mean': float(np.mean(lif)) if lif else None, 'n_lifted': len(lif),
            'near_bias_mean': float(np.mean([r['arms'][arm]['b_near'] for r in cv]))}


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    V = SB.eval_views(); res = {}
    for tag, v in V.items():
        st, gb = load_refs(root, tag, v)
        ex = []
        for j in range(2):
            with np.load(root / 'stageB' / 'streams' / f'stream_{tag}_{j}.npz') as z:
                ex.append({k: z[k] for k in z.files})
        out, g3, reg, Rf, Mc, kap = view_gate(st, gb, v, ['PC4_O', 'CAR_O'], ex)
        for a in ('PC4_O', 'CAR_O'):
            out['arms'][a] = arm_metrics(Mc[a], Rf, g3, kap, Mc['PC4_O'], out['contact_valid'])
        out['cls'] = v['cls']; out['key'] = v['key']
        res[tag] = out
        print('B', tag, v['cls'], out['contact_valid'], {a: (out['arms'][a]['R1L']['failing'], round(100 * (out['arms'][a]['b_near'] or 0), 2)) for a in out['arms']}, flush=True)
    C.jd(res, an / 'stageB_metrics.json')
    rows = list(res.values()); fresh = [r for r in rows if r['cls'] == 'FRESH']
    S = {cls: {a: summary(rr, a) for a in ('PC4_O', 'CAR_O')} for cls, rr in (('ALL', rows), ('FRESH', fresh), ('HELD_SEEN', [r for r in rows if r['cls'] == 'HELD_SEEN']))}
    red = lambda x, y: 1 - y / x if x and x > 0 else float('nan')
    A, Fr = S['ALL'], S['FRESH']
    G = {'G1': red(A['PC4_O']['R1L_failing_total'], A['CAR_O']['R1L_failing_total']) >= 0.70 and red(Fr['PC4_O']['R1L_failing_total'], Fr['CAR_O']['R1L_failing_total']) >= 0.70,
         'G2': red(A['PC4_O']['R1L_max_abs_e_median'], A['CAR_O']['R1L_max_abs_e_median']) >= 0.25,
         'G3': red(A['PC4_O']['ring0_abs_mean'], A['CAR_O']['ring0_abs_mean']) >= 0.25,
         'G4': red(A['PC4_O']['lifted_near_abs_mean'], A['CAR_O']['lifted_near_abs_mean']) >= 0.25 and
               (Fr['PC4_O']['n_lifted'] == 0 or red(Fr['PC4_O']['lifted_near_abs_mean'], Fr['CAR_O']['lifted_near_abs_mean']) >= 0.25)}
    pc = lambda a, k: A[a]['pass_counts'][k]
    G['G5'] = bool(all(pc('CAR_O', k) >= pc('PC4_O', k) for k in ('R5a_bias', 'R5b_bias', 'R1', 'R2')) and
                   all(r['arms']['CAR_O']['pass'].get('R5a_rmse', True) and r['arms']['CAR_O']['pass'].get('R5b_rmse', True) for r in rows))
    G = {k: bool(v) for k, v in G.items()}
    dec = {'summary': S, 'reductions': {'R1L_total_all': red(A['PC4_O']['R1L_failing_total'], A['CAR_O']['R1L_failing_total']),
                                        'R1L_total_fresh': red(Fr['PC4_O']['R1L_failing_total'], Fr['CAR_O']['R1L_failing_total']),
                                        'maxe_median': red(A['PC4_O']['R1L_max_abs_e_median'], A['CAR_O']['R1L_max_abs_e_median']),
                                        'ring0': red(A['PC4_O']['ring0_abs_mean'], A['CAR_O']['ring0_abs_mean']),
                                        'lifted_all': red(A['PC4_O']['lifted_near_abs_mean'], A['CAR_O']['lifted_near_abs_mean']),
                                        'lifted_fresh': red(Fr['PC4_O']['lifted_near_abs_mean'], Fr['CAR_O']['lifted_near_abs_mean']) if Fr['PC4_O']['n_lifted'] else None},
           'rules': G, 'QUALIFIED': bool(all(G.values())),
           'approved_gate_all_pass_views': {a: A[a]['all_pass'] for a in A}, 'max_low_pairing_rel_l1': max(r['low_pairing_rel_l1'] for r in rows)}
    C.jd(dec, an / 'stageB_decision.json')
    print(json.dumps({k: dec[k] for k in ('reductions', 'rules', 'QUALIFIED', 'approved_gate_all_pass_views')}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    import os; sys.stdout.flush(); os._exit(0)
