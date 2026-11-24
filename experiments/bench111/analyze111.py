"""WL111 analysis (CPU, after the evidence chain): tables, frontiers, gate evaluation, SSS near-contact table, figures.

python analyze111.py EVIDENCE_DIR LR3_DIR OUT_DIR

Implements the frozen rules of protocol/bench111.json (metrics.primary, frontier, gate_derivation) and bench111_lr3.json.
Timing for H1 / H2 / S1 / SL frames uses the S0 V0 timing (same scene, resolution and estimator; the protocol times V0 only).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol' / 'bench111.json').read_text(encoding='utf-8'))
EV, LR, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(111)
NB = 2000
G = P['gate_derivation']['initial_margins']
BUDGET = P['gate_derivation']['frame_budgets_ms']


def rd(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


# ------------------------------------------------------------------------------------------------ primary error with CI
def err_from(rows, roi='full', key='vs_full'):
    mse = np.array([r[key][roi]['mse'] for r in rows])
    nv = rows[0][key][roi]['ref_noise_mse']
    mr = rows[0][key][roi]['mean_ref']
    point = math.sqrt(max(mse.mean() - nv, 0.0)) / mr
    if len(mse) > 1:
        bs = [math.sqrt(max(mse[RNG.integers(0, len(mse), len(mse))].mean() - nv, 0.0)) / mr for _ in range(NB)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
    else:
        lo = hi = float('nan')
    return {'E': point, 'lo': float(lo), 'hi': float(hi), 'n': int(len(mse))}


def mean_ci(v):
    v = np.asarray(v, np.float64)
    if v.size < 2:
        return {'mean': float(v.mean()), 'lo': float('nan'), 'hi': float('nan'), 'n': int(v.size)}
    bs = [v[RNG.integers(0, v.size, v.size)].mean() for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'mean': float(v.mean()), 'lo': float(lo), 'hi': float(hi), 'n': int(v.size)}


def med_ci(v):
    v = np.asarray(v, np.float64)
    bs = [np.median(v[RNG.integers(0, v.size, v.size)]) for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'median': float(np.median(v)), 'lo': float(lo), 'hi': float(hi), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99)), 'mean': float(v.mean()), 'n': int(v.size)}


# ------------------------------------------------------------------------------------------------ frontier helpers
def curve(points):
    """points: list of (t_ms, E) for one arm, sorted by t -> monotone-in-t piecewise log-log interpolants."""
    pts = sorted(points)
    return pts


def E_at(pts, t):
    """Arm error at time t (log-log linear between executed points); None outside the executed range."""
    if t < pts[0][0] or t > pts[-1][0]:
        return None
    for (t0, e0), (t1, e1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return min(e0, e1)
            a = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return math.exp((1 - a) * math.log(max(e0, 1e-12)) + a * math.log(max(e1, 1e-12)))
    return pts[-1][1]


def t_to(pts, e):
    """Time at which the arm's interpolated error first reaches e; None when e lies outside the arm's executed error range
    (frozen rule: 'beyond executed range' = not evaluable). Correction disclosed in WL111: the first version returned the
    arm's cheapest time when even that point was better than e."""
    es = [p[1] for p in pts]
    if e > max(es) or e < min(es):
        return None
    if pts[0][1] <= e:
        return pts[0][0]
    for (t0, e0), (t1, e1) in zip(pts[:-1], pts[1:]):
        if e0 > e >= e1:
            a = (math.log(e) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp((1 - a) * math.log(t0) + a * math.log(t1))
    return None


def frontier_eval(rest_t, rest_E, comps):
    """comps: {arm: pts}. Returns frontier values and PATH1 / PATH2 verdicts by the frozen rule."""
    tq = [t_to(p, rest_E / (1 + G['m_q'])) for p in comps.values()]
    tq = [x for x in tq if x is not None]
    tF = min(tq) if tq else None
    eq = [E_at(p, rest_t / (1 + G['m_t'])) for p in comps.values()]
    eq = [x for x in eq if x is not None]
    EF = min(eq) if eq else None
    best_arm_E = min(comps, key=lambda a: (E_at(comps[a], rest_t / (1 + G['m_t'])) or 1e9))
    best_arm_t = min(comps, key=lambda a: (t_to(comps[a], rest_E / (1 + G['m_q'])) or 1e9))
    # descriptive dominance (not a gate rule): best comparator point within 1.5x the ReST time, cheapest comparator point
    # with an error <= the ReST error
    allp = [(a, t, e) for a, p in comps.items() for t, e in p]
    near = [x for x in allp if x[1] <= 1.5 * rest_t]
    better = [x for x in allp if x[2] <= rest_E]
    dom = {'best_within_1p5x_time': min(near, key=lambda x: x[2]) if near else None,
           'cheapest_with_lower_or_equal_error': min(better, key=lambda x: x[1]) if better else None}
    p1 = None if tF is None else bool(rest_t <= (1 - G['s_t']) * tF)
    p2 = None if EF is None else bool(rest_E <= (1 - G['s_q']) * EF)
    return {'t_F_at_E_rest_over_1pmq': tF, 'E_F_at_t_rest_over_1pmt': EF, 'PATH1': p1, 'PATH2': p2,
            'strongest_at_equal_time': best_arm_E, 'strongest_at_equal_quality': best_arm_t,
            'speed_ratio_needed': None if tF is None else rest_t / tF, 'error_ratio': None if EF is None else rest_E / EF,
            'evaluable_arms_time': [a for a, p in comps.items() if t_to(p, rest_E / (1 + G['m_q'])) is not None],
            'evaluable_arms_error': [a for a, p in comps.items() if E_at(p, rest_t / (1 + G['m_t'])) is not None],
            'dominance': dom}


# ------------------------------------------------------------------------------------------------ fixtures
summary = {'fixtures': {}, 'timing': {}, 'dynamic': {}, 'sss_gate': {}, 'gates': {}, 'references': {}}
timing = {}
for f in sorted((EV / 'timing').glob('*.json')):
    t = rd(f)
    key = f.stem
    timing[key] = {k: med_ci(v) for k, v in t['frame_ms'].items()}
    timing[key]['denoise_only'] = med_ci(t['denoise_only_ms'])
    summary['timing'][key] = {'ops': timing[key], 'first_call_ms': t['first_call_ms'], 'exclusivity': t['exclusivity']}


def tkey(state, view):
    return 'N0_V0' if state == 'N0' else 'S0_V0'


for f in sorted((EV / 'quality').glob('*.json')):
    q = rd(f)
    sid, view = q['state'], q['view']
    case = P['states'][sid]['case']
    tk = tkey(sid, view)
    groups = {}
    for r in q['rows']:
        groups.setdefault((r['arm'], r['spp']), []).append(r)
    tab = {}
    for (arm, spp), rows in sorted(groups.items()):
        e = err_from(rows)
        ent = {'E': e, 'E_vs_d6': err_from(rows, 'full', 'vs_d6'),
               'rel_bias': mean_ci([r['vs_full']['full']['rel_bias'] for r in rows]),
               'rel_l1': mean_ci([r['vs_full']['full']['rel_l1'] for r in rows]),
               'ldr_rmse': mean_ci([r['vs_full']['full']['ldr_rmse'] for r in rows]),
               'psnr_ldr': mean_ci([r['vs_full']['full']['psnr_ldr'] for r in rows]),
               'ssim_rep0': next((r.get('ssim_ldr_vs_full') for r in rows if r['rep'] == 0), None),
               'roi': {roi: err_from(rows, roi) for roi in P['metrics']['rois'][case] if roi in rows[0]['vs_full'] and roi != 'full'}}
        tm = timing.get(tk, {}).get(f'{arm}_{spp}')
        ent['ms'] = tm['median'] if tm else None
        ent['ms_ci'] = [tm['lo'], tm['hi']] if tm else None
        if case == 'sss' and 'near_signed' in rows[0]['vs_full']:
            ent['near_signed'] = mean_ci([r['vs_full']['near_signed'] for r in rows])
            ent['near_rel_signed'] = mean_ci([r['vs_full']['near_rel_signed'] for r in rows])
            ent['excess'] = mean_ci([r['vs_full']['near_minus_flank_excess'] for r in rows])
            rings = sorted(k for k in rows[0]['vs_full']['signed'] if k.startswith('ring_'))
            ent['radial'] = {k: mean_ci([r['vs_full']['signed'][k]['diff'] / max(r['vs_full']['signed'][k]['ref'], 1e-12) for r in rows])
                             for k in rings}
        if arm == 'ECRO_CONV':
            ent['hl_vs_full_higher'] = rows[0].get('hl_vs_full_higher')
        tab[f'{arm}_{spp}'] = ent
    # reference self-checks
    rs = q['reference_self']
    summary['references'][f'{sid}_{view}'] = {'half_vs_half_rel_rmse': rs['half_vs_half']['rel_rmse'],
                                              'D6_vs_FULL_rel_bias': rs['D6_vs_FULL']['full']['rel_bias'],
                                              'D6_vs_FULL_E': rs['D6_vs_FULL']['full']['rel_rmse_db'],
                                              'D6_vs_FULL_roi': {k: v['rel_bias'] for k, v in rs['D6_vs_FULL'].items() if isinstance(v, dict) and 'rel_bias' in v}}
    if case == 'sss' and 'near_minus_flank_excess' in rs['D6_vs_FULL']:
        summary['references'][f'{sid}_{view}']['D6_vs_FULL_near_signed'] = rs['D6_vs_FULL']['near_signed']
        summary['references'][f'{sid}_{view}']['D6_vs_FULL_excess'] = rs['D6_vs_FULL']['near_minus_flank_excess']
    summary['fixtures'][f'{sid}_{view}'] = tab
    # frontier + gates at every ECRO operating point
    comps = {}
    for arm in ('PT', 'PTDN'):
        pts = [(tab[k]['ms'], tab[k]['E']['E']) for k in tab if k.startswith(arm + '_') and tab[k]['ms'] is not None]
        if pts:
            comps[arm] = curve(pts)
    gates = {}
    for k, ent in tab.items():
        if not k.startswith('ECRO_') or k.startswith('ECRO_CONV') or ent['ms'] is None:
            continue
        ge = frontier_eval(ent['ms'], ent['E']['E'], comps)
        ge.update({'E_rest': ent['E']['E'], 'E_rest_ci': [ent['E']['lo'], ent['E']['hi']], 't_rest_ms': ent['ms'],
                   'budget_ideal': ent['ms'] <= BUDGET['ideal'], 'budget_desired': ent['ms'] <= BUDGET['desired'],
                   'budget_acceptable': ent['ms'] <= BUDGET['acceptable']})
        gates[k] = ge
    summary['gates'][f'{sid}_{view}'] = gates

# ------------------------------------------------------------------------------------------------ dynamic
dyn = rd(EV / 'dynamic' / 'dynamic.json') if (EV / 'dynamic' / 'dynamic.json').exists() else {}
for name, d in dyn.items():
    if not isinstance(d, dict) or 'raw' not in d:
        continue
    ent = {'edit_to_frame': {k: med_ci([x['ms'] for x in v if x['to_target']]) for k, v in d['raw'].items()},
           'edit_to_frame_reverse': {k: med_ci([x['ms'] for x in v if not x['to_target']]) for k, v in d['raw'].items()},
           'exclusivity': d['exclusivity']}
    sg = {}
    for r in d['stale_quality']:
        sg.setdefault(r['spp'], []).append(r)
    ent['stale_E'] = {spp: err_from(rows) for spp, rows in sg.items()}
    ent['stale_near_signed'] = {spp: mean_ci([r['vs_full']['near_signed'] for r in rows]) for spp, rows in sg.items()
                                if 'near_signed' in rows[0]['vs_full']}
    summary['dynamic'][name] = ent

# ------------------------------------------------------------------------------------------------ LR3
lr3 = {}
if (LR / 'lr3_pt_quality.json').exists():
    q = rd(LR / 'lr3_pt_quality.json')
    tm = rd(LR / 'lr3_pt_timing.json') if (LR / 'lr3_pt_timing.json').exists() else None
    groups = {}
    for r in q['rows']:
        groups.setdefault((r['arm'], r['spp']), []).append(r)
    for (arm, spp), rows in sorted(groups.items()):
        mse = np.array([r['vs_ref65']['mse'] for r in rows])
        ldr = [r['vs_ref65']['ldr_rmse'] for r in rows]
        e = {'E': err_from([{'vs_full': {'full': r['vs_ref65']}} for r in rows]), 'ldr_rmse': mean_ci(ldr),
             'ldr_rmse_vs_ref7': mean_ci([r['vs_ref7']['ldr_rmse'] for r in rows]),
             'rel_bias': mean_ci([r['vs_ref65']['rel_bias'] for r in rows])}
        if tm and f'{arm}_{spp}' in tm['frame_ms']:
            m = med_ci(tm['frame_ms'][f'{arm}_{spp}'])
            e['ms'] = m['median']
            e['ms_ci'] = [m['lo'], m['hi']]
        lr3[f'{arm}_{spp}'] = e
    lr3['_reference_self'] = q['reference_self']
    if tm:
        lr3['_timing_exclusivity'] = tm['exclusivity']
if (LR / 'lr3_transgi_scored.json').exists():
    lr3['_transgi_scored'] = rd(LR / 'lr3_transgi_scored.json')
for f in sorted(LR.glob('bench_*wl111*.json')):
    b = rd(f)
    lr3.setdefault('_transgi_bench', {})[f.stem] = {'frame_ms': b['frame_ms'], 'cuda_event_ms': b['frame_cuda_event_ms_median'],
                                                     'stages': b['stages_ms_median'], 'res': b['res'],
                                                     'gpu_mem_peak_bytes': b.get('gpu_mem_peak_bytes')}
summary['lr3'] = lr3
Path(OUT / 'summary111.json').write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
print('SUMMARY WRITTEN', OUT / 'summary111.json')
