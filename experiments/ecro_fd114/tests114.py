"""WL114 focused tests (before any evidence stream; any failure stops the chain).

T1 depth convention / event ownership: deep pass events 1-6 == WL113 full_pass (and == the depth-6 estimator through it), buckets
   7..24 == WL113 tail; single-bucket energy accounting == sum of buckets; compaction exact.
T2 physical suffix equivalence: deep construction D6 part == WL113 construct4; deeper construction leaves f2 / fh unchanged;
   f2 is camera event 3 (a probe to camera event 3 only has fh == 0).
T3 roulette correctness: rr_from beyond kmax == deterministic (to float64 atomic-summation order); events <= rr_from identical with roulette on.
T4 source linearity / independence: construction targets x2 for a doubled light; superposition of two lights; learned ECRO-S4
   output linear in the source.
T5 energy accounting of the tables: pooled S4 == U (both depths); U_F - U_D6 == the deep-bucket means (>= 0).
T6 spatial target / readout identity: coarse table read as 'coarse' == its dense 4x4 expansion read as 'sub'; zero table -> zero HIGH;
   LOW identical across tables and equal to the physical events 1-2 of the paired deep pass.
T7 dynamic state / revisit: learned dense table at state A, B, A -> A bit-identical.
T8 dataset / reference isolation: no evaluation state within the WL112 exclusion of a D113 configuration; evaluation seeds disjoint
   from training-construction seeds.
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common114 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
S0 = 91149000


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum() / max(np.abs(np.asarray(b)).sum(), 1e-30))


def csum(acc, names):
    return sum(acc.get(f'U_{n}{c}', 0.) for n in names for c in range(3))


def main(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); out = {}
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    ctx = T.Ctx114(geom, 'C')
    cam = dict(C.V0_CAM, res=64)
    # T1
    f, _ = ctx.full_pass(light, cam, 64, 32, S0, b=32)
    a6, _, _, _ = ctx.cam_pass(light, cam, 64, 32, S0, 'cam', b=32)
    d, _ = ctx.deep_pass(light, cam, 64, 32, S0, 24, b=32)
    dn, _ = ctx.deep_pass(light, cam, 64, 32, S0, 24, b=32, compact=False)
    orig = T.bucket_cam
    T.bucket_cam = lambda k: 'tot'
    try:
        one, _ = ctx.deep_pass(light, cam, 64, 32, S0, 24, b=32)
    finally:
        T.bucket_cam = orig
    tot = sum(d[k] for k in ('e1', 'e2', 'h', 'c7_12', 'c13_24'))
    out['T1'] = {'e1e2h_vs_full_pass': max(rel(d[k], f[k]) for k in ('e1', 'e2', 'h')), 'e1e2h_vs_d6_estimator': max(rel(d[k], a6[k]) for k in ('e1', 'e2', 'h')),
                 'tail_vs_full_pass': rel(d['c7_12'] + d['c13_24'], f['tail']), 'single_bucket_vs_sum': rel(one['tot'], tot),
                 'compaction': max(rel(d[k], dn[k]) for k in d)}
    out['T1']['pass'] = bool(out['T1']['e1e2h_vs_full_pass'] <= 1e-6 and out['T1']['e1e2h_vs_d6_estimator'] <= 1e-6 and out['T1']['tail_vs_full_pass'] <= 1e-6
                             and out['T1']['single_bucket_vs_sum'] <= 1e-6 and out['T1']['compaction'] <= 1e-9)
    # T2
    a4 = ctx.construct4(light, 8, S0 + 1, b=8)
    c5, _ = ctx.construct_deep(light, 8, S0 + 1, 5, b=8)
    c23, _ = ctx.construct_deep(light, 8, S0 + 1, 23, b=8)
    c2, _ = ctx.construct_deep(light, 8, S0 + 1, 2, b=8)
    out['T2'] = {'w': rel(c5['U_w'], a4['U_w']), 'f2_vs_construct4': rel(csum(c5, ['f2']), csum(a4, ['f2'])), 'fh_vs_construct4': rel(csum(c5, ['fh']), csum(a4, ['fh'])),
                 'deeper_keeps_f2fh': rel(csum(c23, ['f2', 'fh']), csum(c5, ['f2', 'fh'])), 'probe_to_cam3_fh_zero': float(np.abs(csum(c2, ['fh'])).sum()),
                 'probe_to_cam3_f2_equal': rel(csum(c2, ['f2']), csum(c5, ['f2']))}
    out['T2']['pass'] = bool(out['T2']['w'] == 0 and out['T2']['f2_vs_construct4'] <= 1e-5 and out['T2']['fh_vs_construct4'] <= 1e-5 and
                             out['T2']['deeper_keeps_f2fh'] <= 1e-7 and out['T2']['probe_to_cam3_fh_zero'] == 0 and out['T2']['probe_to_cam3_f2_equal'] <= 1e-7)
    # T3
    r_off, _ = ctx.deep_pass(light, cam, 64, 32, S0 + 2, 48, b=32, rr_from=500)
    det, _ = ctx.deep_pass(light, cam, 64, 32, S0 + 2, 48, b=32)
    r_on, _ = ctx.deep_pass(light, cam, 64, 32, S0 + 2, 48, b=32, rr_from=24)
    out['T3'] = {'rr_beyond_kmax_rel': max(rel(r_off[k], det[k]) for k in det),
                 'events_le_rr_from_identical': max(rel(r_on[k], det[k]) for k in ('e1', 'e2', 'h', 'c7_12', 'c13_24'))}
    out['T3']['pass'] = bool(out['T3']['rr_beyond_kmax_rel'] <= 1e-12 and out['T3']['events_le_rr_from_identical'] <= 1e-12)   # float64 atomic-order only
    # T4
    l2 = [dict(x, rgb=[2 * v for v in x['rgb']]) for x in lc]
    cA, _ = ctx.construct_deep(light, 8, S0 + 3, 23, b=8)
    cB, _ = ctx.construct_deep(PT.Lights(l2), 8, S0 + 3, 23, b=8)
    g1, lc1 = C.fixture('F1')
    lsum = lc + lc1
    cS, _ = ctx.construct_deep(PT.Lights(lsum), 8, S0 + 3, 23, b=8)
    cL1, _ = ctx.construct_deep(PT.Lights(lc1), 8, S0 + 3, 23, b=8)
    names = ['f2', 'fh'] + list(T.DEEP)
    out['T4'] = {'double_light': rel(csum(cB, names), 2 * csum(cA, names)), 'superposition': rel(csum(cS, names), csum(cA, names) + csum(cL1, names))}
    import torch
    import eval113 as E3
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda')); desc = ctx.vhs()
    gs.sup = torch.arange(1270, device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
    model = M3.ECROL4().cuda(); model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda'); model.eval()
    model.load_state_dict(torch.load(C.REPO / 'results/ecro_sss113/primary/stageC/CAND/model.pt', map_location='cuda'))
    s = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
    with torch.no_grad():
        fct = model.factors(gs)
        y1 = model.apply_sub(fct, gs, s); y2 = model.apply_sub(fct, gs, 2 * s)
    out['T4']['learned_source_linearity'] = float((y2 - 2 * y1).abs().sum() / max(float((2 * y1).abs().sum()), 1e-30))
    out['T4']['pass'] = bool(out['T4']['double_light'] <= 1e-6 and out['T4']['superposition'] <= 1e-5 and out['T4']['learned_source_linearity'] <= 1e-5)
    # T5
    ht = T.high_tables(cA, 24)
    w = cA['U_w'].reshape(1270, 16, 64)
    pooled = {tg: (ht[f'S4_{tg}'] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-30)[..., None] for tg in ('D6', 'F')}
    nat = ht['native_c']
    deep_mean = sum(np.stack([cA.get(f'U_{n}{c}', np.zeros(1270 * 16 * 64)) for c in range(3)], -1).reshape(1270, 16, 64, 3).sum(1) for n in T.deep_through(24)) / np.maximum(w.sum(1), 1e-30)[..., None]
    out['T5'] = {'pooled_vs_U_D6': float(np.abs(pooled['D6'] - ht['U_D6'])[nat].max() / max(np.abs(ht['U_D6']).max(), 1e-30)),
                 'pooled_vs_U_F': float(np.abs(pooled['F'] - ht['U_F'])[nat].max() / max(np.abs(ht['U_F']).max(), 1e-30)),
                 'UF_minus_UD6_vs_deep': float(np.abs((ht['U_F'] - ht['U_D6']) - deep_mean)[nat].max() / max(np.abs(ht['U_F']).max(), 1e-30)),
                 'deep_nonnegative': bool((deep_mean >= -1e-7).all())}
    out['T5']['pass'] = bool(out['T5']['pooled_vs_U_D6'] <= 1e-5 and out['T5']['pooled_vs_U_F'] <= 1e-5 and out['T5']['UF_minus_UD6_vs_deep'] <= 1e-5 and out['T5']['deep_nonnegative'])
    # T6
    Tc = ht['U_F']
    tl = [('coarse', PT.fl(np.ascontiguousarray(Tc).ravel()), 'coarse'), ('dense', PT.fl(np.ascontiguousarray(T.T113.to_dense16(Tc, 'coarse')).ravel()), 'sub'),
          ('zero', PT.fl(np.zeros(1270 * 16 * 64 * 3, np.float32)), 'sub')]
    img, _, _, _ = ctx.cam_pass(light, cam, 64, 32, S0 + 4, 'ecro', tables=tl, b=32)
    img2, _, _, _ = ctx.cam_pass(light, cam, 64, 32, S0 + 4, 'ecro', tables=tl[2:], b=32)
    dp, _ = ctx.deep_pass(light, cam, 64, 32, S0 + 4, 24, b=32)
    out['T6'] = {'coarse_vs_dense': rel(img['dense'], img['coarse']), 'zero_table_high': float(np.abs(img['zero']).sum()),
                 'low_independent_of_tables': max(rel(img2[k], img[k]) for k in ('e1', 'e2')), 'low_vs_deep_events12': max(rel(img[k], dp[k]) for k in ('e1', 'e2'))}
    out['T6']['pass'] = bool(out['T6']['coarse_vs_dense'] <= 1e-6 and out['T6']['zero_table_high'] == 0 and out['T6']['low_independent_of_tables'] <= 1e-6
                             and out['T6']['low_vs_deep_events12'] <= 1e-6)
    del ctx
    # T7
    q = C.sequences()['SEQ_D_independent']
    p = C.REPO / 'results/ecro_sss113/primary/stageC/CAND/model.pt'
    tA, _ = E3.learned_table(p, 'dense', q['states'][0], q['light'])
    tB, _ = E3.learned_table(p, 'dense', q['states'][3], q['light'])
    tA2, _ = E3.learned_table(p, 'dense', q['states'][0], q['light'])
    out['T7'] = {'revisit_max_abs': float(np.abs(tA2 - tA).max()), 'state_change_rel': rel(tB, tA)}
    out['T7']['pass'] = bool(out['T7']['revisit_max_abs'] == 0 and out['T7']['state_change_rel'] > 1e-4)
    # T8
    dc = json.loads((C.REPO / 'results/ecro_fd114/design/design_check.json').read_text())
    import eval114 as EV
    eval_seeds = {s for _, (g, l, vl, cs) in EV.states().items() for _, _, ss in vl for s in ss} | {s for _, (g, l, vl, cs) in EV.states().items() for s in cs}
    train_seeds = {({'train': 11320100, 'val': 11320200}[sp]) * 100 + i * 16 + li + o for sp, n in (('train', 192), ('val', 16)) for i in range(n) for li in range(4) for o in (0, 8)}
    out['T8'] = {'eval_state_conflicts': dc['isolation']['any_eval_state_conflict'], 'new_fixture_conflicts': sum(v['train_conflicts'] for v in dc['fixtures'].values()),
                 'seed_overlap': len(eval_seeds & train_seeds)}
    out['T8']['pass'] = bool(out['T8']['eval_state_conflicts'] == 0 and out['T8']['new_fixture_conflicts'] == 0 and out['T8']['seed_overlap'] == 0)
    out['pass'] = bool(all(v['pass'] for k, v in out.items() if k.startswith('T')))
    C.jd(out, root / 'tests114.json')
    print('TESTS', json.dumps(out, indent=1), flush=True)
    if not out['pass']:
        sys.stdout.flush(); os._exit(3)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
