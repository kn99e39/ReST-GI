"""WL67 W1 analysis (protocol baseline_sup67_v1): quality of Cycles Full Recompute ladder vs WL56 F/S/T oracle, same masks/rule.

Env: WL56=<isolated oracle root> (its out/cycles is a read-only link to the WL56 reference), B67_LADDER=<ladder out>,
B67_BOUNCE=<bounce out>, B67_INHERIT=<WL56 out/runs> (second realization), B67_OUT=<analysis out>.
Reuses s56_analyze.group_stats / pred unchanged.
"""
import os, sys, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fnto_sel56'))
import s56 as S
import s56_analyze as A
import s56_fixtures as SF
import k55_common as CM

LAD = Path(os.environ['B67_LADDER']); BOU = Path(os.environ['B67_BOUNCE']); INH = Path(os.environ['B67_INHERIT']); OUT = Path(os.environ['B67_OUT'])
FACE = {'top': 'upper:top', 'gapdown': 'lower:top', 'gapup': 'upper:bot', 'bot': 'lower:bot'}
LADDER = (1, 4, 16, 64, 256, 1024)
STATES = ('rest', 'dent', 'rigid')
CH = 'RGB'


def seed_of(state, li, r): return 670000 + 1000 * STATES.index(state) + 10 * li + r


REF, SS, LAB, SD = {}, {}, {}, {}
for st in STATES:
    SD[st] = SF.scene(st); SS[st] = CM.classify_sceneset(SD[st]); LAB[st] = CM.labels_of(SD[st]); REF[st] = S.extract(st)


def cy_ports(tag, d, st, cam, spp, seed):
    geo = json.load(open(d / f'ccam_{tag}_{st}.json'))[cam]
    ref_geo = json.load(open(S.OUT / 'cycles' / f'ccam_{st}.json'))[cam]
    for k in ref_geo:
        assert np.allclose(np.asarray(geo[k], float), np.asarray(ref_geo[k], float), atol=1e-6), ('camera geometry differs', st, cam, k)
    u = CM.cycles_ports(SS[st], geo, [np.load(d / f'{tag}_{st}_{cam}_spp{spp}_s{seed}.npy')])['amended']
    assert np.array_equal(u['sel'], REF[st][cam]['amended']['sel']), ('mask differs', st, cam)
    return u['mean']


def fnto_ports(st, arm, r, cam, mode='nee_ball'):
    p = (S.OUT / 'runs' if r == 0 else INH) / f'{st}_{arm}_r{r}.npz'
    z = np.load(p); key = CM.dirkey(SD[st]['cams'][cam]['dir'])
    return A.pred(z, key, mode)


def face_stats(st, preds, arm, mode):
    """preds: {cam: [port arrays]} -> per primary face group record."""
    rows = []
    for cam, face in FACE.items():
        u = REF[st][cam]['amended']
        g = [x for x in A.group_stats(preds[cam], u, u['sel'], LAB[st], cam, mode, arm) if x['group'] == face]
        assert len(g) == 1, (st, cam, face)
        g = g[0]; g['cam'] = cam; rows.append(g)
    return rows


def summarize(rows):
    ruled = np.array([r['ruled'] for r in rows]); passed = np.array([r['passed'] for r in rows]); bias = np.array([r['relative_bias'] for r in rows])
    l2 = np.array([r['relative_L2'] for r in rows])
    return dict(M1_passed=int(passed.sum()), ruled=int(ruled.sum()), M2_max_abs_bias=float(np.abs(bias[ruled]).max()),
                M3_rel_L2_max_per_channel=l2.max(0).tolist(),
                faces={r['cam'] + '|' + r['group']: dict(bias=r['relative_bias'].tolist(), se=r['se_total'].tolist(), passed=r['passed'].tolist(),
                                                         ruled=r['ruled'].tolist(), rel_L2=r['relative_L2'].tolist()) for r in rows})


def change_stats(prest, pdent):
    """Stationary receiver lower:top, gapdown, amended: arm dent-rest per-port change vs 4-seed reference change (WL56 definition)."""
    r, d = REF['rest']['gapdown']['amended'], REF['dent']['gapdown']['amended']; lab = LAB['rest']
    mask = r['sel'] & d['sel'] & (lab == 'lower:top'); dm = (d['seeds'][:, mask] - r['seeds'][:, mask]).mean(0); pm = pdent[mask] - prest[mask]
    out = []
    for c in range(3):
        x, y = dm[:, c], pm[:, c]; xx = x - x.mean(); yy = y - y.mean()
        out.append(dict(slope=float(xx @ yy / (xx @ xx)), corr=float(xx @ yy / math.sqrt((xx @ xx) * (yy @ yy))) if yy @ yy > 0 else None,
                        rel_change_L2_error=float(np.linalg.norm(y - x) / np.linalg.norm(x))))
    return dict(n_ports=int(mask.sum()), channels=out)


def ref_floor(st):
    out = {}
    for cam, face in FACE.items():
        u = REF[st][cam]['amended']; m = u['sel'] & (LAB[st] == face); s = u['seeds'][:, m]
        d = (s[:2].mean(0) - s[2:].mean(0)) / 2
        out[cam + '|' + face] = (np.linalg.norm(d, axis=0) / np.linalg.norm(s.mean(0), axis=0)).tolist()
    return out


def cy_timing():
    recs = [json.loads(l) for l in open(LAD / 'timings.jsonl')]; agg = {}
    for st in STATES:
        for spp in LADDER:
            q = [x for x in recs if x['tag'] == 'ladder' and x['state'] == st and x['spp'] == spp]
            agg[f'{st}|{spp}'] = dict(update_seconds=[x['update_seconds'] for x in q], median=float(np.median([x['update_seconds'] for x in q])),
                                     build_seconds=[x['build_seconds'] + x['reset_seconds'] for x in q],
                                     render_seconds={c: [x['cams'][c]['render_seconds'] for x in q] for c in FACE})
    agg['warmup'] = [x['update_seconds'] for x in recs if x['tag'] == 'warmup']
    return agg, recs


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = dict(protocol='baseline_sup67_v1', reference_floor={st: ref_floor(st) for st in STATES})
    # ---- ReST-GI oracle arms: single-realization (r0 re-executed here, r1 inherited WL56) and two-replicate forms
    fn = {}
    for st, arms in (('rest', ('S',)), ('dent', ('F', 'S', 'T')), ('rigid', ('S',))):
        for arm in arms:
            for mode in ('nee_ball', 'bin'):
                per = {}
                for r in (0, 1):
                    if not ((S.OUT / 'runs' if r == 0 else INH) / f'{st}_{arm}_r{r}.npz').exists(): continue
                    per[f'r{r}'] = summarize(face_stats(st, {c: [fnto_ports(st, arm, r, c, mode)] for c in FACE}, arm, mode))
                if len(per) == 2:
                    per['two_replicate'] = summarize(face_stats(st, {c: [fnto_ports(st, arm, r, c, mode) for r in (0, 1)] for c in FACE}, arm, mode))
                fn[f'{st}|{arm}|{mode}'] = per
    res['fnto'] = fn
    # determinism: re-executed r0 vs WL56 r0 outputs
    det = {}
    for st, arm in (('dent', 'F'), ('dent', 'S'), ('dent', 'T'), ('rest', 'S'), ('rigid', 'S')):
        a, b = np.load(S.OUT / 'runs' / f'{st}_{arm}_r0.npz'), np.load(INH / f'{st}_{arm}_r0.npz')
        det[f'{st}|{arm}'] = {k: float(np.abs(a[k] - b[k]).sum() / max(np.abs(b[k]).sum(), 1e-30)) for k in a.files if k in b.files}
    res['reexecution_vs_wl56_r0_relative_L1'] = det
    # WL56 two-replicate reproduction from the inherited runs (both realizations WL56)
    wl = json.load(open(INH.parent / 'analysis.json')); repro = []
    for cam, face in FACE.items():
        for t in wl['fidelity']['dent'][cam]['amended']:
            if t['group'] == face and t['arm'] == 'S' and t['mode'] == 'nee_ball':
                ps = [A.pred(np.load(INH / f'dent_S_r{r}.npz'), CM.dirkey(SD['dent']['cams'][cam]['dir']), 'nee_ball') for r in (0, 1)]
                u = REF['dent'][cam]['amended']
                mine = [x for x in A.group_stats(ps, u, u['sel'], LAB['dent'], cam, 'nee_ball', 'S') if x['group'] == face][0]
                repro.append(float(np.abs(np.asarray(t['relative_bias']) - mine['relative_bias']).max()))
    res['wl56_reproduction_max_abs_bias_diff'] = max(repro)
    # ---- Cycles Full Recompute ladder
    cy = {}; ports = {}
    for st in STATES:
        for li, spp in enumerate(LADDER):
            reps = []
            for r in (0, 1):
                pp = {c: cy_ports('ladder', LAD, st, c, spp, seed_of(st, li, r)) for c in FACE}; ports[(st, spp, r)] = pp; reps.append(pp)
                cy[f'{st}|{spp}|r{r}'] = summarize(face_stats(st, {c: [pp[c]] for c in FACE}, 'CY', 'cycles'))
            cy[f'{st}|{spp}|two_replicate'] = summarize(face_stats(st, {c: [p[c] for p in reps] for c in FACE}, 'CY', 'cycles'))
    res['cycles'] = cy
    # ---- stationary receiver signed change
    ch = {}
    for arm in ('F', 'S', 'T'):
        for r in (0, 1):
            ch[f'{arm}|r{r}|nee_ball'] = change_stats(fnto_ports('rest', 'S', r, 'gapdown'), fnto_ports('dent', arm, r, 'gapdown'))
            ch[f'{arm}|r{r}|bin'] = change_stats(fnto_ports('rest', 'S', r, 'gapdown', 'bin'), fnto_ports('dent', arm, r, 'gapdown', 'bin'))
    for spp in LADDER:
        for r in (0, 1):
            ch[f'CY|{spp}|r{r}'] = change_stats(ports[('rest', spp, r)]['gapdown'], ports[('dent', spp, r)]['gapdown'])
    res['change'] = ch
    # ---- RNA-native bounce/clamp check
    bo = {}
    for tag in ('bounce', 'bouncectl'):
        for cam in ('top', 'bot'):
            u = REF['dent'][cam]['amended']; geo = json.load(open(BOU / f'ccam_{tag}_dent.json'))[cam]
            p = CM.cycles_ports(SS['dent'], geo, [np.load(BOU / f'{tag}_dent_{cam}_spp1024_s679001.npy')])['amended']['mean']
            g = [x for x in A.group_stats([p], u, u['sel'], LAB['dent'], cam, 'cycles', tag) if x['group'] == FACE[cam]][0]
            bo[f'{tag}|{cam}'] = dict(bias=g['relative_bias'].tolist(), se=g['se_total'].tolist(), passed=g['passed'].tolist(), mean=g['prediction_mean'].tolist())
    for cam in ('top', 'bot'):
        bo[f'paired_ratio|{cam}'] = (np.asarray(bo[f'bounce|{cam}']['mean']) / np.asarray(bo[f'bouncectl|{cam}']['mean']) - 1).tolist()
    res['rna_native_bounce_check'] = bo
    # ---- timing and quality matching
    tim, recs = cy_timing(); res['cycles_timing'] = tim
    runs = {p.stem: json.load(open(p)) for p in sorted((S.OUT / 'runs').glob('*.json'))}
    res['fnto_timing'] = {k: dict(total_update_seconds=v['total_update_seconds'], package_seconds=v['package_seconds'], shared=v['update_shared_seconds'],
                                  timings=v['timings'], gpu_processes_before=v['gpu_processes_before'], gpu_allocated_peak=v['gpu_allocated_peak'],
                                  packages={c: dict(recomputed=p['recomputed'], content_hash=p['content_hash']) for c, p in v['packages'].items()})
                          for k, v in runs.items()}
    if (S.OUT / 'resident_rigid.json').exists():
        rr = json.load(open(S.OUT / 'resident_rigid.json'))
        res['fnto_timing']['rigid_S_resident'] = dict(updates=[x['meta']['update_shared_seconds'] for x in rr['records']], initial=rr['initial_cache_load_and_check'] + rr['initial_CSR'])
    qm = {}
    for arm in ('S', 'F'):
        key = f'dent|{arm}|nee_ball'
        a = res['fnto'][key]; worst = dict(M1=min(a[r]['M1_passed'] for r in ('r0', 'r1')), M2=max(a[r]['M2_max_abs_bias'] for r in ('r0', 'r1')),
                                            M3=np.max([a[r]['M3_rel_L2_max_per_channel'] for r in ('r0', 'r1')], 0).tolist())
        mode = key.split('|')[2]
        worst['M4'] = np.max([[c['rel_change_L2_error'] for c in ch[f'{arm}|r{r}|{mode}']['channels']] for r in (0, 1)], 0).tolist()
        match = None
        for spp in LADDER:
            ok = True
            for r in (0, 1):
                c = cy[f'dent|{spp}|r{r}']; m4 = [x['rel_change_L2_error'] for x in ch[f'CY|{spp}|r{r}']['channels']]
                ok &= c['M1_passed'] >= worst['M1'] and c['M2_max_abs_bias'] <= worst['M2'] and all(np.asarray(c['M3_rel_L2_max_per_channel']) <= np.asarray(worst['M3'])) and all(np.asarray(m4) <= np.asarray(worst['M4']))
            if ok: match = spp; break
        qm[arm] = dict(arm_worst_realization=worst, cycles_matched_spp=match)
    contract = None
    for spp in LADDER:
        if all(cy[f'dent|{spp}|r{r}']['M1_passed'] == 12 for r in (0, 1)): contract = spp; break
    qm['contract_12_of_12_cycles_spp'] = contract
    s_dent = runs['dent_S_r0']['total_update_seconds']; f_dent = runs['dent_F_r0']['total_update_seconds']
    for arm, t_arm in (('S', s_dent), ('F', f_dent)):
        spp = qm[arm]['cycles_matched_spp']
        if spp is not None:
            t_cy = tim[f'dent|{spp}']['median']; qm[arm].update(arm_update_seconds=t_arm, cycles_update_seconds=t_cy, cycles_over_arm=t_cy / t_arm,
                                                               class_='C' if t_cy < t_arm else 'B')
    if contract is not None: qm['contract_cycles_update_seconds'] = tim[f'dent|{contract}']['median']
    res['quality_matching'] = qm
    A_ = S.CM.jclean if hasattr(S.CM, 'jclean') else (lambda x: x)
    json.dump(A_(res), open(OUT / 'w1_analysis.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    with open(OUT / 'w1_dent_quality.csv', 'w') as f:
        f.write('arm,realization,M1_passed,ruled,M2_max_abs_bias,M3_R,M3_G,M3_B\n')
        for k, v in list(res['cycles'].items()) + [(k + '|' + r, v[r]) for k, vv in res['fnto'].items() for r, v in vv.items()]:
            if not k.startswith('dent'): continue
            f.write(','.join([k, '', str(v['M1_passed']), str(v['ruled']), f"{v['M2_max_abs_bias']:.6f}"] + [f'{x:.6f}' for x in v['M3_rel_L2_max_per_channel']]) + '\n')
    print(json.dumps(A_(dict(quality_matching=qm, wl56_repro=res['wl56_reproduction_max_abs_bias_diff'])), indent=1, default=str))


if __name__ == '__main__':
    main(); sys.stdout.flush(); os._exit(0)
