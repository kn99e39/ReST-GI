"""WL114A port qualification on LR3 (Mitsuba 3.6.4 venv).

  python qual114a.py OUT [a2] [deep] [field]
a2    : WL102 gate A2 re-run with the ported tracer (D = 6, scene area light, seeds 102001 / 102002, 1,024 spp per half, b = 8):
        (i) cross-version equivalence with the ORIGINAL pt102 result recorded in experiments/ecro_lr3_102/gate_a2.json
            (same seeds, same estimator) and (ii) the frozen WL102 A2 criterion vs Ref-7 (means within 3 %, RMSE <= 1.5 x noise).
deep  : deep camera pass events 1..6 == the D = 6 render (same seeds), and full depth vs Ref-65.
field : deep field (probe) D6 part == field_components (same seeds).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt114a as P  # noqa: E402

XML = r'C:\Users\dna10\TransGI-Reproduction-data\scenes\living-room-3\scene_transgi.xml'
REFD = Path(r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference')
ORIG = json.loads((HERE.parent / 'ecro_lr3_102' / 'gate_a2.json').read_text())


def halves_stats(halves, r, keys):
    res = {}
    for k in keys:
        a, b = halves[0][k], halves[1][k]
        ours = 0.5 * (a + b)
        so = float(np.sqrt(((a - b) ** 2).mean()) / 2); sr = float(np.sqrt(((r[k + '_A'] - r[k + '_B']) ** 2).mean()) / 2)
        rmse = float(np.sqrt(((ours - r[k]) ** 2).mean())); comb = float(np.hypot(so, sr))
        mo, mr = ours.mean((0, 1)), r[k].mean((0, 1)); rel = ((mo - mr) / mr).tolist()
        res[k] = {'mean_ours': mo.tolist(), 'mean_ref': mr.tolist(), 'mean_rel_diff': rel, 'rmse': rmse, 'noise_ours': so, 'noise_ref': sr,
                  'rmse_over_noise': rmse / comb, 'pass': bool(max(abs(x) for x in rel) <= 0.03 and rmse <= 1.5 * comb)}
    return res


def main(out, steps):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ctx = P.Ctx(P.load_scene(XML)); L = P.SceneLights(64)
    res = json.loads((out / 'qual.json').read_text()) if (out / 'qual.json').exists() else {}
    res['N'] = ctx.N
    if 'a2' in steps:
        halves = []; t0 = time.perf_counter()
        for h, seed in enumerate((102001, 102002)):
            acc, T = ctx.render(L, 1024, seed, 'cam', b=8)
            W, H = T['res']; o = P.pack(acc, W, H, 1024)
            halves.append({'di': o['e1'], 'gi': o['e2'] + o['h'], 'full': o['e1'] + o['e2'] + o['h'], 'e1': o['e1'], 'e2': o['e2'], 'h': o['h']})
        np.savez_compressed(out / 'a2_halves.npz', **{f'{k}_{i}': h[k] for i, h in enumerate(halves) for k in ('e1', 'e2', 'h')})
        st = halves_stats(halves, np.load(REFD / 'static_V0_d7.npz'), ('gi', 'full', 'di'))
        cross = {}
        for k in ('gi', 'full', 'di'):
            mo = np.array(st[k]['mean_ours']); mo0 = np.array(ORIG['comp'][k]['mean_ours'])
            se_mean = st[k]['noise_ours'] / np.sqrt(W * H)            # per-channel order of the image-mean SE (pixel noise / sqrt(px))
            cross[k] = {'ported_mean': mo.tolist(), 'original_mean': mo0.tolist(), 'rel_diff': ((mo - mo0) / mo0).tolist(),
                        'abs_diff_over_mean_se': (np.abs(mo - mo0) / max(se_mean, 1e-12)).tolist(),
                        'orig_rmse_over_noise': ORIG['comp'][k]['rmse_over_noise'], 'ported_rmse_over_noise': st[k]['rmse_over_noise']}
        res['a2'] = {'vs_ref7': st, 'A2_pass': bool(st['gi']['pass'] and st['full']['pass']), 'cross_version_vs_original_pt102': cross,
                     'seconds': time.perf_counter() - t0}
        print('A2', json.dumps({k: (v['rel_diff'], v['ported_rmse_over_noise'], v['orig_rmse_over_noise']) for k, v in cross.items()}), res['a2']['A2_pass'], flush=True)
    if 'deep' in steps:
        t0 = time.perf_counter()
        d6, T = ctx.render(L, 64, 114101, 'cam', b=8)
        dp, Td = ctx.render_deep(L, 64, 114101, 96, b=8)
        rel = lambda a, b: float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))
        res['deep_identity_events1to6'] = {k: rel(dp[f'{n}{c}'], d6[f'{n}{c}']) for n in ('e1', 'e2', 'h') for c in range(3) for k in [f'{n}{c}']}
        halves = []
        for h, seed in enumerate((114111, 114112)):
            acc, Td = ctx.render_deep(L, 1024, seed, 96, b=8)
            W, H = Td['res']; o = P.pack(acc, W, H, 1024)
            gi = sum(o[n] for n in ('e2', 'h') + P.DEEP if n in o)
            halves.append({'gi': gi, 'full': o['e1'] + gi, 'di': o['e1']})
            np.savez_compressed(out / f'deep96_h{h}.npz', **o)
        res['deep96_vs_ref65'] = halves_stats(halves, np.load(REFD / 'static_V0.npz'), ('gi', 'full', 'di'))
        res['deep96_seconds'] = time.perf_counter() - t0
        print('DEEP', json.dumps({k: (v['mean_rel_diff'], v['rmse_over_noise']) for k, v in res['deep96_vs_ref65'].items()}),
              max(res['deep_identity_events1to6'].values()), flush=True)
    if 'field' in steps:
        Lp = P.Lights([{'pos': [1.0, 1.2, 1.5], 'rgb': [30.0] * 3}])
        a, b_, cnt = ctx.field_components(Lp, 16, 114201, b=16)
        fd, cnt2 = ctx.field_deep(Lp, 16, 114201, 23, b=16)
        rel = lambda x, y: float(np.abs(x - y).sum() / max(np.abs(y).sum(), 1e-30))
        res['field_identity'] = {'f2': rel(fd['f2'], a), 'fh': rel(fd['fh'], b_), 'count': rel(cnt2, cnt),
                                 'deep_share_over_d6': float(sum(fd[n].sum() for n in P.DEEP if n in fd) / (fd['f2'].sum() + fd['fh'].sum()))}
        print('FIELD', json.dumps(res['field_identity']), flush=True)
    json.dump(res, open(out / 'qual.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or ['a2', 'deep', 'field'])
    sys.stdout.flush(); os._exit(0)
