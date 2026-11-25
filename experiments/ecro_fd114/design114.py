"""WL114 geometry-only design check (before the freeze; no radiance): validity of the new fixtures / SEQ_D states, contact-valid
camera regions, and isolation from the D113 training / validation configurations (WL112 exclusion rule)."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402


def conflict(g, e):
    return (np.linalg.norm(np.asarray(g['S']['c']) - np.asarray(e['S']['c'])) < 0.01 and
            all(np.linalg.norm(np.asarray(g[b]['c']) - np.asarray(e[b]['c'])) < 0.05 for b in 'AB'))


def main(out):
    import tracer114 as T
    res = {'fixtures': {}, 'SEQ_D': [], 'isolation': {}}
    train = [C.train_config('train', i)['geom'] for i in range(192)] + [C.train_config('val', i)['geom'] for i in range(16)]
    for f in C.NEW:
        g, l = C.fixture(f)
        ctx = T.Ctx114(g, 'C')
        rec = {'geometry_valid': bool(C.geometry_valid(g, 0.05)), 'light_valid': bool(C.light_valid(l[0]['pos'], g)), 'views': {}}
        for v, cam in C.views(f).items():
            gb = ctx.gbuffer(dict(cam, res=C.RES), C.RES)
            reg, _ = C.masks(gb, g, l[0]['pos'])
            rec['views'][v] = {'contact_valid': C.contact_valid(reg), 'near_px': int(reg['near'].sum()), 'flank_px': int(reg['flank'].sum())}
        rec['train_conflicts'] = int(sum(conflict(g, e) for e in train))
        res['fixtures'][f] = rec; del ctx
    q = C.sequences()['SEQ_D_independent']
    for st, g in enumerate(q['states']):
        ctx = T.Ctx114(g, 'C')
        rec = {'geometry_valid': bool(C.geometry_valid(g, 0.05)), 'light_valid': bool(C.light_valid(q['light'][0]['pos'], g)), 'views': {}}
        for v in q['views']:
            gb = ctx.gbuffer(dict(C.seq_views(q)[v], res=C.RES), C.RES)
            reg, _ = C.masks(gb, g, q['light'][0]['pos'])
            rec['views'][v] = {'contact_valid': C.contact_valid(reg), 'near_px': int(reg['near'].sum()), 'flank_px': int(reg['flank'].sum())}
        rec['train_conflicts'] = int(sum(conflict(g, e) for e in train))
        res['SEQ_D'].append(rec); del ctx
    res['isolation']['any_eval_state_conflict'] = int(sum(conflict(g, e) for g in C.eval_states() for e in train))
    C.jd(res, out)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
    import os; sys.stdout.flush(); os._exit(0)
