"""WL115 Stage D / E readouts of the learned arms (frozen protocol stage_B_frozen.stage_D_E).

  python evalD115.py streams ROOT [PREFIX ...]   readout-only streams (reference seeds) of B1, B2, CAR1, CAR2 (+ stale_B1 / stale_CAR1 in
                                                 sequences) for HELD_SEEN, FRESH and SELECTION views; per-state tables saved for sequences
  python evalD115.py rebuild ROOT                fresh-process rebuild of every sequence state's CAR1 / B1 tables (incl. the CAR support)
A whole sequence stays in one process (stale tables). No reference, camera or radiance input enters any learned table.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / 'ecro_fd114', HERE.parent / 'ecro_sss113', HERE.parent / 'ecro86', HERE.parent / 'ecro_material100'):
    sys.path.insert(0, str(d))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402
import candidate115 as CA  # noqa: E402
import geomdesc115 as GD  # noqa: E402

PT, dr = T.PT, T.dr
SPP = C.dv(4096, 64)
N, NB = 1270, T.NB
CHILD = CA.child_index_maps()


def arms(root):
    w = C.WL114_ROOT / 'stageC'
    return {'B1': w / 'B1' / 'model.pt', 'B2': w / 'B2' / 'model.pt', 'CAR1': root / 'stageC' / 'CAR1' / 'model.pt', 'CAR2': root / 'stageC' / 'CAR2' / 'model.pt'}


def learned_dense(path, geom, light_cfg):
    import eval113 as E3
    t_, _ = E3.learned_table(path, 'dense', geom, light_cfg)
    return t_


def learned_car(path, geom, light_cfg):
    """(T4 [N,16,NB,3], T16 [P,256,NB,3], w [N,16], pids) of a trained ECRO-CAR at the current geometry and source."""
    import torch
    import model86 as M
    import model115 as MC
    ctx = T.Ctx115(geom, 'C')
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    src = torch.as_tensor(ctx.source(PT.Lights(light_cfg)), device='cuda').unsqueeze(0)
    desc = ctx.vhs()
    gs.sup = torch.as_tensor(np.arange(N), device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
    w, pids, m = CA.support(ctx)
    cd = np.concatenate([GD.vhs_k(ctx, 16, pids), GD.prox_other(ctx, 16, pids=pids, ells=(CA.ELL,))], -1)
    loc = -np.ones(N, np.int64); loc[pids] = np.arange(len(pids))
    ap, asub = np.nonzero(w > 0)
    gs.act_pid = torch.as_tensor(ap, device='cuda'); gs.act_sub = torch.as_tensor(asub, device='cuda')
    gs.cdesc = torch.as_tensor(cd[loc[ap][:, None], CHILD[asub]], device='cuda')
    model = MC.ECROCAR().cuda()
    model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
    with torch.no_grad():
        f = model.factors(gs)
        B, Bs, Bc = model.outputs_car(f, gs, src)
    T4 = Bs[0].float().cpu().numpy()
    T16 = T.upsample(T4[pids], 4, 16).copy()
    Bc = Bc[0].float().cpu().numpy()
    for a_, (p, s) in enumerate(zip(ap, asub)):
        T16[loc[p], CHILD[s]] = Bc[a_]
    del ctx
    return T4.astype(np.float32), T16.astype(np.float32), w, pids


def all_views(root):
    """tag -> dict(key, cls, g, l, cam, seeds, seq) over HELD_SEEN + FRESH (stageB115) and SELECTION (stageA115)."""
    import stageB115 as SB
    import stageA115 as SA
    out = {}
    for tag, v in SB.eval_views().items():
        out[tag] = dict(v)
    for tag, (key, dg, g, l, cam, seeds) in SA.views().items():
        out[tag] = {'key': key, 'cls': 'SELECTION', 'g': g, 'l': l, 'cam': cam, 'seeds': seeds}
    seqs = list(C.sequences())
    for v in out.values():
        v['seq'] = next((s for s in seqs if v['key'].startswith(s + '_')), None)
    return out


def streams(root, only=None):
    d = root / 'stageD'; d.mkdir(parents=True, exist_ok=True)
    A = arms(root)
    V = all_views(root)
    keys = []
    for tag, v in V.items():
        if v['key'] not in keys:
            keys.append(v['key'])
    prev = {}
    for key in keys:
        tags = [t for t, v in V.items() if v['key'] == key and (not only or any(t.startswith(o) for o in only))]
        if not tags:
            prev = {}
            continue
        v0 = V[tags[0]]; seq = v0['seq']
        if all((d / f'stream_{t}_{j}.npz').exists() for t in tags for j in range(2)):
            if seq is not None:              # keep the stale table chain consistent on resume
                prev = {'seq': seq, 't': key, 'B1': np.load(d / f'table_{key}_B1.npy'), 'CAR1': dict(np.load(d / f'table_{key}_CAR1.npz'))}
            continue
        t0 = time.perf_counter()
        tabs = {}
        for a in ('B1', 'B2'):
            tabs[a] = T.pc_uniform(learned_dense(A[a], v0['g'], v0['l']), 4)
        cur = {}
        for a in ('CAR1', 'CAR2'):
            T4, T16, w, pids = learned_car(A[a], v0['g'], v0['l'])
            tabs[a] = CA.rep(T4, T16, w, pids)
            cur[a] = {'T4': T4, 'T16': T16, 'w': w, 'pids': pids}
        if seq is not None:
            Bt = learned_dense(A['B1'], v0['g'], v0['l']); np.save(d / f'table_{key}_B1.npy', Bt)
            np.savez(d / f'table_{key}_CAR1.npz', **cur['CAR1'])
            if prev.get('seq') == seq:
                tabs['stale_B1'] = T.pc_uniform(prev['B1'], 4)
                p_ = prev['CAR1']; tabs['stale_CAR1'] = CA.rep(p_['T4'], p_['T16'], p_['w'], p_['pids'])
            prev = {'seq': seq, 't': key, 'B1': Bt, 'CAR1': cur['CAR1']}
        else:
            prev = {}
        tg = time.perf_counter() - t0
        ctx = T.Ctx115(v0['g'], 'C'); light = PT.Lights(v0['l'])
        for tag in tags:
            v = V[tag]
            for j, s in enumerate(v['seeds']):
                p = d / f'stream_{tag}_{j}.npz'
                if p.exists():
                    continue
                t1 = time.perf_counter()
                img = ctx.readout_multi(light, dict(v['cam'], res=C.RES), C.RES, SPP, s, tabs, b=64)
                rec = {'ro_e1': img['e1'].astype(np.float32), 'ro_e2': img['e2'].astype(np.float32), **{f'hl|{n}': img[n].astype(np.float32) for n in tabs}}
                tmp = p.with_name(p.stem + '.tmp.npz')
                np.savez_compressed(tmp, meta=json.dumps({'tag': tag, 'seed': s, 'reps': list(tabs), 'tables_s': tg, 'seconds': time.perf_counter() - t1}), **rec)
                os.replace(tmp, p)
                print('STREAMD', tag, j, round(tg, 1), round(time.perf_counter() - t1, 1), flush=True)
        del ctx


def rebuild(root):
    d = root / 'stageD'; A = arms(root); res = {}
    for f in sorted(d.glob('table_*_CAR1.npz')):
        key = f.name[len('table_'):-len('_CAR1.npz')]
        V = {v['key']: v for v in all_views(root).values()}
        v = V[key]
        T4, T16, w, pids = learned_car(A['CAR1'], v['g'], v['l'])
        z = np.load(f)
        Bt = learned_dense(A['B1'], v['g'], v['l']); Bref = np.load(d / f'table_{key}_B1.npy')
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
        res[key] = {'CAR1_T4_max_rel': rel(T4, z['T4']), 'CAR1_T16_max_rel': rel(T16, z['T16']) if T16.shape == z['T16'].shape else float('inf'),
                    'CAR1_w_max_abs': float(np.abs(w - z['w']).max()), 'CAR1_pids_equal': bool(np.array_equal(pids, z['pids'])), 'B1_max_rel': rel(Bt, Bref)}
        print('REBUILD', key, res[key], flush=True)
    C.jd(res, d / 'rebuild_check.json')


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'streams': streams, 'rebuild': lambda r, o=None: rebuild(r)}[cmd](Path(root), rest or None) if cmd == 'streams' else rebuild(Path(root))
    sys.stdout.flush(); os._exit(0)
