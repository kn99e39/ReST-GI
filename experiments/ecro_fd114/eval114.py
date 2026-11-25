"""WL114 oracle tables and evaluation streams.

  python eval114.py tables ROOT [STATE_KEY ...]   Stage B deep 4x4 constructions -> U_D6 / U_F / S4_D6 / S4_F per evaluation state
  python eval114.py streamsB ROOT [VIEW_KEY ...]  Stage B streams: deep physical pass to K* + paired readout of the oracle tables
  python eval114.py streamsD ROOT [VIEW_KEY ...]  Stage D readout-only streams (Stage B seeds) of the learned arms (+ stale tables)
  python eval114.py rebuild ROOT                  fresh-process rebuild of the A1r / B1 tables at every sequence state
Selections are prefixes of state keys (F0..F4, N1..N4, <SEQ>_<t>) / view keys (<state>_<view>); a whole sequence must stay in one process (stale tables).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common114 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
N, NB, NS = 1270, T.NB, T.NS
SPP = C.dv(4096, 64)
NPROP = C.dv(1024, 32)
SO = C.SO
STATIC = ['F0', 'F1', 'F2', 'F3', 'F4', 'N1', 'N2', 'N3', 'N4']
SEQS = list(C.sequences())
ORACLES = ('S4_D6', 'S4_F', 'U_D6', 'U_F')

import atexit
from concurrent.futures import ThreadPoolExecutor
_WRITER = ThreadPoolExecutor(max_workers=2)
_PENDING = []
atexit.register(lambda: [f.result() for f in _PENDING])


def write_npz(path, **arrays):
    def _w():
        tmp = path.with_name(path.stem + '.tmp.npz')
        np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
    _PENDING.append(_WRITER.submit(_w))


def flush_writes():
    for f in _PENDING:
        f.result()
    _PENDING.clear()


def kstar(root):
    return json.loads((root / 'analysis' / 'stageA_qualification.json').read_text())['SSS']['K_star']


def states():
    """state key -> (geometry, light cfg, [(view name, camera, [seed j0, seed j1])], construction seeds)"""
    out = {}
    for fi, f in enumerate(STATIC):
        g, l = C.fixture(f); vs = C.views(f)
        names = list(vs) if f in C.FIXTURES_113 else list(C.NEW_VIEWS)
        vlist = []
        for vi, (vn, cam) in enumerate(vs.items()):
            if vn not in names:
                continue
            if f in C.FIXTURES_113:
                seeds = [11310000 + 100 * fi + 10 * vi + j + SO for j in range(2)]
            else:
                k = fi - 5
                seeds = [11420000 + 100 * k + 10 * vi + j + SO for j in range(2)]
            vlist.append((vn, cam, seeds))
        out[f] = (g, l, vlist, [11410000 + 100 * fi + r + SO for r in range(2)])
    for si, sname in enumerate(SEQS):
        q = C.sequences()[sname]; cams = C.seq_views(q)
        for t, g in enumerate(q['states']):
            vlist = [(vn, cams[vn], [11350000 + 1000 * si + 10 * t + 2 * vi + j + SO for j in range(2)]) for vi, vn in enumerate(q['views'])]
            out[f'{sname}_{t}'] = (g, q['light'], vlist, [11411000 + 100 * si + t + SO])
    return out


def view_setup(ctx, geom, light_cfg, cam, out_dir, tag):
    gp = out_dir / f'gbuffer_{tag}.npz'
    if not gp.exists():
        gb = ctx.gbuffer(dict(cam, res=C.RES), C.RES)
        gb['coverage'] = T.T113.silhouette_coverage(ctx, dict(cam, res=C.RES))
        np.savez_compressed(gp, **gb)


def table_path(root, key, r):
    return root / 'stageB' / 'tables' / f'oracle_{key}_r{r}.npz'


def tables(root, only=None):
    K = kstar(root); d = root / 'stageB' / 'tables'; d.mkdir(parents=True, exist_ok=True)
    for key, (g, l, vl, seeds) in states().items():
        if only and not any(key.startswith(o) for o in only):
            continue
        ctx = None
        for r, s in enumerate(seeds):
            p = table_path(root, key, r)
            if p.exists():
                continue
            ctx = ctx or T.Ctx114(g, 'C')
            t0 = time.perf_counter()
            acc, dg = ctx.construct_deep(PT.Lights(l), NPROP, s, K - 1, b=8)
            ht = T.high_tables(acc, K)
            w = acc['U_w'].reshape(N, NS, NB)
            meta = {'state': key, 'seed': s, 'K_star': K, 'n': NPROP, 'seconds': time.perf_counter() - t0, 'diag': dg}
            write_npz(p, meta=json.dumps(meta), w=w.astype(np.float32), **{k: v for k, v in ht.items()})
            print('TABLE', key, r, round(meta['seconds'], 1), flush=True)
        del ctx
    flush_writes()
    (d / f'tables_{"_".join(only) if only else "all"}.done').write_text('DONE')


def oracle_dense(root, key):
    nrep = len(states()[key][3])
    zs = [np.load(table_path(root, key, r)) for r in range(nrep)]
    out = {}
    for nm in ORACLES:
        m = np.mean([z[nm] for z in zs], 0)
        out[nm] = T.T113.to_dense16(m, 'coarse') if nm.startswith('U_') else m
    return out


def readout(ctx, light, cam, seed, tabs, rec):
    tl = [(n_, PT.fl(np.ascontiguousarray(t_, np.float32).ravel()), 'sub') for n_, t_ in tabs]
    img, _, _, _ = ctx.cam_pass(light, cam, C.RES, SPP, seed, 'ecro', tables=tl, b=64)
    dr.flush_malloc_cache()
    rec['ro_e1'] = img['e1'].astype(np.float32); rec['ro_e2'] = img['e2'].astype(np.float32)
    for n_, _ in tabs:
        rec[f'hl|{n_}'] = img[n_].astype(np.float32)


def streamsB(root, only=None):
    K = kstar(root); out = root / 'stageB'; out.mkdir(parents=True, exist_ok=True)
    for key, (g, l, vl, _) in states().items():
        todo = [(vn, cam, s) for vn, cam, s in vl if not only or any(f'{key}_{vn}'.startswith(o) for o in only)]
        if not todo:
            continue
        light = PT.Lights(l); ctx = T.Ctx114(g, 'C'); orc = None
        for vn, cam, seeds in todo:
            tag = f'{key}_{vn}'; cam = dict(cam, res=C.RES)
            view_setup(ctx, g, l, cam, out, tag)
            for j, s in enumerate(seeds):
                p = out / f'stream_{tag}_{j}.npz'
                if p.exists():
                    continue
                orc = orc or oracle_dense(root, key)
                t0 = time.perf_counter(); rec = {}
                img, dg = ctx.deep_pass(light, cam, C.RES, SPP, s, K, b=64)          # same launch partition as the readout (pairing) and WL113
                for k_, v in img.items():
                    rec[f'phys_{k_}'] = v.astype(np.float32)
                tp = time.perf_counter() - t0
                readout(ctx, light, cam, s, list(orc.items()), rec)
                meta = {'seed': s, 'spp': SPP, 'K_star': K, 'phys_seconds': tp, 'seconds': time.perf_counter() - t0, 'diag': dg, 'tables': list(orc)}
                write_npz(p, meta=json.dumps(meta), **rec)
                print('STREAM_B', tag, j, round(meta['seconds'], 1), flush=True)
        del ctx
    flush_writes()
    (out / f'streams_{"_".join(only) if only else "all"}.done').write_text('DONE')


def learned_arms(root):
    w = C.REPO / 'results' / 'ecro_sss113' / 'primary' / 'stageC'
    c = root / 'stageC'
    return {'A': (w / 'CAND' / 'model.pt', 'dense'), 'O_U': (w / 'O_U' / 'model.pt', 'coarse'),
            'A1r': (c / 'A1r' / 'model.pt', 'dense'), 'B1': (c / 'B1' / 'model.pt', 'dense'),
            'A2': (c / 'A2' / 'model.pt', 'dense'), 'B2': (c / 'B2' / 'model.pt', 'dense'), 'OB1': (c / 'OB1' / 'model.pt', 'coarse')}


def learned_tables(root, g, l):
    import eval113 as E3
    tabs = []
    for nm, (p, kind) in learned_arms(root).items():
        t_, _ = E3.learned_table(p, kind, g, l); tabs.append((nm, t_))
    return tabs


def streamsD(root, only=None):
    out = root / 'stageD'; out.mkdir(parents=True, exist_ok=True)
    prev = {}
    for key, (g, l, vl, _) in states().items():
        seq = next((s for s in SEQS if key.startswith(s + '_')), None)
        todo = [(vn, cam, s) for vn, cam, s in vl if not only or any(f'{key}_{vn}'.startswith(o) for o in only)]
        if not todo:
            prev = {} if seq is None else prev
            continue
        tabs = learned_tables(root, g, l)
        cur = {n_: t_ for n_, t_ in tabs}
        if seq is not None:
            for a in ('A1r', 'B1'):
                np.save(out / f'table_{key}_{a}.npy', cur[a])
            t = int(key.rsplit('_', 1)[1])
            if t > 0 and prev.get('seq') == seq:
                tabs += [('stale_A1r', prev['A1r']), ('stale_B1', prev['B1'])]
            prev = {'seq': seq, 'A1r': cur['A1r'], 'B1': cur['B1']}
        light = PT.Lights(l); ctx = T.Ctx114(g, 'C')
        for vn, cam, seeds in todo:
            tag = f'{key}_{vn}'; cam = dict(cam, res=C.RES)
            for j, s in enumerate(seeds):
                p = out / f'stream_{tag}_{j}.npz'
                if p.exists():
                    continue
                t0 = time.perf_counter(); rec = {}
                readout(ctx, light, cam, s, tabs, rec)
                write_npz(p, meta=json.dumps({'seed': s, 'spp': SPP, 'seconds': time.perf_counter() - t0, 'tables': [n_ for n_, _ in tabs]}), **rec)
                print('STREAM_D', tag, j, round(time.perf_counter() - t0, 1), flush=True)
        del ctx
    flush_writes()
    (out / f'streams_{"_".join(only) if only else "all"}.done').write_text('DONE')


def rebuild(root, only=None):
    import eval113 as E3
    out = root / 'stageD'; res = {}
    arms = learned_arms(root)
    for key, (g, l, vl, _) in reversed(list(states().items())):
        if not any(key.startswith(s + '_') for s in SEQS):
            continue
        for a in ('A1r', 'B1'):
            t_, _ = E3.learned_table(*arms[a], g, l)
            ref = np.load(out / f'table_{key}_{a}.npy')
            res[f'{key}_{a}'] = {'table_max_rel': float(np.abs(t_ - ref).max() / max(np.abs(ref).max(), 1e-30))}
    C.jd(res, out / 'rebuild_check.json')


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'tables': tables, 'streamsB': streamsB, 'streamsD': streamsD, 'rebuild': rebuild}[cmd](Path(root), rest or None)
    flush_writes()
    sys.stdout.flush(); os._exit(0)
