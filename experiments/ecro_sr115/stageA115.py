"""WL115 Stage A - spatial error and reconstruction attribution (SELECTION set only).

  python stageA115.py con8   ROOT [KEY ...]   8 x 8 deep U construction of every unique selection geometry (all patches)
  python stageA115.py conF   ROOT [KEY ...]   32 x 32 deep U construction on the diagnostic contact neighbourhood patches
  python stageA115.py capture ROOT [TAG ...]  camera-conditioned (query-weighted) diagnostic capture, 8 x 8 all patches + 32 x 32 contact
  python stageA115.py streams ROOT [TAG ...]  readout-only streams with the WL114 Stage B seeds (paired with the WL114 references)
KEY = selection geometry key (F0..F3, <SEQ>_<t> of the first state with that geometry); TAG = <state key>_<view>.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
N, NB = 1270, T.NB
SPP = C.dv(4096, 64)
N8 = C.dv(1024, 4)
NF = C.dv(1024, 2)
SEED_C8, SEED_CF, SEED_CAP = 11510000, 11511000, 11512000
CAPTURE_VIEWS = ('F0_V0', 'F1_W0', 'F2_W0', 'F3_W0', 'SEQ_A_lift_3_V0', 'SEQ_C_independent_1_W0', 'SEQ_C_independent_3_W0', 'SEQ_B_slide_2_V0')

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


def geoms():
    """sorted unique selection geometries: list of (key, digest, geometry, light)."""
    sel = C.selection_geometries()
    return [(k, d, g, l) for d, (k, g, l) in sorted(sel.items(), key=lambda x: x[1][0])]


def views():
    """selection views: tag -> (state key, geometry digest, geometry, light, camera, [seed j0, seed j1]) with the WL114 seeds."""
    sys.path.insert(0, str(C.REPO / 'experiments' / 'ecro_fd114'))
    import eval114 as EV
    out = {}
    for key, (g, l, vl, _) in EV.states().items():
        if not (key in C.SELECTION_FIX or any(key.startswith(s + '_') for s in C.SELECTION_SEQ)):
            continue
        for vn, cam, seeds in vl:
            out[f'{key}_{vn}'] = (key, C.gdigest(g), g, l, cam, seeds)
    return out


def con8(root, only=None):
    d = root / 'stageA' / 'con8'; d.mkdir(parents=True, exist_ok=True)
    for i, (key, dg, g, l) in enumerate(geoms()):
        if only and key not in only:
            continue
        p = d / f'c8_{dg}.npz'
        if p.exists():
            continue
        ctx = T.Ctx115(g, 'C')
        acc, diag = ctx.construct_k(PT.Lights(l), N8, SEED_C8 + 100 * i + C.SO, 23, 8, b=C.dv(3, 1), verbose=True)
        w = acc['w0'] + acc['w1']
        d6 = np.stack([T.pool(acc[f'D60{c}'] + acc[f'D61{c}'], 8, 4) for c in range(3)], -1)
        write_npz(p, meta=json.dumps({'key': key, 'digest': dg, 'seed': SEED_C8 + 100 * i + C.SO, 'n': N8, 'K': 8, 'diag': diag}),
                  w0=acc['w0'].astype(np.float32), w1=acc['w1'].astype(np.float32),
                  **{f'HF{p_}{c}': acc[f'HF{p_}{c}'].astype(np.float32) for p_ in (0, 1) for c in range(3)}, D6_4=d6.astype(np.float32))
        print('CON8', key, round(diag['seconds'], 1), 'mismatch', diag['mismatch'] / max(diag['counted'], 1), flush=True)
        del ctx, acc
    flush_writes()


def conF(root, only=None):
    d = root / 'stageA' / 'conF'; d.mkdir(parents=True, exist_ok=True)
    for i, (key, dg, g, l) in enumerate(geoms()):
        if only and key not in only:
            continue
        p = d / f'cf_{dg}.npz'
        if p.exists():
            continue
        ctx = T.Ctx115(g, 'C')
        pids = T.contact_patches(ctx.G, g['S']['c'][:2])
        acc, diag = ctx.construct_k(PT.Lights(l), NF, SEED_CF + 100 * i + C.SO, 23, 32, pids=pids, b=C.dv(8, 1), verbose=True)
        write_npz(p, meta=json.dumps({'key': key, 'digest': dg, 'seed': SEED_CF + 100 * i + C.SO, 'n': NF, 'K': 32, 'pids': pids.tolist(), 'diag': diag}),
                  pids=pids, w0=acc['w0'].astype(np.float32), w1=acc['w1'].astype(np.float32),
                  **{f'HF{p_}{c}': acc[f'HF{p_}{c}'].astype(np.float32) for p_ in (0, 1) for c in range(3)})
        print('CONF', key, len(pids), round(diag['seconds'], 1), 'mismatch', diag['mismatch'] / max(diag['counted'], 1), flush=True)
        del ctx, acc
    flush_writes()


def capture(root, only=None):
    d = root / 'stageA' / 'capture'; d.mkdir(parents=True, exist_ok=True)
    V = views()
    for vi, tag in enumerate(CAPTURE_VIEWS):
        if only and tag not in only:
            continue
        p = d / f'cap_{tag}.npz'
        if p.exists():
            continue
        key, dg, g, l, cam, _ = V[tag]
        ctx = T.Ctx115(g, 'C')
        pids = T.contact_patches(ctx.G, g['S']['c'][:2])
        loc = -np.ones(N, np.int64); loc[pids] = np.arange(len(pids))
        t0 = time.perf_counter()
        img, cells = ctx.deep_capture(PT.Lights(l), dict(cam, res=C.RES), C.RES, SPP, SEED_CAP + 10 * vi + C.SO,
                                      {'Q8': (8, None), 'Q32': (32, loc)}, b=16)
        arrays = {f'img_{k}': v.astype(np.float32) for k, v in img.items()}
        for nm, dd in cells.items():
            for k, v in dd.items():
                arrays[f'{nm}_{k}'] = v.astype(np.float32)
        write_npz(p, meta=json.dumps({'tag': tag, 'seed': SEED_CAP + 10 * vi + C.SO, 'spp': SPP, 'pids': pids.tolist(),
                                      'seconds': time.perf_counter() - t0}), pids=pids, **arrays)
        print('CAPTURE', tag, round(time.perf_counter() - t0, 1), flush=True)
        del ctx
    flush_writes()


# ---------------------------------------------------------------------------------------------------- representations
def load_c8(root, dg):
    with np.load(root / 'stageA' / 'con8' / f'c8_{dg}.npz') as z:
        return {k: z[k] for k in z.files}


def load_cf(root, dg):
    with np.load(root / 'stageA' / 'conF' / f'cf_{dg}.npz') as z:
        return {k: z[k] for k in z.files}


def sums(z, parity=None):
    ps = (0, 1) if parity is None else (parity,)
    w = sum(z[f'w{p}'].astype(np.float64) for p in ps)
    L = np.stack([sum(z[f'HF{p}{c}'].astype(np.float64) for p in ps) for c in range(3)], -1)
    return w, L


def oracle_tables(root, dg):
    """U-measure tables of one geometry from the nested 8 x 8 construction and the 32 x 32 contact construction."""
    z8 = load_c8(root, dg); zf = load_cf(root, dg)
    w, L = sums(z8)
    out = {'PC8': T.table_from_sums(w, L, 8), 'PC4': T.table_from_sums(T.pool(w, 8, 4), T.pool(L, 8, 4), 4)}
    for p in (0, 1):
        wp, Lp = sums(z8, p)
        out[f'PC8_h{p}'] = T.table_from_sums(wp, Lp, 8)
    wf, Lf = sums(zf)
    out['F32'] = T.table_from_sums(wf, Lf, 32); out['F16'] = T.table_from_sums(T.pool(wf, 32, 16), T.pool(Lf, 32, 16), 16)
    for p in (0, 1):
        wp, Lp = sums(zf, p)
        out[f'F32_h{p}'] = T.table_from_sums(wp, Lp, 32)
    out['pids'] = zf['pids']
    return out


def query_tables(cap, U8, Uf32):
    """Camera-conditioned (query-weighted) tables Q8 (all patches) / Q4 (pooled) / Q32 / Q16 (contact patches); a cell with no
    query in the capture stream takes the U-measure value (it is then not queried at the capture's density)."""
    def qt(n, t, h, Kk, Ufb):
        T_ = h / np.maximum(t, 1e-30)
        return np.where((n > 0)[..., None] & (t > 0), T_, Ufb).astype(np.float32)
    n8 = cap['Q8_n'].astype(np.float64); t8 = np.stack([cap[f'Q8_t{c}'] for c in range(3)], -1).astype(np.float64)
    h8 = np.stack([cap[f'Q8_h{c}'] for c in range(3)], -1).astype(np.float64)
    nf = cap['Q32_n'].astype(np.float64); tf = np.stack([cap[f'Q32_t{c}'] for c in range(3)], -1).astype(np.float64)
    hf = np.stack([cap[f'Q32_h{c}'] for c in range(3)], -1).astype(np.float64)
    out = {'Q8': qt(n8, t8, h8, 8, U8['PC8'])}
    out['Q4'] = qt(T.pool(n8, 8, 4), T.pool(t8, 8, 4), T.pool(h8, 8, 4), 4, U8['PC4'])
    out['Q32'] = qt(nf, tf, hf, 32, Uf32['F32']); out['Q16'] = qt(T.pool(nf, 32, 16), T.pool(tf, 32, 16), T.pool(hf, 32, 16), 16, Uf32['F16'])
    out['n8'] = n8; out['n32'] = nf
    return out


def representations(root, tag, key, dg, with_capture):
    O = oracle_tables(root, dg)
    pids = O['pids']
    reps = {'PC4': T.pc_uniform(O['PC4'], 4), 'PC8': T.pc_uniform(O['PC8'], 8), 'BL4': T.bl_rep(O['PC4'], 4), 'BL8': T.bl_rep(O['PC8'], 8),
            'R16': T.pc_hybrid(O['PC4'], 4, O['F16'], 16, pids), 'R32': T.pc_hybrid(O['PC4'], 4, O['F32'], 32, pids),
            'R32on8': T.pc_hybrid(O['PC8'], 8, O['F32'], 32, pids)}
    sys.path.insert(0, str(C.REPO / 'experiments' / 'ecro_fd114'))
    import eval114 as EV
    reps['S4F_wl114'] = T.pc_uniform(EV.oracle_dense(C.WL114_ROOT, key)['S4_F'], 4)
    if with_capture:
        reps['PC8_h0'] = T.pc_uniform(O['PC8_h0'], 8); reps['PC8_h1'] = T.pc_uniform(O['PC8_h1'], 8)
        reps['R32_h0'] = T.pc_hybrid(O['PC4'], 4, O['F32_h0'], 32, pids); reps['R32_h1'] = T.pc_hybrid(O['PC4'], 4, O['F32_h1'], 32, pids)
        with np.load(root / 'stageA' / 'capture' / f'cap_{tag}.npz') as z:
            cap = {k: z[k] for k in z.files}
        assert np.array_equal(cap['pids'], pids)
        Q = query_tables(cap, O, O)
        reps['Q4'] = T.pc_uniform(Q['Q4'], 4); reps['Q8'] = T.pc_uniform(Q['Q8'], 8)
        reps['Q16R'] = T.pc_hybrid(Q['Q4'], 4, Q['Q16'], 16, pids); reps['Q32R'] = T.pc_hybrid(Q['Q4'], 4, Q['Q32'], 32, pids)
    return reps


def streams(root, only=None):
    d = root / 'stageA' / 'streams'; d.mkdir(parents=True, exist_ok=True)
    V = views()
    bykey = {}
    for tag, v in V.items():
        if only and not any(tag.startswith(o) for o in only):
            continue
        bykey.setdefault(v[0], []).append(tag)
    for key, tags in bykey.items():
        g, l = V[tags[0]][2], V[tags[0]][3]
        ctx = T.Ctx115(g, 'C'); light = PT.Lights(l)
        for tag in tags:
            _, dg, _, _, cam, seeds = V[tag]
            todo = [(j, s) for j, s in enumerate(seeds) if not (d / f'stream_{tag}_{j}.npz').exists()]
            if not todo:
                continue
            reps = representations(root, tag, key, dg, tag in CAPTURE_VIEWS)
            for j, s in todo:
                t0 = time.perf_counter()
                img = ctx.readout_multi(light, dict(cam, res=C.RES), C.RES, SPP, s, reps, b=64)
                rec = {'ro_e1': img['e1'].astype(np.float32), 'ro_e2': img['e2'].astype(np.float32)}
                for nm in reps:
                    rec[f'hl|{nm}'] = img[nm].astype(np.float32)
                write_npz(d / f'stream_{tag}_{j}.npz', meta=json.dumps({'tag': tag, 'seed': s, 'spp': SPP, 'reps': list(reps),
                                                                         'seconds': time.perf_counter() - t0}), **rec)
                print('STREAM', tag, j, len(reps), round(time.perf_counter() - t0, 1), flush=True)
            del reps
        del ctx
    flush_writes()


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'con8': con8, 'conF': conF, 'capture': capture, 'streams': streams}[cmd](Path(root), rest or None)
    flush_writes()
    sys.stdout.flush(); os._exit(0)
