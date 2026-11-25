"""WL114 Stage A (SSS): full-depth target qualification. Per Stage A view: deterministic deep passes to camera event 192 (2
replicas) and verification-roulette passes from camera event 6 (6 replicas); per Stage A construction: deep probe constructions
(2 deterministic, 4 roulette replicas). Region / patch-class sums only plus a few images for figures. Resumable; atomic outputs.

  python stageA114.py ROOT [VIEW_INDEX ...]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
SPP = C.dv(2048, 64)
NDET, NRR = 2, C.dv(6, 3)
NCON = C.dv(64, 8)
REG = ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full')
N = 1270


def region_sums(img, reg):
    return {r: [float(img[..., c][reg[r]].sum()) for c in range(3)] + [int(reg[r].sum())] for r in REG if r in reg}


def atomic_json(obj, path):
    tmp = path.with_name(path.stem + '.tmp.json'); C.jd(obj, tmp); os.replace(tmp, path)


def views_main(root, only=None):
    out = root / 'stageA'; out.mkdir(parents=True, exist_ok=True)
    for vi, (tag, geom, lc, cam) in enumerate(C.stageA_views()):
        if only and str(vi) not in only:
            continue
        p = out / f'view_{tag}.json'
        if p.exists():
            continue
        light = PT.Lights(lc); ctx = T.Ctx114(geom, 'C'); cam = dict(cam, res=C.RES)
        gb = ctx.gbuffer(cam, C.RES); gb['coverage'] = T.T113.silhouette_coverage(ctx, cam)
        reg, _ = C.masks(gb, geom, lc[0]['pos'])
        rec = {'tag': tag, 'contact_valid': C.contact_valid(reg), 'spp': SPP, 'det': [], 'rr': []}
        imgs = {}
        for j in range(NDET):
            t0 = time.perf_counter()
            img, dg = ctx.deep_pass(light, cam, C.RES, SPP, 11400000 + 100 * vi + j + C.SO, 192, b=64)
            rec['det'].append({'seed': 11400000 + 100 * vi + j + C.SO, 'seconds': time.perf_counter() - t0, 'diag': dg,
                               'sums': {k: region_sums(v, reg) for k, v in img.items()}})
            if j == 0:
                imgs.update({f'det0_{k}': v.astype(np.float32) for k, v in img.items()})
        for j in range(NRR):
            t0 = time.perf_counter()
            img, dg = ctx.deep_pass(light, cam, C.RES, SPP, 11401000 + 100 * vi + j + C.SO, 1024, b=64, rr_from=6)
            rec['rr'].append({'seed': 11401000 + 100 * vi + j + C.SO, 'seconds': time.perf_counter() - t0, 'diag': dg,
                              'sums': {k: region_sums(v, reg) for k, v in img.items()}})
            if j == 0:
                imgs['rr0_rr'] = img['rr'].astype(np.float32)
        np.savez_compressed(out / f'images_{tag}.npz', **imgs, **{f'mask_{k}': v for k, v in reg.items()})
        atomic_json(rec, p)
        print('VIEW', tag, round(sum(d['seconds'] for d in rec['det'] + rec['rr']), 1), 's', flush=True)
        del ctx
    (out / f'views_{"_".join(only) if only else "all"}.done').write_text('DONE')


def con_sums(acc, sets):
    """Per patch set: summed bucket energy over counted proposals (luminance) and slot counts."""
    w = acc['U_w'].reshape(N, -1)
    res = {}
    for sname, sel in sets.items():
        d = {'U_w': float(w[sel].sum())}
        for k in acc:
            if k.startswith('U_') and k[-1] in '012' and k != 'U_w':
                d[k] = float(acc[k].reshape(N, -1)[sel].sum())
        res[sname] = d
    return res


def con_main(root):
    out = root / 'stageA'; out.mkdir(parents=True, exist_ok=True)
    for ci, (tag, geom, lc) in enumerate(C.stageA_constructions()):
        p = out / f'con_{tag}.json'
        if p.exists():
            continue
        light = PT.Lights(lc); ctx = T.Ctx114(geom, 'C')
        sets = {'all': np.ones(N, bool), 'ring0': T.ring0(ctx.G, geom['S']['c'][:2])}
        rec = {'tag': tag, 'n': NCON, 'ring0_patches': int(sets['ring0'].sum()), 'det': [], 'rr': []}
        for j in range(NDET):
            t0 = time.perf_counter()
            acc, dg = ctx.construct_deep(light, NCON, 11402000 + 10 * ci + j + C.SO, 191, b=8)
            rec['det'].append({'seconds': time.perf_counter() - t0, 'diag': dg, 'sums': con_sums(acc, sets)})
        for j in range(C.dv(8, 2)):
            t0 = time.perf_counter()
            acc, dg = ctx.construct_deep(light, NCON, 11403000 + 10 * ci + j + C.SO, 1023, b=8, rr_from=6)
            rec['rr'].append({'seconds': time.perf_counter() - t0, 'diag': dg, 'sums': con_sums(acc, sets)})
        atomic_json(rec, p)
        print('CON', tag, round(sum(d['seconds'] for d in rec['det'] + rec['rr']), 1), 's', flush=True)
        del ctx
    (out / 'con.done').write_text('DONE')


if __name__ == '__main__':
    root = Path(sys.argv[1]); what = sys.argv[2]
    if what == 'views':
        views_main(root, sys.argv[3:] or None)
    else:
        con_main(root)
    sys.stdout.flush(); os._exit(0)
