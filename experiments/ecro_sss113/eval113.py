"""WL113 evaluation streams. Per stream: depth-6 physical pass (e1, e2, h), independent deep tail pass (events 7-24),
paired readout of every table (dense 4x4 expansion), luminance unit-marker query weights per region over 4x4 sub-cells.

  python eval113.py stageA ROOT [FIX ...]        oracle arms (U, S2_U, S4_U, L4_U, R0_U, decorrelated S4) on F0-F3
  python eval113.py stageD ROOT [FIX_VIEW ...]   learned arms; F0-F3 readout-only on the Stage A seeds, F4 complete
  python eval113.py stageE ROOT [SEQ ...]        dynamic sequences
  python eval113.py rebuild ROOT                 fresh-process candidate rebuild at every sequence state
  python eval113.py oppoint ROOT                 operating-point frames (16 spp x 8 replicates) for O_U and the candidate
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import tracer113 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
N, NB, NS = 1270, T.NB, T.NS
SPP = C.dv(4096, 64)
LAUNCH = int(os.environ.get('WL113_LAUNCH', '64'))
SO = C.SO
WREG = ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full')
FIXI = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4}
NREP = C.dv(4, 2)
NPROP = C.dv(1024, 32)


import atexit
from concurrent.futures import ThreadPoolExecutor
_WRITER = ThreadPoolExecutor(max_workers=2)        # compression (zlib releases the GIL) overlaps the next GPU pass
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


def load_acc(p):
    with np.load(p) as z:
        return {k: z[k].astype(np.float64) for k in z.files if k != 'meta'}


def view_setup(ctx, geom, light_cfg, cam, out_dir, tag):
    gp = out_dir / f'gbuffer_{tag}.npz'
    if not gp.exists():
        gb = ctx.gbuffer(dict(cam, res=C.RES), C.RES)
        gb['coverage'] = T.silhouette_coverage(ctx, dict(cam, res=C.RES))
        np.savez_compressed(gp, **gb)
    with np.load(gp) as z:
        gb = {k: z[k] for k in z.files}
    reg, _ = C.masks(gb, geom, light_cfg[0]['pos'])
    return reg


def render_stream(ctx, light, cam, seed, tseed, tables, regions, path, phys=True):
    if path.exists():
        return
    t0 = time.perf_counter(); rec = {}; meta = {'seed': seed, 'spp': SPP, 'depth': T.KMAX_FULL, 'tables': [t[0] for t in tables]}
    if phys:
        img, bad = ctx.full_pass(light, cam, C.RES, SPP, seed, b=LAUNCH)      # events 1-6 (D6, paired with the readout) + 7-24 tail
        for k in ('e1', 'e2', 'h', 'tail'):
            rec[f'phys_{k}' if k != 'tail' else 'tail'] = img[k].astype(np.float32)
        meta['bad_origin'] = bad; meta['bad_floor'] = ctx.last_bad_floor; meta['paths'] = C.RES * C.RES * SPP
        meta['phys_seconds'] = time.perf_counter() - t0
    tl = [(n_, PT.fl(np.ascontiguousarray(t_, np.float32).ravel()), 'sub') for n_, t_ in tables]
    rm = {k: regions[k] for k in WREG}
    img, W, _, _ = ctx.cam_pass(light, cam, C.RES, SPP, seed, 'ecro', tables=tl, b=LAUNCH, region_masks=rm)
    dr.flush_malloc_cache()
    rec['ro_e1'] = img['e1'].astype(np.float32); rec['ro_e2'] = img['e2'].astype(np.float32)
    for n_, _ in tables:
        rec[f'hl|{n_}'] = img[n_].astype(np.float32)
    for r_, v in W.items():
        rec[f'W|{r_}'] = (T.LUM[0] * v[0] + T.LUM[1] * v[1] + T.LUM[2] * v[2]).astype(np.float32)
    meta['seconds'] = time.perf_counter() - t0
    if not phys and path.with_name(path.stem + '_phys.json').exists():
        pass
    write_npz(path, meta=json.dumps(meta), **rec)
    print('STREAM', path.name, round(meta['seconds'], 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ Stage A oracle tables
def oracle_tables(root, fix, ctx):
    A = root / 'stageA'
    desc = np.load(A / f'vhs_{fix}.npy')
    sup = T.support_from_desc(desc, ctx.G); r0 = T.support_ring0(ctx.G)
    tabs, native = [], {}
    for r in range(NREP):
        Tc, T2, T4, nat = T.tables4(load_acc(A / f'{fix}_r{r}.npz'))
        tabs += [(f'U_r{r}', T.to_dense16(Tc, 'coarse')), (f'S2_r{r}', T.to_dense16(T2, 'sub2')), (f'S4_r{r}', T4),
                 (f'L4_r{r}', T.to_dense16(T4, 'compact', sup, Tc)), (f'R0_r{r}', T.to_dense16(T4, 'compact', r0, Tc))]
        nc = np.repeat(nat['c'][:, None], NS, 1); n2 = np.repeat(np.repeat(nat['2'].reshape(N, 2, 2, NB), 2, axis=1), 2, axis=2).reshape(N, NS, NB)
        n4 = nat['4'] & nc
        native[f'U_r{r}'] = nc; native[f'S2_r{r}'] = n2 & nc; native[f'S4_r{r}'] = n4
        native[f'L4_r{r}'] = np.where(sup[:, None, None], n4, nc); native[f'R0_r{r}'] = np.where(r0[:, None, None], n4, nc)
        dp = A / f'{fix}_dec_r{r}.npz'
        if dp.exists():
            _, _, T4d, natd = T.tables4(load_acc(dp))
            tabs.append((f'S4D_r{r}', T4d)); native[f'S4D_r{r}'] = natd['4'] & np.repeat(natd['c'][:, None], NS, 1)
    return tabs, native, sup


def stageA(root, only=None):
    root = Path(root); out = root / 'stageA'
    for fix in ('F0', 'F1', 'F2', 'F3'):
        if only and fix not in only:
            continue
        fi = FIXI[fix]
        geom, lc = C.fixture(fix); light = PT.Lights(lc)
        ctx = T.Ctx113(geom, 'C')
        tabs, native, sup = oracle_tables(root, fix, ctx)
        np.savez_compressed(out / f'native_{fix}.npz', **native)
        for vi, (vname, cam) in enumerate(C.views(fix).items()):
            cam = dict(cam, res=C.RES)
            reg = view_setup(ctx, geom, lc, cam, out, f'{fix}_{vname}')
            for j in range(2):
                s = 11310000 + 100 * fi + 10 * vi + j + SO
                render_stream(ctx, light, cam, s, s + 1000 + 500000, tabs, reg, out / f'stream_{fix}_{vname}_{j}.npz')
        del ctx
    flush_writes()
    (out / f'streams_{"_".join(only) if only else "all"}.done').write_text('DONE')


# ------------------------------------------------------------------------------------------------ learned tables
def learned_table(path, kind, geom, light_cfg):
    """Dense-4x4 readout table of a trained model at the current geometry and source + its support (no reference input)."""
    import torch
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    ctx = T.Ctx113(geom, 'C')
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    src = torch.as_tensor(ctx.source(PT.Lights(light_cfg)), device='cuda').unsqueeze(0)
    sup = None
    if kind == 'coarse':
        model = M.ECRO().cuda()
    else:
        desc = ctx.vhs()
        sup = np.ones(N, bool) if kind == 'dense' else T.support_from_desc(desc, ctx.G)
        model = M3.ECROL4().cuda()
        gs.sup = torch.as_tensor(np.flatnonzero(sup), device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
    model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
    with torch.no_grad():
        f = model.factors(gs)
        Bc = model.apply(f, gs, src)
        if kind == 'coarse':
            dense = T.to_dense16(Bc[0].float().cpu().numpy(), 'coarse')
        else:
            Bs = model.apply_sub(f, gs, src)[0].float().cpu().numpy()            # [R,16,NB,3]
            T4 = np.repeat(Bc[0].float().cpu().numpy()[:, None], NS, 1); T4[sup] = Bs
            dense = T4
    del ctx
    return dense.astype(np.float32), sup


def trained_arms(root):
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    arms = {'A_learned': (C.REPO / 'results/ecro_material100/sss/model.pt', 'coarse'), 'O_U': (root / 'stageC' / 'O_U' / 'model.pt', 'coarse')}
    if dec['selected'] in ('L4', 'S4'):
        arms['CAND'] = (root / 'stageC' / 'CAND' / 'model.pt', 'compact' if dec['selected'] == 'L4' else 'dense')
    return arms, dec


def oracle_current(ctx, light, dec, seed, n):
    """Current-state physical oracle of the selected representation and the coarse U oracle (same samples)."""
    acc = ctx.construct4(light, n, seed, b=8)
    Tc, T2, T4, _ = T.tables4(acc)
    sup = T.support_from_desc(ctx.vhs(), ctx.G) if dec['selected'] == 'L4' else np.ones(N, bool)
    return [('oracle', T.to_dense16(T4, 'compact', sup, Tc)), ('oracle_U', T.to_dense16(Tc, 'coarse')), ('oracle_S4', T4)], sup


def stageD(root, only=None):
    root = Path(root); out = root / 'stageD'; out.mkdir(parents=True, exist_ok=True)
    arms, dec = trained_arms(root)
    for fix in ('F0', 'F1', 'F2', 'F3', 'F4'):
        fi = FIXI[fix]
        geom, lc = C.fixture(fix); light = PT.Lights(lc)
        sel = [v for v in C.views(fix) if not only or f'{fix}_{v}' in only]
        if not sel:
            continue
        tabs = []
        sups = {}
        for nm, (p, kind) in arms.items():
            t_, s_ = learned_table(p, kind, geom, lc); tabs.append((nm, t_)); sups[nm] = s_
        ctx = T.Ctx113(geom, 'C')
        if fix == 'F4':
            orc, osup = oracle_current(ctx, light, dec, 11341000 + SO, NPROP)
            tabs += orc
        np.savez_compressed(out / f'support_{fix}.npz', **{k: v for k, v in sups.items() if v is not None})
        for vi, (vname, cam) in enumerate(C.views(fix).items()):
            if vname not in sel:
                continue
            cam = dict(cam, res=C.RES)
            reg = view_setup(ctx, geom, lc, cam, out, f'{fix}_{vname}')
            for j in range(2):
                s = 11310000 + 100 * fi + 10 * vi + j + SO
                render_stream(ctx, light, cam, s, s + 1000 + 500000, tabs, reg, out / f'stream_{fix}_{vname}_{j}.npz', phys=(fix == 'F4'))
        del ctx
    flush_writes()
    (out / f'streams_{"_".join(only) if only else "all"}.done').write_text('DONE')


def stageE(root, only=None):
    root = Path(root); out = root / 'stageE'; out.mkdir(parents=True, exist_ok=True)
    arms, dec = trained_arms(root)
    cand = 'CAND'
    for si_, (sname, seq) in enumerate(C.sequences().items()):
        if only and sname not in only:
            continue
        light = PT.Lights(seq['light']); cams = C.views(seq['fixture'])
        prev = None
        for st, geom in enumerate(seq['states']):
            tabs = []; sups = {}
            for nm, (p, kind) in arms.items():
                t_, s_ = learned_table(p, kind, geom, seq['light']); tabs.append((nm, t_)); sups[nm] = s_
            cur = [t for t in tabs if t[0] == cand][0]
            np.save(out / f'table_{sname}_{st}_{cand}.npy', cur[1])
            if sups.get(cand) is not None:
                np.save(out / f'support_{sname}_{st}.npy', sups[cand])
            if prev is not None:
                tabs.append(('stale', prev))
            ctx = T.Ctx113(geom, 'C')
            orc, _ = oracle_current(ctx, light, dec, 11351000 + 100 * si_ + st + SO, NPROP)
            tabs += orc
            for vi, vname in enumerate(seq['views']):
                cam = dict(cams[vname], res=C.RES)
                reg = view_setup(ctx, geom, seq['light'], cam, out, f'{sname}_{st}_{vname}')
                for j in range(2):
                    s = 11350000 + 1000 * si_ + 10 * st + 2 * vi + j + SO
                    render_stream(ctx, light, cam, s, s + 500000, tabs, reg, out / f'stream_{sname}_{st}_{vname}_{j}.npz')
            prev = cur[1]
            del ctx
    flush_writes()
    (out / f'streams_{"_".join(only) if only else "all"}.done').write_text('DONE')


def rebuild(root, only=None):
    root = Path(root); out = root / 'stageE'
    arms, dec = trained_arms(root)
    res = {}
    for sname, seq in C.sequences().items():
        for st in reversed(range(len(seq['states']))):
            t_, s_ = learned_table(*arms['CAND'], seq['states'][st], seq['light'])
            ref = np.load(out / f'table_{sname}_{st}_CAND.npy')
            res[f'{sname}_{st}'] = {'table_max_rel': float(np.abs(t_ - ref).max() / max(np.abs(ref).max(), 1e-30))}
            if s_ is not None and (out / f'support_{sname}_{st}.npy').exists():
                res[f'{sname}_{st}']['support_identical'] = bool(np.array_equal(s_, np.load(out / f'support_{sname}_{st}.npy')))
    C.jd(res, out / 'rebuild_check.json')


def oppoint(root, only=None):
    """Operating-point method frames: 16 spp x 8 independent replicates of the complete method frame (LOW + lookup)."""
    root = Path(root); out = root / 'stageD'
    arms, dec = trained_arms(root)
    res = {}
    for fix in ('F0', 'F1', 'F2', 'F3', 'F4'):
        geom, lc = C.fixture(fix); light = PT.Lights(lc)
        tabs = [(nm, learned_table(p, k, geom, lc)[0]) for nm, (p, k) in arms.items() if nm in ('O_U', 'CAND')]
        ctx = T.Ctx113(geom, 'C')
        tl = [(n_, PT.fl(np.ascontiguousarray(t_, np.float32).ravel()), 'sub') for n_, t_ in tabs]
        for vi, (vname, cam) in enumerate(C.views(fix).items()):
            fr = {n_: [] for n_, _ in tabs}
            for rep in range(8):
                img, _, _, _ = ctx.cam_pass(light, dict(cam, res=C.RES), C.RES, 16, 11360000 + 1000 * FIXI[fix] + 10 * vi + rep + SO, 'ecro', tables=tl, b=16)
                for n_, _ in tabs:
                    fr[n_].append((img['e1'] + img['e2'] + img[n_]).astype(np.float32))
            np.savez_compressed(out / f'oppoint_{fix}_{vname}.npz', **{n_: np.stack(v) for n_, v in fr.items()})
            print('OPPOINT', fix, vname, flush=True)
        del ctx
    (out / 'oppoint.done').write_text('DONE')


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'stageA': stageA, 'stageD': stageD, 'stageE': stageE, 'rebuild': rebuild, 'oppoint': oppoint}[cmd](root, rest or None)
    flush_writes()
    sys.stdout.flush(); os._exit(0)
