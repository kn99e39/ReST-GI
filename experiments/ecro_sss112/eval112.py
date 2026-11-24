"""WL112 evaluation streams: physical references (C and N semantics) and paired readouts of any set of tables.

  python eval112.py stageB ROOT [VIEW ...]
  python eval112.py stageD ROOT [CONFIG_VIEW ...]
  python eval112.py stageE ROOT [SEQ ...]
Per stream: phys_C (e1, e2, h), optional phys_N, readout HIGH of every table (same seed -> identical rays / first two events),
per-region unit-marker query weights over sub-cells. Outputs are atomic and skipped when present (resumable).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT
SPP = C.dv(4096, 64)
PVSPP = C.dv(1024, 32)
SO = C.SO
LAUNCH = 32
WREG = ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full')
N, NB, NS = 1270, T.NB, T.NSUB


def load_acc(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files if k != 'meta'}


def measure_tables(acc, meas, ufall=None):
    """Coarse and 2x2 tables with the frozen fallback chain; returns dict and native-support masks."""
    Tc, wc, _, _ = T.tables_from(acc, meas, 'coarse')
    Ts, ws, _, _ = T.tables_from(acc, meas, 'sub')
    Tc = Tc.reshape(N, NB, 3); Ts = Ts.reshape(N, NS, NB, 3)
    wc = wc.reshape(N, NB); ws = ws.reshape(N, NS, NB)
    natc = wc > 0
    if ufall is not None:
        Tc = np.where(natc[..., None], Tc, ufall)
    nats = ws > 0
    Ts = np.where(nats[..., None], Ts, Tc[:, None])
    return Tc, Ts, natc, nats & natc[:, None]


def learned_table(model_path, geom, light_cfg, kind='coarse'):
    """Table of a trained ECRO model at the current geometry and source (never reads any reference)."""
    import torch
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model112 as M2
    ctx = T.Ctx112(geom, 'C')
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    src = torch.as_tensor(ctx.source(PT.Lights(light_cfg)), device='cuda').unsqueeze(0)
    model = (M2.ECROS2() if kind == 'sub' else M.ECRO()).cuda()
    model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    model.load_state_dict(torch.load(model_path, map_location='cuda')); model.eval()
    with torch.no_grad():
        f = model.factors(gs)
        Bt = model.apply(f, gs, src, use_single=True, use_global=True)[0]
    del ctx
    return Bt.float().cpu().numpy()


def render_stream(ctxC, ctxN, light, cam, seed, tables, regions, path, phys=True):
    if path.exists():
        return
    t0 = time.perf_counter()
    rm = {k: regions[k] for k in WREG}
    rec = {}
    meta = {'seed': seed, 'spp': SPP, 'launch': LAUNCH, 'tables': [t[0] for t in tables]}
    if phys:
        img, _, _, bad = ctxC.cam_pass(light, cam, C.RES, SPP, seed, 'cam', b=LAUNCH)
        for k in ('e1', 'e2', 'h'):
            rec[f'phys_C_{k}'] = img[k].astype(np.float32)
        meta['bad_origin_C'] = bad; meta['bad_floor_C'] = ctxC.last_bad_floor; meta['paths'] = C.RES * C.RES * SPP
        if ctxN is not None:
            img, _, _, bad = ctxN.cam_pass(light, cam, C.RES, SPP, seed, 'cam', b=LAUNCH)
            for k in ('e1', 'e2', 'h'):
                rec[f'phys_N_{k}'] = img[k].astype(np.float32)
            meta['bad_origin_N'] = bad; meta['bad_floor_N'] = ctxN.last_bad_floor
    tl = [(n_, PT.fl(np.ascontiguousarray(t_, np.float32).ravel()), k_) for n_, t_, k_ in tables]
    img, W, _, bad = ctxC.cam_pass(light, cam, C.RES, SPP, seed, 'ecro', tables=tl, b=LAUNCH, region_masks=rm)
    rec['ro_e1'] = img['e1'].astype(np.float32); rec['ro_e2'] = img['e2'].astype(np.float32)
    for n_, _, _ in tables:
        rec[f'hl|{n_}'] = img[n_].astype(np.float32)
    for r_, v in W.items():
        rec[f'W|{r_}'] = v.astype(np.float32)
    meta['seconds'] = time.perf_counter() - t0
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, meta=json.dumps(meta), **rec)
    os.replace(tmp, path)
    print('STREAM', path.name, round(meta['seconds'], 1), 's', flush=True)


def view_setup(ctxC, geom, light_cfg, cam, out_dir, vname):
    gpath = out_dir / f'gbuffer_{vname}.npz'
    if not gpath.exists():
        gb = ctxC.gbuffer(dict(cam, res=C.RES), C.RES)
        np.savez_compressed(gpath, **gb)
    with np.load(gpath) as z:
        gb = {k: z[k] for k in z.files}
    reg, bands = C.masks(gb, geom, light_cfg[0]['pos'])
    return reg


# ------------------------------------------------------------------------------------------------ Stage B
def stageB_tables(root, view_pv=None):
    A = root / 'stageA'
    tabs = []; native = {}
    for r in range(C.RC):
        acc = load_acc(A / f'C_r{r}.npz')
        Uc, Us, nUc, nUs = measure_tables(acc, 'U')
        tabs += [(f'U_r{r}', Uc, 'coarse'), (f'S_U_r{r}', Us, 'sub')]
        native[f'U_r{r}'] = nUc; native[f'S_U_r{r}'] = nUs
        for m in ('V', 'G'):
            Tc, Ts, nc, ns = measure_tables(acc, m, ufall=Uc)
            tabs += [(f'{m}_r{r}', Tc, 'coarse'), (f'S_{m}_r{r}', Ts, 'sub')]
            native[f'{m}_r{r}'] = nc; native[f'S_{m}_r{r}'] = ns
    for r in range(C.RN):
        acc = load_acc(A / f'N_r{r}.npz')
        Uc, _, nUc, _ = measure_tables(acc, 'U')
        _, Gs, _, nGs = measure_tables(acc, 'G', ufall=Uc)
        tabs += [(f'UN_r{r}', Uc, 'coarse'), (f'S_GN_r{r}', Gs, 'sub')]
        native[f'UN_r{r}'] = nUc; native[f'S_GN_r{r}'] = nGs
    for fr in ('frame_normal', 'frame_patch'):
        acc = load_acc(A / f'{fr}.npz')
        Uc, _, nUc, _ = measure_tables(acc, 'U')
        Gc, _, nGc, _ = measure_tables(acc, 'G', ufall=Uc)
        tabs += [(f'{fr}_U', Uc, 'coarse'), (f'{fr}_G', Gc, 'coarse')]
        native[f'{fr}_U'] = nUc; native[f'{fr}_G'] = nGc
    return tabs, native


def pv_tables(cells, Gc_fall):
    """Camera-conditioned throughput-weighted projection P_v (coarse) and S_v (2x2), fallback G."""
    t = np.stack([cells[f't{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
    h = np.stack([cells[f'h{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
    tc, hc = t.sum(1), h.sum(1)
    natc = tc.min(-1) > 0
    Pc = np.where(natc[..., None], hc / np.maximum(tc, 1e-30), Gc_fall)
    nats = (t.min(-1) > 0) & natc[:, None]
    Ps = np.where(nats[..., None], h / np.maximum(t, 1e-30), Pc[:, None])
    return Pc, Ps, natc, nats


def stageB(root, only=None):
    root = Path(root); out = root / 'stageB'; out.mkdir(parents=True, exist_ok=True)
    geom, light_cfg = C.fixture('F0'); light = PT.Lights(light_cfg)
    cams = C.views('F0')
    tabs, native = stageB_tables(root)
    tabs.append(('A_learned', learned_table(C.REPO / 'results/ecro_material100/sss/model.pt', geom, light_cfg).reshape(N, NB, 3), 'coarse'))
    np.savez_compressed(out / 'native_support.npz', **native)
    ctxC = T.Ctx112(geom, 'C'); ctxN = T.Ctx112(geom, 'N')
    G0 = [t for t in tabs if t[0] == 'G_r0'][0][1]
    for vi, (vname, cam) in enumerate(cams.items()):
        if only and vname not in only:
            continue
        cam = dict(cam, res=C.RES)
        reg = view_setup(ctxC, geom, light_cfg, cam, out, vname)
        pvt = []
        for r in range(2):
            pp = out / f'pv_cells_{vname}_r{r}.npz'
            if not pp.exists():
                t0 = time.perf_counter()
                _, _, cells, _ = ctxC.cam_pass(light, cam, C.RES, PVSPP, 11211000 + 10 * vi + r + SO, 'cam', b=LAUNCH, capture_cells=True)
                np.savez_compressed(pp, **{k: v.astype(np.float64) for k, v in cells.items()})
                print('PV', vname, r, round(time.perf_counter() - t0, 1), flush=True)
            with np.load(pp) as z:
                Pc, Ps, nc, ns = pv_tables({k: z[k] for k in z.files}, G0)
            pvt += [(f'P_v_r{r}', Pc, 'coarse'), (f'S_v_r{r}', Ps, 'sub')]
            np.savez_compressed(out / f'pv_native_{vname}_r{r}.npz', P=nc, S=ns)
        for j in range(2):
            render_stream(ctxC, ctxN, light, cam, 11210000 + 10 * vi + j + SO, tabs + pvt, reg, out / f'stream_{vname}_{j}.npz')
    (out / 'stageB_streams.done').write_text('DONE')


# ------------------------------------------------------------------------------------------------ Stage D / E (learned arms)
def arm_tables(root, geom, light_cfg, arms):
    """Learned tables of the trained arms at this geometry (camera-independent; no reference input)."""
    out = []
    for name, (path, kind) in arms.items():
        Bt = learned_table(path, geom, light_cfg, kind)
        out.append((name, Bt.reshape(N, NS, NB, 3) if kind == 'sub' else Bt.reshape(N, NB, 3), kind))
    return out


def trained_arms(root):
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    arms = {'A_learned': (C.REPO / 'results/ecro_material100/sss/model.pt', 'coarse'),
            'O_U': (root / 'stageC' / 'O_U' / 'model.pt', 'coarse'),
            'O_G': (root / 'stageC' / 'O_G' / 'model.pt', 'coarse')}
    if dec['outcome'].startswith('SPATIAL'):
        arms['S2'] = (root / 'stageC' / 'S2' / 'model.pt', 'sub')
    return arms, dec


def oracle_table(ctxC, light, dec, seed, n=256):
    """Physical camera-independent table of the Stage-B-selected representation at the current geometry (class C)."""
    acc = ctxC.construct(light, n, seed, b=32)
    Uc, _, _, _ = measure_tables(acc, 'U')
    meas = 'U' if dec['outcome'] == 'SPATIAL_U' else 'G'
    Tc, Ts, _, _ = measure_tables(acc, meas, ufall=Uc if meas != 'U' else None)
    return (Ts, 'sub') if dec['outcome'].startswith('SPATIAL') else (Tc, 'coarse')


def stageD(root, only=None):
    root = Path(root); out = root / 'stageD'; out.mkdir(parents=True, exist_ok=True)
    arms, dec = trained_arms(root)
    ci = 0
    for fix in ('F0', 'F1'):
        geom, light_cfg = C.fixture(fix); light = PT.Lights(light_cfg)
        ctxC = T.Ctx112(geom, 'C')
        tabs = arm_tables(root, geom, light_cfg, arms)
        orc, kind = oracle_table(ctxC, light, dec, 11241000 + ci + SO, n=C.dv(256, 16))
        tabs.append(('oracle', orc, kind))
        for vname, cam in C.views(fix).items():
            if only and f'{fix}_{vname}' not in only:
                ci += 1; continue
            cam = dict(cam, res=C.RES)
            reg = view_setup(ctxC, geom, light_cfg, cam, out, f'{fix}_{vname}')
            for j in range(2):
                render_stream(ctxC, None, light, cam, 11240000 + 10 * ci + j + SO, tabs, reg, out / f'stream_{fix}_{vname}_{j}.npz')
            ci += 1
    (out / 'stageD_streams.done').write_text('DONE')


def stageE(root, only=None):
    root = Path(root); out = root / 'stageE'; out.mkdir(parents=True, exist_ok=True)
    arms, dec = trained_arms(root)
    cand = 'S2' if 'S2' in arms else 'O_G'
    for si_, (sname, seq) in enumerate(C.sequences().items()):
        if only and sname not in only:
            continue
        light = PT.Lights(seq['light']); cams = C.views(seq['fixture'])
        prev = None
        for st, geom in enumerate(seq['states']):
            ctxC = T.Ctx112(geom, 'C')
            tabs = arm_tables(root, geom, seq['light'], arms)
            cur = [t for t in tabs if t[0] == cand][0]
            np.save(out / f'table_{sname}_{st}_{cand}.npy', cur[1].astype(np.float32))
            if prev is not None:
                tabs.append(('stale', prev[1], prev[2]))
            orc, kind = oracle_table(ctxC, light, dec, 11251000 + 100 * si_ + st + SO, n=C.dv(256, 16))
            tabs.append(('oracle', orc, kind))
            for vi, vname in enumerate(seq['views']):
                cam = dict(cams[vname], res=C.RES)
                reg = view_setup(ctxC, geom, seq['light'], cam, out, f'{sname}_{st}_{vname}')
                for j in range(2):
                    render_stream(ctxC, None, light, cam, 11250000 + 1000 * si_ + 10 * st + 2 * vi + j + SO, tabs, reg,
                                  out / f'stream_{sname}_{st}_{vname}_{j}.npz')
            prev = cur
            del ctxC
    (out / 'stageE_streams.done').write_text('DONE')


def rebuild_check(root):
    """Fresh-process full rebuild of the candidate table at every sequence state; compared with the in-sequence table."""
    root = Path(root); out = root / 'stageE'
    arms, dec = trained_arms(root)
    cand = 'S2' if 'S2' in arms else 'O_G'
    res = {}
    for sname, seq in C.sequences().items():
        for st in reversed(range(len(seq['states']))):            # reverse order: no shared history with the sequence run
            geom = seq['states'][st]
            Bt = arm_tables(root, geom, seq['light'], {cand: arms[cand]})[0][1]
            ref = np.load(out / f'table_{sname}_{st}_{cand}.npy')
            res[f'{sname}_{st}'] = float(np.abs(Bt - ref).max() / max(np.abs(ref).max(), 1e-30))
    C.jd(res, out / 'rebuild_check.json')


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'stageB': stageB, 'stageD': stageD, 'stageE': stageE, 'rebuild': lambda r, o=None: rebuild_check(r)}[cmd](root, rest or None)
    sys.stdout.flush(); os._exit(0)
