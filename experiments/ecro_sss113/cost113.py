"""WL113 Stage G cost accounting (local RTX 5080; Mitsuba 3.5.1 / Dr.Jit 0.4.6 unoptimised runtime, recorded as such).

  python cost113.py ROOT -> ROOT/analysis/cost.json
"""
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402


def med(fn, n=10, warm=2):
    import torch
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return {'median_ms': 1e3 * statistics.median(ts), 'p90_ms': 1e3 * sorted(ts)[int(0.9 * (len(ts) - 1))]}


def main(root):
    root = Path(root)
    import torch
    import tracer113 as T
    import eval113 as E
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    PT, dr = T.PT, T.dr
    arms, dec = E.trained_arms(root)
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    geom2 = C.with_sphere(geom, 0.9, 0.4, 0.01)
    cam = dict(C.V0_CAM, res=C.RES)
    out = {'runtime_note': 'Dr.Jit 0.4.6 re-traces every launch; latencies include Python tracing (WL101 frozen runtime not ported)', 'arms': {}}
    for name in ('O_U', 'CAND'):
        if name not in arms:
            continue
        path, kind = arms[name]
        model = (M.ECRO() if kind == 'coarse' else M3.ECROL4()).cuda()
        model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
        model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
        ctx = T.Ctx113(geom, 'C')
        hit, bn, _ = ctx.state_rays(); gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
        src = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
        row = {'params': int(sum(x.numel() for x in model.parameters()))}
        if kind != 'coarse':
            t0 = time.perf_counter(); desc = ctx.vhs(); row['support_vhs_ms_first'] = 1e3 * (time.perf_counter() - t0)
            row['support_vhs'] = med(lambda: ctx.vhs(), n=5, warm=1)
            sup = T.support_from_desc(desc, ctx.G) if kind == 'compact' else np.ones(1270, bool)
            gs.sup = torch.as_tensor(np.flatnonzero(sup), device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
            row['refined_patches'] = int(sup.sum())
            row['spatial_outputs'] = int(1270 + 15 * sup.sum())
        else:
            row['spatial_outputs'] = 1270
        row['active_bins'] = 64
        torch.cuda.reset_peak_memory_stats()

        def gen():
            with torch.no_grad():
                f = model.factors(gs)
                return model.outputs(f, gs, src) if kind != 'coarse' else (model.apply(f, gs, src), None)
        B, Bs = gen()
        row['table_floats'] = int(B.numel() + (Bs.numel() if Bs is not None else 0))
        row['table_mib'] = row['table_floats'] * 4 / 2 ** 20
        row['index_metadata_bytes'] = 1270 * 4 if kind != 'coarse' else 0          # one int32 slot per patch
        row['generator'] = med(gen); row['generator_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
        dense = np.repeat(B[0].float().cpu().numpy()[:, None], 16, 1)
        if Bs is not None:
            dense[gs.sup.cpu().numpy()] = Bs[0].float().cpu().numpy()
        tl = [(name, PT.fl(np.ascontiguousarray(dense, np.float32).ravel()), 'sub')]
        row['frame_512_1spp'] = med(lambda: ctx.cam_pass(light, cam, C.RES, 1, 11370001, 'ecro', tables=tl, b=1), n=8)

        def update():
            c2 = T.Ctx113(geom2, 'C'); h2, b2, _ = c2.state_rays(); g2 = M.GeoState(c2.G, h2, b2, torch.device('cuda'))
            s2 = torch.as_tensor(c2.source(light), device='cuda').unsqueeze(0)
            if kind != 'coarse':
                d2 = c2.vhs(); su = T.support_from_desc(d2, c2.G) if kind == 'compact' else np.ones(1270, bool)
                g2.sup = torch.as_tensor(np.flatnonzero(su), device='cuda'); g2.desc = torch.as_tensor(d2, device='cuda')
            with torch.no_grad():
                f = model.factors(g2)
                o = model.outputs(f, g2, s2) if kind != 'coarse' else (model.apply(f, g2, s2), None)
            dr.sync_thread(); return c2, o
        row['transport_update'] = med(lambda: update(), n=5, warm=1)
        tj = root / 'stageC' / name / 'training.json'
        if tj.exists():
            t = json.loads(tj.read_text()); row['training_s'] = t['train_s']; row['training_peak_mib'] = t['peak_mib']
        out['arms'][name] = row
        print('COST', name, json.dumps(row), flush=True)
        del model, ctx; torch.cuda.empty_cache()
    gl = sorted((root / 'logs').glob('data_*.log')) if (root / 'logs').exists() else []
    out['dataset_bytes'] = int(sum(p.stat().st_size for p in (root / 'stageC' / 'data').rglob('g*.npz')))
    C.jd(out, root / 'analysis' / 'cost.json')


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
