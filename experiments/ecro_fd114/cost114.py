"""WL114 Stage F cost accounting (local RTX 5080; Mitsuba 3.5.1 / Dr.Jit 0.4.6 unoptimised runtime, recorded as such).
Target generation (WL113 D6 construct4 vs WL114 deep construction to K*, same configurations / proposals / launch size), training
cost / memory (training.json), generator latency, table size, static frame and geometry update for a D6 and a full-depth arm of the
same architecture.

  python cost114.py ROOT -> ROOT/analysis/cost.json
"""
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common114 as C  # noqa: E402


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
    import tracer114 as T
    import eval114 as EV
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    PT, dr = T.PT, T.dr
    K = EV.kstar(root)
    out = {'runtime_note': 'Dr.Jit 0.4.6 re-traces every launch; latencies include Python tracing (WL101 frozen runtime not ported)', 'K_star': K}
    # ---- target generation (uncontended, 3 D113 training configurations x 1 light, n = 32, b = 16)
    tg = {'d6_construct4_s': [], 'deep_construct_s': []}
    for i in (0, 1, 2):
        cfg = C.train_config('train', i); ctx = T.Ctx114(cfg['geom'], 'C'); light = PT.Lights(cfg['lights'][0])
        ctx.construct4(light, 32, 91148000 + i, b=16); ctx.construct_deep(light, 32, 91148100 + i, K - 1, b=16)   # warm (compile)
        dr.sync_thread(); t0 = time.perf_counter(); ctx.construct4(light, 32, 91148200 + i, b=16); dr.sync_thread(); tg['d6_construct4_s'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ctx.construct_deep(light, 32, 91148300 + i, K - 1, b=16); dr.sync_thread(); tg['deep_construct_s'].append(time.perf_counter() - t0)
        del ctx
    tg['ratio_deep_over_d6'] = float(np.median(tg['deep_construct_s']) / np.median(tg['d6_construct4_s']))
    out['target_generation_per_light_n32'] = tg
    out['dataset_bytes_D114'] = int(sum(p.stat().st_size for p in (root / 'stageC' / 'data').rglob('g*.npz')))
    out['dataset_bytes_D113'] = int(sum(p.stat().st_size for p in (C.REPO / 'results/ecro_sss113/primary/stageC/data').rglob('g*.npz')))
    gl = root / 'logs' / 'data.log'
    if gl.exists():
        lines = [l for l in gl.read_text().splitlines() if l.startswith('GEN ')]
        out['D114_generation_wall_s'] = float(lines[-1].split()[-1]) if lines else None
    out['D113_generation_wall_s_two_processes'] = 'WL113 log: data step 03:47:32 -> 04:19:16 (2 processes)'
    # ---- arms
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    geom2 = C.with_sphere(geom, 0.9, 0.4, 0.01)
    cam = dict(C.V0_CAM, res=C.RES)
    out['arms'] = {}
    for name in ('A1r', 'B1', 'OB1'):
        path, kind = EV.learned_arms(root)[name]
        model = (M.ECRO() if kind == 'coarse' else M3.ECROL4()).cuda()
        model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
        model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
        ctx = T.Ctx114(geom, 'C')
        hit, bn, _ = ctx.state_rays(); gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
        src = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
        row = {'params': int(sum(x.numel() for x in model.parameters()))}
        if kind != 'coarse':
            desc = ctx.vhs(); gs.sup = torch.arange(1270, device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
        torch.cuda.reset_peak_memory_stats()

        def gen():
            with torch.no_grad():
                f = model.factors(gs)
                return model.outputs(f, gs, src) if kind != 'coarse' else (model.apply(f, gs, src), None)
        B, Bs = gen()
        row['table_floats'] = int(B.numel() + (Bs.numel() if Bs is not None else 0)); row['table_mib'] = row['table_floats'] * 4 / 2 ** 20
        row['generator'] = med(gen); row['generator_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
        dense = np.repeat(B[0].float().cpu().numpy()[:, None], 16, 1)
        if Bs is not None:
            dense[:] = Bs[0].float().cpu().numpy()
        tl = [(name, PT.fl(np.ascontiguousarray(dense, np.float32).ravel()), 'sub')]
        row['frame_512_1spp'] = med(lambda: ctx.cam_pass(light, cam, C.RES, 1, 11470001, 'ecro', tables=tl, b=1), n=8)

        def update():
            c2 = T.Ctx114(geom2, 'C'); h2, b2, _ = c2.state_rays(); g2 = M.GeoState(c2.G, h2, b2, torch.device('cuda'))
            s2 = torch.as_tensor(c2.source(light), device='cuda').unsqueeze(0)
            if kind != 'coarse':
                d2 = c2.vhs(); g2.sup = torch.arange(1270, device='cuda'); g2.desc = torch.as_tensor(d2, device='cuda')
            with torch.no_grad():
                f = model.factors(g2)
                o = model.outputs(f, g2, s2) if kind != 'coarse' else (model.apply(f, g2, s2), None)
            dr.sync_thread(); return c2, o
        row['transport_update'] = med(lambda: update(), n=5, warm=1)
        t = json.loads((root / 'stageC' / name / 'training.json').read_text())
        row['training_s'] = t['train_s']; row['training_peak_mib'] = t['peak_mib']
        out['arms'][name] = row
        print('COST', name, json.dumps(row), flush=True)
        del model, ctx; torch.cuda.empty_cache()
    out['training_all'] = {a: json.loads((root / 'stageC' / a / 'training.json').read_text())['train_s'] for a in ('A1r', 'B1', 'A2', 'B2', 'OB1')}
    C.jd(out, root / 'analysis' / 'cost.json')


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
