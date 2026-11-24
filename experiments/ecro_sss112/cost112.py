"""WL112 Stage F cost accounting (local RTX 5080, Mitsuba 3.5.1 / Dr.Jit 0.4.6 unoptimised runtime; WL101's frozen dr.freeze runtime
is not ported to sub-patch tables, so absolute latencies include Python re-tracing and are not comparable with WL101's 11-16 ms).

  python cost112.py ROOT     -> ROOT/analysis/cost.json
"""
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402


def med(fn, n=12, warm=2):
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
    import tracer112 as T
    import eval112 as E
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model112 as M2
    PT, dr = T.PT, T.dr
    arms, dec = E.trained_arms(root)
    out = {'runtime_note': __doc__.strip().splitlines()[1].strip(), 'arms': {}}
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    geom2 = C.with_sphere(geom, 0.9, 0.4, 0.04)
    cam = dict(C.V0_CAM, res=C.RES)
    for name, (path, kind) in arms.items():
        model = (M2.ECROS2() if kind == 'sub' else M.ECRO()).cuda()
        model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
        model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
        ctx = T.Ctx112(geom, 'C')
        hit, bn, _ = ctx.state_rays(); gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
        src = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
        torch.cuda.reset_peak_memory_stats()

        def gen():
            with torch.no_grad():
                f = model.factors(gs); return model.apply(f, gs, src)
        row = {'params': int(sum(x.numel() for x in model.parameters())), 'table_floats': int(gen().numel()),
               'table_mib_float32': gen().numel() * 4 / 2 ** 20, 'generator': med(gen), 'generator_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
        tab = gen()[0].float().reshape(-1).contiguous().cpu().numpy()
        tl = [(name, PT.fl(tab), kind)]

        def update():                       # geometry edit -> scene rebuild + relation state + source + generator (transport update)
            c2 = T.Ctx112(geom2, 'C'); h2, b2, _ = c2.state_rays(); g2 = M.GeoState(c2.G, h2, b2, torch.device('cuda'))
            s2 = torch.as_tensor(c2.source(light), device='cuda').unsqueeze(0)
            with torch.no_grad():
                f = model.factors(g2); B = model.apply(f, g2, s2)
            dr.sync_thread(); return c2, B
        row['transport_update'] = med(lambda: update(), n=6, warm=1)

        def frame():
            ctx.cam_pass(light, cam, C.RES, 1, 11260001, 'ecro', tables=tl, b=1)
        row['frame_512_1spp'] = med(frame, n=10, warm=2)

        def edit_to_frame():
            c2, B = update()
            c2.cam_pass(light, cam, C.RES, 1, 11260002, 'ecro', tables=[(name, PT.fl(B[0].float().reshape(-1).contiguous().cpu().numpy()), kind)], b=1)
        row['edit_to_frame'] = med(edit_to_frame, n=6, warm=1)
        tj = root / 'stageC' / name / 'training.json'
        if tj.exists():
            t = json.loads(tj.read_text()); row['training_s'] = t['train_s']; row['training_peak_mib'] = t['peak_mib']
        out['arms'][name] = row
        del model, ctx
        torch.cuda.empty_cache()
        print('COST', name, json.dumps(row), flush=True)
    # readout cost: coarse vs sub lookup at equal frame (same tracer)
    ctx = T.Ctx112(geom, 'C')
    n = ctx.N * T.NB * 3
    c1 = [('c', PT.fl(np.ones(n, np.float32)), 'coarse')]; s1 = [('s', PT.fl(np.ones(4 * n, np.float32)), 'sub')]
    out['readout_coarse_frame'] = med(lambda: ctx.cam_pass(light, cam, C.RES, 1, 11260003, 'ecro', tables=c1, b=1), n=10)
    out['readout_sub_frame'] = med(lambda: ctx.cam_pass(light, cam, C.RES, 1, 11260003, 'ecro', tables=s1, b=1), n=10)
    out['low_only_frame'] = med(lambda: ctx.cam_pass(light, cam, C.RES, 1, 11260003, 'ecro', tables=[], b=1), n=10)
    t0 = time.perf_counter(); ctx.construct(light, 32, 11260004, b=32); out['training_field_construction_s'] = time.perf_counter() - t0
    logs = sorted((root / 'stageC' / 'data').glob('*/done_*'))
    out['dataset_files'] = len(list((root / 'stageC' / 'data' / 'train').glob('g*.npz')))
    out['dataset_bytes'] = int(sum(p.stat().st_size for p in (root / 'stageC' / 'data').rglob('g*.npz')))
    C.jd(out, root / 'analysis' / 'cost.json')


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
