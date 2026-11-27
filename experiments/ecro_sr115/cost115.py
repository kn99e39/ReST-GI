"""WL115 Stage G cost accounting (local RTX 5080; Mitsuba 3.5.1 / Dr.Jit 0.4.6 unoptimised experiment-local runtime, recorded as such;
measured with the GPU otherwise idle).

Arms: canonical coarse full-depth OB1 (WL114), dense 4 x 4 full-depth B1 (WL114), CAR1 (WL115). Per arm: parameters, table floats /
storage, metadata (support), generator latency and peak memory, support + descriptor computation, readout of a 512^2 1-spp frame, edit
(geometry update) to new table, training time / peak; target generation per light (dense 4 x 4 vs CAR-16 children).
  python cost115.py ROOT -> ROOT/analysis/cost.json
"""
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / 'ecro_fd114', HERE.parent / 'ecro_sss113', HERE.parent / 'ecro86', HERE.parent / 'ecro_material100'):
    sys.path.insert(0, str(d))
import common115 as C  # noqa: E402


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
    import tracer115 as T
    import candidate115 as CA
    import geomdesc115 as GD
    import model86 as M
    import model113 as M3
    import model115 as MC
    PT, dr = T.PT, T.dr
    out = {'runtime_note': 'Dr.Jit 0.4.6 re-traces every launch; latencies include Python tracing (experiment-local runtime, not an optimised renderer)'}
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    geom2 = C.with_sphere(geom, 0.9, 0.4, 0.01)
    cam = dict(C.C113.V0_CAM, res=C.RES)
    # ---- target generation per light (n = 32), D114-style dense 4 x 4 vs CAR-16 children (uncontended)
    tg = {'dense4_s': [], 'car16_children_s': [], 'support_and_descriptors_s': []}
    for i in (0, 1, 2):
        cfg = C.C114.train_config('train', i); ctx = T.Ctx115(cfg['geom'], 'C'); lt = PT.Lights(cfg['lights'][0])
        w, pids, m = CA.support(ctx)
        ctx.construct_k(lt, 4, 91158000 + i, 23, 4, b=4); ctx.construct_k(lt, 4, 91158100 + i, 23, 16, pids=pids, b=4)
        dr.sync_thread(); t0 = time.perf_counter(); ctx.construct_k(lt, 32, 91158200 + i, 23, 4, b=16); dr.sync_thread(); tg['dense4_s'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ctx.construct_k(lt, 32, 91158300 + i, 23, 16, pids=pids, b=4); dr.sync_thread(); tg['car16_children_s'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); CA.support(ctx); GD.vhs_k(ctx, 16, pids); GD.prox_other(ctx, 16, pids=pids, ells=(CA.ELL,)); dr.sync_thread()
        tg['support_and_descriptors_s'].append(time.perf_counter() - t0)
        del ctx
    out['target_generation_per_light_n32'] = tg
    out['dataset_bytes_D114'] = int(sum(p.stat().st_size for p in (C.WL114_ROOT / 'stageC' / 'data').rglob('g*.npz')))
    out['dataset_bytes_D115_children'] = int(sum(p.stat().st_size for p in (root / 'stageC' / 'd115').rglob('g*.npz')))
    # ---- arms
    arms = {'OB1': (C.WL114_ROOT / 'stageC' / 'OB1' / 'model.pt', 'coarse'), 'B1': (C.WL114_ROOT / 'stageC' / 'B1' / 'model.pt', 'dense'),
            'CAR1': (root / 'stageC' / 'CAR1' / 'model.pt', 'car')}
    out['arms'] = {}
    for name, (path, kind) in arms.items():
        model = {'coarse': M.ECRO, 'dense': M3.ECROL4, 'car': MC.ECROCAR}[kind]().cuda()
        model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
        model.load_state_dict(torch.load(path, map_location='cuda')); model.eval()
        row = {'params': int(sum(x.numel() for x in model.parameters()))}

        def prepare(g):
            ctx = T.Ctx115(g, 'C')
            hit, bn, _ = ctx.state_rays(); gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
            src = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
            extra = {}
            if kind != 'coarse':
                gs.sup = torch.arange(1270, device='cuda'); gs.desc = torch.as_tensor(ctx.vhs(), device='cuda')
            if kind == 'car':
                w, pids, m = CA.support(ctx)
                cd = np.concatenate([GD.vhs_k(ctx, 16, pids), GD.prox_other(ctx, 16, pids=pids, ells=(CA.ELL,))], -1)
                loc = -np.ones(1270, np.int64); loc[pids] = np.arange(len(pids)); ap, asub = np.nonzero(w > 0)
                gs.act_pid = torch.as_tensor(ap, device='cuda'); gs.act_sub = torch.as_tensor(asub, device='cuda')
                gs.cdesc = torch.as_tensor(cd[loc[ap][:, None], CA.child_index_maps()[asub]], device='cuda')
                extra = {'w': w, 'pids': pids, 'ap': ap, 'asub': asub, 'loc': loc}
            dr.sync_thread()
            return ctx, gs, src, extra

        ctx, gs, src, extra = prepare(geom)
        torch.cuda.reset_peak_memory_stats()

        def gen():
            with torch.no_grad():
                f = model.factors(gs)
                if kind == 'coarse':
                    return model.apply(f, gs, src), None, None
                if kind == 'dense':
                    B, Bs = model.outputs(f, gs, src); return B, Bs, None
                return model.outputs_car(f, gs, src)
        B, Bs, Bc = gen()
        row['table_floats'] = int(B.numel() + (Bs.numel() if Bs is not None else 0) + (Bc.numel() if Bc is not None else 0))
        row['table_mib'] = row['table_floats'] * 4 / 2 ** 20
        if kind == 'car':
            row['active_cells'] = int(len(extra['ap'])); row['support_patches'] = int(len(extra['pids']))
            row['child_table_mib'] = Bc.numel() * 4 / 2 ** 20
            row['metadata_mib'] = (extra['w'].nbytes + extra['pids'].nbytes + gs.cdesc.numel() * 4) / 2 ** 20
        row['generator'] = med(gen); row['generator_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
        if kind == 'car':
            row['support_compute'] = med(lambda: CA.support(ctx), n=5, warm=1)
            row['child_descriptor_compute'] = med(lambda: (GD.vhs_k(ctx, 16, extra['pids']), GD.prox_other(ctx, 16, pids=extra['pids'], ells=(CA.ELL,))), n=5, warm=1)
        row['dense_descriptor_compute'] = med(lambda: ctx.vhs(), n=5, warm=1) if kind != 'coarse' else None
        # readout representation and 1-spp frame
        if kind == 'coarse':
            rep = T.pc_uniform(np.repeat(B[0].float().cpu().numpy()[:, None], 16, 1), 4)
        elif kind == 'dense':
            rep = T.pc_uniform(Bs[0].float().cpu().numpy(), 4)
        else:
            T4 = Bs[0].float().cpu().numpy(); T16 = T.upsample(T4[extra['pids']], 4, 16).copy(); bc = Bc[0].float().cpu().numpy()
            for a_, (p, s) in enumerate(zip(extra['ap'], extra['asub'])):
                T16[extra['loc'][p], CA.child_index_maps()[s]] = bc[a_]
            rep = CA.rep(T4, T16, extra['w'], extra['pids'])
            row['readout_table_mib'] = float(np.asarray(rep['T']).size * 4 / 2 ** 20)
        row['frame_512_1spp'] = med(lambda: ctx.readout_multi(light, cam, C.RES, 1, 11570001, {'x': rep}, b=1), n=8)

        def update():
            c2, g2, s2, ex2 = prepare(geom2)
            with torch.no_grad():
                f = model.factors(g2)
                o = model.apply(f, g2, s2) if kind == 'coarse' else (model.outputs(f, g2, s2) if kind == 'dense' else model.outputs_car(f, g2, s2))
            dr.sync_thread(); return o
        row['edit_to_table'] = med(lambda: update(), n=5, warm=1)
        tj = (C.WL114_ROOT if kind != 'car' else root) / 'stageC' / name / 'training.json'
        t = json.loads(tj.read_text()); row['training_s'] = t['train_s']; row['training_peak_mib'] = t['peak_mib']
        out['arms'][name] = row
        print('COST', name, json.dumps(row), flush=True)
        del model, ctx; torch.cuda.empty_cache()
    out['training_all_s'] = {a: json.loads(((root if a.startswith('CAR') else C.WL114_ROOT) / 'stageC' / a / 'training.json').read_text())['train_s']
                             for a in ('B1', 'B2', 'CAR1', 'CAR2', 'OB1')}
    C.jd(out, root / 'analysis' / 'cost.json')


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
