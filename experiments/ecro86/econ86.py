"""WL86 Stage 4: measured end-to-end economics, ECRO vs full frozen-ER path tracing (run ALONE on the GPU).

  python econ86.py <root> <model_tag> [--dev]
For E1 (primary) and E2: time-to-quality ladders of the plain path tracer (events <= 6) and of plain ECRO (LOW + table lookup) against the
pooled REF-ER image of T1, and every ECRO cost after an edit: geometry update, selective / full state rays, source encoding, generator +
application (torch), table transfer, render. Costs are medians over repeated edits; cold initialisation is measured first in this fresh
process. No reference or target data is used by the ECRO path (the references only score the outputs).
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecro86 as EC  # noqa: E402
PT, M, S = EC.PT, EC.M, EC.S
mi, dr = PT.mi, PT.dr
FX = PT.FX
E = S.P['stage4']
DEV = '--dev' in sys.argv
dev = torch.device('cuda')


def foreign():
    me = str(os.getpid())
    return ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)


def tk(T):
    return {k: T[k] for k in ['trace_s', 'eval_s', 'gpu_ms', 'wall_s', 'launches', 'spp']}


def timed(fn):
    EC.sync(); dr.kernel_history()
    t0 = time.perf_counter()
    r = fn()
    EC.sync()
    return r, time.perf_counter() - t0, sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())


def ref_image(root, cfg):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{cfg}_r*.npz')))
    acc = 0
    for fn in fns:
        z = np.load(fn); acc = acc + z['e1'].astype(np.float64) + z['e2'] + z['h']
    return acc / len(fns)


def main():
    root, tag = sys.argv[1], sys.argv[2]
    out = os.path.join(root, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cam = FX['camera']; res = cam['res']
    R = {'foreign_before': foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(),
         'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__}, 'model': tag}
    lad = E['ladder_spp'] if not DEV else [4, 64]
    reps = E['ladder_replicates'] if not DEV else 1
    for ei, e in enumerate(('E1', 'E2')):
        ex = FX[e]; lights = FX['light_sets'][ex['light_set']]
        Re = {}
        # ---- cold initialisation (first workload only: fresh process)
        t0 = time.perf_counter(); model = EC.load_model(root, tag); EC.sync(); t_model = time.perf_counter() - t0
        t0 = time.perf_counter(); ctx = PT.Ctx(ex['T0']); lt = PT.Lights(lights); dr.sync_thread(); t_scene = time.perf_counter() - t0
        (st, t_state, g_state) = timed(lambda: EC.build_state(ctx))
        (s, t_src, g_src) = timed(lambda: ctx.source(lt))
        ((B, f), t_inf, _) = timed(lambda: EC.respond(model, st, s))
        (Bd, t_xfer, _) = timed(lambda: PT.fl(B))
        acc, Tc = ctx.render(lt, 'ER', cam, res, 64, 86800, 'ecro', B=Bd, plain=True, b=64)
        accp, Tp = ctx.render(lt, 'ER', cam, res, 64, 86800, 'cam', plain=True, b=64)
        Re['cold'] = {'model_load_s': t_model, 'scene_build_s': t_scene, 'state_build_s': t_state, 'source_s': t_src, 'inference_s': t_inf, 'transfer_s': t_xfer,
                      'first_ecro_render_64spp': tk(Tc), 'first_pt_render_64spp': tk(Tp), 'first_process': ei == 0}
        # ---- repeated edits (warm): T0 <-> T1
        ed = []
        seq = ['T1', 'T0'] * (E['repeated_edits'] if not DEV else 1) + ['T1']
        for cfg in seq:
            (_, t_geo, _) = timed(lambda: ctx.set_geom(ex[cfg]))
            ((st2, nd), t_sel, g_sel) = timed(lambda: EC.update_state(ctx, st, ex['moved']))
            (stf, t_full, g_full) = timed(lambda: EC.build_state(ctx))
            same = bool(np.array_equal(st2.hit, stf.hit) and np.array_equal(st2.dbin, stf.dbin))
            st = st2
            (s, t_src, g_src) = timed(lambda: ctx.source(lt))
            ((B, f), t_inf, _) = timed(lambda: EC.respond(model, st, s))
            (Bd, t_xfer, _) = timed(lambda: PT.fl(B))
            _, Tr = ctx.render(lt, 'ER', cam, res, 16, 86810, 'ecro', B=Bd, plain=True, b=16)      # first frame after the edit (acceleration structure refit included)
            _, Tq = ctx.render(lt, 'ER', cam, res, 16, 86811, 'ecro', B=Bd, plain=True, b=16)      # second frame (resident)
            ed.append({'cfg': cfg, 'geometry_update_s': t_geo, 'state_selective_s': t_sel, 'state_selective_gpu_ms': g_sel, 'dirty_rays': int(nd),
                       'state_full_s': t_full, 'state_full_gpu_ms': g_full, 'selective_equals_full': same, 'source_s': t_src, 'source_gpu_ms': g_src,
                       'inference_s': t_inf, 'transfer_s': t_xfer, 'first_frame_16spp': tk(Tr), 'second_frame_16spp': tk(Tq)})
        Re['edits'] = ed
        med = lambda k: float(np.median([x[k] for x in ed[1:]]))
        Re['edit_median'] = {k: med(k) for k in ['geometry_update_s', 'state_selective_s', 'state_selective_gpu_ms', 'state_full_s', 'state_full_gpu_ms', 'source_s',
                                                  'source_gpu_ms', 'inference_s', 'transfer_s']}
        Re['edit_median']['first_frame_extra_wall_s'] = float(np.median([x['first_frame_16spp']['wall_s'] - x['second_frame_16spp']['wall_s'] for x in ed[1:]]))
        # geometry-only edit for the path tracer (same in-place update + first frame)
        pe = []
        for cfg in ['T0', 'T1'] * 2:
            (_, t_geo, _) = timed(lambda: ctx.set_geom(ex[cfg]))
            _, T1_ = ctx.render(lt, 'ER', cam, res, 16, 86812, 'cam', plain=True, b=16)
            _, T2_ = ctx.render(lt, 'ER', cam, res, 16, 86813, 'cam', plain=True, b=16)
            pe.append({'geometry_update_s': t_geo, 'first_frame_extra_wall_s': T1_['wall_s'] - T2_['wall_s']})
        Re['pt_edit_median'] = {k: float(np.median([x[k] for x in pe[1:]])) for k in pe[0]}
        Re['memory'] = {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'process': PT.gpu_mem(),
                        'state_bytes': int(st.hit.nbytes + st.dbin.nbytes + st.seg.nbytes), 'factor_bytes': int(sum(v.numel() * 4 for v in f.values())),
                        'table_bytes': int(B.size * 4), 'model_params': sum(p.numel() for p in model.parameters())}
        # ---- time-to-quality ladders at T1 (state of T1 is current after the edit sequence)
        ref = ref_image(root, f'{e}_T1')
        mz = np.load(os.path.join(root, 'results', 'masks', f'{e}.npz'))
        mk = {k: mz[k] for k in ('stationary_all', 'A', 'C', 'D')}
        err = lambda img: {k: float(np.abs(img[m] - ref[m]).sum() / ref[m].sum()) for k, m in mk.items()}
        L = {'pt': [], 'ecro': []}
        for spp in lad:
            for q in range(reps):
                b = min(128, spp)
                a, T_ = ctx.render(lt, 'ER', cam, res, spp, E['ladder_seed_base'] + 97 * spp + q, 'cam', plain=True, b=b)
                L['pt'].append({'spp': spp, 'rep': q, **tk(T_), **{'err_' + k: v for k, v in err(PT.pack(a, res, spp)['tot'].astype(np.float64)).items()}})
                a, T_ = ctx.render(lt, 'ER', cam, res, spp, E['ladder_seed_base'] + 97 * spp + q + 50, 'ecro', B=Bd, plain=True, b=b)
                L['ecro'].append({'spp': spp, 'rep': q, **tk(T_), **{'err_' + k: v for k, v in err(PT.pack(a, res, spp)['tot'].astype(np.float64)).items()}})
            print(e, 'ladder', spp, json.dumps(L['pt'][-1]), json.dumps(L['ecro'][-1]), flush=True)
        Re['ladder'] = L
        Re['ladder_median'] = {}
        for arm in L:
            by = {}
            for row in L[arm]:
                by.setdefault(row['spp'], []).append(row)
            Re['ladder_median'][arm] = [{'spp': s_, **{k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ('spp', 'rep', 'launches')}} for s_, rows in sorted(by.items())]
        # ---- second view with the same state (multi-view amortisation): render only
        _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 86820, 'ecro', B=Bd, plain=True, b=64)
        _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 86821, 'ecro', B=Bd, plain=True, b=64)
        _, Tw = ctx.render(lt, 'ER', FX['camera2'], res, 64, 86822, 'cam', plain=True, b=64)
        _, Tw = ctx.render(lt, 'ER', FX['camera2'], res, 64, 86823, 'cam', plain=True, b=64)
        Re['view2_64spp'] = {'ecro': tk(Tv), 'pt': tk(Tw)}
        R[e] = Re
    R['foreign_after'] = foreign()
    # ---- matched-quality comparison
    qj = json.load(open(os.path.join(root, 'results', 'qualify.json')))
    for e in ('E1', 'E2'):
        g = qj[e]['regions']['stationary_all']['g_ER_vs_P_T1']
        Re = R[e]
        Re['g_stationary'] = g
        em = Re['edit_median']
        fixed_wall = em['geometry_update_s'] + em['state_selective_s'] + em['source_s'] + em['inference_s'] + em['transfer_s'] + max(em['first_frame_extra_wall_s'], 0.0)
        fixed_wall_full = fixed_wall - em['state_selective_s'] + em['state_full_s']
        pt_fixed = Re['pt_edit_median']['geometry_update_s'] + max(Re['pt_edit_median']['first_frame_extra_wall_s'], 0.0)
        Re['ecro_fixed_wall_s'] = fixed_wall; Re['ecro_fixed_wall_full_state_s'] = fixed_wall_full; Re['pt_fixed_wall_s'] = pt_fixed

        def at(arm, q, ck):
            med = Re['ladder_median'][arm]
            ee = [m['err_stationary_all'] for m in med]; cc = [m[ck] for m in med]
            if q >= ee[0]:
                return {'value': cc[0], 'spp': med[0]['spp'], 'how': 'first ladder point already below q'}
            for i in range(len(ee) - 1):
                if ee[i] >= q >= ee[i + 1] and ee[i + 1] < ee[i]:
                    s_ = math.log(q / ee[i]) / math.log(ee[i + 1] / ee[i])
                    return {'value': float(cc[i] * (cc[i + 1] / cc[i]) ** s_), 'spp': float(med[i]['spp'] * (med[i + 1]['spp'] / med[i]['spp']) ** s_), 'how': 'interpolated',
                            'region_err': {k: float(med[i]['err_' + k] * (med[i + 1]['err_' + k] / med[i]['err_' + k]) ** s_) for k in ('A', 'C', 'D')}}
            return {'value': None, 'how': 'not reached on the ladder', 'best_error': min(ee)}
        Re['matched'] = {}
        for name, q in [('0.10', 0.10), ('0.05', 0.05), ('0.03', 0.03), ('g', g)]:
            row = {'q': q}
            for arm in ('pt', 'ecro'):
                row[arm] = {'gpu_ms': at(arm, q, 'gpu_ms'), 'wall_s': at(arm, q, 'wall_s')}
            p, c = row['pt'], row['ecro']
            if p['wall_s']['value'] is not None and c['wall_s']['value'] is not None:
                row['resident_frame_speedup_gpu'] = p['gpu_ms']['value'] / c['gpu_ms']['value']
                row['resident_frame_speedup_wall'] = p['wall_s']['value'] / c['wall_s']['value']
                row['edit_to_first_frame_speedup_wall'] = (p['wall_s']['value'] + pt_fixed) / (c['wall_s']['value'] + fixed_wall)
                row['edit_to_first_frame_speedup_wall_full_state'] = (p['wall_s']['value'] + pt_fixed) / (c['wall_s']['value'] + fixed_wall_full)
                fixed_gpu = em['state_selective_gpu_ms'] + em['source_gpu_ms'] + 1e3 * em['inference_s']
                row['edit_to_first_frame_speedup_gpu'] = p['gpu_ms']['value'] / (c['gpu_ms']['value'] + fixed_gpu)
                rp, rc = p['wall_s'].get('region_err'), c['wall_s'].get('region_err')
                row['region_matched'] = bool(rp is not None and rc is not None and all(rc[k] <= E['region_tolerance'] * rp[k] for k in rp)) if rp and rc else None
            Re['matched'][name] = row
    json.dump(R, open(os.path.join(out, 'econ86.json'), 'w'), indent=1)
    print(json.dumps({e: {k: {kk: vv for kk, vv in v.items() if 'speedup' in kk or kk == 'region_matched'} for k, v in R[e]['matched'].items()} for e in ('E1', 'E2')}), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
