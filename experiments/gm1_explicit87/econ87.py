"""WL87 Stage 3: quality-matched economics of all arms (run ALONE on the GPU; protocol stage3).

  python econ87.py <R87> <R86> [--dev]
Arms: PT, PT_RR (physical path tracing), ECRO_s8601 (frozen learned table), EXP (explicit same-state transport), TF<b> (physical table
fill at b samples per slot), C3 (sender-side deposition control). Table arms share the unchanged WL86 LOW + lookup renderer.
Per edit: warm costs over repeated T0 <-> T1 edits (medians), source-only update, first-frame extra; ladders at T1 against the pooled
REF-ER image; matched quality by log-log interpolation (no extrapolation). References only score outputs.
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx87  # noqa: E402
import quad87 as Q  # noqa: E402
import xport87 as X  # noqa: E402
import pt87  # noqa: E402
PT = fx87.PT
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P87 = fx87.P87
E3S = P87['stage3']
EXPP = P87['explicit']
DEV = '--dev' in sys.argv
NB, K = S.NB, S.K


def foreign():
    me = str(os.getpid())
    return ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)


def tk(T):
    return {k: T[k] for k in ['trace_s', 'eval_s', 'gpu_ms', 'wall_s', 'launches', 'spp']}


def timed(fn):
    X.sync(); dr.kernel_history()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    X.sync()
    return r, time.perf_counter() - t0, sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())


def ref_image(root, cfg):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{cfg}_r*.npz')))
    acc = 0
    for fn in fns:
        z = np.load(fn); acc = acc + z['e1'].astype(np.float64) + z['e2'] + z['h']
    return acc / len(fns), fns


def main():
    R87, R86 = sys.argv[1], sys.argv[2]
    import ecro86 as EC
    fx87.inject(R87)
    out = os.path.join(R87, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cam = FX['camera']; res = cam['res']
    adm = json.load(open(os.path.join(R87, 'results', 'e3_admission.json'))) if os.path.exists(os.path.join(R87, 'results', 'e3_admission.json')) else {}
    edits = [e for e in E3S['edits'] if e != 'E3' or adm.get('ADMITTED') or (os.environ.get('WL87_DEV_FORCE_E3') and adm)]      # override: dev smoke only
    R = {'foreign_before': foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(),
         'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__}, 'edits': edits}
    lad = E3S['ladder_spp'] if not DEV else [4, 64]
    reps = E3S['replicates'] if not DEV else 1
    budgets = E3S['tf_budgets'] if not DEV else [16, 64]
    nedit = 3 if not DEV else 1
    t0 = time.perf_counter(); model = EC.load_model(R86, 'ecro_s8601'); X.sync(); t_model = time.perf_counter() - t0
    for ei, e in enumerate(edits):
        ex = FX[e]; lights = FX['light_sets'][ex['light_set']]
        Re = {}
        # ---- cold initialisation (fresh process for the first edit)
        t0 = time.perf_counter(); ctx = PT.Ctx(ex['T0']); lt = PT.Lights(lights); dr.sync_thread(); t_scene = time.perf_counter() - t0
        t0 = time.perf_counter()
        quad = Q.Quad(ctx.u_state_np, EXPP['M_ray'], EXPP['seed_ray'], EXPP['M_dense'], EXPP['seed_dense'])
        Wt = torch.as_tensor(quad.W.astype(np.float32), device=X.dev); X.sync(); t_quad = time.perf_counter() - t0
        t0 = time.perf_counter(); dense = quad.dense(); t_dense = time.perf_counter() - t0
        (st, t_state, g_state) = timed(lambda: EC.build_state(ctx))
        (s, t_src, g_src) = timed(lambda: ctx.source(lt))
        ((Bc, fac), t_inf, _) = timed(lambda: EC.respond(model, st, s))
        (q, t_q, g_q) = timed(lambda: Q.qtable(ctx, lt, EXPP['P'], EXPP['Mq'], EXPP['seed_q']))
        (Bx, t_app, _) = timed(lambda: X.recurrence([X.XOp(ctx.G, st.hit, st.dbin, Wt)], q)[0].cpu().numpy())
        _, Tc = ctx.render(lt, 'ER', cam, res, 64, 87800, 'ecro', B=PT.fl(Bx), plain=True, b=64)
        _, Tp = ctx.render(lt, 'ER', cam, res, 64, 87800, 'cam', plain=True, b=64)
        _, Tr = pt87.render_rr(ctx, lt, cam, res, 64, 87800, 3, b=64)
        Re['cold'] = {'first_process': ei == 0, 'model_load_s': t_model if ei == 0 else None, 'scene_build_s': t_scene, 'material_W_s': t_quad,
                      'dense_table_s': t_dense, 'state_build_s': t_state, 'source_s': t_src, 'ecro_inference_s': t_inf, 'q_s': t_q, 'exp_apply_s': t_app,
                      'first_table_render_64spp': tk(Tc), 'first_pt_render_64spp': tk(Tp), 'first_rr_render_64spp': tk(Tr)}
        print(e, 'cold', json.dumps(Re['cold']), flush=True)
        # ---- repeated edits (warm)
        ed = []
        seq = ['T1', 'T0'] * nedit + ['T1']
        for cfg in seq:
            (_, t_geo, _) = timed(lambda: ctx.set_geom(ex[cfg]))
            ((st2, nd), t_sel, g_sel) = timed(lambda: EC.update_state(ctx, st, ex['moved']))
            st = st2
            (s, t_src, g_src) = timed(lambda: ctx.source(lt))
            ((Bc, fac), t_inf, _) = timed(lambda: EC.respond(model, st, s))
            (Bcd, t_xc, _) = timed(lambda: PT.fl(Bc))
            (q, t_q, g_q) = timed(lambda: Q.qtable(ctx, lt, EXPP['P'], EXPP['Mq'], EXPP['seed_q']))
            (Bx, t_app, _) = timed(lambda: X.recurrence([X.XOp(ctx.G, st.hit, st.dbin, Wt)], q)[0].cpu().numpy())
            (Bxd, t_xx, _) = timed(lambda: PT.fl(Bx))
            ((B3, _), t_c3, g_c3) = timed(lambda: X.sender_side(ctx, q, dense, ctx.N * K, EXPP['seed_c3']))
            _, Tf1 = ctx.render(lt, 'ER', cam, res, 16, 87810, 'ecro', B=Bxd, plain=True, b=16)          # first frame after the edit
            _, Tf2 = ctx.render(lt, 'ER', cam, res, 16, 87811, 'ecro', B=Bxd, plain=True, b=16)          # second frame (resident)
            ed.append({'cfg': cfg, 'geometry_update_s': t_geo, 'state_selective_s': t_sel, 'state_selective_gpu_ms': g_sel, 'dirty_rays': int(nd),
                       'source_s': t_src, 'source_gpu_ms': g_src, 'ecro_inference_s': t_inf, 'ecro_transfer_s': t_xc,
                       'q_s': t_q, 'q_gpu_ms': g_q, 'exp_apply_s': t_app, 'exp_transfer_s': t_xx, 'c3_s': t_c3, 'c3_gpu_ms': g_c3,
                       'table_first_frame_extra_s': Tf1['wall_s'] - Tf2['wall_s']})
        Re['edits'] = ed
        keys = [k for k in ed[0] if k not in ('cfg', 'dirty_rays')]
        Re['edit_median'] = {k: float(np.median([x[k] for x in ed[1:]])) for k in keys}
        # PT / RR: geometry-only edit + first frame
        pe = []
        for cfg in ['T0', 'T1'] * 2:
            (_, t_geo, _) = timed(lambda: ctx.set_geom(ex[cfg]))
            _, T1_ = ctx.render(lt, 'ER', cam, res, 16, 87812, 'cam', plain=True, b=16)
            _, T2_ = ctx.render(lt, 'ER', cam, res, 16, 87813, 'cam', plain=True, b=16)
            pe.append({'geometry_update_s': t_geo, 'first_frame_extra_s': T1_['wall_s'] - T2_['wall_s']})
        Re['pt_edit_median'] = {k: float(np.median([x[k] for x in pe[1:]])) for k in pe[0]}
        # current geometry is T1 (state st is T1's)
        # ---- table fill: cost per budget (warm, repeated)
        tf = {}
        TFB = {}
        for bi, b in enumerate(budgets):
            rows = []
            for rep in range(2 if not DEV else 1):
                (Bt, t_tf, g_tf) = timed(lambda: pt87.table_fill(ctx, lt, b, E3S['tf_seed_base'] + 10 * bi + rep))
                rows.append({'fill_s': t_tf, 'fill_gpu_ms': g_tf})
            TFB[b] = Bt
            tf[b] = {'fill_s': float(np.median([r['fill_s'] for r in rows])), 'fill_gpu_ms': float(np.median([r['fill_gpu_ms'] for r in rows]))}
        Re['tf_cost'] = tf
        # ---- source-only update (same geometry T1, another light: the LMIX pair at the same intensity scale)
        lt2 = PT.Lights(FX['light_sets']['LMIX'])
        (s2, t_s2, g_s2) = timed(lambda: ctx.source(lt2))
        ((_, _), t_app2, _) = timed(lambda: EC.respond(model, st, s2, factors=fac))
        (q2, t_q2, g_q2) = timed(lambda: Q.qtable(ctx, lt2, EXPP['P'], EXPP['Mq'], EXPP['seed_q']))
        (_, t_x2, _) = timed(lambda: X.recurrence([X.XOp(ctx.G, st.hit, st.dbin, Wt)], q2)[0].cpu().numpy())
        ((_, _), t_c32, _) = timed(lambda: X.sender_side(ctx, q2, dense, ctx.N * K, EXPP['seed_c3']))
        Re['source_update'] = {'ecro_source_s': t_s2, 'ecro_application_s': t_app2, 'exp_q_s': t_q2, 'exp_apply_s': t_x2, 'c3_s': t_c32,
                               'tf_fill_s': {b: v['fill_s'] for b, v in tf.items()}}
        Re['memory'] = {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'process': PT.gpu_mem(),
                        'state_bytes': int(st.hit.nbytes + st.dbin.nbytes + st.seg.nbytes), 'ecro_factor_bytes': int(sum(v.numel() * 4 for v in fac.values())),
                        'exp_W_bytes': int(quad.W.astype(np.float32).nbytes), 'dense_table_bytes': int(dense.astype(np.float32).nbytes),
                        'table_bytes': int(Bx.size * 4), 'ecro_params': sum(p.numel() for p in model.parameters())}
        # ---- ladders at T1
        tables = {'ECRO_s8601': Bc, 'EXP': Bx, 'C3': B3}
        for b in budgets:
            tables[f'TF{b}'] = TFB[b]
        ref_root = R87 if e == 'E3' else R86
        ref, fns = ref_image(ref_root, f'{e}_T1')
        mz = np.load(os.path.join(ref_root, 'results', 'masks', f'{e}.npz'))
        mk = {k: mz[k] for k in ('stationary_all', 'A', 'C', 'D')}
        err = lambda img: {k: float(np.abs(img[m] - ref[m]).sum() / ref[m].sum()) for k, m in mk.items()}
        L = {}
        arms = ['PT', 'PT_RR'] + list(tables)
        for spp in lad:
            for qr in range(reps):
                sd = E3S['seed_base'] + 97 * spp + qr
                bsz = min(64, spp)
                for ai, arm in enumerate(arms):
                    if arm == 'PT':
                        a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd, 'cam', plain=True, b=bsz)
                    elif arm == 'PT_RR':
                        a, T_ = pt87.render_rr(ctx, lt, cam, res, spp, sd + 7, 3, b=bsz)
                    else:
                        a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd + 50, 'ecro', B=PT.fl(tables[arm]), plain=True, b=bsz)
                    img = PT.pack(a, res, spp)['tot'].astype(np.float64)
                    L.setdefault(arm, []).append({'spp': spp, 'rep': qr, **tk(T_), **{'err_' + k: v for k, v in err(img).items()}})
            print(e, 'ladder', spp, json.dumps({a: round(L[a][-1]['err_stationary_all'], 4) for a in arms}), flush=True)
        Re['ladder'] = L
        Re['ladder_median'] = {}
        for arm in L:
            by = {}
            for row in L[arm]:
                by.setdefault(row['spp'], []).append(row)
            Re['ladder_median'][arm] = [{'spp': s_, **{k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ('spp', 'rep', 'launches')}}
                                        for s_, rows in sorted(by.items())]
        # ---- second view (amortisation over views): render only, 64 spp, warm
        v2 = {}
        for arm in ('PT', 'EXP'):
            for rep in range(2):
                if arm == 'PT':
                    _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 87820 + rep, 'cam', plain=True, b=64)
                else:
                    _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 87820 + rep, 'ecro', B=PT.fl(Bx), plain=True, b=64)
            v2[arm] = tk(Tv)
        Re['view2_64spp'] = v2
        Re['ref_files'] = [os.path.basename(f) for f in fns]
        R[e] = Re
        json.dump(R, open(os.path.join(out, 'econ87_partial.json'), 'w'), indent=1)
    R['foreign_after'] = foreign()
    json.dump(R, open(os.path.join(out, 'econ87.json'), 'w'), indent=1)
    print('ECON_DONE', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
