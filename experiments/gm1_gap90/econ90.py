"""WL90 cost measurements for the feasibility probe (run ALONE on the GPU). python econ90.py <R90> [--dev] -> results/econ/econ90.json

Warm, synchronised wall clock; every timed operation is executed once untimed first (compilation excluded); medians over repeats and sources.
Measured on W1 T1 (T0 <-> T1 edits) for the deployed relations:
  per source   q table (WL87)                     | exact once-scattered radiance per relation ray (R0 rays, C3UR rays)
               table recurrence (4 orders)        | first gather from rays + 3 table applications
  per edit     relation update (R0 state / C3UR)  | + probe-ray buffer rebuild (the buffer is geometry-dependent, source-independent)
Diagnostic operators (pooled exact reads, hybrids) are not deployable and are not timed as candidates; their wall time is in the table metadata.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx90  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx90.PT, fx90.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P90, E87 = fx90.P90, fx90.E87
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def timed(fn, warm=False):
    if warm:
        fn()
    sync(); dr.kernel_history()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return r, time.perf_counter() - t0, sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())


def main():
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    import gap90 as GP
    import econ89
    R = sys.argv[1]
    R8.inject_choice(R)
    fx90.fx88.setup('W1_T0')
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    EP = P90['economics']
    Rr = {'foreign_before': econ89.foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip()}
    ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']])
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    mat = R8.materials(ctx)
    S1 = fx90.P89['stage1']
    st = EC.build_state(ctx)
    rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    nrep = 2 if DEV else EP['n_repeat']
    ed = []
    for ei, cfg in enumerate(['T1', 'T0'] * (nrep + 1) + ['T1']):
        row = {'cfg': cfg}
        ctx.set_geom(ex[cfg])
        ((st2, nd), row['R0_state_update_s'], _) = timed(lambda: EC.update_state(ctx, st, ex['moved']))
        st = st2
        (r0, row['R0_coeff_s'], _) = timed(lambda: X.XOp(ctx.G, st.hit, st.dbin, quad.W))
        (_, row['C3UR_update_s'], _) = timed(lambda: rc.update(ctx, ex['moved']))
        ((idx, rb0), row['R0_raybuf_s'], _) = timed(lambda: GP.r0_raybuf(st))
        (rbc, row['C3UR_raybuf_s'], _) = timed(lambda: GP.c3ur_raybuf(rc))
        (q, row['q_s'], row['q_gpu_ms']) = timed(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']))
        (rad0, row['R0_exactL1_s'], row['R0_exactL1_gpu_ms']) = timed(lambda: rb0.radiance(ctx, lt))
        (radc, row['C3UR_exactL1_s'], row['C3UR_exactL1_gpu_ms']) = timed(lambda: rbc.radiance(ctx, lt))
        (_, row['R0_apply4_s'], _) = timed(lambda: RL.recurrence([r0], q)[0].cpu().numpy())
        (_, row['C3UR_apply4_s'], _) = timed(lambda: RL.recurrence([rc], q)[0].cpu().numpy())
        (_, row['R0X1_first_plus_3_s'], _) = timed(lambda: GP.rec_from(r0, GP.r0_first_exact(r0, ctx.N, idx, rad0)).cpu().numpy())
        (_, row['C3URX1_first_plus_3_s'], _) = timed(lambda: GP.rec_from(rc, GP.c3ur_first(rc, radc)).cpu().numpy())
        row['rays'] = {'R0_hit_rays': int(len(idx)), 'C3UR_landed_rays': int(rc.n_landed)}
        if ei >= 2:
            ed.append(row)
        print('edit', cfg, json.dumps({k: round(v, 5) for k, v in row.items() if isinstance(v, float)}), flush=True)
    Rr['edits'] = ed
    Rr['edit_median'] = {k: float(np.median([e[k] for e in ed])) for k in ed[0] if isinstance(ed[0][k], float)}
    # source-only updates at fixed geometry (T1): relation and ray buffers resident
    srcs = [FX['light_sets'][k] for k in ('LMIX89', 'LSPOT89')] + econ89.random_lights(2 if DEV else EP['n_sources'], EP['source_seed'], ex['T1'])
    rows = []
    for li, ls in enumerate(srcs):
        lt2 = PT.Lights(ls)
        row = {'lights': len(ls)}
        (q2, row['q_s'], _) = timed(lambda: Q.qtable(ctx, lt2, E87['P'], E87['Mq'], E87['seed_q']), warm=li < 2)
        (rad0, row['R0_exactL1_s'], _) = timed(lambda: rb0.radiance(ctx, lt2), warm=li < 2)
        (radc, row['C3UR_exactL1_s'], _) = timed(lambda: rbc.radiance(ctx, lt2), warm=li < 2)
        (_, row['C3UR_apply4_s'], _) = timed(lambda: RL.recurrence([rc], q2)[0].cpu().numpy(), warm=li < 2)
        (_, row['C3URX1_first_plus_3_s'], _) = timed(lambda: GP.rec_from(rc, GP.c3ur_first(rc, radc)).cpu().numpy(), warm=li < 2)
        (_, row['R0_apply4_s'], _) = timed(lambda: RL.recurrence([r0], q2)[0].cpu().numpy(), warm=li < 2)
        (_, row['R0X1_first_plus_3_s'], _) = timed(lambda: GP.rec_from(r0, GP.r0_first_exact(r0, ctx.N, idx, rad0)).cpu().numpy(), warm=li < 2)
        rows.append(row)
    Rr['source_updates'] = rows
    one = [r for r in rows if r['lights'] == 1]
    Rr['source_update_median_single_light'] = {k: float(np.median([r[k] for r in one])) for k in one[0] if isinstance(one[0][k], float)}
    Rr['memory'] = {'R0_raybuf_bytes': int(rb0.n * 6 * 4), 'C3UR_raybuf_bytes': int(rbc.n * 6 * 4), 'per_ray_radiance_bytes_R0': int(rb0.n * 3 * 4),
                    'per_ray_radiance_bytes_C3UR': int(rbc.n * 3 * 4), 'q_table_bytes': int(ctx.N * NB * 3 * 4), 'C3UR_relation_bytes': rc.nbytes(),
                    'R0_state_bytes': int(st.hit.nbytes + st.dbin.nbytes + st.seg.nbytes), 'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
    Rr['foreign_after'] = econ89.foreign()
    json.dump(Rr, open(os.path.join(out, 'econ90.json'), 'w'), indent=1)
    print('ECON_DONE', json.dumps({k: round(v * 1000, 2) for k, v in Rr['edit_median'].items()}), flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
