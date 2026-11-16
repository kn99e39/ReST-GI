"""WL98 critic X3 (systems / cost): first-principles recomputation of candidate cost claims.

Design input only, not evidence of record. numpy only, local CPU, reads recorded JSON (read-only):
  results/gm1_c3urx1_91/results/econ/{ladder91,life91,scale91}.json   (WL91, RTX 3080 Ti)
  results/ecro_util96a/cost_s1.0.json                                  (WL96A)
  results/audit97/E5/e5_summary.json  + numbers transcribed from results/audit97/E5/e5_tables.txt (section 4)
Writes results/wl98_calc/X3/cost_check.json.

Sections
  A  measured per-unit rates (tracing, ER evaluation, gathers, frame per spp, generator scaling exponent)
  B  WL91 ladder: bias-floor / noise-constant decomposition err^2 = f^2 + c^2/spp per arm, draw, region
  C  quality-matched utility bound for any table-space learned replacement on GM1 (frame-dominated)
  D  candidate counts: FLS rays / readout, XQT nnz / apply / readout, ECRO-FC generator, RB-DTO apply, LIEM vs walk
"""
import json, math, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
out = {}

# ---------------- A. measured rates [MEAS: E5 section 4, run1 s1.0 unless noted] ----------------
E5 = {  # N: (one_order_gpu_ms, q_exec_ms, gen_gpu_ms, dirty_test_torch_ms, retrace_drjit_ms, dirty_rays, build_full_drjit_ms, relation_rays, frame64_exec_ms)
    1270: (0.528, 1.844, 4.204, 4.434, 0.053, 49936, 0.176, 650240, 22.670),
    2802: (0.685, 4.088, 8.668, 7.274, 0.078, 98664, 0.339, 1434624, 23.723),
    5080: (0.911, 7.245, 15.694, 13.120, 0.104, 199814, 0.590, 2600960, 23.055),
    11430: (2.282, 16.674, 37.676, 43.537, 0.226, 449970, 1.502, 5852160, 23.543),
}
rates = {}
for N, (o, q, g, dt, rt, dr, bf, rr, fr) in E5.items():
    rates[N] = dict(
        ns_per_gather_entry=o * 1e6 / rr,                # one binned order: N*K relation entries
        ns_per_retraced_ray=rt * 1e6 / dr,               # Dr.Jit exec of the selective re-trace
        ns_per_ray_full_build=bf * 1e6 / rr,             # Dr.Jit exec of the full relation build (torch part not profiled)
        ns_per_relation_entry_dirty_test=dt * 1e6 / rr,  # torch dirty test over every relation ray
        ps_per_ER_eval_qtable=q * 1e9 / (N * 16 * 1024), # q table: N*16 source points * 1024 ER evals (20.8 M at N 1270)
        generator_ms=g, frame64_ms=fr,
        frame_ns_per_sample_per_spp=fr * 1e6 / (512 * 512 * 64),
    )
Ns = np.array(sorted(E5)); gen = np.array([E5[n][2] for n in Ns])
expl = np.array([E5[n][1] + 4 * E5[n][0] for n in Ns])          # q (ER half) + 4 orders, kernel
out['A_rates'] = rates
out['A_generator_exponent'] = float(np.polyfit(np.log(Ns), np.log(gen), 1)[0])
out['A_generator_over_replaced_explicit'] = {int(n): float(g / e) for n, g, e in zip(Ns, gen, expl)}

# ---------------- B. ladder decomposition [MEAS: ladder91.json; fit is ARITH] ----------------
lad = J('results/gm1_c3urx1_91/results/econ/ladder91.json')['configs']
dec = {}
for cfg, v in lad.items():
    for arm, rows in v['ladder'].items():
        for reg in ('stationary_all', 'R_mirror', 'C_ss'):
            spp = np.array([r['spp'] for r in rows], float); e = np.array([r['err_' + reg] for r in rows])
            gpu = np.array([r['gpu_ms'] for r in rows])
            m = (spp >= 64) & (spp <= 4096)
            A = np.stack([np.ones(m.sum()), 1.0 / spp[m]], 1)
            sol, *_ = np.linalg.lstsq(A, e[m] ** 2, rcond=None)
            f2, c2 = sol
            ms_per_spp = float(np.median(gpu[spp >= 64] / spp[spp >= 64]))
            dec[f'{cfg}|{arm}|{reg}'] = dict(floor=float(math.sqrt(max(f2, 0.0))), floor2_raw=float(f2),
                                              c=float(math.sqrt(max(c2, 0))), ms_per_spp_kernel=ms_per_spp)
out['B_ladder_decomposition'] = dec

def spp_to_tier(f, c, tier):
    return math.inf if tier <= f else c * c / (tier * tier - f * f)

# ---------------- C. quality-matched bound for a table-space learned replacement ----------------
# Explicit comparator allowed to pool m relation draws (WL91: pooled X1 = oracle in expectation).
# Engineered table cost model per edit at N 1,270 (kernel basis), per pooled draw:
r = rates[1270]
draw_ms = dict(retrace_full=r['ns_per_ray_full_build'] * 650240 / 1e6,          # full relation re-trace per draw
               x1_eval=1.844 * 650240 / 20.8e6,                                  # ER evals per landed ray (q-table rate)
               gather=0.528)                                                     # one exact first gather
per_draw = sum(draw_ms.values())
C = {'per_draw_ms_engineered': per_draw, 'parts': draw_ms}
shared_kernel = 22.67 / 64  # frame ms per spp (kernel)
fixed_kernel = 4.5 + 2.0 + 1.84 + 3 * 0.528     # relation update + XOp + q + 3 tail orders (kernel, shared by all table arms)
rows = []
for m in (1, 4, 16, 64):
    for spp in (64, 256, 1024, 4096):
        Lexp = fixed_kernel + m * per_draw + spp * shared_kernel
        Llearn0 = fixed_kernel + spp * shared_kernel   # learned table at zero cost reaching the same floor
        rows.append(dict(draws=m, spp=spp, U_zero_cost_learned=Lexp / Llearn0))
C['U_bound_rows'] = rows
# tier example from the decomposition (W1_T1 R_mirror, C3URX1 / C3UR / R0 deployed draw s86001)
ex = {}
for arm in ('C3URX1_s86001', 'C3UR_s86001', 'R0_s86001', 'PT'):
    d = dec[f'W1_T1|{arm}|R_mirror']
    ex[arm] = {f'tier_{t}': dict(spp=spp_to_tier(d['floor'], d['c'], t),
                                 frame_ms=spp_to_tier(d['floor'], d['c'], t) * d['ms_per_spp_kernel'])
               for t in (0.10, 0.06, 0.04, 0.03)}
C['W1_T1_R_mirror_time_to_tier_kernel'] = ex
out['C_quality_matched'] = C

# ---------------- D. candidate counts [PRED from stated design parameters] ----------------
D = {}
N, Kst = 1270, 512
# FLS (DESIGN_REVIEW): K lobes, Q cubature nodes, J sweeps, state-dependent rays traced every update
for (Kl, Q, Js, L) in [(8, 13, 4, 1), (8, 13, 4, 4), (8, 33, 4, 1)]:
    for n in (1270, 11430):
        rays = n * Kl * Q * Js * L
        D[f'FLS_rays_N{n}_K{Kl}_Q{Q}_J{Js}_Lstates{L}'] = dict(
            rays_per_update=rays, kernel_ms_at_measured_trace_rate=rays * rates[1270]['ns_per_ray_full_build'] / 1e6,
            vs_explicit_selective_retrace=rays / (49936 if n == 1270 else 449970))
# FLS per-lobe flux quantum: one cubature node carries 1/Q of a lobe
D['FLS_node_flux_fraction'] = {Q: 1.0 / Q for Q in (13, 33, 65)}
# Frame readout: bytes fetched per camera sample at x2
samples = 512 * 512 * 64
D['readout_bytes_per_frame64_GB'] = {
    'table_fetch_3f': samples * 12 / 1e9,
    'FLS_8_lobes_12f': samples * 8 * 12 * 4 / 1e9,
    'RBDTO_16_lobes_10f_plus_broad_64f': samples * (16 * 10 + 64) * 4 / 1e9,
    'XQT_sum_64_incident_cells_6f': samples * 64 * 6 * 4 / 1e9,
}
D['readout_ms_at_900GBps_lower_bound'] = {k: v / 900 * 1e3 for k, v in D['readout_bytes_per_frame64_GB'].items()}
# XQT (D1): nnz, apply per flight, regeneration
for n_nb in (9, 25):
    nnz = N * Kst * n_nb
    D[f'XQT_nnz_nnb{n_nb}'] = dict(nnz=nnz, MB_values_fp32=nnz * 4 / 1e6, MB_with_int32_index=nnz * 8 / 1e6,
                                   apply_ms_per_flight_at_gather_rate=nnz * rates[1270]['ns_per_gather_entry'] / 1e6,
                                   apply_ms_per_flight_at_N11430_rate=nnz * rates[11430]['ns_per_gather_entry'] / 1e6,
                                   break_even_eta_vs_X1_draws=n_nb)
# indirect exact sample: sum over nonzero incident cells per landed ray
for cells in (64, 256):
    evals = 526287 * cells
    D[f'XQT_indirect_eval_cells{cells}'] = dict(evals=evals, ms_at_qtable_ER_rate=evals * rates[1270]['ps_per_ER_eval_qtable'] / 1e9)
# ECRO-FC: heads run over N*NB tokens; scale generator by NB ratio (upper end) and message-passing share (lower end)
for n in (1270, 11430):
    g = E5[n][2]
    D[f'ECROFC_generator_8x32_N{n}_ms'] = dict(current_64bins=g, if_heads_dominate_x4=4 * g, state_rays_per_bin=512 / 256)
# RB-DTO (D3): incident samples per body per iteration
for (Nb, Kext, agg) in [(400, 512, None), (400, 512, 64), (400, 512, 6)]:
    M_in = Nb * (agg if agg else Kext)
    for tq in (5, 20):
        D[f'RBDTO_Nb{Nb}_agg{agg}_tq{tq}ns'] = dict(M_in_per_body=M_in, ms_per_body_per_iter=M_in * tq / 1e6,
                                                     ms_20bodies_4iter=20 * 4 * M_in * tq / 1e6)
# explicit intra-body twin with a CACHED (rigid-invariant) body relation: gathers only, measured gather rates
for (nb, Nb, Ki, Ji) in [(2, 200, 4096, 8), (20, 400, 4096, 8), (20, 400, 16384, 16)]:
    ent = nb * Nb * Ki
    D[f'RBDTO_twin1_cached_b{nb}_Nb{Nb}_K{Ki}_J{Ji}'] = dict(
        entries_per_order=ent,
        ms_at_0p81ns=Ji * ent * 0.81e-6, ms_at_0p39ns=Ji * ent * rates[11430]['ns_per_gather_entry'] * 1e-6,
        GB_relation_8B=ent * 8 / 1e9)
# LIEM-S vs explicit walk per exit sample (ray casts at measured full-build rate; MLP eval cost assumed)
ray_ns = rates[1270]['ns_per_ray_full_build']
for ev in (2.5, 10, 30, 50):
    for mlp_kflop in (20, 200):
        net_ns = mlp_kflop * 1e3 / 10e12 * 1e9   # 10 TFLOP/s effective
        D[f'LIEM_events{ev}_mlp{mlp_kflop}k'] = dict(walk_ns=ev * ray_ns, liem_ns=net_ns + 2 * ray_ns,
                                                   ratio_walk_over_liem=(ev * ray_ns) / (net_ns + 2 * ray_ns))
out['D_candidates'] = D

os.makedirs(os.path.join(ROOT, 'results', 'wl98_calc', 'X3'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'wl98_calc', 'X3', 'cost_check.json'), 'w', newline='\n') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: out[k] for k in ('A_generator_exponent', 'A_generator_over_replaced_explicit')}, indent=1))
for N_, rr in rates.items():
    print(N_, {k: round(v, 3) for k, v in rr.items()})
for k in ('W1_T1', 'W2_T1', 'W1_T1_LMIX89'):
    for arm in ('PT', 'R0_s86001', 'C3UR_s86001', 'C3URX1_s86001', 'R0_s89201', 'C3UR_s89201', 'C3URX1_s89201', 'C3URX1_s89202', 'C3UR_s89202'):
        for reg in ('R_mirror', 'stationary_all'):
            d = dec[f'{k}|{arm}|{reg}']
            print(k, arm, reg, 'floor %.4f (f2 %.2e) c %.3f ms/spp %.3f' % (d['floor'], d['floor2_raw'], d['c'], d['ms_per_spp_kernel']))
print('per_draw', C['per_draw_ms_engineered'], C['parts'])
for row in C['U_bound_rows']:
    print(row)
print(json.dumps(C['W1_T1_R_mirror_time_to_tier_kernel'], indent=0))
for k, v in D.items():
    print(k, v)
