"""WL98 critic X3 (systems / cost), part 2: decisive recomputations used in the X3 report.

Design input only, not evidence of record. numpy only, local CPU. Reads recorded JSON (read-only):
  results/gm1_c3urx1_91/results/econ/{ladder91,life91,scale91}.json   (WL91, RTX 3080 Ti, LabServer63)
  results/ecro_util96a/cost_s1.0.json                                  (WL96A)
  results/wl98_calc/D3/cost_model.json                                 (designer D3 scratch, for its body_regimes inputs)
Kernel numbers of WL97 E5 section 4 are transcribed from results/audit97/E5/e5_tables.txt (run1 s1.0 and run6 s3.0).
Writes results/wl98_calc/X3/cost_check2.json.

Sections
  F  kernel-basis Amdahl ceilings for table-level learned roles at N 1,270 and 11,430 (frame / no frame / 1 spp,
     recorded dirty test vs an assumed engineered dirty test)
  G  ECRO generator FLOP split (dense message passing vs per-(patch,bin) heads) and ECRO-FC at 256 bins
  H  XQT per-flight cost (indirect exact ER sum + sparse apply) and per-edit W regeneration
  I  quality-matched lever through the frame: time-to-tier from the WL91 ladders and its sensitivity to the floor
  J  explicit draw pooling across steady-state frames (latency window of a learned advantage)
  K  RB-DTO regime map recomputed with symmetric sample counts and the best explicit twin
  L  LIEM per-sample break-even vs the explicit walk
"""
import json, math, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
out = {}

# ---- recorded kernel components [MEAS, WL97 E5 section 4; torch_gpu or drjit_exec, ms] ----
E5K = {
    1270: dict(dirty_test=4.434, retrace=0.053, xop=2.015, q=1.844, order=0.528, gen=4.204, ecro_apply=0.758,
               frame64=22.670, frame16=5.618, frame4=1.470, frame1=0.469, geom=0.043, relation_rays=650240, dirty_rays=49936),
    11430: dict(dirty_test=43.537, retrace=0.226, xop=21.823, q=16.674, order=2.282, gen=37.676, ecro_apply=3.812,
                frame64=23.543, frame16=5.946, frame4=1.560, frame1=0.461, geom=0.054, relation_rays=5852160, dirty_rays=449970),
}

# ---------------- F. Amdahl ceilings, kernel basis ----------------
F = {}
for N, k in E5K.items():
    for dt_label, dirty in (('recorded_dirty_test', k['dirty_test']),
                            ('engineered_dirty_test_0p5ns', 0.5e-6 * k['relation_rays'])):
        shared_noframe = k['geom'] + dirty + k['retrace'] + k['xop']
        replaceable = k['q'] + 4 * k['order']                     # q ER-half (upper: whole q) + 4 orders
        for fl in ('frame64', 'frame16', 'frame4', 'frame1', None):
            fr = k[fl] if fl else 0.0
            L = shared_noframe + replaceable + fr
            U0 = L / (L - replaceable)
            # learned cost c allowed for U >= 1.25 and 1.1
            c125 = replaceable - L * (1 - 1 / 1.25)
            c110 = replaceable - L * (1 - 1 / 1.10)
            F[f'N{N}|{dt_label}|{fl or "noframe"}'] = dict(L_ms=round(L, 3), replaceable_ms=round(replaceable, 3),
                                                          U_zero_cost=round(U0, 3),
                                                          max_learned_cost_ms_for_U1p25=round(c125, 3),
                                                          max_learned_cost_ms_for_U1p10=round(c110, 3))
out['F_ceilings_kernel'] = F

# ---------------- G. ECRO generator FLOP split ----------------
d, R, NB = 128, 16, 64
head = (d + 16) * d + d * d + d * 3 * R          # one head_c or head_a MLP per (patch, bin) token, MAC
G = {'head_mac_per_token_each': head}
for N in (1270, 2802, 5080, 11430, 50000):
    for nb in (64, 256):
        heads = 2 * head * N * nb
        msg = 3 * N * N * d                        # Ft @ msg(t), 3 rounds
        G[f'N{N}_NB{nb}'] = dict(heads_GMAC=heads / 1e9, msgpass_GMAC=msg / 1e9, heads_share=heads / (heads + msg),
                                 Ft_dense_MB=N * N * 4 / 1e6)
# measured generator kernel ms -> GMAC/s-equivalent at NB 64, then predicted NB 256 time
for N, k in E5K.items():
    tot = (2 * head * N * 64 + 3 * N * N * d) / 1e9
    rate = tot / k['gen']                          # GMAC per ms
    t256 = (2 * head * N * 256 + 3 * N * N * d) / 1e9 / rate
    G[f'ECRO_FC_256bins_pred_N{N}'] = dict(gen64_meas_ms=k['gen'], eff_GMAC_per_ms=rate, gen256_pred_ms=t256,
                                           replaced_explicit_ms=k['q'] + 4 * k['order'],
                                           state_rays_per_bin_at_K512=512 / 256)
# kernel-basis edit->frame64 utility for ECRO-FC vs FC explicit at 256 bins with K raised to 2048 (8 rays / bin)
for N, k in E5K.items():
    scale = 4.0                                    # relation 4x for 8 rays / bin at 256 bins
    shared = k['geom'] + k['dirty_test'] * scale + k['retrace'] * scale + k['xop'] * scale + k['frame64']
    expl = shared + k['q'] + 4 * k['order'] * scale
    t256 = G[f'ECRO_FC_256bins_pred_N{N}']['gen256_pred_ms']
    lrn = shared + t256 + k['ecro_apply'] * scale
    G[f'ECRO_FC_U_edit_frame64_N{N}'] = dict(L_explicit_FC256_ms=expl, L_ECROFC_ms=lrn, U=expl / lrn)
out['G_ecro_generator'] = G

# ---------------- H. XQT per-flight cost ----------------
H = {}
ps_er = 88.6e-12 * 1e3       # ms per ER evaluation (q-table rate, s per eval -> ms)
landed = 526287
for N, gr in ((1270, 0.812e-6), (11430, 0.390e-6)):   # ms per gather entry (E5 one_order / relation rays)
    scale = N / 1270
    for cells, glossy_frac in ((64, 0.15), (256, 0.15), (64, 1.0)):
        er = landed * scale * glossy_frac * cells * ps_er
        for nnb in (9, 25):
            nnz = landed * scale * nnb
            apply = nnz * gr
            H[f'N{N}_cells{cells}_glossy{glossy_frac}_nnb{nnb}'] = dict(
                indirect_ER_ms=er, sparse_apply_ms=apply, per_flight_ms=er + apply,
                explicit_order_ms=E5K[N]['order'], ratio_to_explicit_order=(er + apply) / E5K[N]['order'],
                W_MB_val_idx=nnz * 8 / 1e6)
# W regeneration after an edit: dirty fraction of rays x neighbourhood spread
for spread in (1, 3, 9):
    f = min(1.0, 49936 / 650240 * spread)
    ev = f * landed * 9
    H[f'W_regen_spread{spread}'] = dict(dirty_entry_fraction=f, mlp_evals_M=ev / 1e6,
                                        ms_at_20kFLOP_10TFLOPs=ev * 2e4 / 1e13 * 1e3)
out['H_xqt'] = H

# ---------------- I. quality-matched lever via the frame (WL91 ladders) ----------------
lad = J('results/gm1_c3urx1_91/results/econ/ladder91.json')['configs']
def fit(rows, reg):
    spp = np.array([r['spp'] for r in rows], float); e = np.array([r['err_' + reg] for r in rows])
    m = (spp >= 64) & (spp <= 4096)
    A = np.stack([np.ones(m.sum()), 1.0 / spp[m]], 1)
    (f2, c2), *_ = np.linalg.lstsq(A, e[m] ** 2, rcond=None)
    return max(f2, 0.0), max(c2, 0.0)
def spp_at(f2, c2, t):
    return math.inf if t * t <= f2 else c2 / (t * t - f2)
I = {}
for cfg in ('W1_T1', 'W2_T1'):
    for reg, tiers in (('R_mirror', (0.10, 0.08, 0.06, 0.05, 0.04)), ('stationary_all', (0.06, 0.03, 0.02, 0.015))):
        row = {}
        for arm in ('R0_s86001', 'C3UR_s86001', 'C3URX1_s86001', 'C3URX1_s89201', 'C3URX1_s89202'):
            f2, c2 = fit(lad[cfg]['ladder'][arm], reg)
            row[arm] = dict(floor=math.sqrt(f2), **{f't{t}': spp_at(f2, c2, t) for t in tiers})
        # sensitivity: perturb C3URX1 deployed-draw floor by +-0.005 (order of a draw / reference-noise shift)
        f2, c2 = fit(lad[cfg]['ladder']['C3URX1_s86001'], reg)
        f = math.sqrt(f2)
        row['sens_C3URX1_s86001_floor_pm0.005'] = {f't{t}': [spp_at(max(f + s, 0) ** 2, c2, t) for s in (-0.005, 0.0, 0.005)]
                                                   for t in tiers}
        I[f'{cfg}|{reg}'] = row
out['I_time_to_tier_spp'] = I
# frame kernel ms per spp (E5) to convert spp to ms
out['I_frame_kernel_ms_per_spp_512sq'] = E5K[1270]['frame64'] / 64

# ---------------- J. draw pooling across steady-state frames ----------------
per_draw_ms = 0.176 + 0.058 + 0.528   # full relation re-trace (drjit exec) + X1 evals + one gather [ARITH, section C of cost_check]
J_ = {}
for fps_frame_spp in (1, 4, 16, 64):
    frame_ms = E5K[1270]['frame64'] / 64 * fps_frame_spp
    for m in (4, 16, 64):
        J_[f'frame{fps_frame_spp}spp_draws{m}'] = dict(
            extra_ms_per_frame_if_one_draw_per_frame=per_draw_ms,
            overhead_fraction_per_frame=per_draw_ms / frame_ms,
            frames_to_pool=m)
out['J_draw_pooling'] = J_

# ---------------- K. RB-DTO regime map, symmetric counts ----------------
K = {}
shared_kernel = E5K[1270]['dirty_test'] + E5K[1270]['xop'] + E5K[1270]['q'] + 4 * E5K[1270]['order'] + E5K[1270]['frame64']
for (nb, Nb, Jint, Kext) in ((2, 200, 3, 512), (2, 200, 8, 512), (20, 400, 8, 512), (20, 400, 16, 512), (20, 400, 25, 512)):
    for Kint, label in ((512, 'sender_side_twin_K512'), (4096, 'D3_uniform_K4096')):
        for tg in (0.39e-6, 0.81e-6):
            expl = Jint * nb * Nb * Kint * tg
            for Jext in (1, 4):
                for tq_ns in (5, 20):
                    M_in = Nb * Kext * 0.8                 # landed C_ext rays per body per outer iteration
                    lrn = Jext * nb * M_in * tq_ns * 1e-6
                    K[f'b{nb}_Nb{Nb}_Jint{Jint}_{label}_tg{tg*1e6:.2f}_Jext{Jext}_tq{tq_ns}'] = dict(
                        explicit_intra_ms=expl, learned_apply_ms=lrn,
                        U_edit_frame64=(shared_kernel + expl) / (shared_kernel + lrn))
# per-patch break-even: J_int*K_int*t_gather = s_in*t_q
K['breakeven_JintKint_per_patch'] = {f's_in{s}_tq{t}_tg{g}': s * t / g for s in (128, 410) for t in (5, 20) for g in (0.39, 0.81)}
D3 = J('results/wl98_calc/D3/cost_model.json')['body_regimes']
K['D3_inputs_M_in_per_body'] = [r['M_in'] for r in D3]
K['landed_Cext_rays_per_body_Nb400_K512'] = 400 * 512 * 0.8
out['K_rbdto'] = K

# ---------------- L. LIEM per-sample break-even ----------------
Lr = {}
t_ray = 0.271   # ns, Dr.Jit full-build exec per ray at N 1,270 [E5]; a walk event ~ one boundary/free-flight query
for ev in (2.5, 10, 12, 30, 47):
    for kflop in (20, 100, 200):
        for tflops in (10, 30):
            net = kflop * 1e3 / (tflops * 1e12) * 1e9
            Lr[f'events{ev}_net{kflop}k_{tflops}TF'] = dict(walk_ns=ev * t_ray, liem_ns=net + 2 * t_ray,
                                                           walk_over_liem=(ev * t_ray) / (net + 2 * t_ray))
out['L_liem'] = Lr

os.makedirs(os.path.join(ROOT, 'results', 'wl98_calc', 'X3'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'wl98_calc', 'X3', 'cost_check2.json'), 'w', newline='\n') as fh:
    json.dump(out, fh, indent=1, default=lambda x: None if x == math.inf else x)

def p(sec, keys=None):
    for k, v in out[sec].items():
        if keys is None or any(s in k for s in keys):
            print(sec, k, json.dumps(v, default=str)[:400])
p('F_ceilings_kernel')
p('G_ecro_generator', ['N1270', 'N11430', 'ECRO_FC'])
p('H_xqt')
p('I_time_to_tier_spp')
p('K_rbdto', ['b20_Nb400_Jint8', 'b20_Nb400_Jint25', 'breakeven', 'D3_inputs', 'landed'])
p('L_liem', ['net20k_10TF', 'net200k_10TF'])
