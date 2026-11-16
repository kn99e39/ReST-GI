"""WL98 D3 scratch cost model (numpy only; no GPU, no rendering, no training).

Inputs are RECORDED component medians from WL96A / WL97 E5 / WL97 D (N = 1,270, CK2, RTX 3080 Ti):
 wall ms / GPU-kernel ms. Outputs: Amdahl ceilings U = L / (L - A + c) for the amortisable share A that
 each learned-role could remove, under several lifecycle conventions; and a parametric model of when
 an intra-body (rigid-body self-transport) amortisation could exceed a utility threshold.
All outputs are [PRED]/[ARITH]; nothing here is a measurement.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'wl98_calc', 'D3')
os.makedirs(OUT, exist_ok=True)

# recorded (E5 run1 isolated components; D section 2.4) -- (wall_ms, kernel_ms)
C = {
    'geom_update':   (30.14, 0.04),   # set_geom (numpy Geom 23.9 + params.update 2.7 + tables 3.1)
    'rel_update':    (18.80, 4.49),   # selective relation update (dirty_rays torch 4.43 + retrace 0.06)
    'rel_coeff':     (5.47, 2.01),    # XOp build
    'q_table':       (24.13, 1.84),   # source projection, 1 light, 20.8 M ER evals
    'prefix_order':  (1.90, 0.53),    # one binned order
    'tail_3orders':  (3.17, 1.58),
    'handover':      (0.20, 0.20),
    'frame64':       (58.40, 22.67),  # 512^2 x 64 spp table-lookup frame
    'ecro_apply':    (0.76, 0.74),
    'ecro_factors':  (4.35, 4.20),
    'src_encode':    (5.25, 0.05),    # WL87 source encoding (CPU bincount)
    'lambda24':      (0.146, 0.105),
}
FRAME_SPP_KERNEL = {64: 22.67, 16: 5.7, 4: 1.5, 1: 0.4}   # WL87 E2 ladder kernel (64 from E5)

def life(kind, basis, spp=64, frame=True):
    i = 0 if basis == 'wall' else 1
    parts = ['q_table', 'prefix_order', 'tail_3orders', 'handover']
    if kind == 'edit':
        parts = ['geom_update', 'rel_update', 'rel_coeff'] + parts
    t = sum(C[p][i] for p in parts)
    if frame:
        t += C['frame64'][i] if (basis == 'wall' or spp == 64) else FRAME_SPP_KERNEL[spp]
        if basis == 'wall' and spp != 64:
            t += 0  # wall frame is overhead-dominated (~35.7 ms fixed); keep 64-spp wall as conservative
    return t

roles = {
    # role: (amortisable parts, replacement cost parts)
    'resolvent_only (WL96A role)': (['tail_3orders'], ['lambda24']),
    'full ECRO-type operator (q + all orders replaced)': (['q_table', 'prefix_order', 'tail_3orders'], ['src_encode', 'ecro_apply']),
    'full operator + per-edit generator': (['q_table', 'prefix_order', 'tail_3orders'], ['src_encode', 'ecro_apply', 'ecro_factors']),
}
res = {'ceilings': []}
for basis in ('wall', 'kernel'):
    for kind in ('edit', 'source'):
        for frame, spp in ((True, 64), (True, 4), (False, 64)):
            if basis == 'wall' and spp == 4:
                continue
            L = life(kind, basis, spp, frame)
            for rname, (amo, rep) in roles.items():
                if kind == 'source' and 'generator' in rname:
                    continue
                i = 0 if basis == 'wall' else 1
                A = sum(C[p][i] for p in amo); c = sum(C[p][i] for p in rep)
                U = L / (L - A + c)
                res['ceilings'].append(dict(basis=basis, event=kind, frame=(f'{spp}spp' if frame else 'none'), role=rname,
                                            lifecycle_ms=round(L, 2), amortisable_ms=round(A, 2), replacement_ms=round(c, 2), U=round(U, 3)))

# Required amortisable share for a utility threshold u: A - c >= L (1 - 1/u)
res['required_share'] = {str(u): round(1 - 1 / u, 3) for u in (1.1, 1.25, 1.5, 2.0)}

# --- Parametric explicit cost of uniform (receiver-side) binned transport vs angular sharpness -------------
# Equal-solid-angle bins needed to put >= b bins across a GGX lobe of roughness alpha:
# lobe solid angle ~ pi * (c alpha)^2 with c ~ 2 (half-width containing most energy, [PRED]);
# bins NB ~ b * 2 pi / (pi (c alpha)^2).  kappa rays per bin.  Per-order cost ~ N * NB * kappa entries.
ns_per_entry_kernel = C['tail_3orders'][1] / 3 / (1270 * 512) * 1e6   # recorded ~0.8 ns / entry (includes launch overhead)
alphas = np.array([0.3, 0.18, 0.1, 0.05, 0.03, 0.02, 0.01])
rows = []
for a in alphas:
    NB = 4 * 2 * np.pi / (np.pi * (2 * a) ** 2)
    NB = max(NB, 64)
    for N in (1270, 10000, 100000):
        entries = N * NB * 8
        rows.append(dict(alpha=float(a), N=N, bins=int(NB), entries_M=round(entries / 1e6, 1),
                         ms_per_order_kernel=round(entries * ns_per_entry_kernel * 1e-6, 2),
                         relation_GB_at_8B=round(entries * 8 / 1e9, 2)))
res['uniform_binned_scaling'] = rows
res['ns_per_entry_kernel'] = round(ns_per_entry_kernel, 3)

# --- Rigid-body self-transport amortisation (the D3 candidate's lever) ----------------------------------
# Explicit twin per update (incident field changed at body b): J_int orders over the cached intra-body
# relation of N_b patches x K_int rays (sender-side lobe sampling keeps K_int moderate), + per-update
# variance cost: to reach relative error eps on a glossy multi-bounce exit distribution with MC,
# samples ~ 1/eps^2 per output cell.  Learned: application ~ M_in * t_inf + splat.
# All constants are [PRED] placeholders; the point is the SHAPE of the regime map, not the numbers.
t_inf_ns = np.array([5., 20.])           # per incident-sample inference, fused MLP vs prototype ([PRED])
def body_regime(n_bodies, N_b, K_int, J_int, M_in):
    expl = n_bodies * J_int * N_b * K_int * ns_per_entry_kernel * 1e-6         # ms, kernel basis
    lrn = n_bodies * M_in * t_inf_ns * 1e-6                                       # ms
    return expl, lrn
reg = []
for (nb, Nb, Kint, J, Min) in [(2, 200, 512, 3, 2e4), (2, 200, 4096, 8, 2e4), (20, 400, 4096, 8, 5e4),
                               (20, 400, 16384, 16, 5e4), (100, 400, 16384, 16, 5e4)]:
    e, l = body_regime(nb, Nb, Kint, J, Min)
    L_fixed_kernel = life('edit', 'kernel', 64, True) - C['tail_3orders'][1] - C['prefix_order'][1]
    reg.append(dict(bodies=nb, patches_per_body=Nb, K_int=Kint, J_int=J, M_in=Min,
                    explicit_intra_ms_kernel=round(e, 2), learned_apply_ms=[round(x, 2) for x in l],
                    U_edit_kernel_frame64=[round((L_fixed_kernel + e) / (L_fixed_kernel + x), 2) for x in l]))
res['body_regimes'] = reg

json.dump(res, open(os.path.join(OUT, 'cost_model.json'), 'w'), indent=1)
for r in res['ceilings']:
    print(f"{r['basis']:6s} {r['event']:6s} frame={r['frame']:6s} L={r['lifecycle_ms']:7.2f} A={r['amortisable_ms']:6.2f} c={r['replacement_ms']:5.2f} U={r['U']:.3f}  {r['role']}")
print('required share', res['required_share'], 'ns/entry', res['ns_per_entry_kernel'])
for r in res['uniform_binned_scaling']:
    print(r)
for r in res['body_regimes']:
    print(r)

# --- ECRO generator dense form-factor message passing: O(N^2) memory (model86.form_factors builds N*N) ----
gen = []
for N in (1270, 5080, 11430, 50000, 100000):
    gen.append(dict(N=N, Ft_dense_MB_fp32=round(N * N * 4 / 2**20, 1),
                    msg_pass_GFLOP_per_round_d128=round(2 * N * N * 128 / 1e9, 2)))
res['ecro_generator_dense_ff'] = gen

# --- Learning-free twin of a rigid-body self-transfer: explicit precomputed sparse binned transfer --------
# in-dim = N_b patches x NB(alpha) incident bins; each input column spreads to s output (patch,bin) cells.
# For a glossy multi-bounce exit distribution, s grows with bounce count; take s in {64, 1024} ([PRED]).
tw = []
for a in (0.18, 0.1, 0.05, 0.03):
    NB = max(4 * 2 / (2 * a) ** 2, 64)
    for Nb in (200, 1000):
        for s in (64, 1024):
            nnz = Nb * NB * s
            tw.append(dict(alpha=a, N_b=Nb, bins=int(NB), spread=s, nnz_M=round(nnz / 1e6, 1), GB_fp16_idx32=round(nnz * 6 / 1e9, 2)))
res['explicit_precomputed_body_transfer'] = tw
json.dump(res, open(os.path.join(OUT, 'cost_model.json'), 'w'), indent=1, default=float)
print(gen)
for r in tw: print(r)
