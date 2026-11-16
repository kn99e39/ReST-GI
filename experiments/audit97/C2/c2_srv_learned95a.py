"""WL97 audit C2 (server, short GPU job): the WL95A learned S1 sender (OWN s8601 / s8602, read-only checkpoints)
(1) distribution of the per-patch directional albedo (the worklog reports only the maximum over 42 x 64 x 3 entries),
(2) sensitivity of the learned-sender box-A delivery to the 16 fixed source points (64 source-point redraws, deployed relation).
Production modules are imported ONLY to build the scene / state / source field and to evaluate the frozen model factors; every statistic is
computed here in numpy.  Usage: python c2_srv_learned95a.py <CODE/experiments> <R (own)> <R92A> <out.json>"""
import json, math, os, sys
import numpy as np

EXP, R, R92A, OUT = sys.argv[1:5]
sys.path.insert(0, os.path.join(EXP, 'ecro_mat94'))
import mat94 as MT  # noqa: E402

PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
import torch  # noqa: E402
import ecro86 as EC  # noqa: E402
import model86 as M86  # noqa: E402

G = ctx.G; N = G.N
NB = 64; KAPPA = S.KAPPA; K = S.K
S1 = np.arange(1088, 1130); BOXA = np.arange(1130, 1184)
WT = ((np.arange(NB) // 16) + 0.5) / 4 * 2 * math.pi / NB
st = EC.build_state(ctx)
u0 = ctx.u_state_np.copy(); us0 = ctx.u_src_np.copy()
s16 = ctx.source(lt).astype(np.float64)
Mq = MT.mq_ext().astype(np.float64)
M1 = Mq[1]
pm = np.asarray(G.PMAT)
z = np.load(os.path.join(os.path.expanduser('~/rest_x94'), 'results', 'fields', 'CKF_none.npz'))
ref = 0.5 * (z['e2_p_a'].astype(np.float64) + z['e2_p_b'].astype(np.float64))
ref_ex = (ref[BOXA] * WT[None, :, None]).sum((1, 2))

# own numpy flight + receive for the box-A receivers only (deployed relation)
hit = np.asarray(st.hit).astype(np.int64).reshape(N, K)[BOXA]          # [54, K]
dbin = np.asarray(st.dbin).astype(np.int64).reshape(N, K)[BOXA]
k = np.arange(K)
rw = (((k // KAPPA) // 16 + u0[2, 0]) / 4) * (2 * math.pi / NB) / KAPPA   # cos dOmega / kappa per state ray
is_s1 = (hit >= 1088) & (hit < 1130)
Mrcv = Mq[pm[BOXA]]                                                     # [54, in, out, 3]


def deliver(Bs1):
    """box-A exitance [54] delivered by an S1-only sender table Bs1 [42, NB, 3] through the deployed relation and the exact receiver."""
    val = np.where(is_s1[..., None], Bs1[np.clip(hit - 1088, 0, 41), dbin], 0.0)     # [54, K, 3]
    E1 = (val * rw[None, :, None]).reshape(len(BOXA), NB, KAPPA, 3).sum(2)        # [54, in, 3]
    o = np.einsum('nmc,nmdc->ndc', E1, Mrcv)
    return (o * WT[None, :, None]).sum((1, 2))


def dm(e):
    return {'ratio': float(e.sum() / ref_ex.sum()), 'abs_rel_l1': float(np.abs(e - ref_ex).sum() / ref_ex.sum())}


out = {'rays_boxA_to_S1': int(is_s1.sum()), 'M64': {}}
# source redraws (the WL95A rule: rng 951000 + d, 16 points per patch)
src = []
for d in range(64):
    ctx.u_src_np = np.random.default_rng(951000 + d).random(us0.shape).astype(np.float32); ctx.set_tables(G)
    src.append(ctx.source(lt).astype(np.float64)[S1])
ctx.u_src_np = us0; ctx.set_tables(G)
src = np.asarray(src)                                                    # [64, 42, NB, 3]
out['source_support_bins_s16'] = np.nonzero(s16[S1].sum((0, 2)))[0].tolist()
out['source_support_bins_all_draws'] = np.nonzero(src.sum((0, 1, 3)))[0].tolist()
out['source_irradiance_total_cv_over_draws'] = float(src.sum((1, 2, 3)).std(ddof=1) / src.sum((1, 2, 3)).mean())
alb_M = (M1 * WT[None, :, None]).sum(1)                                  # [in, 3]
out['M64']['albedo_max'] = float(alb_M.max())
e = deliver(np.einsum('nmc,mdc->ndc', s16[S1], M1)); out['M64']['deployed_s16'] = dm(e)
per = [dm(deliver(np.einsum('nmc,mdc->ndc', x, M1))) for x in src]
out['M64']['source_draws'] = {q: {'mean': float(np.mean([p[q] for p in per])), 'sd': float(np.std([p[q] for p in per], ddof=1)),
                                  'min': float(np.min([p[q] for p in per])), 'max': float(np.max([p[q] for p in per]))} for q in ('ratio', 'abs_rel_l1')}
dev = torch.device('cuda')
gs = M86.GeoState(G, st.hit, st.dbin, dev)
for sd in (8601, 8602):
    model = MT.load_frozen(R92A, sd)
    with torch.no_grad():
        F = model.factors(gs)
        c_all, a_all = F['c'].double().cpu().numpy(), F['a'].double().cpu().numpy()      # [N, NB, 3, R]
    c, a = c_all[S1], a_all[S1]
    lop = np.einsum('nmcr,ndcr->nmdc', c, a)                                             # [42, in, out, 3]
    alb = (lop * WT[None, None, :, None]).sum(2)                                         # [42, in, 3]
    r = {'rank_R': int(c.shape[-1]), 'min_entry': float(lop.min()), 'albedo_max': float(alb.max()),
         'albedo_quantiles_50_90_99_999': [float(np.quantile(alb, q)) for q in (0.5, 0.9, 0.99, 0.999)],
         'albedo_mean': float(alb.mean()), 'frac_entries_gt_1': float((alb > 1 + 1e-6).mean()), 'n_entries': int(alb.size),
         'frac_patches_any_gt_1': float((alb.max((1, 2)) > 1 + 1e-6).mean()),
         'argmax_patch_bin_ch': [int(v) for v in np.unravel_index(alb.argmax(), alb.shape)],
         'albedo_max_by_incident_ring': [float(alb[:, 16 * i:16 * i + 16].max()) for i in range(4)],
         'frac_gt_1_by_incident_ring': [float((alb[:, 16 * i:16 * i + 16] > 1 + 1e-6).mean()) for i in range(4)],
         'albedo_mean_by_incident_ring': [float(alb[:, 16 * i:16 * i + 16].mean()) for i in range(4)],
         'M64_albedo_mean_by_incident_ring': [float(alb_M[16 * i:16 * i + 16].mean()) for i in range(4)]}
    # albedo actually exercised on CK2: source-weighted effective albedo per patch and overall (what the deployed light excites)
    s = s16[S1]
    num = (s * alb).sum((1, 2)); den = s.sum((1, 2))
    r['effective_albedo_ck2_overall'] = float(num.sum() / den.sum())
    r['effective_albedo_ck2_patch_max'] = float((num / np.maximum(den, 1e-30)).max())
    r['effective_albedo_ck2_M64_overall'] = float((s * alb_M[None]).sum() / s.sum())
    lit = s.sum(2) > 0                                                                   # [42, in]
    r['albedo_max_on_lit_bins'] = float(alb[lit].max()); r['frac_lit_entries_gt_1'] = float((alb[lit] > 1 + 1e-6).mean())
    # all patches (context): how common is albedo > 1 over the whole scene's per-patch operators
    lop_all_alb = np.einsum('nmcr,ncr->nmc', c_all, np.einsum('ndcr,d->ncr', a_all, WT))
    r['all_patches'] = {'albedo_max': float(lop_all_alb.max()), 'frac_entries_gt_1': float((lop_all_alb > 1 + 1e-6).mean()),
                        'frac_patches_any_gt_1': float((lop_all_alb.max((1, 2)) > 1 + 1e-6).mean()),
                        'albedo_q99': float(np.quantile(lop_all_alb, 0.99))}
    # delivery: deployed source and 64 source redraws (deployed relation)
    Bl = np.einsum('ndcr,ncr->ndc', a, np.einsum('nmcr,nmc->ncr', c, s))
    r['deployed_s16'] = dm(deliver(Bl))
    per = [dm(deliver(np.einsum('ndcr,ncr->ndc', a, np.einsum('nmcr,nmc->ncr', c, x)))) for x in src]
    r['source_draws'] = {q: {'mean': float(np.mean([p[q] for p in per])), 'sd': float(np.std([p[q] for p in per], ddof=1)),
                             'min': float(np.min([p[q] for p in per])), 'max': float(np.max([p[q] for p in per]))} for q in ('ratio', 'abs_rel_l1')}
    r['deployed_ratio_rank_in_source_draws'] = float(np.mean([p['ratio'] < r['deployed_s16']['ratio'] for p in per]))
    out[f'A0_s{sd}'] = r
    del model
out['gpu_mem_peak_mib_torch'] = torch.cuda.max_memory_allocated() / 2 ** 20
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
os._exit(0)
