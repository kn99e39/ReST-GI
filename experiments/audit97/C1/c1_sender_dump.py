"""WL97 audit, track C1: raw S1-sender quantities of the WL94D models, dumped for an independent metric recomputation.

Uses the frozen PRODUCTION model / flight code read-only (mat94.load_spec, model86.factors, eval93b.flight, mat94.mq_ext, c94.load_items)
and none of the audited METRIC code (s94.sender / an94). Nothing is written outside <OUT>.

  PYTHONDONTWRITEBYTECODE=1 python c1_sender_dump.py <CODE> <R94D> <R86> <R92A> <OUT>
For each model (own, D, T, R, P x seeds 8601 / 8602), on CK2 (untextured) and on the 24 WL86 validation items (original materials):
  Bl, Bm [L, 42, 64, 3]      learned / M64 once-scattered sender radiance on the S1 panel patches (float32)
  el_*, em_* [L, N]          exitance of the single flight delivered from the S1 sender through the EXACT receiver, for the learned (l) and
                             the M64 (m) sender, under the coded (8 x 8) and the true (4 x 16) cos dOmega weights
  operator sums              learned S1 local operator A = sum_r a c^T vs M64 S1: pooled flux-L1, per-colour flux-Frobenius, mass
CK2 additionally: full delivered fields Dl, Dm [N, 64, 3], the operator A [42, 64, 64, 3], and M64 S1.
"""
import json, os, sys
import numpy as np

CODE, R, R86, R92A, OUT = sys.argv[1:6]
for d in ('ecro_mat94', 'ecro_stage93b'):
    sys.path.insert(0, os.path.join(CODE, 'experiments', d))
import mat94 as MT  # noqa: E402
import c94 as C94  # noqa: E402
import torch  # noqa: E402
import eval93b as EB  # noqa: E402
import ecro86 as EC  # noqa: E402
import model86 as M86  # noqa: E402
import scene86 as S  # noqa: E402

dev = torch.device('cuda')
NB = S.NB
assert (S.NZ, S.NPHI, NB) == (4, 16, 64)
PANEL = list(range(1088, 1130))
W_CODED = torch.as_tensor(((np.arange(NB) // 8) + 0.5) / 8 * (2 * np.pi / NB), device=dev, dtype=torch.float64)
W_TRUE = torch.as_tensor(((np.arange(NB) // 16) + 0.5) / 4 * (2 * np.pi / NB), device=dev, dtype=torch.float64)
ZB = torch.as_tensor(((np.arange(NB) // 16) + 0.5) / 4, device=dev, dtype=torch.float64)
os.makedirs(OUT, exist_ok=True)

Mq = torch.as_tensor(MT.mq_ext().astype(np.float32), device=dev)                 # [11, NB_in, NB_out, 3]
M1 = Mq[1].double()
np.save(os.path.join(OUT, 'M64_S1.npy'), Mq[1].cpu().numpy())
np.save(os.path.join(OUT, 'M64_all.npy'), Mq.cpu().numpy())

PT, S_, A_, FX, ctx, lt = MT.ctx_ck2(R)
st = EC.build_state(ctx)
s_ck = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, NB, 3)
MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
gs_ck = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
assert (ctx.G.PSID[PANEL] == 5).all() and int((ctx.G.PSID == 5).sum()) == len(PANEL)
np.savez_compressed(os.path.join(OUT, 'ck2_geom.npz'), PSID=ctx.G.PSID, PMAT=ctx.G.PMAT, PAREA=ctx.G.PAREA, s=s_ck[0].cpu().numpy(),
                    hit=st.hit, dbin=st.dbin)
val = C94.load_items(R, R86, 'val', 'orig')
print('val items', len(val), flush=True)


def one(model, gs, s, full):
    F = model.factors(gs)
    c, a = F['c'][PANEL], F['a'][PANEL]
    Bl = torch.einsum('ndcr,lncr->lndc', a, torch.einsum('nmcr,lnmc->lncr', c, s[:, PANEL]))
    Bm = torch.einsum('lnmc,nmdc->lndc', s[:, PANEL], Mq[gs.mat[PANEL]])
    fl = torch.zeros(s.shape[0], gs.N, NB, 3, device=dev); fm = torch.zeros_like(fl)
    fl[:, PANEL] = Bl; fm[:, PANEL] = Bm
    Mn = Mq[gs.mat]
    Dl = torch.einsum('lnmc,nmdc->lndc', EB.flight(model.rw, gs, fl), Mn).double()
    Dm = torch.einsum('lnmc,nmdc->lndc', EB.flight(model.rw, gs, fm), Mn).double()
    o = {'Bl': Bl.cpu().numpy().astype(np.float32), 'Bm': Bm.cpu().numpy().astype(np.float32),
         'el_coded': torch.einsum('lndc,d->ln', Dl, W_CODED).cpu().numpy(), 'em_coded': torch.einsum('lndc,d->ln', Dm, W_CODED).cpu().numpy(),
         'el_true': torch.einsum('lndc,d->ln', Dl, W_TRUE).cpu().numpy(), 'em_true': torch.einsum('lndc,d->ln', Dm, W_TRUE).cpu().numpy(),
         'psid': None}
    Aop = torch.einsum('nmcr,ndcr->nmdc', c, a).double()                           # [42, bi, bo, 3]
    fw = ZB.view(1, -1, 1, 1) * ZB.view(1, 1, -1, 1)
    T = M1.unsqueeze(0).expand_as(Aop)
    o['op'] = {'l1_num': float(((Aop - T).abs() * fw).sum()), 'l1_den': float((T * fw).sum()), 'mass_num': float((Aop * fw).sum()),
               'fro_num_c': [float((((Aop - T) * fw)[..., k] ** 2).sum()) for k in range(3)],
               'fro_den_c': [float(((T * fw)[..., k] ** 2).sum()) for k in range(3)],
               # per-patch flux-Frobenius relative residual per colour (WL92 mat_metrics convention), then averaged
               'fro_rel_c_patchmean': [float((torch.sqrt((((Aop - T) * fw)[..., k] ** 2).sum((1, 2))) / torch.sqrt(((T * fw)[..., k] ** 2).sum((1, 2)))).mean()) for k in range(3)],
               # operator-level peak retention: learned value at the argmax-out bin of every (incident bin, colour) row of M64 (flux weighted)
               'peak_num': float(torch.gather(Aop * fw, 2, (T * fw).argmax(2, keepdim=True)).sum()),
               'peak_den': float((T * fw).amax(2).sum())}
    if full:
        o['Dl'] = Dl[0].cpu().numpy().astype(np.float32); o['Dm'] = Dm[0].cpu().numpy().astype(np.float32)
        o['A'] = Aop.cpu().numpy().astype(np.float32)
    del o['psid']
    return o


specs = [(f'{k}_s{sd}', (f'own:{sd}' if k == 'own' else f'cand:{k}_s{sd}')) for sd in (8601, 8602) for k in ('own', 'D', 'T', 'R', 'P')]
meta = {}
with torch.no_grad():
    for tag, spec in specs:
        model = MT.load_spec(R, R92A, spec)
        ck = one(model, gs_ck, s_ck, True)
        out = {f'ck2|{k}': v for k, v in ck.items() if k != 'op'}
        ops = {'ck2': ck['op'], 'val': []}
        for i, it in enumerate(val):
            gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
            v = one(model, gs, s, False)
            for k, x in v.items():
                if k != 'op':
                    out[f'val{i:02d}|{k}'] = x
            ops['val'].append(v['op'])
        np.savez_compressed(os.path.join(OUT, f'SND_{tag}.npz'), **out)
        meta[tag] = {'spec': spec, 'op': ops}
        print('DUMP', tag, 'gpu peak MiB', round(torch.cuda.max_memory_allocated() / 2 ** 20), flush=True)
json.dump(meta, open(os.path.join(OUT, 'sender_dump_meta.json'), 'w'), indent=1)
open(os.path.join(OUT, 'done'), 'w').write('ok\n')
print('DONE', flush=True)
