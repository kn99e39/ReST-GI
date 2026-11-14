"""WL93D checkerboard addendum (rev. 2), post hoc table diagnostic: single-flight S and global G of the checker patches, textured / untextured.
  python diag_sg93d.py <R> <R92A> <R93A> <variant>      -> <R>/results/viz/sg_<variant>.json
Frozen own s8601 / s8602; arms C0 and C2_rcv; ratios of sums over the checker patches of each material id."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checker93d as CK  # noqa: E402


def main(R, R92A, R93A, variant):
    import torch
    import diag92 as DG
    import ecro86 as EC
    import model86 as M86
    OP = CK.OP
    fx92, PT, S, FX, ctx, lt = CK.ctx_of(variant, R)
    _, Mq = DG.local_matrices(); Mq = CK.mq_ext(Mq)
    dev = torch.device('cuda')
    st = EC.build_state(ctx)
    pm_t = ctx.G.PMAT.copy()
    pm_u = CK.default_pmat(ctx.G)
    w = np.nonzero(pm_t != pm_u)[0] if (pm_t != pm_u).any() else np.zeros(0, int)
    tg = np.concatenate([np.arange(ctx.G.RBASE[r], ctx.G.RBASE[r] + ctx.G.RG[r][0] * ctx.G.RG[r][1]) for r in CK.TARGET_RECTS])
    w = tg
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    out = {}
    with torch.no_grad():
        for sd in CK.SEEDS:
            M = OP.load_family(R92A, R93A, 'own', sd); CK.extend_model(M['own']); rw = M['own'].rw
            res = {}
            for tag, pm in (('tex', pm_t), ('unt', pm_u)):
                ctx.G.PMAT[:] = pm
                gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
                Mn = torch.as_tensor(Mq[pm].astype(np.float32), device=dev)
                f = OP.local_factors(M, gs); G = OP.global_term(M, gs, s)[0].cpu().numpy()
                for arm in ('C0', 'C2_rcv'):
                    S_, _, _, _ = OP.arm_eval(M, f, f, Mn, gs, s, torch.zeros_like(s), arm, rw)
                    res[(tag, arm)] = (S_[0].cpu().numpy(), G)
            ctx.G.PMAT[:] = pm_t
            o = {}
            for arm in ('C0', 'C2_rcv'):
                St, Gt = res[('tex', arm)]; Su, Gu = res[('unt', arm)]
                for m in np.unique(pm_t[w]):
                    sel = w[pm_t[w] == m]
                    o[f'{arm}:id{m}(base id{pm_u[sel][0]})'] = {'S_ratio': float(St[sel].sum() / Su[sel].sum()), 'G_ratio': float(Gt[sel].sum() / Gu[sel].sum()),
                                         'G_share_untextured': float(Gu[sel].sum() / (Su[sel].sum() + Gu[sel].sum())),
                                         'B_ratio': float((St[sel].sum() + Gt[sel].sum()) / (Su[sel].sum() + Gu[sel].sum()))}
            out[f's{sd}'] = o
    fn = os.path.join(R, 'results', 'viz', f'sg_{variant}.json')
    json.dump({'variant': variant, 'albedo_ratio_vs_base': {str(m): CK.ALB[int(m)] / CK.ALB[int(pm_u[w][0])] for m in np.unique(pm_t[w])}, 'by_seed': out}, open(fn, 'w'), indent=1)
    print('SG', variant, json.dumps(out), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
