"""WL97 audit, track E4: explicit tails from path-traced prefixes with the comparator under test (xport87.XOp / quad87.Quad),
an OWN gather-scatter sweep (numpy float64) and an OWN estimate of the ER receiver weights, per-order and per-last-sender-class output.

Usage: python e4_tail.py <CODE/experiments> <CKF_none.npz> <orig tails.npz> <out prefix> <TF_1.npz> [<TF_2.npz> ...]
Audit code: no production file is modified; nothing is written outside <out prefix>*.
"""
import json, math, os, sys, time
import numpy as np
import torch

CODE, CKF, OTAILS, OUT = sys.argv[1:5]
TFS = sys.argv[5:]
for d in ('ecro86', 'gm1_explicit87'):
    sys.path.insert(0, os.path.join(CODE, d))
import fx87  # noqa: E402
import quad87 as Q  # noqa: E402
import xport87 as X  # noqa: E402
PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
DOM = 2 * math.pi / NB
E87 = fx87.P87['explicit']
SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
dev = X.dev
CLS = ('floor', 'panel', 'sphere', 'lambert')


def load_prefixes():
    z = np.load(CKF)
    g = lambda k: z[k].astype(np.float64)
    P = {'ckf_a': g('e2_p_a') + g('e2_o_a'), 'ckf_b': g('e2_p_b') + g('e2_o_b')}
    P['ckf'] = 0.5 * (P['ckf_a'] + P['ckf_b'])
    new = []
    for i, fn in enumerate(TFS):
        t = np.load(fn)
        e2 = t['k2'].astype(np.float64).sum(2)
        P[f'new{i}'] = e2; new.append(e2)
    if new:
        P['new'] = np.mean(new, 0)
        P['all'] = (2 * P['ckf'] + len(new) * P['new']) / (2 + len(new))
    return P


def build(ctx, us, seed_ray):
    X.set_realization(ctx, us)
    hit, dbin, seg = ctx.state_rays()
    quad = Q.Quad(us, E87['M_ray'], seed_ray)
    op = X.XOp(ctx.G, hit, dbin, quad.W)
    h = np.asarray(hit).reshape(ctx.N, K).astype(np.int64)
    pm = np.minimum(np.asarray(ctx.G.PMAT)[np.clip(h, 0, None)], 3)
    cm = [torch.as_tensor(((h >= 0) & (pm == c)).astype(np.float32)[..., None], device=dev) for c in range(4)]
    return op, cm, quad, h, np.asarray(dbin).reshape(ctx.N, K).astype(np.int64)


def orders(op, cm, P):
    """explicit orders L3..L5 split by the class of the LAST sender: [3 orders, 4 classes, N, NB, 3] float64."""
    L = torch.as_tensor(np.asarray(P, np.float32), device=dev)
    out = []
    for j in range(3):
        g = op.gather(L)
        parts = [op.scatter(g * m) for m in cm]
        out.append(torch.stack(parts))
        L = op.apply(L)
    return torch.stack(out).double().cpu().numpy()


def own_dirs(us):
    k = np.arange(K); bn = k // KAPPA
    z = (bn // NPHI + us[2, 0].astype(np.float64)) / NZ
    phi = (bn % NPHI + us[3, 0].astype(np.float64)) * (2 * math.pi / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def own_W_er(state, dirs, M, seed, chunk=64):
    """OWN receiver weights of one ER material, native closure orientation (wi = outgoing viewer v, wo = light direction w_r):
    W[r, bo] = (1 / kappa) int_{v in bo} f(v, w_r) cos(w_r) dv ~= DOM / kappa * mean_{v uniform in bo} eval(wi = v, wo = w_r)."""
    n = len(dirs)
    cl = CD.DrClosure83('ER', state)
    acc = np.zeros((n * NB, 3))
    dx, dy, dz = (PT.fl(dirs[:, k]) for k in range(3))
    done, it = 0, 0
    while done < M:
        m = min(chunk, M - done)
        lanes = n * NB * m
        li = dr.arange(U, lanes)
        slot = li // m
        di = slot // NB; bo = slot % NB
        w = mi.Vector3f(dr.gather(F, dx, di), dr.gather(F, dy, di), dr.gather(F, dz, di))
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 104729 + it), initseq=mi.UInt64(9797))
        v = PT.Ctx.dir_in_bin(bo, rng.next_float32(), rng.next_float32())
        ok = (w.z > 1e-6) & (v.z > 1e-6)
        f = cl.eval(dr.select(ok, v, mi.Vector3f(0, 0, 1)), dr.select(ok, w, mi.Vector3f(0, 0, 1)), ok)
        chs = [dr.select(ok, c, 0.0) for c in (f.x, f.y, f.z)]
        dr.eval(chs, slot); dr.sync_thread()
        for c in range(3):
            b = dr.zeros(F, n * NB)
            dr.scatter_reduce(dr.ReduceOp.Add, b, chs[c], slot)
            dr.eval(b); dr.sync_thread()
            acc[:, c] += np.asarray(b, np.float64)
        done += m; it += 1
    return (acc / M).reshape(n, NB, 3) * (DOM / KAPPA)


def own_sweep(G, hit, dbin, W, L):
    """OWN gather-scatter (numpy float64): out[i, bo, c] = sum_r W[wid_i, r, bo, c] L[hit_ir, dbin_ir, c] (miss / back face: 0)."""
    wid = np.asarray(G.PMAT).copy()
    sph = np.asarray(G.PSID) == S.SID_SPHERE
    wid[sph] = 6 + np.asarray(G.PSPH)[sph, 1].astype(np.int64)
    Lr = L[np.clip(hit, 0, None), dbin] * (hit >= 0)[..., None]                     # [N, K, 3]
    out = np.zeros((G.N, NB, 3))
    for m in np.unique(wid):
        ix = np.nonzero(wid == m)[0]
        out[ix] = np.einsum('nkc,kbc->nbc', Lr[ix], W[m])
    return out


def main():
    t0 = time.time()
    ctx = PT.Ctx(SC['geom'])
    G = ctx.G
    P = load_prefixes()
    ot = np.load(OTAILS)
    rep = {'prefixes': sorted(P), 'tf_files': TFS}
    # ---- deployed realization
    us = X.realization_us(86001)
    op, cm, quad, hit, dbin = build(ctx, us, E87['seed_ray'])
    rep['deployed_rays_to_panel'] = int(cm[1].sum().item())
    save = {}
    for name, pre in P.items():
        o = orders(op, cm, pre)                                   # [3, 4, N, NB, 3]
        save[f'dep|{name}'] = o.astype(np.float32)
    o = save['dep|ckf'].astype(np.float64)
    mine_p = o[:, 1].sum(0); mine_o = o[:, [0, 2, 3]].sum((0, 1))
    rl = lambda a, b: float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))
    rep['repro_dep_o_vs_stored'] = rl(mine_o, ot['dep_o'].astype(np.float64))
    rep['repro_dep_p_vs_stored'] = rl(mine_p, ot['dep_p'].astype(np.float64))
    # non-overlap / no double count: split sums to the unsplit operator; orders are a pure recurrence
    L2 = torch.as_tensor(P['ckf'].astype(np.float32), device=dev)
    L3 = op.apply(L2); L4 = op.apply(L3); L5 = op.apply(L4)
    full = (L3 + L4 + L5).double().cpu().numpy()
    rep['split_sum_vs_unsplit'] = rl(o.sum((0, 1)), full)
    rep['order_sums_exitance'] = {f'L{j + 3}': float(o[j].sum()) for j in range(3)}
    # ---- OWN sweep with the comparator's W (checks gather / scatter / indexing), then with OWN weights
    Wq = np.asarray(quad.W, np.float64)
    own3 = own_sweep(G, hit, dbin, Wq, P['ckf'])
    rep['own_sweep_vs_xop_L3'] = rl(own3, L3.double().cpu().numpy())
    dirs = own_dirs(us)
    rep['own_dirs_vs_quad_max'] = float(np.abs(dirs - quad.dirs).max())
    st, lam, alb = Q.mats()
    Wown = Wq.copy()
    wchk = {}
    for m in range(6):
        if lam[m]:
            Wown[m] = (alb[m] / math.pi * dirs[:, 2] * DOM / KAPPA)[:, None, None]
            wchk[f'lambert{m}'] = {'max_rel': float(np.abs(Wown[m] - Wq[m]).max() / np.abs(Wq[m]).max())}
    for m, M_ in ((0, 2048), (1, 8192), (2, 2048)):
        t1 = time.time()
        Wm = own_W_er(st[m], dirs, M_, 97100 + m)
        wchk[f'er{m}'] = {'M': M_, 'sum_ratio': float(Wm.sum() / Wq[m].sum()), 'rel_l1': rl(Wm, Wq[m]),
                          'per_ray_albedo_rel_l1': rl(Wm.sum((1, 2)), Wq[m].sum((1, 2))), 'seconds': time.time() - t1}
        if m < 2:
            Wown[m] = Wm                                          # flat ER patches (floor, panel); sphere rows (curved) stay the comparator's
    rep['own_W_check'] = wchk
    own3b = own_sweep(G, hit, dbin, Wown, P['ckf'])
    pmr = np.asarray(G.PMAT)
    x3 = L3.double().cpu().numpy()
    rep['own_W_sweep_vs_xop_L3'] = {'all': rl(own3b, x3), 'floor_rcv': rl(own3b[pmr == 0], x3[pmr == 0]), 'panel_rcv': rl(own3b[pmr == 1], x3[pmr == 1]),
                                    'lambert_rcv': rl(own3b[pmr >= 3], x3[pmr >= 3]),
                                    'floor_rcv_flux': float(own3b[pmr == 0].sum() / x3[pmr == 0].sum()), 'panel_rcv_flux': float(own3b[pmr == 1].sum() / x3[pmr == 1].sum())}
    save['own_L3_ckf'] = own3b.astype(np.float32)
    del op
    torch.cuda.empty_cache()
    # ---- pooled realizations (the original rule), prefixes ckf and all
    pool = {k: [] for k in ('ckf', 'all') if k in P}
    for d in range(16):
        opd, cmd, _, _, _ = build(ctx, X.realization_us(960000 + d), E87['seed_ray'] + 1000 * (d + 1))
        for k in pool:
            pool[k].append(orders(opd, cmd, P[k]).astype(np.float32))
        if d == 0:
            od = pool['ckf'][0].astype(np.float64)
            rep['repro_pool0_o_vs_stored'] = rl(od[:, [0, 2, 3]].sum((0, 1)), ot['pool_o'][0].astype(np.float64))
        del opd
        torch.cuda.empty_cache()
    for k in pool:
        A = np.stack(pool[k])                                     # [16, 3, 4, N, NB, 3]
        save[f'pool|{k}'] = A.mean(0).astype(np.float32)
        save[f'poolreal_o|{k}'] = A[:, :, [0, 2, 3]].sum((1, 2)).astype(np.float32)      # per-realization e3_o tails [16, N, NB, 3]
    rep['repro_pool_o_mean_vs_stored'] = rl(save['pool|ckf'].astype(np.float64)[:, [0, 2, 3]].sum((0, 1)), ot['pool_o'].astype(np.float64).mean(0))
    rep['seconds'] = time.time() - t0
    np.savez_compressed(OUT + '.npz', **save)
    json.dump(rep, open(OUT + '.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    print('E4TAIL DONE', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
