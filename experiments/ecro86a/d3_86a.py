"""WL86A amendment 2: can the frozen geometry-state relation carry the sharp-sender flight at all? (diagnostic; frozen WL86 state rays)

  python d3_86a.py run <root86> <out> <cfg>
  python d3_86a.py analyze <root86> <out>
For the light -> panel -> Lambert patch family (exactly 2 scattering events, last sender on the panel) the irradiance of a Lambert receiver
patch is computed three ways and compared with the path-traced truth (SPLIT table e2_p):
  BIN   the deployed 512 state rays reading the TRUE once-scattered radiance of the panel patches, averaged per (patch, outgoing bin):
        the best a single flight of the WL86 operator family can do with an exact sender table
  RAY   the same 512 state rays reading the exact (un-binned) radiance of the panel at each ray's hit point: isolates the ray count
  learned single flight (panel senders) of the frozen operators, from amendment 1
A Lambert patch of albedo rho has outgoing radiance rho / pi x irradiance in every bin.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86a.json')))
A2 = json.load(open(os.path.join(HERE, 'protocol', 'amendment2.json')))
SEEDS = P['seeds']


def qfield(A, ctx, lt, spp, seed):
    """Once-scattered (direct-lit) outgoing radiance per (patch, outgoing bin): [N, NB, 3]."""
    PT = A.PT; mi, dr, CD = PT.mi, PT.dr, PT.CD
    F, U, NB = PT.F, PT.U, PT.NB
    ns = ctx.N * NB
    N = ns * spp
    rng = mi.PCG32(size=N, initstate=mi.UInt64(seed), initseq=mi.UInt64(0))
    slot = dr.arange(U, N) % ns
    p, nn, (t, bv, n) = ctx.patch_point(slot // NB, rng.next_float32(), rng.next_float32())
    wl0 = ctx.dir_in_bin(slot % NB, rng.next_float32(), rng.next_float32())
    w = t * wl0.x + bv * wl0.y + n * wl0.z
    act = dr.dot(w, nn) > 1e-3
    si = ctx.scene.ray_intersect(mi.Ray3f(p + w * 2e-3, -w), act)
    act &= si.is_valid() & (dr.norm(si.p - p) < 1e-3)
    mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
    wi = si.to_local(w)
    okp = mi.Bool(act)
    act &= wi.z > 1e-6
    cl = CD.DrClosure83('ER', ctx.mats.states, mid)
    c = mi.Color3f(0.0)
    for l in range(lt.n):
        dl = lt.pos[l] - si.p
        d2 = dr.squared_norm(dl); wl = dl * dr.rsqrt(d2)
        wo_l = si.to_local(wl)
        on = act & (wo_l.z > 1e-6) & (dr.dot(-wl, lt.axis[l]) > lt.cc[l])
        vis = on & ~ctx.scene.ray_test(si.spawn_ray_to(lt.pos[l]), on)
        c += dr.select(vis, cl.eval(dr.select(act, wi, mi.Vector3f(0, 0, 1)), dr.select(vis, wo_l, mi.Vector3f(0, 0, 1)), vis) * lt.rgb[l] * dr.rcp(d2), 0.0)
    a = ctx.accum({'f0': c.x, 'f1': c.y, 'f2': c.z, 'cnt': dr.select(okp, F(1.0), F(0.0))}, slot, ns)
    cnt = a['cnt'].reshape(ctx.N, NB)
    return np.stack([a[f'f{k}'] for k in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]


def run(root, out, cfg):
    import pt86a as A
    import ecro86 as EC
    PT = A.PT; mi, dr = PT.mi, PT.dr
    NB, K = PT.NB, PT.K
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(PT.FX['light_sets'][ls])
    st = EC.build_state(ctx)
    G = ctx.G
    q = 0.5 * (qfield(A, ctx, lt, A2['qfield_spp_per_half'], 87800) + qfield(A, ctx, lt, A2['qfield_spp_per_half'], 87805))
    # exact radiance along every state ray whose hit is on the panel
    hitp = (st.hit >= 0) & (G.PMAT[np.clip(st.hit, 0, None)] == 1)
    idx = np.nonzero(hitp)[0]
    o = st.seg[idx, :3].astype(np.float64); e = st.seg[idx, 3:].astype(np.float64)
    d = e - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
    ray = mi.Ray3f(mi.Point3f(*[PT.fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[PT.fl(d[:, k]) for k in range(3)]))
    n = len(idx)
    sd = dr.opaque(mi.UInt64, 1)
    rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(0))
    outv, _ = ctx.trace(ray, n, rng, sd, 'ER', 1, lt, 'cam')
    L = np.stack([np.asarray(outv[f'e1{c}'], np.float64) for c in range(3)], 1)
    np.savez_compressed(os.path.join(out, 'results', 'fields', f'D3_{cfg}.npz'), q=q.astype(np.float32), ray_idx=idx, ray_L=L.astype(np.float32),
                        hit=st.hit, dbin=st.dbin)
    print('d3', cfg, 'panel-hit rays', n, 'of', st.hit.size, flush=True)


def analyze(root, out):
    import scene86 as S
    import model86 as M
    NB, K = S.NB, S.K
    R = os.path.join(out, 'results')
    rw = M.ray_weights().astype(np.float64)                         # cos dOmega / kappa per ray slot
    mats = np.load(os.path.join(HERE, '..', 'ecro86', 'data', 'mats86.npz'))
    res = {}
    A0 = json.load(open(os.path.join(R, 'attribution.json')))
    for cfg, rk in [tuple(c) for c in P['cases']['primary']]:
        e, t = cfg.split('_')
        G = S.Geom(S.FX[e][t])
        z = np.load(os.path.join(R, 'fields', f'D3_{cfg}.npz'))
        N = G.N
        hit = z['hit'].astype(np.int64); dbin = z['dbin'].astype(np.int64)
        kk = np.arange(N * K) % K; pid = np.arange(N * K) // K
        hp = (hit >= 0) & (G.PMAT[np.clip(hit, 0, None)] == 1)
        q = z['q'].astype(np.float64)
        Ebin = np.zeros((N, 3)); Eray = np.zeros((N, 3))
        np.add.at(Ebin, pid[hp], q[hit[hp], dbin[hp]] * rw[kk[hp]][:, None])
        np.add.at(Eray, pid[z['ray_idx']], z['ray_L'].astype(np.float64) * rw[kk[z['ray_idx']]][:, None])
        lam = mats['lambert'][G.PMAT] > 0
        rho = mats['albedo'][G.PMAT].astype(np.float64)
        Lbin = Ebin * (rho / math.pi)[:, None]; Lray = Eray * (rho / math.pi)[:, None]
        sp = np.load(os.path.join(R, 'fields', f'D_SPLIT_{cfg}.npz'))
        Ta = sp['e2_p_a'].astype(np.float64).mean(1); Tb = sp['e2_p_b'].astype(np.float64).mean(1)      # [N, 3] mean over bins
        T = 0.5 * (Ta + Tb)
        W = np.load(os.path.join(out, 'stageB', f'importance_{cfg}.npz'))[rk].astype(np.float64).reshape(N, NB, 3).sum(1)
        zo = np.load(os.path.join(root, 'results', 'fields', f'F_{cfg}.npz')); Bo = 0.5 * (zo['fa'].astype(np.float64) + zo['fb'])
        full = (np.load(os.path.join(out, 'stageB', f'importance_{cfg}.npz'))[rk].astype(np.float64).reshape(N, NB, 3) * Bo).sum()
        rd = lambda X: float((W[lam] * X[lam]).sum() / full)       # share of what the region reads (of the whole higher-order reading)
        row = {'truth_e2p_read_all_receivers': float((W * T).sum() / full), 'truth_e2p_read_lambert': rd(T), 'truth_halves': [rd(Ta), rd(Tb)],
               'BIN_read': rd(Lbin), 'RAY_read': rd(Lray), 'panel_hit_rays_per_patch_mean': float(hp.reshape(N, K).sum(1)[lam].mean())}
        m = lam & (W.sum(1) > 0.002 * W.sum())                       # Lambert receiver patches that matter for the region
        tt = T[m].sum(1)
        for name, X in (('BIN', Lbin), ('RAY', Lray)):
            x = X[m].sum(1)
            row[f'{name}_patch_rel_rms'] = float(math.sqrt(((x - tt) ** 2).sum() / max((tt ** 2).sum(), 1e-30)))
            row[f'{name}_patch_corr'] = float(np.corrcoef(x, tt)[0, 1]) if m.sum() > 2 else None
        row['truth_patch_noise_rel_rms'] = float(math.sqrt(((Ta[m].sum(1) - Tb[m].sum(1)) ** 2).sum() / 4 / max((tt ** 2).sum(), 1e-30)))
        row['patches'] = int(m.sum())
        row['panel_hit_rays_important_patches'] = [int(v) for v in np.sort(hp.reshape(N, K).sum(1)[m])[[0, len(tt) // 2, -1]]]
        for sd in SEEDS:
            mm = np.load(os.path.join(R, 'fields', f'M2_{sd}_{cfg}.npz'))['single_panel'].astype(np.float64).mean(1)
            row[f'{sd}_learned_single_panel_read'] = rd(mm)
        res[f'{cfg}:{rk}'] = row
    cl = {}
    ok_bin = [abs(r['BIN_read'] / r['truth_e2p_read_lambert'] - 1) <= 0.2 and r['BIN_patch_rel_rms'] <= 0.3 for r in res.values() if r['truth_e2p_read_lambert'] > 0.02]
    ok_ray = [abs(r['RAY_read'] / r['truth_e2p_read_lambert'] - 1) <= 0.2 and r['RAY_patch_rel_rms'] <= 0.3 for r in res.values() if r['truth_e2p_read_lambert'] > 0.02]
    bad_ray = [abs(r['RAY_read'] / r['truth_e2p_read_lambert'] - 1) > 0.2 or r['RAY_patch_rel_rms'] > 0.5 for r in res.values() if r['truth_e2p_read_lambert'] > 0.02]
    cl['cases_with_resolvable_family'] = len(ok_bin)
    cl['state_relation'] = 'ADEQUATE' if (ok_bin and all(ok_bin) and all(ok_ray)) else ('SAMPLING-LIMITED' if (bad_ray and all(bad_ray)) else ('BIN-LIMITED' if (ok_ray and all(ok_ray) and not all(ok_bin)) else 'MIXED'))
    res['classification'] = cl
    json.dump(res, open(os.path.join(R, 'attribution3.json'), 'w'), indent=1)
    for k, r in res.items():
        if k != 'classification':
            print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
    print(json.dumps(cl))


if __name__ == '__main__':
    cmd, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == 'run':
        run(root, out, sys.argv[4])
        os._exit(0)
    analyze(root, out)
