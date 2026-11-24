"""WL112 post-hoc diagnostic (labelled; written after the frozen Stage B outcome NONE; never changes a verdict).

  python diag112.py ROOT
(a) query-weighted decomposition of the near / flank HIGH readout of U, G, S_U, S_G by queried shape (and sphere ring), from the
    stored unit-marker weights of the Stage B streams and the Stage A tables (no rendering);
(b) one extra resolution level: 4 x 4 sub-patches under U and G (4 replicas x 1,024 proposals per sub-cell, C semantics, seeds
    11208001+), read on the SAME Stage B stream seeds (paired with the stored physical references; LOW must reproduce) at the five
    contact-valid F0 views; endpoints with the Stage B formulas (incl. Amendment 1 kappa).
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import metrics112 as MX  # noqa: E402

N, NB = 1270, 64
LUMW = MX.LUMW


def part_a(root):
    import eval112 as E
    tabs, _ = E.stageB_tables(root)
    tab = {n: t for n, t, k in tabs}
    g = __import__('analyzeAB112').geom112(C.fixture('F0')[0])
    cls = np.array(['floor', 'ceiling', 'wall', 'wall', 'wall', 'panel', 'sphere', 'boxA', 'boxB'])[g.PSID]
    ring = np.where(g.PSID == 6, g.PSPH[:, 1].astype(int), -1)
    labels = np.where(cls == 'sphere', np.char.add('sphere_ring', ring.astype(str)), cls)
    out = {}
    for v in ('V0', 'H1', 'H2', 'H3', 'H4'):
        st = [np.load(root / 'stageB' / f'stream_{v}_{j}.npz') for j in range(2)]
        gb = dict(np.load(root / 'stageB' / f'gbuffer_{v}.npz'))
        reg, _ = C.masks(gb, C.fixture('F0')[0], C.fixture('F0')[1][0]['pos'])
        row = {}
        for r in ('near', 'flank'):
            W = 0.5 * (st[0][f'W|{r}'].astype(np.float64) + st[1][f'W|{r}'].astype(np.float64))     # [3, N*4*64]
            npx = reg[r].sum()
            Wc = W.T.reshape(N, 4, NB, 3)
            per = {}
            for arm in ('U', 'G', 'S_U', 'S_G'):
                vals = []
                for rr in range(C.RC):
                    T_ = tab[f'{arm}_r{rr}']
                    if T_.ndim == 3:                                   # coarse: broadcast over sub-cells
                        T_ = np.repeat(T_[:, None], 4, axis=1)
                    contrib = (Wc * T_).sum((1, 2)) @ LUMW / npx       # [N]
                    vals.append({lab: float(contrib[labels == lab].sum()) for lab in np.unique(labels)})
                per[arm] = {lab: float(np.mean([x[lab] for x in vals])) for lab in vals[0]}
            share = {lab: float((Wc.sum((1, 2)) @ LUMW)[labels == lab].sum() / (Wc.sum((1, 2)) @ LUMW).sum()) for lab in np.unique(labels)}
            row[r] = {'query_weight_share': share, 'readout_by_shape': per,
                      'S_G_minus_G_by_shape': {lab: per['S_G'][lab] - per['G'][lab] for lab in share},
                      'G_minus_U_by_shape': {lab: per['G'][lab] - per['U'][lab] for lab in share}}
        out[v] = row
    return out


def part_b(root):
    import tracer112 as T
    mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT
    K = 4
    T.NSUB = K * K                                                   # module-level sub-cell count used by cam_pass
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)

    class Ctx4(T.Ctx112):
        def subcell(self, si, pid):
            g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
            o = mi.Point3f(g(0), g(1), g(2)); e1 = mi.Vector3f(g(3), g(4), g(5)); e2 = mi.Vector3f(g(6), g(7), g(8))
            rel = si.p - o
            a = dr.dot(rel, e1) / dr.maximum(dr.squared_norm(e1), 1e-30); b = dr.dot(rel, e2) / dr.maximum(dr.squared_norm(e2), 1e-30)
            q = (si.p - self.cvec) * (1.0 / self.SR)
            zf = (q.z + 1.0) * (0.5 * T.S.SPH_NZ) - g(20); pf = (dr.atan2(q.y, q.x) + math.pi) * (T.S.SPH_NPHI / (2 * math.pi)) - g(19)
            sph = g(18) > 0.5
            a = dr.select(sph, pf, a); b = dr.select(sph, zf, b)
            su = U(dr.clamp(dr.floor(a * K), 0, K - 1)); sv = U(dr.clamp(dr.floor(b * K), 0, K - 1))
            return su + K * sv

        def construct4(self, n_sub, seed, b=8):
            ncs = self.N * K * K * NB
            acc = {}
            for launch, done in enumerate(range(0, n_sub, b)):
                bb = min(b, n_sub - done); Nl = ncs * bb
                sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
                rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
                slot = dr.arange(U, Nl) % ncs
                bn = slot % NB; ps = slot // NB; sub = ps % (K * K); pid = ps // (K * K)
                u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
                a = (F(sub % K) + u0) * (1.0 / K); bpar = (F(sub // K) + u1) * (1.0 / K)
                sid = dr.gather(U, self.PSIDU, pid)
                t, bv, n = self.patch_frame(pid)
                local = self.dir_in_bin(bn, u2, u3)
                z = 1.25 * u2 - 1.
                ph = (F(bn % T.S.NPHI) + u3) * (2 * math.pi / T.S.NPHI) - math.pi
                rr = dr.sqrt(dr.maximum(1 - z * z, 0.))
                local = dr.select(dr.eq(sid, T.S.SID_SPHERE) & (bn < T.S.NPHI), mi.Vector3f(rr * dr.cos(ph), rr * dr.sin(ph), z), local)
                w = t * local.x + bv * local.y + n * local.z
                si, ssid, nn = self.first_si(pid, a, bpar, w)
                bur = self.buried(si.p, sid, nn); sing = ~bur & self.singular_point(si.p, sid, nn)
                ray = mi.Ray3f(si.p + w * 2e-3, -w)
                out, cap = self.trace112(ray, Nl, rng, sd, 5, light, 'probe', first=si, first_ok=~bur & ~sing)
                cnt = cap['count']
                fray, fs = self.spawn(si, w)
                fh = self.scene.ray_intersect(fray, cnt & ~fs)
                recv = cnt & ~fs & fh.is_valid() & (fh.to_local(-w).z > 1e-6)
                vals = {}
                for meas, wt in (('U', dr.select(cnt, F(1.), F(0.))), ('G', dr.select(recv, dr.dot(w, nn), F(0.)))):
                    vals[f'{meas}_w'] = wt
                    for c in range(3):
                        vals[f'{meas}_L{c}'] = wt * (out[f'f2{c}'] + out[f'fh{c}'])
                keys = sorted(vals); vv = [vals[k] for k in keys]
                dr.eval(vv, slot); dr.sync_thread()
                if not acc:
                    acc = {k: dr.zeros(mi.Float64, ncs) for k in keys}
                for k, v in zip(keys, vv):
                    dr.scatter_reduce(dr.ReduceOp.Add, acc[k], mi.Float64(v), slot)
                dr.eval([acc[k] for k in keys]); dr.sync_thread()
            return {k: np.asarray(v) for k, v in acc.items()}

    ctx = Ctx4(geom, 'C')
    tabs = []
    for r in range(4):
        t0 = time.perf_counter()
        acc = ctx.construct4(1024, 11208001 + r, b=8)
        Uw = acc['U_w']; Uc = np.stack([acc[f'U_L{c}'] for c in range(3)], -1)
        Gw = acc['G_w']; Gc = np.stack([acc[f'G_L{c}'] for c in range(3)], -1)
        # coarse fallbacks from the same samples (pooled), then 4x4 tables with parent fallback (frozen chain)
        Ucoarse = (Uc.reshape(N, 16, NB, 3).sum(1) / np.maximum(Uw.reshape(N, 16, NB).sum(1), 1e-30)[..., None])
        Gcw = Gw.reshape(N, 16, NB).sum(1)
        Gcoarse = np.where((Gcw > 0)[..., None], Gc.reshape(N, 16, NB, 3).sum(1) / np.maximum(Gcw, 1e-30)[..., None], Ucoarse)
        U4 = np.where((Uw.reshape(N, 16, NB) > 0)[..., None], (Uc / np.maximum(Uw, 1e-30)[:, None]).reshape(N, 16, NB, 3), Ucoarse[:, None])
        G4 = np.where((Gw.reshape(N, 16, NB) > 0)[..., None], (Gc / np.maximum(Gw, 1e-30)[:, None]).reshape(N, 16, NB, 3), Gcoarse[:, None])
        tabs += [(f'S4_U_r{r}', U4.astype(np.float32), 'sub'), (f'S4_G_r{r}', G4.astype(np.float32), 'sub')]
        print('construct4', r, round(time.perf_counter() - t0, 1), flush=True)
    res = {}
    cams = C.views('F0')
    for vi, (v, cam) in enumerate(cams.items()):
        if v == 'H5':
            continue
        tl = [(n_, PT.fl(np.ascontiguousarray(t_).ravel()), k_) for n_, t_, k_ in tabs]
        imgs = []
        for j in range(2):
            seed = 11210000 + 10 * vi + j
            img, _, _, _ = ctx.cam_pass(light, dict(cam, res=C.RES), C.RES, 4096, seed, 'ecro', tables=tl, b=32)
            imgs.append(img)
        st = [np.load(root / 'stageB' / f'stream_{v}_{j}.npz') for j in range(2)]
        low_rel = max(float((np.abs(imgs[j]['e1'] - st[j]['ro_e1']).sum() + np.abs(imgs[j]['e2'] - st[j]['ro_e2']).sum()) /
                            (np.abs(st[j]['ro_e1']).sum() + np.abs(st[j]['ro_e2']).sum())) for j in range(2))
        h = [st[j]['phys_C_h'].astype(np.float64) for j in range(2)]
        gb = dict(np.load(root / 'stageB' / f'gbuffer_{v}.npz'))
        reg, bands = C.masks(gb, geom, lc[0]['pos'])
        kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
        row = {'low_pairing_rel_l1': low_rel}
        for arm in ('S4_U', 'S4_G'):
            rs = [[imgs[j][f'{arm}_r{r}'] - h[j] for j in range(2)] for r in range(4)]
            row[arm] = {'endpoints': MX.endpoint_stats(rs, reg['near'], reg['flank'], kappa=kap), 'bands': MX.band_profile(rs, bands),
                        'rmse': {k: MX.rmse_debiased(rs, h, reg[k])[0] for k in ('near', 'flank', 'sphere_boundary', 'contact_shadow', 'floor', 'full')}}
        res[v] = row
        print('view', v, json.dumps({a: (round(row[a]['endpoints']['near']['estimate'], 4), round(row[a]['endpoints']['contrast']['estimate'], 4)) for a in ('S4_U', 'S4_G')}), flush=True)
    np.savez_compressed(root / 'analysis' / 'diag_4x4_tables_r0.npz', S4_U=tabs[0][1], S4_G=tabs[1][1])
    return res


def main(root):
    root = Path(root)
    out = {'label': 'POST HOC diagnostic after the frozen Stage B outcome NONE; descriptive only; no verdict changes'}
    out['a_shape_decomposition'] = part_a(root)
    C.jd(out, root / 'analysis' / 'diag112.json')
    out['b_resolution_4x4'] = part_b(root)
    C.jd(out, root / 'analysis' / 'diag112.json')
    print('DIAG done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
