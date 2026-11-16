"""WL97 audit track E2, supplementary measurements (own code; uses the functions of e2_chain.py).

  python e2_extra.py mquad  <R>    sensitivity of the bin-averaged operator (the M64 object) to its incident quadrature -> results/mquad.json
  python e2_extra.py binres <R>    outgoing-direction projection at finer equal-solid-angle grids (per ray)           -> results/binres.npz
"""
import json, math, os, sys, time
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_chain as C  # noqa: E402

CMD, R = sys.argv[1], sys.argv[2]
OUT = C.OUT
NB, K = C.NB, C.K
S1, BOXA = C.S1, C.BOXA


def deliver(E, rw, idx, L, D):
    rec = idx // K
    e = np.zeros((D * 54, 3))
    for c in range(3):
        e[:, c] = np.bincount(rec, weights=rw[idx] * L[:, c], minlength=D * 54)
    return (E.rho * e).reshape(D, 54, 3)


def rays_for(E, D, seed0, chunk=32):
    for a in range(0, D, chunk):
        b = min(D, a + chunk)
        U4 = np.stack([np.random.default_rng(seed0 + d).random((4, 54, K), np.float32) for d in range(a, b)]).astype(np.float64)
        o, w, rw = C.make_rays(E, U4)
        sid, hp = C.trace(E, o, w)
        idx, x, v, pid, bo, ok = C.panel_hits(E, sid, hp, w)
        assert ok
        yield b - a, rw, idx, x, v, pid, bo


def m_rows(E, bins, mode, n_out, seed, sub=8):
    """Rows of the bin-averaged operator for the incident bins `bins`: mean over the incident bin (mode: 'mid' = sub x sub cell centres,
    'mc' = uniform random) and over the outgoing bin (uniform random) of f = eval(v, w) / cos(w)."""
    mi, dr = E.mi, E.dr
    rng = np.random.default_rng(seed)
    cl = E.closure(1)
    M = np.zeros((NB, NB, 3))
    bo = np.repeat(np.arange(NB), n_out)
    for bi in bins:
        if mode == 'mid':
            # one launch per group of incident cells (the first version launched once per cell and was stopped after 9 min; see report)
            n_each = max(1, n_out // (sub * sub))
            ncell = sub * sub
            cells_per = max(1, (1 << 22) // (NB * n_each))
            acc = np.zeros((NB, 3))
            for c0 in range(0, ncell, cells_per):
                cc = np.arange(c0, min(ncell, c0 + cells_per))
                m = len(cc) * NB * n_each
                cell = np.repeat(cc, NB * n_each)
                bo2 = np.tile(np.repeat(np.arange(NB), n_each), len(cc))
                w = C.dir_in_bin_np(np.full(m, bi), (cell // sub + 0.5) / sub, (cell % sub + 0.5) / sub)
                u = rng.random((2, m))
                v = C.dir_in_bin_np(bo2, u[0], u[1])
                f = cl.eval(E.V3(v), E.V3(w), True)
                fc = [f[c] for c in range(3)]
                dr.eval(fc); dr.sync_thread()
                for c in range(3):
                    acc[:, c] += (np.asarray(fc[c], np.float64) / w[:, 2]).reshape(len(cc), NB, n_each).mean(2).sum(0)
            M[bi] = acc / ncell
        else:
            u = rng.random((4, NB * n_out))
            w = C.dir_in_bin_np(np.full(NB * n_out, bi), u[0], u[1])
            v = C.dir_in_bin_np(bo, u[2], u[3])
            ok = w[:, 2] > 1e-6
            f = cl.eval(E.V3(v), E.V3(w), mi.Bool(np.ascontiguousarray(ok)))
            fc = [f[c] for c in range(3)]
            dr.eval(fc); dr.sync_thread()
            for c in range(3):
                M[bi, :, c] = (np.asarray(fc[c], np.float64) / np.maximum(w[:, 2], 1e-6)).reshape(NB, n_out).mean(1)
    return M


def mquad(E):
    z = np.load(os.path.join(OUT, 'senders_e2.npz'))
    spool, A4, q = z['spool'], z['A4'], z['q']
    bins = np.nonzero(spool.sum((0, 2)) > 0)[0]
    info = {'incident_bins_with_energy': bins.tolist(), 'energy_share': (spool.sum((0, 2))[bins] / spool.sum()).tolist()}
    V = {'A4': A4}
    t0 = time.time()
    print('incident bins', bins.tolist(), flush=True)
    V['mid8'] = m_rows(E, bins, 'mid', 1 << 19, 9770001, 8)
    print('mid8', round(time.time() - t0, 1), flush=True)
    V['mid32'] = m_rows(E, bins, 'mid', 1 << 19, 9770002, 32)
    print('mid32', round(time.time() - t0, 1), flush=True)
    V['mid128'] = m_rows(E, bins, 'mid', 1 << 19, 9770003, 128)
    print('mid128', round(time.time() - t0, 1), flush=True)
    V['mc_a'] = m_rows(E, bins, 'mc', 1 << 17, 9770004)
    V['mc_b'] = m_rows(E, bins, 'mc', 1 << 17, 9770005)
    print('mc', round(time.time() - t0, 1), flush=True)
    V['mc'] = 0.5 * (V['mc_a'] + V['mc_b'])
    info['seconds_build'] = time.time() - t0
    zb = ((np.arange(NB) // C.NPHI) + 0.5) / C.NZ
    wt = (zb * C.DOM)[None, :, None]
    T = {k: np.einsum('pmc,mdc->pdc', spool, M) for k, M in V.items()}
    info['field_energy_vs_q'] = {k: float((t * wt).sum() / (q * wt).sum()) for k, t in T.items()}
    info['row_flux_ratio_vs_A4'] = {k: float((M[bins] * zb[None, :, None]).sum() / (A4[bins] * zb[None, :, None]).sum()) for k, M in V.items()}
    T['q'] = q
    D = 512
    acc = {k: [] for k in T}
    for d, rw, idx, x, v, pid, bo in rays_for(E, D, 9780000):
        for k, t in T.items():
            acc[k].append(deliver(E, rw, idx, t[pid, bo], d))
    ex = {k: np.concatenate(a).sum(-1) for k, a in acc.items()}
    ar = E.rho * np.load(os.path.join(OUT, 'aref_main.npz'))['E'].sum(2).mean(0)
    info['delivered'] = {}
    for k, e in ex.items():
        r = e.sum(1) / ar.sum()
        info['delivered'][k] = {'ratio_vs_area': float(e.mean(0).sum() / ar.sum()), 'ratio_se': float(r.std(ddof=1) / math.sqrt(D)),
                                'abs_rel_l1_vs_area': float(np.abs(e.mean(0) - ar).sum() / ar.sum())}
    for k in ('mid8', 'mid32', 'mid128', 'mc', 'mc_a', 'mc_b'):
        dlt = (ex[k] - ex['A4']).sum(1) / ar.sum()
        info['delivered'][k]['paired_d_ratio_vs_A4'] = float(dlt.mean()); info['delivered'][k]['paired_se'] = float(dlt.std(ddof=1) / math.sqrt(D))
    C.jd(info, os.path.join(OUT, 'mquad.json'))
    print(json.dumps(info, indent=1, default=float), flush=True)


def binres(E):
    """Per ray: exact hit point, outgoing direction redrawn uniformly inside its cell of an nz x nphi equal-solid-angle grid (4 redraws);
    and the same with the position redrawn uniformly in the hit patch ('both')."""
    G = E.G
    grids = [(4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
    D = 768
    rng = np.random.default_rng(9790001)
    acc = {}
    t0 = time.time()
    for d, rw, idx, x, v, pid, bo in rays_for(E, D, 9790100):
        loc = C.to_local(E, v)
        m = len(idx)
        res = {'X1': C.er_radiance(E, x, v)}
        for nz, nphi in grids:
            iz = np.clip(np.floor(loc[:, 2] * nz), 0, nz - 1)
            ip = np.clip(np.floor((np.arctan2(loc[:, 1], loc[:, 0]) + math.pi) * (nphi / (2 * math.pi))), 0, nphi - 1)
            a_pos = np.zeros((m, 3)); a_both = np.zeros((m, 3))
            for _ in range(4):
                u = rng.random((4, m))
                zc = (iz + u[0]) / nz; ph = (ip + u[1]) * (2 * math.pi / nphi) - math.pi
                rr = np.sqrt(np.maximum(1 - zc * zc, 0))
                vr = C.to_world(E, np.stack([rr * np.cos(ph), rr * np.sin(ph), zc], 1))
                xr = G.PO[S1][pid] + u[2][:, None] * G.PE1[S1][pid] + u[3][:, None] * G.PE2[S1][pid]
                a_pos += C.er_radiance(E, x, vr); a_both += C.er_radiance(E, xr, vr)
            res[f'pos_exact_dir_{nz}x{nphi}'] = a_pos / 4; res[f'pos_patchavg_dir_{nz}x{nphi}'] = a_both / 4
        for k, L in res.items():
            acc.setdefault(k, []).append(deliver(E, rw, idx, L, d).astype(np.float32))
    np.savez_compressed(os.path.join(OUT, 'binres.npz'), **{k: np.concatenate(a) for k, a in acc.items()})
    ar = E.rho * np.load(os.path.join(OUT, 'aref_main.npz'))['E'].sum(2).mean(0)
    info = {'D': D, 'seconds': time.time() - t0, 'seed0': 9790100, 'arms': {}}
    for k, a in acc.items():
        e = np.concatenate(a).astype(np.float64).sum(-1)
        r = e.sum(1) / ar.sum()
        h = len(e) // 2
        info['arms'][k] = {'ratio_vs_area': float(e.mean(0).sum() / ar.sum()), 'ratio_se': float(r.std(ddof=1) / math.sqrt(len(e))),
                           'abs_rel_l1_vs_area': float(np.abs(e.mean(0) - ar).sum() / ar.sum()),
                           'abs_rel_l1_half_draws_mean': float(0.5 * (np.abs(e[:h].mean(0) - ar).sum() + np.abs(e[h:].mean(0) - ar).sum()) / ar.sum())}
    C.jd(info, os.path.join(OUT, 'binres.json'))
    print(json.dumps(info, indent=1, default=float), flush=True)


if __name__ == '__main__':
    E = C.Env()
    {'mquad': mquad, 'binres': binres}[CMD](E)
    print('DONE', CMD, flush=True)
    os._exit(0)
