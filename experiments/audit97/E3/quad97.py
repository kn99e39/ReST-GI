"""WL97 audit track E3: deterministic area-form quadrature of the exactly-2-event panel-mediated irradiance on the 54 box-A patches of
CK2 'none' (light -> S1 panel -> Lambert box A), with NO relation sampling and no Monte Carlo. Independent code (own ER port er97, own
geometry geo97); float64 torch.

  E_j,c = (1 / A_j) int_{A_j} dA_r int_{panel} dA_s  L_c(x_s -> x_r) cos_s cos_r / r^2 V(x_s, x_r)          (irradiance, per channel)
  exitance_j,c = rho_A E_j,c  (rho_A = 0.5, applied by the analysis)
  V = [cos_s > 0][cos_r > 0]  (no other occluder between panel, box A and light: val97_dr.py T_VIS); light -> panel unoccluded.

Sender models (identical quadrature nodes):
  exact   L = eval_ER(wi = dir(x_s -> x_r), wo = dir(x_s -> light)) I / |light - x_s|^2          (eval = f cos(wo); NEE convention of pt86)
  anchor  as exact but the incident state (light direction, cosine, distance) is taken at the CENTRE of the 20 cm panel patch of x_s;
          the outgoing direction stays the exact x_s -> x_r direction                              (emission-anchor effect alone)
  table   L = Q[patch(x_s), bin(dir(x_s -> x_r) in the patch frame)]   for every table Q [42, 64, 3] in the tables file (binned senders)

Tables (subcommand `tables`):
  qbin    Q[i, b] = patch-average over x and solid-angle average over the outgoing bin b of the exact once-scattered radiance
          (exact incident direction per point) - "what a patch x 64-bin table of the true q delivers"
  m64     Q[i, bo] = sum_bi s[i, bi] Mq[bi, bo]; s = patch-average irradiance per incident bin; Mq = incident-bin average (n_c x n_c cell
          centres, uniform solid angle) of the outgoing-bin average of f (no cosine) - the M64(s) construction, built here from ER
  Outgoing-bin integrals use a half-vector parameterisation v = 2 (w.h) h - w, dv = 4 (w.h) dOmega_h, theta_h = (pi / 2) t^p grid
  (resolves the GGX lobes); validated against a brute-force uniform (z, phi) grid and against the Lambert closed form (subcommand `tabtest`).

  python quad97.py tabtest <CODE> <OUT>
  python quad97.py tables  <CODE> <OUT> <tag> <npp> <nth> <nph> <ncell>
  python quad97.py deliver <CODE> <OUT> <tag> <hs_mm> <nr> [--tables=<file>] [--off=<a>,<b>] [--roff=<a>,<b>] [--noanchor] [--bindir=<nth>,<nph>]
  bindir  L = bin-average over the outgoing bin of the exact once-scattered radiance AT THE EXACT SENDER POINT (angular binning alone)
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import er97  # noqa: E402
import geo97  # noqa: E402

DT = torch.float64
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NB, NZ, NPHI, DOM = geo97.NB, geo97.NZ, geo97.NPHI, geo97.DOM
TWO_PI = 2 * math.pi
PAIR_CHUNK = 3_000_000


def T(a):
    return torch.as_tensor(np.asarray(a, np.float64), dtype=DT, device=DEV)


def setup(CODE):
    fx, sc = geo97.load(CODE)
    st = np.load(os.path.join(CODE, 'experiments', 'ecro86', 'data', 'mats86.npz'))['states']
    LP = np.array(sc['lights'][0]['pos'], float); I = np.array(sc['lights'][0]['rgb'], float)
    pan = geo97.panel(fx); faces = geo97.box_faces(sc['geom']['A'])
    assert np.allclose(pan['a1'], [0, 1, 0]) and np.allclose(pan['a2'], [0, 0, 1]) and np.allclose(pan['n'], [1, 0, 0])
    return fx, sc, st, LP, I, pan, faces


def bin_of(x, y, z):
    iz = torch.clamp(torch.floor(z * NZ), 0, NZ - 1)
    ip = torch.clamp(torch.floor((torch.atan2(y, x) + math.pi) * (NPHI / TWO_PI)), 0, NPHI - 1)
    return (iz * NPHI + ip).long()


def panel_points(pan, nu, nv, off=(0.5, 0.5)):
    """Grid nodes on the panel: in-plane coordinates (u along +y, v along +z, origin at the panel corner), world points, patch index."""
    du = 2 * pan['h'][0] / nu; dv = 2 * pan['h'][1] / nv
    u = (torch.arange(nu, dtype=DT, device=DEV) + off[0]) * du
    v = (torch.arange(nv, dtype=DT, device=DEV) + off[1]) * dv
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    uu = uu.reshape(-1); vv = vv.reshape(-1)
    c = pan['c']
    X = torch.stack([torch.full_like(uu, c[0]), c[1] - pan['h'][0] + uu, c[2] - pan['h'][1] + vv], 1)
    pu = 2 * pan['h'][0] / pan['nu']; pv = 2 * pan['h'][1] / pan['nv']
    iu = torch.clamp(torch.floor(uu / pu), 0, pan['nu'] - 1); iv = torch.clamp(torch.floor(vv / pv), 0, pan['nv'] - 1)
    patch = (iu + pan['nu'] * iv).long()
    return X, patch, du * dv


def light_state(X, LP, I):
    """Local (panel frame: x = +y, y = +z, z = +x) unit light direction and I / d^2 per channel at points X [n, 3]."""
    d = T(LP)[None] - X
    d2 = (d * d).sum(1); inv = torch.rsqrt(d2)
    return (d[:, 1] * inv, d[:, 2] * inv, d[:, 0] * inv), d2


# ------------------------------------------------------------------------------------------------ outgoing-bin integrals
def hgrid(nth, nph, p=2.0):
    t = (torch.arange(nth, dtype=DT, device=DEV) + 0.5) / nth
    th = (math.pi / 2) * t ** p; dth = (math.pi / 2) * p * t ** (p - 1) / nth
    ph = (torch.arange(nph, dtype=DT, device=DEV) + 0.5) / nph * TWO_PI - math.pi
    TH, PH = torch.meshgrid(th, ph, indexing='ij')
    W = (torch.sin(TH) * dth[:, None] * (TWO_PI / nph)).reshape(-1)
    hx = (torch.sin(TH) * torch.cos(PH)).reshape(-1); hy = (torch.sin(TH) * torch.sin(PH)).reshape(-1); hz = torch.cos(TH).reshape(-1)
    return hx, hy, hz, W


def fbar_eval(er, w, nth, nph, chunk=2_500_000):
    """[n, NB, 3]: (1 / DOM) int_{v in bin} eval(wi = v, wo = w) dv for light directions w [n, 3] (local frame), half-vector quadrature."""
    hx, hy, hz, W = hgrid(nth, nph)
    H = hx.numel(); n = w.shape[0]
    out = torch.zeros(n * NB, 3, dtype=DT, device=DEV)
    step = max(1, chunk // H)
    for a in range(0, n, step):
        ww = w[a:a + step]; m = ww.shape[0]
        wx, wy, wz = ww[:, 0:1], ww[:, 1:2], ww[:, 2:3]
        wh = wx * hx[None] + wy * hy[None] + wz * hz[None]
        vx = 2 * wh * hx[None] - wx; vy = 2 * wh * hy[None] - wy; vz = 2 * wh * hz[None] - wz
        ok = (wh > 0) & (vz > 1e-9)
        vzc = torch.where(ok, vz, torch.ones_like(vz))
        f = er.eval((vx, vy, vzc), (wx, wy, wz))
        jac = torch.where(ok, 4 * wh * W[None], torch.zeros_like(wh))
        idx = (torch.arange(m, device=DEV)[:, None] + a) * NB + bin_of(vx, vy, vzc)
        for c in range(3):
            out[:, c].index_add_(0, idx.reshape(-1), (f[c] * jac).reshape(-1))
    return out.reshape(n, NB, 3) / DOM


def fbar_brute(er, w, nz, nphi):
    """Same quantity by a uniform (z, phi) midpoint grid inside every bin (nz x nphi nodes per bin); one direction w [3]."""
    out = torch.zeros(NB, 3, dtype=DT, device=DEV)
    a = (torch.arange(nz, dtype=DT, device=DEV) + 0.5) / nz; b = (torch.arange(nphi, dtype=DT, device=DEV) + 0.5) / nphi
    A, B = torch.meshgrid(a, b, indexing='ij')
    wo = (w[0].reshape(1, 1), w[1].reshape(1, 1), w[2].reshape(1, 1))
    for bn in range(NB):
        z = (bn // NPHI + A) / NZ; ph = (bn % NPHI + B) * (TWO_PI / NPHI) - math.pi
        r = torch.sqrt(torch.clamp(1 - z * z, min=0))
        f = er.eval((r * torch.cos(ph), r * torch.sin(ph), z), wo)
        for c in range(3):
            out[bn, c] = f[c].mean()
    return out


def tabtest(CODE, OUT):
    fx, sc, st, LP, I, pan, faces = setup(CODE)
    res = {}
    wt = T(geo97.bin_weights())
    # Lambert closed form: every bin = rho / pi cos(theta_l)
    lam = er97.ER(st[4], DEV)
    w = T([[0.3, -0.5, math.sqrt(1 - 0.34)], [-0.9, 0.42, math.sqrt(1 - 0.81 - 0.1764)], [0.6, 0.79, 0.12]])
    fb = fbar_eval(lam, w, 384, 256)
    ref = (0.5 / math.pi * w[:, 2])[:, None, None]
    res['lambert_bin_max_rel'] = {f'{nth}x{nph}': float((fbar_eval(lam, w, nth, nph) / ref - 1).abs().max()) for nth, nph in ((192, 128), (384, 256), (768, 512))}
    # S1: CK2-like grazing light directions (panel frame); brute force at two resolutions vs half-vector quadrature at three
    s1 = er97.ER(st[1], DEV)
    Xc, _, _ = panel_points(pan, 6, 7)
    (lx, ly, lz), d2 = light_state(Xc, LP, I)
    pick = [0, 5, 17, 20, 36, 41]
    ws = torch.stack([lx, ly, lz], 1)[pick]
    res['test_dirs'] = ws.cpu().numpy().round(5).tolist()
    br = {}
    for nz, nphi in ((256, 2048), (512, 4096)):
        t0 = time.time()
        br[f'{nz}x{nphi}'] = torch.stack([fbar_brute(s1, ws[k], nz, nphi) for k in range(len(pick))])
        res[f'brute_{nz}x{nphi}_s'] = time.time() - t0
    bref = br["512x4096"]
    alb_ref = (bref * wt[None, :, None]).sum(1)                                   # [k, 3] directional albedo x cos_l  (= int eval dv)
    res['brute_self_convergence'] = {'albedo_max_rel': float(((br["256x2048"] * wt[None, :, None]).sum(1) / alb_ref - 1).abs().max()),
                                     'bin_max_abs_over_bin_max': float((br["256x2048"] - bref).abs().max() / bref.max())}
    for nth, nph in ((192, 128), (384, 256), (768, 512), (1536, 1024)):
        t0 = time.time()
        fb = fbar_eval(s1, ws, nth, nph)
        res[f'hv_{nth}x{nph}'] = {'albedo_max_rel_vs_brute': float(((fb * wt[None, :, None]).sum(1) / alb_ref - 1).abs().max()),
                                  'bin_L1_rel_vs_brute': float((fb - bref).abs().sum() / bref.abs().sum()),
                                  'bin_max_abs_over_bin_max': float((fb - bref).abs().max() / bref.max()), 's': time.time() - t0}
    res['albedo_times_cos'] = alb_ref.cpu().numpy().tolist()
    res['directional_albedo'] = (alb_ref / ws[:, 2:3]).cpu().numpy().tolist()
    json.dump(res, open(os.path.join(OUT, 'tabtest.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


def tables(CODE, OUT, tag, npp, nth, nph, ncell):
    fx, sc, st, LP, I, pan, faces = setup(CODE)
    s1 = er97.ER(st[1], DEV)
    It = T(I)
    t0 = time.time()
    # qbin: npp x npp midpoints per panel patch
    X, patch, dA = panel_points(pan, pan['nu'] * npp, pan['nv'] * npp)
    (lx, ly, lz), d2 = light_state(X, LP, I)
    assert bool((lz > 1e-6).all())
    w = torch.stack([lx, ly, lz], 1)
    fb = fbar_eval(s1, w, nth, nph)                                              # [n, NB, 3]  bin-average of eval (cos_l included)
    L = fb * (It[None, None, :] / d2[:, None, None])
    Q = torch.zeros(42, NB, 3, dtype=DT, device=DEV)
    Q.index_add_(0, patch, L)
    Q /= npp * npp
    t1 = time.time()
    # m64: s from a fine patch grid (64 x 64 per patch), Mq from ncell x ncell cell centres per incident bin
    Xf, pf, _ = panel_points(pan, pan['nu'] * 64, pan['nv'] * 64)
    (fx_, fy_, fz_), fd2 = light_state(Xf, LP, I)
    bi = bin_of(fx_, fy_, fz_)
    s = torch.zeros(42 * NB, dtype=DT, device=DEV)
    s.index_add_(0, pf * NB + bi, fz_ / fd2)
    s = (s / (64 * 64)).reshape(42, NB)                                          # per unit intensity
    used = torch.nonzero(s.sum(0) > 0).reshape(-1)
    a = (torch.arange(ncell, dtype=DT, device=DEV) + 0.5) / ncell
    A, B = torch.meshgrid(a, a, indexing='ij')
    Mq = torch.zeros(NB, NB, 3, dtype=DT, device=DEV)
    for b_ in used.tolist():
        z = ((b_ // NPHI) + A.reshape(-1)) / NZ; ph = ((b_ % NPHI) + B.reshape(-1)) * (TWO_PI / NPHI) - math.pi
        r = torch.sqrt(1 - z * z)
        wc = torch.stack([r * torch.cos(ph), r * torch.sin(ph), z], 1)
        fc = fbar_eval(s1, wc, nth, nph) / z[:, None, None]                      # bin-average of f (no cosine)
        Mq[b_] = fc.mean(0)
    Qm = torch.einsum('ib,boc->ioc', s, Mq) * It[None, None, :]
    t2 = time.time()
    np.savez_compressed(os.path.join(OUT, f'tables_{tag}.npz'), qbin=Q.cpu().numpy(), m64=Qm.cpu().numpy(), s=s.cpu().numpy(), Mq=Mq.cpu().numpy(),
                        used=used.cpu().numpy(), meta=json.dumps({'npp': npp, 'nth': nth, 'nph': nph, 'ncell': ncell, 'qbin_s': t1 - t0, 'm64_s': t2 - t1}))
    wt = T(geo97.bin_weights())
    print('TABLES', tag, 'qbin', round(t1 - t0, 1), 's  m64', round(t2 - t1, 1), 's  used incident bins', used.tolist(),
          ' flux qbin / m64 (sum_b W Q, patch-summed, rgb-summed):', float((Q * wt[None, :, None]).sum()), float((Qm * wt[None, :, None]).sum()), flush=True)


# ------------------------------------------------------------------------------------------------ delivery
def deliver(CODE, OUT, tag, hs_mm, nr, tabfile=None, off=(0.5, 0.5), roff=(0.5, 0.5), anchor=True, bindir=None):
    fx, sc, st, LP, I, pan, faces = setup(CODE)
    s1 = er97.ER(st[1], DEV)
    hs = hs_mm * 1e-3
    nu = int(round(2 * pan['h'][0] / hs)); nv = int(round(2 * pan['h'][1] / hs))
    X, patch, dA = panel_points(pan, nu, nv, off)
    (lx, ly, lz), d2 = light_state(X, LP, I)
    Xc, pc, _ = panel_points(pan, pan['nu'], pan['nv'])                             # patch centres; pc = patch index of each centre node
    assert sorted(pc.tolist()) == list(range(42))
    inv_pc = torch.empty_like(pc); inv_pc[pc] = torch.arange(42, device=DEV)        # FIX (v1 indexed the 'ij'-ordered nodes by patch id)
    (cx, cy, cz), cd2 = light_state(Xc[inv_pc], LP, I)
    ax, ay, az, ad2 = cx[patch], cy[patch], cz[patch], cd2[patch]
    assert float((Xc[inv_pc][patch] - X).abs().max()) <= 0.5 * math.hypot(2 * pan['h'][0] / pan['nu'], 2 * pan['h'][1] / pan['nv']) + 1e-9
    nodetab = None
    if bindir:
        t1 = time.time()
        nodetab = (fbar_eval(s1, torch.stack([lx, ly, lz], 1), bindir[0], bindir[1]) * (T(I)[None, None, :] / d2[:, None, None])).reshape(-1, 3)
        print('bindir node table', X.shape[0], 'nodes', round(time.time() - t1, 1), 's', flush=True)
    tabs = {}
    if tabfile:
        zt = np.load(tabfile)
        tabs = {k: T(zt[k]).reshape(42 * NB, 3) for k in ('qbin', 'm64')}
    models = ['exact'] + (['anchor'] if anchor else []) + list(tabs) + (['bindir'] if bindir else [])
    E = {m: torch.zeros(54, 3, dtype=DT, device=DEV) for m in models}
    npairs = 0
    t0 = time.time()
    for fi, f in enumerate(faces):
        nf = T(f['n'])
        smask = ((X - T(f['c'])[None]) @ nf) > 0                                    # cos_r > 0 <=> sender in front of the face plane
        if not bool(smask.any()):
            continue
        sidx = torch.nonzero(smask).reshape(-1)
        O, e1, e2, area = geo97.patch_rects(f)
        a = (torch.arange(nr, dtype=DT, device=DEV) + roff[0]) / nr; b = (torch.arange(nr, dtype=DT, device=DEV) + roff[1]) / nr
        A, B = torch.meshgrid(a, b, indexing='ij')
        A = A.reshape(-1); B = B.reshape(-1)
        R = A.numel()
        step = max(1, PAIR_CHUNK // R)
        for j in range(9):
            Xr = T(O[j])[None] + A[:, None] * T(e1)[None] + B[:, None] * T(e2)[None]
            acc = {m: torch.zeros(3, dtype=DT, device=DEV) for m in models}
            for s0 in range(0, sidx.numel(), step):
                ii = sidx[s0:s0 + step]
                xs = X[ii]
                dx = Xr[None, :, 0] - xs[:, 0:1]; dy = Xr[None, :, 1] - xs[:, 1:2]; dz = Xr[None, :, 2] - xs[:, 2:3]
                r2 = dx * dx + dy * dy + dz * dz
                inv = torch.rsqrt(r2)
                wx, wy, wz = dx * inv, dy * inv, dz * inv
                cos_s = wx                                                          # panel normal +x
                cos_r = -(wx * nf[0] + wy * nf[1] + wz * nf[2])
                Gm = torch.clamp(cos_s, min=0) * torch.clamp(cos_r, min=0) / r2
                wi = (wy, wz, torch.clamp(wx, min=1e-12))                            # panel frame (s, t, n) = (+y, +z, +x)
                fe = s1.eval(wi, (lx[ii, None], ly[ii, None], lz[ii, None]))
                sc_ = Gm / d2[ii, None]
                for c in range(3):
                    acc['exact'][c] += (fe[c] * sc_).sum()
                if anchor:
                    fa = s1.eval(wi, (ax[ii, None], ay[ii, None], az[ii, None]))
                    sa = Gm / ad2[ii, None]
                    for c in range(3):
                        acc['anchor'][c] += (fa[c] * sa).sum()
                if tabs or bindir:
                    bo = bin_of(wi[0], wi[1], wi[2])
                    idx = patch[ii, None] * NB + bo
                    for k, Q in tabs.items():
                        acc[k] += (Q[idx.reshape(-1)] * Gm.reshape(-1, 1)).sum(0)
                    if bindir:                                                      # exact sender POSITION, bin-averaged outgoing direction
                        acc['bindir'] += (nodetab[(ii[:, None] * NB + bo).reshape(-1)] * Gm.reshape(-1, 1)).sum(0)
                npairs += ii.numel() * R
            pj = f['base'] - geo97.BOXA0 + j
            for m in models:
                scale = dA / R
                E[m][pj] = acc[m] * scale
    It = T(I)
    out = {}
    for m in models:
        out[m] = (E[m] * (It[None] if m in ('exact', 'anchor') else 1.0)).cpu().numpy()
    secs = time.time() - t0
    meta = {'tag': tag, 'hs_mm': hs_mm, 'nu': nu, 'nv': nv, 'nr': nr, 'off': list(off), 'roff': list(roff), 'pairs': npairs, 'seconds': secs, 'tables': tabfile, 'bindir': bindir,
            'gpu_peak_mib': (torch.cuda.max_memory_allocated() / 2 ** 20 if DEV == 'cuda' else 0.0)}
    np.savez_compressed(os.path.join(OUT, f'deliv_{tag}.npz'), meta=json.dumps(meta), **{f'E_{m}': v for m, v in out.items()})
    print('DELIVER', json.dumps(meta), {m: float(0.5 * v.sum()) for m, v in out.items()}, flush=True)


def main():
    cmd, CODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[4:] if a.startswith('--') and '=' in a}
    args = [a for a in sys.argv[4:] if not a.startswith('--')]
    os.makedirs(OUT, exist_ok=True)
    with torch.no_grad():
        if cmd == 'tabtest':
            tabtest(CODE, OUT)
        elif cmd == 'tables':
            tables(CODE, OUT, args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]))
        elif cmd == 'deliver':
            pf = lambda s: tuple(float(v) for v in s.split(','))
            deliver(CODE, OUT, args[0], float(args[1]), int(args[2]), opt.get('--tables'), pf(opt.get('--off', '0.5,0.5')), pf(opt.get('--roff', '0.5,0.5')),
                    anchor='--noanchor' not in sys.argv, bindir=(tuple(int(v) for v in opt['--bindir'].split(',')) if '--bindir' in opt else None))


if __name__ == '__main__':
    main()
