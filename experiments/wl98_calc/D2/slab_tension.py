"""WL98 D2 scratch (design input only, not evidence of record): the 'learned-value tension' of an intrinsic learned exit
measure, on a toy infinite homogeneous slab with SMOOTH dielectric faces (eta 1.5) and a Henyey-Greenstein phase function.

Question: does a regime exist in which the explicit twin is expensive (many internal events per exit, i.e. long walks the
learned macro-step would replace) AND the collided exit response is still angularly high-frequency (so a continuous /
adaptive angular representation matters)? For each (albedo, g, optical thickness tau, incidence cos_i) report:
  - mean number of volume-scattering events per exiting unit of flux (twin cost proxy), collided exits only;
  - collided exit flux on the entry side (R) and far side (T), per unit transmitted-in flux;
  - half-maximum solid angle of the collided exit radiance on each side and the rel L1 loss of the best flux-conserving
    4 x 16 and 16 x 64 equal-solid-angle piecewise-constant projection against a 32 x 128 histogram (angular high-frequency
    proxy; the 32 x 128 reference itself smears anything below ~1.5e-3 sr);
  - RMS lateral exit displacement in mean free paths (spatial spread).
Ballistic (zero-collision) exits are a separate path family (Architecture section 3) and are excluded. Smooth faces make
the exit lobe of a directional internal field as sharp as refraction allows; rough faces would broaden it. Toy only:
infinite slab, no edges, grey, float64, numpy, CPU.
"""
import json, math, os
import numpy as np

ETA = 1.5
NPH = 600000


def fresnel(ci, eta_i, eta_t):
    """unpolarised Fresnel reflectance for incidence cosine ci (>0) from medium eta_i into eta_t; returns R, cos_t."""
    s2t = (eta_i / eta_t) ** 2 * np.maximum(0.0, 1.0 - ci * ci)
    tir = s2t >= 1.0
    ct = np.sqrt(np.maximum(0.0, 1.0 - s2t))
    rs = (eta_i * ci - eta_t * ct) / (eta_i * ci + eta_t * ct + 1e-300)
    rp = (eta_t * ci - eta_i * ct) / (eta_t * ci + eta_i * ct + 1e-300)
    R = np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))
    return R, ct


def hg(d, g, rng):
    n = d.shape[0]
    u = rng.random(n)
    if abs(g) < 1e-6:
        ct = 1 - 2 * u
    else:
        ct = (1 + g * g - ((1 - g * g) / (1 - g + 2 * g * u)) ** 2) / (2 * g)
    st = np.sqrt(np.maximum(0, 1 - ct * ct))
    ph = 2 * math.pi * rng.random(n)
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t = np.cross(d, a); t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(d, t)
    return ct[:, None] * d + (st * np.cos(ph))[:, None] * t + (st * np.sin(ph))[:, None] * b


def walk(albedo, g, tau, cos_i, n, rng, max_ev=4000):
    # enter through the top face z = tau going down; weight = 1 per unit TRANSMITTED-in flux (entry reflection is local K)
    si = math.sqrt(1 - cos_i * cos_i)
    ct_in = math.sqrt(1 - (si / ETA) ** 2)
    d = np.tile([si / ETA, 0.0, -ct_in], (n, 1))
    x = np.zeros((n, 3)); x[:, 2] = tau
    w = np.ones(n); ev = np.zeros(n, np.int64); alive = np.ones(n, bool)
    rec = []          # (side, dir_out[3], weight, events, lateral r)
    while alive.any():
        idx = np.where(alive)[0]
        s = -np.log(1 - rng.random(idx.size))
        dz = d[idx, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            tb = np.where(dz < 0, (0 - x[idx, 2]) / dz, np.where(dz > 0, (tau - x[idx, 2]) / dz, np.inf))
        hit = tb <= s
        # boundary hits
        hi = idx[hit]
        if hi.size:
            x[hi] += d[hi] * tb[hit][:, None]
            ci = np.abs(d[hi, 2])
            R, ct = fresnel(ci, ETA, 1.0)
            refl = rng.random(hi.size) < R
            # reflect (stay inside)
            d[hi[refl], 2] *= -1
            # exit
            ex = hi[~refl]
            if ex.size:
                ctx = ct[~refl]
                dout = np.empty((ex.size, 3))
                dout[:, :2] = d[ex, :2] * ETA
                side_top = d[ex, 2] > 0
                dout[:, 2] = np.where(side_top, ctx, -ctx)
                rec.append(np.column_stack([side_top.astype(float), dout, w[ex], ev[ex].astype(float),
                                            np.hypot(x[ex, 0], x[ex, 1])]))
                alive[ex] = False
        # scattering events
        si_ = idx[~hit]
        if si_.size:
            x[si_] += d[si_] * s[~hit][:, None]
            ev[si_] += 1
            w[si_] *= albedo
            d[si_] = hg(d[si_], g, rng)
            # roulette
            low = si_[w[si_] < 0.05]
            if low.size:
                surv = rng.random(low.size) < 0.5
                w[low[surv]] *= 2.0
                w[low[~surv]] = 0.0
                alive[low[~surv]] = False
            stop = si_[ev[si_] >= max_ev]
            alive[stop] = False
    return np.concatenate(rec, 0) if rec else np.zeros((0, 7))


def ang_stats(dirs, wts, side_top):
    # local frame: z = outward normal of the exit face
    z = np.abs(dirs[:, 2]); ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    NZ, NP = 32, 128
    iz = np.clip((z * NZ).astype(int), 0, NZ - 1); ip = np.clip(((ph + math.pi) / (2 * math.pi) * NP).astype(int), 0, NP - 1)
    H = np.zeros((NZ, NP)); np.add.at(H, (iz, ip), wts)          # flux per fine bin
    if H.sum() <= 0:
        return None
    dw = (1 / NZ) * (2 * math.pi / NP)
    zc = ((np.arange(NZ) + 0.5) / NZ)[:, None]
    L = H / (zc * dw)                                               # radiance per fine bin
    hw = float((L > 0.5 * L.max()).sum() * dw)
    out = dict(half_max_sr=hw)
    for nz, nphi in [(4, 16), (16, 64)]:
        bz, bp = NZ // nz, NP // nphi
        Hb = H.reshape(nz, bz, nphi, bp).sum((1, 3), keepdims=True)
        cb = np.broadcast_to(zc.reshape(nz, bz, 1, 1), (nz, bz, nphi, bp)).sum((1, 3), keepdims=True)
        Hh = np.broadcast_to(Hb / cb * zc.reshape(nz, bz, 1, 1), (nz, bz, nphi, bp)).reshape(NZ, NP)
        out[f'loss_{nz}x{nphi}'] = float(np.abs(Hh - H).sum() / H.sum())
    return out


def main():
    rng = np.random.default_rng(98003)
    res = []
    for albedo in [0.8, 0.99, 0.999]:
        for g in [0.0, 0.9]:
            for tau in [1.0, 10.0]:
                for cos_i in [0.1, 0.9]:
                    rec = walk(albedo, g, tau, cos_i, NPH, rng)
                    coll = rec[rec[:, 5] >= 1]
                    row = dict(albedo=albedo, g=g, tau=tau, cos_i=cos_i)
                    W = coll[:, 4]
                    row['collided_flux_per_unit_in'] = float(W.sum() / NPH)
                    row['mean_events_per_exit_flux'] = float((coll[:, 5] * W).sum() / max(W.sum(), 1e-30))
                    row['rms_lateral_mfp'] = float(math.sqrt((coll[:, 6] ** 2 * W).sum() / max(W.sum(), 1e-30)))
                    for side, nm in [(1.0, 'R'), (0.0, 'T')]:
                        m = coll[:, 0] == side
                        row[f'{nm}_flux'] = float(coll[m, 4].sum() / NPH)
                        row[f'{nm}_ang'] = ang_stats(coll[m, 1:4], coll[m, 4], side == 1.0)
                        # noise-floor control: the same weights on cosine-distributed (Lambertian) directions
                        nn = int(m.sum())
                        if nn:
                            u1, u2 = rng.random(nn), rng.random(nn)
                            zz = np.sqrt(u1); rr = np.sqrt(1 - zz * zz); pp = 2 * math.pi * u2
                            dl = np.stack([rr * np.cos(pp), rr * np.sin(pp), zz], -1)
                            row[f'{nm}_ang_lambert_noise_floor'] = ang_stats(dl, coll[m, 4], True)
                    res.append(row)
                    print('a %.3f g %.1f tau %4.1f ci %.1f | ev %7.1f | R %.3f T %.3f | R hm %s 4x16 %s 16x64 %s | T hm %s 4x16 %s | lat %.2f | floor R4x16 %s T4x16 %s' % (
                        albedo, g, tau, cos_i, row['mean_events_per_exit_flux'], row['R_flux'], row['T_flux'],
                        '%.4f' % row['R_ang']['half_max_sr'] if row['R_ang'] else '-',
                        '%.3f' % row['R_ang']['loss_4x16'] if row['R_ang'] else '-',
                        '%.3f' % row['R_ang']['loss_16x64'] if row['R_ang'] else '-',
                        '%.4f' % row['T_ang']['half_max_sr'] if row['T_ang'] else '-',
                        '%.3f' % row['T_ang']['loss_4x16'] if row['T_ang'] else '-',
                        row['rms_lateral_mfp'],
                        '%.3f' % row['R_ang_lambert_noise_floor']['loss_4x16'] if row.get('R_ang_lambert_noise_floor') else '-',
                        '%.3f' % row['T_ang_lambert_noise_floor']['loss_4x16'] if row.get('T_ang_lambert_noise_floor') else '-'), flush=True)
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'wl98_calc', 'D2', 'slab_tension.json')
    with open(os.path.abspath(out), 'w', newline='\n') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
