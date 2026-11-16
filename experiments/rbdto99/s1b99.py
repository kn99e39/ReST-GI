"""WL99 Stage 1b: learning-free representations of N_b on the direct-light input (Sd), their error / memory / cost frontier.

Arms (protocol rbdto99_v1 stage1):
  E2  TwinP: per-body precomputed transfer. Incident cells (main-face spatial cell h_x x projected-solid-angle direction cell, N_d^2 per
      hemisphere) -> n_e light-side self paths per cell, stored as self-incident photons (landing plate, uv, incoming direction, flux per
      unit incident flux). Applied to incident samples (explicit NEE with external visibility) by nearest-cell lookup.
  E3  TwinP ladder of memory budgets (h_x, N_d, n_e) + first-order offset variant 'J' (per-cell least-squares Jacobian of the landing
      position with respect to the in-cell incident offset) + fp16 storage accounting.
  E4  dense limit of the precomputed family: per-update light-side tracing of n_e paths from every exact incident sample (no cells).
  E1c cached self-relation explicit (Twin 1), order 2: records = (y-cell, stratified cosine direction) -> first body hit, traced once in the
      body frame; per update Sd2(y, wo) = sum_d f(y; wo, d) cos / pdf * L1(x1(d) -> y), L1 by exact ER at x1 from an incident map with
      explicit visibility. Orders >= 3 are NOT included (reported as a known omission; S3 / Sd is reported per workload).
Delivery of photons to camera queries: exact ER at the landing point, box kernel of half-width r on the landing face (area-normalised).
Queries: ROI_body pixels x 4 stratified sub-pixel positions (the reference integrates the pixel).
  python s1b99.py <stage0_dir> <out_dir> <workload_id> [--dev]
"""
import json, math, os, sys, time
import numpy as np
import pt99, wl99
import scene99 as S

mi, dr = pt99.mi, pt99.dr
F, U = mi.Float, mi.UInt32
fl = pt99.fl
RES = wl99.RES


# ------------------------------------------------------------------------------------------------ body frame
class Body:
    """Body-only Mitsuba scene in the body frame (plates as cubes) + plate frames for main-face coordinates."""

    def __init__(self, bd, states):
        self.plates = S.body_plates(bd['kind'], bd['prm'])
        d = {'type': 'scene'}
        for j, p in enumerate(self.plates):
            d[f'p{j}'] = {'type': 'cube', 'to_world': mi.ScalarTransform4f(p['M'].tolist())}
        self.scene = mi.load_dict(d)
        by = {s.id(): mi.ShapePtr(s) for s in self.scene.shapes()}
        self.ptrs = [by[f'p{j}'] for j in range(len(self.plates))]
        self.mat = S.MAT_ID[bd['mat']]
        self.states = states
        self.C = np.array([p['c'] for p in self.plates]); self.Nn = np.array([p['n'] for p in self.plates])
        self.Uu = np.array([p['u'] for p in self.plates]); self.Vv = np.array([p['v'] for p in self.plates])
        self.H = np.array([[p['hu'], p['hv']] for p in self.plates])

    def pid(self, si):
        j = U(0)
        for i, p in enumerate(self.ptrs):
            j = dr.select(dr.eq(si.shape, p), U(i), j)
        return j

    def frame_np(self, j):
        return self.Uu[j], self.Vv[j], self.Nn[j]

    def closure(self, n):
        return pt99.CD.DrClosure83('ER', self.states, U(np.full(n, self.mat, np.uint32)))


def gat(a, idx):
    return dr.gather(F, fl(a), idx)


def plate_frame_dr(body, j):
    """Dr.Jit vectors u, v, n, c of plate j (UInt32 array)."""
    g = lambda A, k: dr.gather(F, fl(A[:, k]), j)
    return [mi.Vector3f(g(A, 0), g(A, 1), g(A, 2)) for A in (body.Uu, body.Vv, body.Nn, body.C)]


def to_local(v, fu, fv, fn):
    return mi.Vector3f(dr.dot(v, fu), dr.dot(v, fv), dr.dot(v, fn))


def to_world(v, fu, fv, fn):
    return fu * v.x + fv * v.y + fn * v.z


def walk(body, j, x, wl, flux, depth, rng, record):
    """Light-side self walk from points x (body frame, on main faces of plates j) with light-side direction wl (body frame, unit,
    pointing away from the surface) and flux [RGB per unit incident flux]. Records self-incident photons on main faces.
    record(order, plate, u, v, w_in_local, phi) receives Dr.Jit arrays (one call per order)."""
    n = dr.width(x)
    active = mi.Bool(True)
    phi = mi.Color3f(flux)
    fu, fv, fn, _ = plate_frame_dr(body, j)
    wl_loc = to_local(wl, fu, fv, fn)
    pos, jj = x, j
    on_main = mi.Bool(True)
    for order in range(2, depth + 1):
        cl = body.closure(n)
        u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
        active &= wl_loc.z > 1e-6
        wo, pdf, w, valid = cl.sample(dr.select(active, wl_loc, mi.Vector3f(0, 0, 1)), u1, u2, active)
        # flux weight: f_r(wo, wl) cos(wo) / pdf(wo | wl);  eval(wi=wo, wo=wl) = f_r cos(wl)
        ev = cl.eval(dr.select(valid, wo, mi.Vector3f(0, 0, 1)), dr.select(valid, wl_loc, mi.Vector3f(0, 0, 1)), valid)
        k = dr.select(valid & (pdf > 0), wo.z * dr.rcp(dr.maximum(pdf, 1e-20)) * dr.rcp(dr.maximum(wl_loc.z, 1e-6)), 0.0)
        phi = phi * ev * k
        active &= valid
        dw = to_world(wo, fu, fv, fn)
        ray = mi.Ray3f(pos + dw * 1e-5, dw)
        si = body.scene.ray_intersect(ray, active)
        active &= si.is_valid()
        jn = body.pid(si)
        fu2, fv2, fn2, c2 = plate_frame_dr(body, jn)
        main = dr.dot(si.n, fn2) > 0.99
        win = to_local(-dw, fu2, fv2, fn2)                       # incoming light-side direction at the landing point
        rel = si.p - c2
        uu, vv = dr.dot(rel, fu2), dr.dot(rel, fv2)
        rec = active & main & (win.z > 1e-6)
        record(order, rec, jn, uu, vv, win, phi)
        # continue (from main and non-main faces alike; non-main faces are not recorded)
        nloc = si.to_local(-dw)
        pos = si.p
        fu, fv, fn = dr.select(main, fu2, si.sh_frame.s), dr.select(main, fv2, si.sh_frame.t), dr.select(main, fn2, si.sh_frame.n)
        wl_loc = to_local(-dw, fu, fv, fn)
        active &= wl_loc.z > 1e-6


class Recorder:
    def __init__(self):
        self.parts = []

    def __call__(self, order, rec, j, u, v, w, phi):
        dr.eval(rec, j, u, v, w, phi)
        m = np.asarray(rec)
        if not m.any():
            return
        idx = np.nonzero(m)[0]
        self.parts.append({'order': np.full(len(idx), order, np.int8), 'lane': idx.astype(np.int64), 'j': np.asarray(j)[idx].astype(np.int8),
                           'u': np.asarray(u)[idx], 'v': np.asarray(v)[idx],
                           'w': np.stack([np.asarray(w.x)[idx], np.asarray(w.y)[idx], np.asarray(w.z)[idx]], 1),
                           'phi': np.stack([np.asarray(phi.x)[idx], np.asarray(phi.y)[idx], np.asarray(phi.z)[idx]], 1)})

    def cat(self):
        if not self.parts:
            return None
        return {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}


def concentric_dir(a, b):
    """Unit square -> cosine-weighted hemisphere (equal projected solid angle); numpy."""
    a = 2 * a - 1; b = 2 * b - 1
    r = np.where(np.abs(a) > np.abs(b), a, b)
    phi = np.where(np.abs(a) > np.abs(b), (np.pi / 4) * (b / np.where(a == 0, 1, a)), np.pi / 2 - (np.pi / 4) * (a / np.where(b == 0, 1, b)))
    x, y = r * np.cos(phi), r * np.sin(phi)
    return np.stack([x, y, np.sqrt(np.maximum(1 - x * x - y * y, 0))], -1)


def dir_cell(wloc, nd):
    """Local unit direction (z up) -> projected-solid-angle cell index in an nd x nd concentric grid."""
    x, y = wloc[:, 0], wloc[:, 1]
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    phi = np.where(phi < -np.pi / 4, phi + 2 * np.pi, phi)
    a = np.zeros_like(r); b = np.zeros_like(r)
    s1 = phi < np.pi / 4; s2 = (phi >= np.pi / 4) & (phi < 3 * np.pi / 4); s3 = (phi >= 3 * np.pi / 4) & (phi < 5 * np.pi / 4); s4 = phi >= 5 * np.pi / 4
    a[s1] = r[s1]; b[s1] = phi[s1] * 4 / np.pi * r[s1]
    b[s2] = r[s2]; a[s2] = -(phi[s2] - np.pi / 2) * 4 / np.pi * r[s2]
    a[s3] = -r[s3]; b[s3] = -(phi[s3] - np.pi) * 4 / np.pi * r[s3]
    b[s4] = -r[s4]; a[s4] = (phi[s4] - 3 * np.pi / 2) * 4 / np.pi * r[s4]
    ia = np.clip(((a + 1) / 2 * nd).astype(int), 0, nd - 1); ib = np.clip(((b + 1) / 2 * nd).astype(int), 0, nd - 1)
    return ia * nd + ib


# ------------------------------------------------------------------------------------------------ TwinP precompute (E2 / E3)
def twinp_build(body, hx, nd, ne, depth, seed, chunk=4_000_000):
    """Returns table dict: cell layout + photons sorted by cell (CSR) + per-cell offset Jacobians; timings."""
    t0 = time.perf_counter()
    cells = []
    for j, p in enumerate(body.plates):
        nu, nv = max(1, int(round(2 * p['hu'] / hx))), max(1, int(round(2 * p['hv'] / hx)))
        cells.append((nu, nv))
    base = np.concatenate([[0], np.cumsum([nu * nv * nd * nd for nu, nv in cells])])
    ncell = int(base[-1])
    rec_all = []
    lane_cell, lane_off = [], []
    rng_np = np.random.default_rng(seed)
    total_lanes = ncell * ne
    for start in range(0, total_lanes, chunk):
        lanes = np.arange(start, min(total_lanes, start + chunk))
        cell = lanes // ne
        j = np.searchsorted(base, cell, side='right') - 1
        loc = cell - base[j]
        nu = np.array([cells[q][0] for q in j]); nv = np.array([cells[q][1] for q in j])
        sp, dc = loc // (nd * nd), loc % (nd * nd)
        iu, iv = sp % nu, sp // nu
        ia, ib = dc // nd, dc % nd
        ju, jv, ja, jb = (rng_np.random(len(lanes)) for _ in range(4))
        hu, hv = body.H[j, 0], body.H[j, 1]
        cu = -hu + (iu + ju) * (2 * hu / nu); cv = -hv + (iv + jv) * (2 * hv / nv)
        wloc = concentric_dir((ia + ja) / nd, (ib + jb) / nd)
        x = body.C[j] + cu[:, None] * body.Uu[j] + cv[:, None] * body.Vv[j]
        wl = wloc[:, :1] * body.Uu[j] + wloc[:, 1:2] * body.Vv[j] + wloc[:, 2:3] * body.Nn[j]
        # in-cell offsets relative to the cell centre (for the J variant): du, dv (m), da, db (cell-normalised direction offset)
        off = np.stack([(ju - 0.5) * (2 * hu / nu), (jv - 0.5) * (2 * hv / nv), ja - 0.5, jb - 0.5], 1)
        rec = Recorder()
        rng = mi.PCG32(size=len(lanes), initstate=mi.UInt64(seed + start), initseq=mi.UInt64(1))
        walk(body, U(j.astype(np.uint32)), mi.Point3f(*[fl(x[:, k]) for k in range(3)]), mi.Vector3f(*[fl(wl[:, k]) for k in range(3)]),
             mi.Color3f(1.0 / ne), depth, rng, rec)
        ph = rec.cat()
        if ph is not None:
            ph['cell'] = cell[ph['lane'] - start]
            ph['off'] = off[ph['lane'] - start]
            del ph['lane']
            rec_all.append(ph)
    ph = {k: np.concatenate([p[k] for p in rec_all]) for k in rec_all[0]}
    order = np.argsort(ph['cell'], kind='stable')
    ph = {k: v[order] for k, v in ph.items()}
    ptr = np.searchsorted(ph['cell'], np.arange(ncell + 1))
    # per-cell first-order landing Jacobian (least squares of (u, v) on the 4 offsets), only for cells with >= 8 photons on one plate
    J = np.zeros((ncell, 2, 4), np.float32)
    cnt = np.diff(ptr)
    # batched weighted normal equations per cell: A = [off(4), 1], targets (u, v); only cells with >= 8 photons all on one plate
    A = np.concatenate([ph['off'], np.ones((len(ph['u']), 1))], 1)
    wts = np.maximum(ph['phi'].mean(1), 0).astype(np.float64)
    c = ph['cell']
    AtA = np.zeros((ncell, 5, 5)); AtY = np.zeros((ncell, 5, 2))
    for a in range(5):
        for b in range(a, 5):
            s_ = np.bincount(c, wts * A[:, a] * A[:, b], minlength=ncell)
            AtA[:, a, b] = s_; AtA[:, b, a] = s_
        AtY[:, a, 0] = np.bincount(c, wts * A[:, a] * ph['u'], minlength=ncell)
        AtY[:, a, 1] = np.bincount(c, wts * A[:, a] * ph['v'], minlength=ncell)
    jmin = np.full(ncell, 127); jmax = np.full(ncell, -1)
    np.minimum.at(jmin, c, ph['j'].astype(int)); np.maximum.at(jmax, c, ph['j'].astype(int))
    good = np.nonzero((cnt >= 8) & (jmin == jmax))[0]
    if len(good):
        M = AtA[good] + 1e-12 * np.eye(5)[None]
        ok = np.linalg.cond(M) < 1e10
        sol = np.zeros((len(good), 5, 2))
        sol[ok] = np.linalg.solve(M[ok], AtY[good][ok])
        J[good] = np.transpose(sol[:, :4, :], (0, 2, 1))
    t_build = time.perf_counter() - t0
    mem_photon = 1 + 2 * 2 + 2 * 2 + 3 * 2                              # plate id, uv fp16, octahedral direction fp16x2, rgb fp16 -> 15 B
    table = {'hx': hx, 'nd': nd, 'ne': ne, 'cells': cells, 'base': base, 'ptr': ptr, 'ph': ph, 'J': J,
             'n_photons': int(len(ph['u'])), 'n_cells': ncell,
             'mem_bytes': int(len(ph['u']) * mem_photon + (ncell + 1) * 4),
             'mem_bytes_J': int(len(ph['u']) * mem_photon + (ncell + 1) * 4 + ncell * 8 * 2),
             'build_s': t_build}
    return table


# ------------------------------------------------------------------------------------------------ per-update incident samples
def incident_samples(ctx, body, pose, hs):
    """Stratified incident samples on every main face (body frame) with explicit NEE (world scene, includes self-shadowing and
    occluders). Returns plate j, uv, light-side local direction, flux RGB (W), body-frame point; one sample per (point, light)."""
    Mp = S.pose_matrix(pose)
    R, t = Mp[:3, :3], Mp[:3, 3]
    out = []
    for jp, p in enumerate(body.plates):
        nu, nv = max(1, int(round(2 * p['hu'] / hs))), max(1, int(round(2 * p['hv'] / hs)))
        uu = -p['hu'] + (np.arange(nu) + 0.5) * (2 * p['hu'] / nu); vv = -p['hv'] + (np.arange(nv) + 0.5) * (2 * p['hv'] / nv)
        U_, V_ = np.meshgrid(uu, vv, indexing='ij')
        U_, V_ = U_.ravel(), V_.ravel()
        xb = p['c'] + U_[:, None] * p['u'] + V_[:, None] * p['v']
        xw = xb @ R.T + t
        nw = R @ p['n']
        dA = (2 * p['hu'] / nu) * (2 * p['hv'] / nv)
        for l in ctx.cfg['lights']:
            lp = np.asarray(l['pos'], float)
            d = lp - xw; r2 = (d ** 2).sum(1); dn = d / np.sqrt(r2)[:, None]
            cos = dn @ nw
            o = xw + nw[None] * 1e-4
            ray = mi.Ray3f(mi.Point3f(*[fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[fl(dn[:, k]) for k in range(3)]))
            ray.maxt = fl(np.sqrt(r2) * (1 - 1e-4))
            occ = np.asarray(ctx.scene.ray_test(ray))
            vis = (cos > 1e-6) & ~occ
            E = np.asarray(l['rgb'], float)[None] * (cos * vis / r2)[:, None]
            wb = dn @ R                                                      # body-frame direction
            wloc = np.stack([wb @ p['u'], wb @ p['v'], wb @ p['n']], 1)
            keep = vis
            out.append({'j': np.full(keep.sum(), jp), 'u': U_[keep], 'v': V_[keep], 'w': wloc[keep], 'phi': E[keep] * dA, 'xb': xb[keep]})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def apply_twinp(table, body, inc, variant):
    """Photons of the incident samples' cells, scaled by the samples' flux; variant 'nn' (nearest cell) or 'J' (first-order shift)."""
    cells, base, nd = table['cells'], table['base'], table['nd']
    j = inc['j']
    nu = np.array([cells[q][0] for q in j]); nv = np.array([cells[q][1] for q in j])
    hu, hv = body.H[j, 0], body.H[j, 1]
    fu = (inc['u'] + hu) / (2 * hu) * nu; fv = (inc['v'] + hv) / (2 * hv) * nv
    iu = np.clip(fu.astype(int), 0, nu - 1); iv = np.clip(fv.astype(int), 0, nv - 1)
    dc = np.zeros(len(j), int)
    for jp in np.unique(j):
        m = j == jp
        dc[m] = dir_cell(inc['w'][m], nd)
    cell = base[j] + (iv * nu + iu) * nd * nd + dc
    ptr, ph = table['ptr'], table['ph']
    n0, n1 = ptr[cell], ptr[cell + 1]
    cnt = n1 - n0
    tot = int(cnt.sum())
    rep = np.repeat(np.arange(len(j)), cnt)
    pidx = np.repeat(n0, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    u, v = ph['u'][pidx].astype(np.float64), ph['v'][pidx].astype(np.float64)
    if variant == 'J':
        # in-cell offset of the actual sample: spatial (m) and direction (cell-normalised) relative to the cell centre
        du = (fu - iu - 0.5) * (2 * hu / nu); dv = (fv - iv - 0.5) * (2 * hv / nv)
        # direction offset: recompute fractional concentric coordinates
        a_b = np.zeros((len(j), 2))
        for jp in np.unique(j):
            m = j == jp
            a_b[m] = frac_concentric(inc['w'][m], nd)
        off = np.stack([du, dv, a_b[:, 0], a_b[:, 1]], 1)
        Jc = table['J'][cell[rep]]
        sh = np.einsum('nij,nj->ni', Jc, off[rep])
        u = u + sh[:, 0]; v = v + sh[:, 1]
    return {'j': ph['j'][pidx], 'u': u, 'v': v, 'w': ph['w'][pidx], 'phi': ph['phi'][pidx] * inc['phi'][rep], 'order': ph['order'][pidx]}


def frac_concentric(wloc, nd):
    """Fractional in-cell offset (-0.5..0.5) of directions in the nd x nd concentric grid."""
    x, y = wloc[:, 0], wloc[:, 1]
    r = np.sqrt(x * x + y * y); phi = np.arctan2(y, x)
    phi = np.where(phi < -np.pi / 4, phi + 2 * np.pi, phi)
    a = np.zeros_like(r); b = np.zeros_like(r)
    s1 = phi < np.pi / 4; s2 = (phi >= np.pi / 4) & (phi < 3 * np.pi / 4); s3 = (phi >= 3 * np.pi / 4) & (phi < 5 * np.pi / 4); s4 = phi >= 5 * np.pi / 4
    a[s1] = r[s1]; b[s1] = phi[s1] * 4 / np.pi * r[s1]
    b[s2] = r[s2]; a[s2] = -(phi[s2] - np.pi / 2) * 4 / np.pi * r[s2]
    a[s3] = -r[s3]; b[s3] = -(phi[s3] - np.pi) * 4 / np.pi * r[s3]
    b[s4] = -r[s4]; a[s4] = (phi[s4] - 3 * np.pi / 2) * 4 / np.pi * r[s4]
    fa = (a + 1) / 2 * nd; fb = (b + 1) / 2 * nd
    return np.stack([fa - np.floor(fa) - 0.5, fb - np.floor(fb) - 0.5], 1)


def dense_photons(body, inc, ne, depth, seed):
    """E4: per-update light-side tracing of ne paths from every exact incident sample (no cells)."""
    n = len(inc['j'])
    rep = np.repeat(np.arange(n), ne)
    j = inc['j'][rep]
    x = inc['xb'][rep]
    wl = inc['w'][rep]
    wlb = wl[:, :1] * body.Uu[j] + wl[:, 1:2] * body.Vv[j] + wl[:, 2:3] * body.Nn[j]
    rec = Recorder()
    rng = mi.PCG32(size=len(rep), initstate=mi.UInt64(seed), initseq=mi.UInt64(3))
    phi0 = inc['phi'][rep] / ne
    walk(body, U(j.astype(np.uint32)), mi.Point3f(*[fl(x[:, k]) for k in range(3)]), mi.Vector3f(*[fl(wlb[:, k]) for k in range(3)]),
         mi.Color3f(*[fl(phi0[:, k]) for k in range(3)]), depth, rng, rec)
    ph = rec.cat()
    return {'j': ph['j'], 'u': ph['u'].astype(np.float64), 'v': ph['v'].astype(np.float64), 'w': ph['w'], 'phi': ph['phi'], 'order': ph['order']}


# ------------------------------------------------------------------------------------------------ queries and delivery
def queries(ctx, body, cfg, roi, nsub=4, seed=7):
    """ROI pixels x nsub stratified sub-pixel camera rays -> hits on body main faces: pixel id, plate, uv, local wo."""
    cam = cfg['camera']
    o, fwd, right, up, s = pt99.cam_basis(cam)
    pix = np.nonzero(roi.ravel())[0]
    rng = np.random.default_rng(seed)
    q = int(math.sqrt(nsub))
    rows = []
    Mp = S.pose_matrix(cfg['body']['pose']); R, t = Mp[:3, :3], Mp[:3, 3]
    for a in range(q):
        for b in range(q):
            jx = (a + rng.random(len(pix))) / q; jy = (b + rng.random(len(pix))) / q
            ii, jj = pix // RES, pix % RES
            x = (jj + jx) * (2.0 / RES) - 1.0; yv = 1.0 - (ii + jy) * (2.0 / RES)
            d = fwd[None] + s * x[:, None] * right[None] + s * yv[:, None] * up[None]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            ray = mi.Ray3f(mi.Point3f(*[fl(np.full(len(pix), o[k])) for k in range(3)]), mi.Vector3f(*[fl(d[:, k]) for k in range(3)]))
            si = ctx.scene.ray_intersect(ray)
            sid = np.asarray(ctx.shape_id(si)); ok = np.asarray(si.is_valid())
            p = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], 1)
            pb = (p - t) @ R
            db = (-d) @ R
            plate = sid - 5                                                  # body slots follow the 5 room rectangles
            okb = ok & (plate >= 0) & (plate < len(body.plates))
            plate = np.clip(plate, 0, len(body.plates) - 1)
            nb = body.Nn[plate]
            main = okb & (np.abs(((pb - body.C[plate]) * nb).sum(1)) < 2e-4)
            uu = ((pb - body.C[plate]) * body.Uu[plate]).sum(1); vv = ((pb - body.C[plate]) * body.Vv[plate]).sum(1)
            wo = np.stack([(db * body.Uu[plate]).sum(1), (db * body.Vv[plate]).sum(1), (db * nb).sum(1)], 1)
            rows.append({'pix': pix, 'j': plate, 'u': uu, 'v': vv, 'wo': wo, 'main': main})
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, nsub


def deliver(body, ph, qr, r, chunk=40_000_000):
    """Sd at queries: sum over photons on the same plate within the box |du|,|dv| <= r of f_r(y; wo, w_in) phi / A_box(y)."""
    out = np.zeros((len(qr['u']), 3))
    t_pairs = 0
    for jp in range(len(body.plates)):
        mq = np.nonzero(qr['main'] & (qr['j'] == jp))[0]
        mp = np.nonzero(ph['j'] == jp)[0]
        if len(mq) == 0 or len(mp) == 0:
            continue
        hu, hv = body.H[jp]
        gx = np.floor((ph['u'][mp] + hu) / r).astype(np.int64); gy = np.floor((ph['v'][mp] + hv) / r).astype(np.int64)
        ny = int(np.ceil(2 * hv / r)) + 2
        key = gx * ny + gy
        o = np.argsort(key, kind='stable'); key_s = key[o]; mp_s = mp[o]
        qx = np.floor((qr['u'][mq] + hu) / r).astype(np.int64); qy = np.floor((qr['v'][mq] + hv) / r).astype(np.int64)
        qi, pi = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (qx + dx) * ny + (qy + dy)
                lo = np.searchsorted(key_s, k, 'left'); hi = np.searchsorted(key_s, k, 'right')
                c = hi - lo
                qi.append(np.repeat(mq, c))
                pi.append(mp_s[np.repeat(lo, c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))])
        qi = np.concatenate(qi); pi = np.concatenate(pi)
        m = (np.abs(ph['u'][pi] - qr['u'][qi]) <= r) & (np.abs(ph['v'][pi] - qr['v'][qi]) <= r)
        qi, pi = qi[m], pi[m]
        t_pairs += len(qi)
        uq, vq = qr['u'][mq], qr['v'][mq]
        area = (np.minimum(uq + r, hu) - np.maximum(uq - r, -hu)) * (np.minimum(vq + r, hv) - np.maximum(vq - r, -hv))
        inv_area = np.zeros(len(out)); inv_area[mq] = 1.0 / np.maximum(area, 1e-12)
        for s0 in range(0, len(qi), chunk):
            a, b = qi[s0:s0 + chunk], pi[s0:s0 + chunk]
            n = len(a)
            cl = body.closure(n)
            wo = qr['wo'][a]; wi = ph['w'][b]
            wov = mi.Vector3f(*[fl(wo[:, k]) for k in range(3)]); wiv = mi.Vector3f(*[fl(wi[:, k]) for k in range(3)])
            ok = (wov.z > 1e-6) & (wiv.z > 1e-6)
            ev = cl.eval(wov, wiv, ok)                                       # f_r(wo, w_in) cos(w_in)
            ev = ev * dr.rcp(dr.maximum(wiv.z, 1e-6))
            dr.eval(ev)
            f = np.stack([np.asarray(ev.x), np.asarray(ev.y), np.asarray(ev.z)], 1)
            contrib = f * ph['phi'][b] * inv_area[a][:, None]
            for c in range(3):
                out[:, c] += np.bincount(a, contrib[:, c], minlength=len(out))
    return out, t_pairs


def to_image(qr, vals, nsub):
    img = np.zeros((RES * RES, 3)); cnt = np.zeros(RES * RES)
    np.add.at(img, qr['pix'], vals); np.add.at(cnt, qr['pix'], 1)
    img /= np.maximum(cnt, 1)[:, None]
    return img.reshape(RES, RES, 3)


# ------------------------------------------------------------------------------------------------ E1c (order 2)
def e1c_build(body, hy, nd2, seed):
    """Records per main-face y-cell: nd2 stratified cosine directions from the cell centre -> first body hit (plate, uv, local w at hit)."""
    t0 = time.perf_counter()
    ys, js, cells = [], [], []
    for jp, p in enumerate(body.plates):
        nu, nv = max(1, int(round(2 * p['hu'] / hy))), max(1, int(round(2 * p['hv'] / hy)))
        cells.append((nu, nv))
    base = np.concatenate([[0], np.cumsum([a * b for a, b in cells])])
    q = int(round(math.sqrt(nd2)))
    a, b = np.meshgrid((np.arange(q) + 0.5) / q, (np.arange(q) + 0.5) / q, indexing='ij')
    dloc = concentric_dir(a.ravel(), b.ravel())                              # pdf = cos / pi
    recs = []
    for jp, p in enumerate(body.plates):
        nu, nv = cells[jp]
        uu = -p['hu'] + (np.arange(nu) + 0.5) * (2 * p['hu'] / nu); vv = -p['hv'] + (np.arange(nv) + 0.5) * (2 * p['hv'] / nv)
        V_, U_ = np.meshgrid(vv, uu, indexing='ij')
        y = p['c'] + U_.ravel()[:, None] * p['u'] + V_.ravel()[:, None] * p['v']
        ny = len(y)
        dw = dloc[:, :1] * p['u'] + dloc[:, 1:2] * p['v'] + dloc[:, 2:3] * p['n']
        Y = np.repeat(y, len(dloc), 0); D = np.tile(dw, (ny, 1))
        ray = mi.Ray3f(mi.Point3f(*[fl(Y[:, k] + 1e-5 * D[:, k]) for k in range(3)]), mi.Vector3f(*[fl(D[:, k]) for k in range(3)]))
        si = body.scene.ray_intersect(ray)
        hit = np.asarray(si.is_valid())
        jn = np.asarray(body.pid(si))
        P_ = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], 1)
        main = hit & (np.abs(((P_ - body.C[jn]) * body.Nn[jn]).sum(1)) < 2e-4)
        win = -D                                                             # direction at the hit towards y (light-side "wo" at x1 is
        loc_w = np.stack([(win * body.Uu[jn]).sum(1), (win * body.Vv[jn]).sum(1), (win * body.Nn[jn]).sum(1)], 1)   # toward y)
        recs.append({'hit': main, 'j1': jn, 'u1': ((P_ - body.C[jn]) * body.Uu[jn]).sum(1), 'v1': ((P_ - body.C[jn]) * body.Vv[jn]).sum(1),
                     'wy': np.tile(dloc, (ny, 1)), 'w1': loc_w, 'x1b': P_})
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    return {'hy': hy, 'nd2': nd2, 'cells': cells, 'base': base, 'rec': rec, 'build_s': time.perf_counter() - t0,
            'mem_bytes': int(len(rec['hit']) * (1 + 2 * 2 + 2 * 2))}


def e1c_apply(ctx, body, tab, qr, pose, chunk=30_000_000):
    """Per update: exact NEE at every record hit (world scene, explicit visibility), exact ER at x1 and at y; order 2 only."""
    rec = tab['rec']; nd2 = tab['nd2']
    Mp = S.pose_matrix(pose); R, t = Mp[:3, :3], Mp[:3, 3]
    hit = np.nonzero(rec['hit'])[0]
    x1w = rec['x1b'][hit] @ R.T + t
    L1 = np.zeros((len(rec['hit']), 3))
    for l in ctx.cfg['lights']:
        lp = np.asarray(l['pos'], float)
        d = lp - x1w; r2 = (d ** 2).sum(1); dn = d / np.sqrt(r2)[:, None]
        jn = rec['j1'][hit]
        nw = body.Nn[jn] @ R.T
        cos = (dn * nw).sum(1)
        o = x1w + nw * 1e-4
        ray = mi.Ray3f(mi.Point3f(*[fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[fl(dn[:, k]) for k in range(3)]))
        ray.maxt = fl(np.sqrt(r2) * (1 - 1e-4))
        vis = (cos > 1e-6) & ~np.asarray(ctx.scene.ray_test(ray))
        db = dn @ R
        wl = np.stack([(db * body.Uu[jn]).sum(1), (db * body.Vv[jn]).sum(1), (db * body.Nn[jn]).sum(1)], 1)
        cl = body.closure(len(hit))
        wy = rec['w1'][hit]
        ev = cl.eval(mi.Vector3f(*[fl(wy[:, k]) for k in range(3)]), mi.Vector3f(*[fl(wl[:, k]) for k in range(3)]), mi.Bool(vis))
        dr.eval(ev)
        f = np.stack([np.asarray(ev.x), np.asarray(ev.y), np.asarray(ev.z)], 1)        # f_r(w_y, wl) cos(wl)
        L1[hit] += f * (np.asarray(l['rgb'], float)[None] * (vis / r2)[:, None])
    # gather at queries: records of the query's y-cell
    cells, base = tab['cells'], tab['base']
    j = qr['j']; ok = qr['main']
    nu = np.array([cells[q][0] for q in j]); nv = np.array([cells[q][1] for q in j])
    hu, hv = body.H[j, 0], body.H[j, 1]
    iu = np.clip(((qr['u'] + hu) / (2 * hu) * nu).astype(int), 0, nu - 1); iv = np.clip(((qr['v'] + hv) / (2 * hv) * nv).astype(int), 0, nv - 1)
    cell = base[j] + iv * nu + iu
    out = np.zeros((len(j), 3))
    qidx = np.nonzero(ok)[0]
    for s0 in range(0, len(qidx), max(1, chunk // nd2)):
        qq = qidx[s0:s0 + max(1, chunk // nd2)]
        rr = (cell[qq][:, None] * nd2 + np.arange(nd2)[None]).ravel()
        a = np.repeat(qq, nd2)
        wo = qr['wo'][a]; wy = rec['wy'][rr]
        cl = body.closure(len(a))
        ev = cl.eval(mi.Vector3f(*[fl(wo[:, k]) for k in range(3)]), mi.Vector3f(*[fl(wy[:, k]) for k in range(3)]),
                     mi.Bool(wo[:, 2] > 1e-6))
        dr.eval(ev)
        f = np.stack([np.asarray(ev.x), np.asarray(ev.y), np.asarray(ev.z)], 1)       # f_r(wo, d) cos(d)
        contrib = f * (np.pi / np.maximum(wy[:, 2:3], 1e-6)) / nd2 * L1[rr]            # / pdf(d) = pi / cos(d)
        for c in range(3):
            out[:, c] += np.bincount(a, contrib[:, c], minlength=len(out))
    return out


# ------------------------------------------------------------------------------------------------ driver
def metrics(img, R, m):
    body, roi_self, sharp = m
    sd = R['Sd']
    return {'rel_rmse_self': wl99.rel_l2(img, sd, roi_self), 'rel_rmse_sharp': wl99.rel_l2(img, sd, sharp),
            'ratio_sum_self': float(img[roi_self].sum() / max(sd[roi_self].sum(), 1e-300))}


def main():
    s0, out, wid = sys.argv[1], sys.argv[2], sys.argv[3]
    DEV = '--dev' in sys.argv
    os.makedirs(out, exist_ok=True)
    q = json.load(open(os.path.join(s0, 'qual99.json')))['workloads'][wid]
    w = {x['id']: x for x in wl99.WL}[wid]
    cf = wl99.configs(w)
    ctx = pt99.Ctx(cf['T0'])
    body = Body(cf['T0']['body'], ctx.states)
    depth = 8
    ladder = [(0.02, 8, 16), (0.02, 16, 16), (0.01, 16, 16), (0.01, 16, 64), (0.005, 16, 64), (0.01, 32, 64)] if not DEV else [(0.04, 8, 4)]
    res = {'workload': wid, 'arms': []}
    tables = {}
    for hx, nd, ne in ladder:
        t = twinp_build(body, hx, nd, ne, depth, seed=992001)
        tables[(hx, nd, ne)] = t
        res.setdefault('builds', []).append({'hx': hx, 'nd': nd, 'ne': ne, 'n_cells': t['n_cells'], 'n_photons': t['n_photons'],
                                             'mem_MB': t['mem_bytes'] / 2 ** 20, 'mem_MB_J': t['mem_bytes_J'] / 2 ** 20, 'build_s': t['build_s']})
        json.dump(res, open(os.path.join(out, f's1b_{wid}.json'), 'w'), indent=1)
    e1c = [e1c_build(body, hy, nd2, 992101) for hy, nd2 in ([(0.005, 1024), (0.005, 4096)] if not DEV else [(0.02, 64)])]
    for state in ('T0', 'T1'):
        cfg = cf[state]
        ctx.set_cfg(cfg)
        spp = q['spp_per_half']
        A = dict(np.load(os.path.join(s0, 'refs', f'{wid}_{state}_A_{spp}.npz'))); B = dict(np.load(os.path.join(s0, 'refs', f'{wid}_{state}_B_{spp}.npz')))
        R = {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}
        m = wl99.masks(R)
        qr, nsub = queries(ctx, body, cfg, m[0])
        for hs in ([0.0025] if not DEV else [0.01]):
            t0 = time.perf_counter(); inc = incident_samples(ctx, body, cfg['body']['pose'], hs); t_inc = time.perf_counter() - t0
            for key, tab in tables.items():
                for variant in ('nn', 'J'):
                    for r in ((0.002, 0.004, 0.008) if not DEV else (0.01,)):
                        t1 = time.perf_counter()
                        ph = apply_twinp(tab, body, inc, variant)
                        vals, npairs = deliver(body, ph, qr, r)
                        t_ap = time.perf_counter() - t1
                        img = to_image(qr, vals, nsub)
                        row = {'arm': 'E2/E3 TwinP', 'state': state, 'hx': key[0], 'nd': key[1], 'ne': key[2], 'variant': variant, 'r': r, 'hs': hs,
                               'mem_MB': (tab['mem_bytes_J'] if variant == 'J' else tab['mem_bytes']) / 2 ** 20,
                               'apply_wall_s': t_ap + t_inc, 'n_photons_applied': int(len(ph['u'])), 'pairs': int(npairs), **metrics(img, R, m)}
                        res['arms'].append(row)
                        print(json.dumps(row), flush=True)
                        np.save(os.path.join(out, f"img_{wid}_{state}_twinp_{key[0]}_{key[1]}_{key[2]}_{variant}_{r}.npy"), img.astype(np.float32))
                json.dump(res, open(os.path.join(out, f's1b_{wid}.json'), 'w'), indent=1)
            for ne in ((16, 64) if not DEV else (4,)):
                for r in ((0.002, 0.004, 0.008) if not DEV else (0.01,)):
                    t1 = time.perf_counter()
                    ph = dense_photons(body, inc, ne, depth, 992201 + ne)
                    vals, npairs = deliver(body, ph, qr, r)
                    t_ap = time.perf_counter() - t1
                    img = to_image(qr, vals, nsub)
                    row = {'arm': 'E4 dense-limit (per-update light tracing)', 'state': state, 'ne': ne, 'r': r, 'hs': hs, 'mem_MB': 0.0,
                           'apply_wall_s': t_ap + t_inc, 'n_photons_applied': int(len(ph['u'])), 'pairs': int(npairs), **metrics(img, R, m)}
                    res['arms'].append(row); print(json.dumps(row), flush=True)
                    np.save(os.path.join(out, f"img_{wid}_{state}_e4_{ne}_{r}.npy"), img.astype(np.float32))
        for tab in e1c:
            t1 = time.perf_counter()
            vals = e1c_apply(ctx, body, tab, qr, cfg['body']['pose'])
            t_ap = time.perf_counter() - t1
            img = to_image(qr, vals, nsub)
            R2 = dict(R); R2['Sd'] = R['S2']
            row = {'arm': 'E1c cached self-relation (order 2)', 'state': state, 'hy': tab['hy'], 'nd2': tab['nd2'], 'mem_MB': tab['mem_bytes'] / 2 ** 20,
                   'build_s': tab['build_s'], 'apply_wall_s': t_ap, **metrics(img, R, m), 'vs_S2_only': metrics(img, R2, m)}
            res['arms'].append(row); print(json.dumps(row), flush=True)
            np.save(os.path.join(out, f"img_{wid}_{state}_e1c_{tab['hy']}_{tab['nd2']}.npy"), img.astype(np.float32))
        json.dump(res, open(os.path.join(out, f's1b_{wid}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
