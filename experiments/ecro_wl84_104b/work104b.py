"""WL104B worker for one WL84 image (= one material). Run one process per image in parallel (chain104b.py).
  <venv-cu128 python> work104b.py <image> <raw_root>
Stages (each resumable by its output file):
  contracts : C0 state hashes; C5a WL83 per-lane metric reproduction; C3 torch vs Dr.Jit closure and ER reciprocity; S_target WL83 fit targets
              vs a new high-walk layered reference at the same pairs
  response  : WL83 eval set (8 theta x 2 phi x 1024 mixture directions, WL83 high-precision layered reference) metrics per lane for ER_L and
              ER_DF (el1, ealb, eshape, signed albedo ratio); specular polar grids and hemisphere atlases (ER deterministic, LAY walks)
  centre    : centre-ray first-hit geometry; render contract (tracer E1 vs closure / layered prediction); per-pixel predicted E1 and its
              energy / shape split (per-pixel albedo by mixture MC with common samples)
  images    : R replicates x SPP of LAY / ER_L / ER_DF with the pt102 tracer (D = 6; e1, e2, h), shared seeds; ER_DF Mitsuba depth-7 halves (C2)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import h104b as H

C, RT, P, mi, dr, R81, DA, CL, E82 = H.C, H.RT, H.P, H.mi, H.dr, H.R81, H.DA, H.CL, H.E82
image = sys.argv[1]
ROOT = Path(sys.argv[2]).resolve()
OUT = ROOT / image; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get('WL104B_SMOKE') == '1'
PR = json.loads((H.HERE / 'protocol' / 'wl104b.json').read_text(encoding='utf-8'))
S = PR['sampling']
if SMOKE:
    S = {**S, 'R': 4, 'spp': 256, 'grid_walks': 64, 'polar_walks': 128, 'target_walks': 256, 'pixel_walks': 512, 'albedo_samples': 64,
         'albedo_walks': 8, 'lay_centre_passes': 8, 'mitsuba_spp_half': 256, 'evalset_walks_d64': 64}
tag, light, st, k = C.image_spec(image)
ii = C.IMAGES.index(image)
mat = H.material(tag); H.register(tag, mat)
desc = mat['desc']
log = lambda *a: print(time.strftime('%H:%M:%S'), image, *a, flush=True)
f32 = lambda a: np.asarray(a, np.float32)


def save_json(name, d):
    C.write_json(OUT / name, d)


# ======================================================================================================== contracts
if not (OUT / 'contracts.json').exists():
    res = {'C0': {'ER_L_sha256': mat['ER_L']['sha256'], 'ER_DF_sha256': mat['ER_DF']['sha256'],
                  'ER_L_matches_WL84_audit': mat['ER_L']['sha256'] == C.STATE_SHA[tag], 'ER_DF_matches_WL84_audit': mat['ER_DF']['sha256'] == H.DF_STATE_SHA[tag]}}
    ev = np.load(H.IN63B / 'data' / f'eval83_{st}.npz')
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    sel = ev['m'] == k
    wi, wo = ev['wi'][sel].astype(np.float64), ev['wo'][sel].astype(np.float64)
    f = ev['f'][sel].astype(np.float64); fA = ev['fA'][sel].astype(np.float64); fB = ev['fB'][sel].astype(np.float64)
    w = (ev['use'][sel].astype(np.float64) / ev['q'][sel].astype(np.float64))
    G = nth * nph
    T = lambda a: torch.as_tensor(a, device=H.dev, dtype=torch.float64)
    lanes = np.load(H.IN63B / 'eval' / f'lanes83_{st}.npz')
    resp = {'theta_deg': DA.A['eval_theta_deg'], 'n_phi': nph, 'nwo': nwo}
    preds = {}
    for arm, key in (('ER_L', 'L_ER_s8301'), ('ER_DF', 'DF_ER')):
        pr_ = H.er_eval_torch(mat[arm]['p32'], ev['wi'][sel], ev['wo'][sel]); preds[arm] = pr_      # float32 as eval83
        gm, a_cl, a_ref = E82.group_metrics(T(pr_).view(G, nwo, 3), T(f).view(G, nwo, 3), T(w).view(G, nwo, 1), torch.ones(G, 3, dtype=torch.float64, device=H.dev))
        rep = {m_: float(np.abs(gm[m_] - lanes[f'{key}__{m_}'][k].reshape(-1)).max()) for m_ in ('el1', 'ealb', 'eshape')}
        res.setdefault('C5a_wl83_lane_reproduction_max_abs', {})[arm] = rep
        resp[arm] = {m_: gm[m_].reshape(nth, nph).tolist() for m_ in ('el1', 'ealb', 'eshape')}
        al, ar = a_cl @ H.LUMA, a_ref @ H.LUMA
        resp[arm]['albedo_ratio_luma'] = (al / ar).reshape(nth, nph).tolist()
        resp[arm]['albedo_ratio_rgb'] = (a_cl / a_ref).reshape(nth, nph, 3).tolist()
        resp[arm]['albedo_ref_luma'] = ar.reshape(nth, nph).tolist()
    nz, aA, aB = E82.group_metrics(T(fA).view(G, nwo, 3), T(fB).view(G, nwo, 3), T(w).view(G, nwo, 1), torch.ones(G, 3, dtype=torch.float64, device=H.dev))
    resp['noise_half'] = {m_: (nz[m_] / 2).reshape(nth, nph).tolist() for m_ in ('el1', 'ealb', 'eshape')}
    resp['noise_albedo_ratio_luma_half'] = (0.5 * np.abs((aA @ H.LUMA) / (aB @ H.LUMA) - 1)).reshape(nth, nph).tolist()
    # walk-depth control: the same directions with the in-scene plugin's depth 64 (vs the WL83 reference at depth 1024)
    gA, gB = H.lay_eval(desc, wi, wo, S['evalset_walks_d64'], 1042500 + ii, max_depth=64)
    g64 = 0.5 * (gA + gB)
    ONES = torch.ones(G, 3, dtype=torch.float64, device=H.dev)
    gm, a64, aref = E82.group_metrics(T(g64).view(G, nwo, 3), T(f).view(G, nwo, 3), T(w).view(G, nwo, 1), ONES)
    resp['LAY64_vs_LAY1024'] = {m_: gm[m_].reshape(nth, nph).tolist() for m_ in ('el1', 'ealb', 'eshape')}
    resp['LAY64_vs_LAY1024']['albedo_ratio_luma'] = ((a64 @ H.LUMA) / (aref @ H.LUMA)).reshape(nth, nph).tolist()
    nz64, b64A, b64B = E82.group_metrics(T(gA).view(G, nwo, 3), T(gB).view(G, nwo, 3), T(w).view(G, nwo, 1), ONES)
    resp['LAY64_noise_albedo_ratio_luma_half'] = (0.5 * np.abs((b64A @ H.LUMA) / (b64B @ H.LUMA) - 1)).reshape(nth, nph).tolist()
    for arm in ('ER_L', 'ER_DF'):
        gm, a_cl, a64_ = E82.group_metrics(T(preds[arm]).view(G, nwo, 3), T(g64).view(G, nwo, 3), T(w).view(G, nwo, 1), ONES)
        resp[f'{arm}_vs_LAY64'] = {m_: gm[m_].reshape(nth, nph).tolist() for m_ in ('el1', 'ealb', 'eshape')}
        resp[f'{arm}_vs_LAY64']['albedo_ratio_luma'] = ((a_cl @ H.LUMA) / (a64_ @ H.LUMA)).reshape(nth, nph).tolist()
    # C3: compiled runtime (Dr.Jit) vs torch closure on the same directions; ER reciprocity (declared by construction)
    c3 = {}
    for arm in ('ER_L', 'ER_DF'):
        dr_ = H.er_eval_dr(mat[arm]['state'], wi, wo)
        c3[arm] = {'max_abs_over_max': float(np.abs(dr_ - preds[arm]).max() / max(np.abs(preds[arm]).max(), 1e-30)),
                   'rel_l1': float(np.abs(dr_ - preds[arm]).sum() / np.abs(preds[arm]).sum())}
        fw = H.er_eval_torch(mat[arm]['p'], wi, wo) / np.maximum(wo[:, 2:3], 1e-12)
        bw = H.er_eval_torch(mat[arm]['p'], wo, wi) / np.maximum(wi[:, 2:3], 1e-12)
        ok = (wi[:, 2] > 0.05) & (wo[:, 2] > 0.05)
        c3[arm]['reciprocity_max_rel'] = float((np.abs(fw - bw)[ok]).max() / max(np.abs(fw[ok]).max(), 1e-30))
    res['C3'] = c3
    # S_target: WL83 fit targets of this material vs a new layered reference at the same pairs
    fz = np.load(H.IN63B / 'data' / f'fit82_{st}.npz')
    ts = np.nonzero(fz['m'] == k)[0]
    rng = np.random.default_rng([104201, ii])
    ts = np.sort(rng.choice(ts, min(len(ts), S['target_pairs']), replace=False))
    twi, two = fz['wi'][ts].astype(np.float64), fz['wo'][ts].astype(np.float64)
    tw = fz['use'][ts].astype(np.float64) / fz['q'][ts].astype(np.float64)
    nA, nB = H.lay_eval(desc, twi, two, S['target_walks'], 1042010 + ii)
    new = 0.5 * (nA + nB)
    old, oA, oB = fz['f'][ts].astype(np.float64), fz['fA'][ts].astype(np.float64), fz['fB'][ts].astype(np.float64)
    lw = lambda x: (H.lum(x) * tw)
    res['S_target'] = {'pairs': int(len(ts)), 'target_walks_new': S['target_walks'],
                       'energy_ratio_target_over_new': float(lw(old).sum() / lw(new).sum()),
                       'rel_l1_target_vs_new': float(np.abs(lw(old) - lw(new)).sum() / np.abs(lw(new)).sum()),
                       'rel_l1_expected_from_target_halves': float(0.5 * np.abs(lw(oA) - lw(oB)).sum() / np.abs(lw(new)).sum()),
                       'rel_l1_new_halves': float(0.5 * np.abs(lw(nA) - lw(nB)).sum() / np.abs(lw(new)).sum()),
                       'energy_ratio_se': float(np.std(lw(old) - lw(new), ddof=1) * np.sqrt(len(ts)) / lw(new).sum())}
    save_json('contracts.json', res); save_json('response_wl83set.json', resp)
    log('contracts', json.dumps({k_: v for k_, v in res.items() if k_ != 'S_target'})[:400])

# ======================================================================================================== response grids
if not (OUT / 'response_grids.npz').exists():
    store = {}
    hd, hw, HT, HP = H.hemi_grid(S['grid_dth_deg'], S['grid_dph_deg'])
    for ti, th in enumerate(S['probe_theta_deg']):
        for phi in S['probe_phi_deg']:
            wi1 = H.sph(np.radians(th), np.radians(phi))
            key = f't{th}_p{phi}'
            pdirs, pdw, rdeg, psi = H.polar_grid(H.mirror(wi1), S['polar_rmax_deg'], S['polar_dr_deg'], S['polar_naz'])
            up = pdirs[:, 2] > 1e-3
            for gname, dirs, dw, nwalk in (('polar', pdirs[up], pdw[up], S['polar_walks']), ('hemi', hd, hw, S['grid_walks'])):
                W = np.repeat(wi1[None], len(dirs), 0)
                lA, lB = H.lay_eval(desc, W, dirs, nwalk, 1043000 + 1000 * ii + 10 * ti + phi // 90 + (500 if gname == 'hemi' else 0))
                store[f'{key}_{gname}_dirs'] = f32(dirs); store[f'{key}_{gname}_dw'] = f32(dw)
                store[f'{key}_{gname}_LAY_A'] = f32(lA); store[f'{key}_{gname}_LAY_B'] = f32(lB)
                for arm in ('ER_L', 'ER_DF'):
                    store[f'{key}_{gname}_{arm}'] = f32(H.er_eval_torch(mat[arm]['p'], W, dirs))
            store[f'{key}_polar_rdeg'] = f32(rdeg[up]); store[f'{key}_polar_psi'] = f32(psi[up]); store[f'{key}_wi'] = f32(wi1)
    np.savez_compressed(OUT / 'response_grids.npz', **store)
    log('response grids')

# ======================================================================================================== centre rays
if not (OUT / 'centre.npz').exists():
    pos_, I = C.LIGHTS[light]
    ctxs = {a: RT.Ctx(H.scene(tag, a)) for a in ('ER_L', 'ER_DF', 'LAY', 'LAY1024')}
    ctx = ctxs['ER_L']
    ray, pix, _ = ctx.gen_rays(1, None)
    si, sid, pid, ok = ctx.surface(ray, mi.Bool(True))
    Lp = mi.Point3f(*[float(v) for v in pos_])
    dl = Lp - si.p; d2 = dr.squared_norm(dl); wl = dl * dr.rsqrt(d2)
    wo_l = si.to_local(wl)
    on = ok & (si.wi.z > 1e-6) & (wo_l.z > 1e-6)
    vis = on & ~ctx.scene.ray_test(si.spawn_ray_to(Lp), on)
    dr.eval(si.wi, wo_l, d2, vis, sid)
    wiL = np.stack([np.asarray(si.wi[q], np.float64) for q in range(3)], 1)
    woL = np.stack([np.asarray(wo_l[q], np.float64) for q in range(3)], 1)
    scale = np.where(np.asarray(vis), I / np.asarray(d2, np.float64), 0.0)
    names = [m_[1] for m_ in ctx.meshes]
    sidn = np.asarray(sid); validn = np.asarray(ok)
    objm = validn & np.isin(sidn, [names.index('sphere'), names.index('cube')]) & np.asarray(vis)
    idx = np.nonzero(objm)[0]
    pred = {}
    for arm in ('ER_L', 'ER_DF'):
        pred[arm] = np.zeros((len(scale), 3)); pred[arm][idx] = H.er_eval_dr(mat[arm]['state'], wiL[idx], woL[idx]) * scale[idx, None]
        pred[arm + '_torch'] = np.zeros((len(scale), 3)); pred[arm + '_torch'][idx] = H.er_eval_torch(mat[arm]['p'], wiL[idx], woL[idx]) * scale[idx, None]
    lA, lB = H.lay_eval(desc, wiL[idx], woL[idx], S['pixel_walks'], 1044000 + ii)
    pred['LAY_A'] = np.zeros((len(scale), 3)); pred['LAY_A'][idx] = lA * scale[idx, None]
    pred['LAY_B'] = np.zeros((len(scale), 3)); pred['LAY_B'][idx] = lB * scale[idx, None]
    lA, lB = H.lay_eval(desc, wiL[idx], woL[idx], S['pixel_walks'], 1044500 + ii, max_depth=64)
    pred['LAY64_A'] = np.zeros((len(scale), 3)); pred['LAY64_A'][idx] = lA * scale[idx, None]
    pred['LAY64_B'] = np.zeros((len(scale), 3)); pred['LAY64_B'][idx] = lB * scale[idx, None]
    # render contract: tracer E1 at the same centre rays (k = 1 only)
    Lt = RT.point_lights(light)
    rend = {}
    for arm in ('ER_L', 'ER_DF'):
        c_ = ctxs[arm]
        r_, p_, _ = c_.gen_rays(1, None)
        N = C.RES * C.RES
        rng = mi.PCG32(size=N, initstate=mi.UInt64(1045000 + ii), initseq=mi.UInt64(0))
        out, _ = c_.trace(r_, N, rng, mi.UInt64(1045000 + ii), 'MI', 1, Lt, 'cam')
        a_ = c_.accum(out, p_, N)
        rend[arm] = np.stack([a_[f'e1{q}'] for q in range(3)], 1)
    for lay_arm in ('LAY', 'LAY1024'):
        c_ = ctxs[lay_arm]
        accs = []
        for q in range(S['lay_centre_passes']):
            R81.PASS_SEED['v'] = mi.UInt64(1046000 + 1000 * ii + q + (500 if lay_arm == 'LAY1024' else 0)); dr.make_opaque(R81.PASS_SEED['v'])
            r_, p_, _ = c_.gen_rays(1, None)
            N = C.RES * C.RES
            rng = mi.PCG32(size=N, initstate=mi.UInt64(1047000 + ii + q), initseq=mi.UInt64(0))
            out, _ = c_.trace(r_, N, rng, mi.UInt64(1047000 + ii), 'MI', 1, Lt, 'cam')
            a_ = c_.accum(out, p_, N)
            accs.append(np.stack([a_[f'e1{q_}'] for q_ in range(3)], 1))
        rend[f'{lay_arm}_passes'] = np.stack(accs)
    # per-pixel albedo for the energy / shape split: mixture directions (common to all arms), ER deterministic, LAY walks
    nS = S['albedo_samples']
    rng = np.random.default_rng([104401, ii])
    WI = np.repeat(wiL[idx], nS, 0)
    WO, Q, USE = DA.mixture(np.repeat(desc[None], len(WI), 0), WI, rng)
    wgt = (USE / Q)[:, None]
    alb = {}
    for arm in ('ER_L', 'ER_DF'):
        alb[arm] = (H.er_eval_dr(mat[arm]['state'], WI, WO) * wgt).reshape(len(idx), nS, 3).mean(1)
    aA, aB = H.lay_eval(desc, WI, WO, S['albedo_walks'], 1048000 + ii)
    alb['LAY'] = (0.5 * (aA + aB) * wgt).reshape(len(idx), nS, 3).mean(1)
    alb['LAY_A'] = (aA * wgt).reshape(len(idx), nS, 3).mean(1); alb['LAY_B'] = (aB * wgt).reshape(len(idx), nS, 3).mean(1)
    np.savez_compressed(OUT / 'centre.npz', idx=idx, wi=f32(wiL), wo=f32(woL), scale=scale, sid=sidn, valid=validn, names=np.array(names),
                        **{f'pred_{a}': f32(v) for a, v in pred.items()}, **{f'rend_{a}': f32(v) for a, v in rend.items()},
                        **{f'alb_{a}': f32(v) for a, v in alb.items()})
    log('centre')

# ======================================================================================================== images
d = OUT / 'rep'; d.mkdir(exist_ok=True)
ctxs = {a: RT.Ctx(H.scene(tag, a)) for a in ('ER_L', 'ER_DF', 'LAY', 'LAY1024')}
t0 = time.perf_counter()
for r in range(S['R']):
    fn = d / f'r{r:03d}.npz'
    if fn.exists():
        continue
    sd = 1049000 + 10000 * ii + r
    o = {}
    for arm in ('LAY', 'LAY1024', 'ER_L', 'ER_DF'):
        x = H.HA.render(ctxs[arm], image, S['spp'], sd, 'cam', pass_seed=arm.startswith('LAY'))
        for c_ in ('e1', 'e2', 'h'):
            o[f'{arm}_{c_}'] = f32(x[c_])
    np.savez(fn.with_suffix('.tmp.npz'), **o); os.replace(fn.with_suffix('.tmp.npz'), fn)
    if r % 8 == 0:
        log('rep', r, round(time.perf_counter() - t0, 1))
if not (OUT / 'mitsuba_ERDF7.npz').exists():
    hs = []
    for h in (0, 1):
        sc = H.scene(tag, 'ER_DF', light=light, with_sensor=False)
        m_, _, _, _ = RT.render_ref(sc, S['mitsuba_spp_half'], 104990 + 10 * ii + h, 7, batch=C.BATCH['ref'])   # render_g: seed * 7919 must fit uint32
        hs.append(f32(m_))
    np.savez(OUT / 'mitsuba_ERDF7.npz', h0=hs[0], h1=hs[1])
(OUT / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf-8')
log('DONE')
os._exit(0)
