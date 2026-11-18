"""WL104B analysis: contracts, event table, response metrics, angular grids, correspondence, classification, figures, gallery.
  <venv-cu128 python> analyze104b.py <raw_root> <reviewer_dir>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import h104b as H

sys.path.insert(0, str(H.HERE.parent / 'ecro_wl84_104a'))
import stats104a as S  # noqa: E402
import stats104b as SB  # noqa: E402

C, RT, R81 = H.C, H.RT, H.R81
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
PR = json.loads((H.HERE / 'protocol' / 'wl104b.json').read_text(encoding='utf-8'))
E104 = C.ROOT / 'evidence'
LUMA = H.LUMA
lum = lambda x: np.asarray(x) @ LUMA
EV = ['E1', 'E2', 'HIGH']
KEY = {'E1': 'e1', 'E2': 'e2', 'HIGH': 'h'}
ARMS = ['ER_L', 'ER_DF']
# comparison pairs (X, reference): ER error attribution uses the data-consistent physical reference (walk depth 1024, as the WL83 fitting and
# evaluation reference); 'vs_LAY' reproduces the WL84 / WL104 in-scene reference semantics (lay81 plugin, walk depth 64); LAY_vs_LAY1024 is the
# reference's own walk truncation
PAIRS = {'ER_L_vs_LAY1024': ('ER_L', 'LAY1024'), 'ER_DF_vs_LAY1024': ('ER_DF', 'LAY1024'), 'ER_L_vs_LAY': ('ER_L', 'LAY'),
         'ER_DF_vs_LAY': ('ER_DF', 'LAY'), 'LAY_vs_LAY1024': ('LAY', 'LAY1024')}
WL104_OBJ = {'test83_iid_1_L2': 0.035, 'test83_iid_0_L2': 0.027, 'test83_hard_1_L1': 0.030}


def rel_l1(a, b, m):
    return float(np.abs(a - b)[m].sum() / max(np.abs(b)[m].sum(), 1e-30))


ms_loo, jk_ci = SB.ms_loo, SB.jk_ci


def masks_for(image, tag, light):
    sc = H.scene(tag, 'LAY', light=light, with_sensor=False)
    obj, ndv = R81.masks(sc)
    cz = np.load(ROOT / image / 'centre.npz')
    names = list(cz['names']); sid = cz['sid'].reshape(C.RES, C.RES); valid = cz['valid'].reshape(C.RES, C.RES)
    sph = valid & (sid == names.index('sphere')); cub = valid & (sid == names.index('cube'))
    ref = 0.5 * sum(np.load(E104 / 'refs' / image / f'LAY65_h{h}.npz')['mean'].astype(np.float64) for h in (0, 1))
    L = lum(ref)
    objm = sph | cub
    core = objm & (L >= 3 * np.median(L[objm]))
    ring = sph & ndimage.binary_dilation(core & sph, structure=np.ones((3, 3)), iterations=6) & ~core
    return {'objects': objm, 'sphere': sph, 'cube': cub, 'highlight_core': core, 'highlight_ring': ring,
            'grazing': objm & (ndv <= 0.25), 'full': np.ones_like(objm)}, obj


res = {'images': {}, 'protocol_sha256': C.sha(H.HERE / 'protocol' / 'wl104b.json')}
VIS = {}
for ii, image in enumerate(C.IMAGES):
    tag, light, st, k = C.image_spec(image)
    mat = H.material(tag); H.register(tag, mat)
    D = ROOT / image
    con = json.loads((D / 'contracts.json').read_text(encoding='utf-8'))
    rsp = json.loads((D / 'response_wl83set.json').read_text(encoding='utf-8'))
    M, obj_r81 = masks_for(image, tag, light)
    files = sorted((D / 'rep').glob('r*.npz'))
    Z = {}
    for f in files:
        z = np.load(f)
        for kk in z.files:
            Z.setdefault(kk, []).append(z[kk].astype(np.float64))
    Z = {kk: np.stack(v) for kk, v in Z.items()}
    R = len(files)
    comp = lambda arm, e: (Z[f'{arm}_e1'] + Z[f'{arm}_e2'] + Z[f'{arm}_h']) if e == 'TOTAL' else (Z[f'{arm}_e1'] + Z[f'{arm}_e2'] if e == 'LOW' else Z[f'{arm}_{KEY[e]}'])
    out = {'R': R, 'roi_counts': {kk: int(v.sum()) for kk, v in M.items()}}
    # ---------------- contracts
    cz = np.load(D / 'centre.npz')
    idx = cz['idx']
    rc = {}
    for arm in ARMS:
        pr_, rd = cz[f'pred_{arm}'][idx].astype(np.float64), cz[f'rend_{arm}'][idx].astype(np.float64)
        rc[arm] = {'max_abs_over_max': float(np.abs(pr_ - rd).max() / max(np.abs(pr_).max(), 1e-30)),
                   'torch_vs_dr_centre': float(np.abs(cz[f'pred_{arm}_torch'][idx] - pr_).max() / max(np.abs(pr_).max(), 1e-30))}
    pl = 0.5 * (cz['pred_LAY_A'][idx].astype(np.float64) + cz['pred_LAY_B'][idx].astype(np.float64))          # depth 1024
    pl64 = 0.5 * (cz['pred_LAY64_A'][idx].astype(np.float64) + cz['pred_LAY64_B'][idx].astype(np.float64))   # depth 64
    for lay_arm, pk in (('LAY', 'LAY64'), ('LAY1024', 'LAY')):      # each plugin against the reference of its own walk depth
        passes = cz[f'rend_{lay_arm}_passes'][:, idx].astype(np.float64)
        pA, pB = cz[f'pred_{pk}_A'][idx].astype(np.float64), cz[f'pred_{pk}_B'][idx].astype(np.float64)
        sums = lum(passes).sum(1)
        var_r = sums.var(ddof=1) / len(sums)
        var_p = ((lum(pA).sum() - lum(pB).sum()) / 2) ** 2
        pm = lum(0.5 * (pA + pB)).sum()
        rc[lay_arm] = {'z_object_energy': float((sums.mean() - pm) / np.sqrt(var_r + var_p)), 'ratio': float(sums.mean() / pm), 'reference_depth': 64 if pk == 'LAY64' else 1024}
    se_d = np.sqrt(((lum(cz['pred_LAY64_A'][idx]).sum() - lum(cz['pred_LAY64_B'][idx]).sum()) / 2) ** 2 + ((lum(cz['pred_LAY_A'][idx]).sum() - lum(cz['pred_LAY_B'][idx]).sum()) / 2) ** 2)
    rc['walk_depth_bias_E1_objects'] = {'ratio_d64_over_d1024': float(lum(pl64).sum() / lum(pl).sum()), 'z': float((lum(pl64).sum() - lum(pl).sum()) / se_d)}
    c1 = all(rc[a]['max_abs_over_max'] <= 1e-5 for a in ARMS) and all(abs(rc[a]['z_object_energy']) <= 4 for a in ('LAY', 'LAY1024'))
    c2 = {}
    for arm, src in (('LAY', 'LAY7'), ('ER_L', 'ER7'), ('ER_DF', None)):
        tr = comp(arm, 'TOTAL')
        if src:
            h = [np.load(E104 / 'refs' / image / f'{src}_h{q}.npz')['mean'].astype(np.float64) for q in (0, 1)]
        else:
            zz = np.load(D / 'mitsuba_ERDF7.npz'); h = [zz['h0'].astype(np.float64), zz['h1'].astype(np.float64)]
        rm, tm = 0.5 * (h[0] + h[1]), tr.mean(0)
        o = M['objects']
        lim = 1.5 * 0.5 * float(np.hypot(rel_l1(tr[:R // 2].mean(0), tr[R // 2:].mean(0), o), rel_l1(h[0], h[1], o)))
        g = {'obj_mean_ratio': float(lum(tm)[o].mean() / lum(rm)[o].mean()), 'full_mean_ratio': float(lum(tm).mean() / lum(rm).mean()),
             'obj_rel_l1': rel_l1(tm, rm, o), 'limit': lim}
        g['pass'] = bool(abs(g['obj_mean_ratio'] - 1) <= 0.01 and abs(g['full_mean_ratio'] - 1) <= 0.01 and g['obj_rel_l1'] <= lim)
        c2[arm] = g
    c3 = con['C3']
    c3_ok = all(c3[a]['max_abs_over_max'] <= 1e-5 and c3[a]['reciprocity_max_rel'] <= 1e-4 for a in ARMS)
    c5a = con['C5a_wl83_lane_reproduction_max_abs']
    c5a_ok = all(v <= 1e-4 for a in ARMS for v in c5a[a].values())
    c5b_val = rel_l1(comp('ER_L', 'TOTAL').mean(0), comp('LAY', 'TOTAL').mean(0), M['objects'])
    st_ = con['S_target']
    st_ok = (abs(st_['energy_ratio_target_over_new'] - 1) <= 3 * st_['energy_ratio_se']
             and st_['rel_l1_target_vs_new'] <= 1.25 * np.hypot(st_['rel_l1_expected_from_target_halves'], st_['rel_l1_new_halves']))
    out['contracts'] = {'C0': con['C0'], 'C0_pass': bool(con['C0']['ER_L_matches_WL84_audit'] and con['C0']['ER_DF_matches_WL84_audit']),
                        'C1_render': rc, 'C1_pass': bool(c1), 'C2': c2, 'C2_pass': bool(all(v['pass'] for v in c2.values())),
                        'C3': c3, 'C3_pass': bool(c3_ok), 'C5a': c5a, 'C5a_pass': bool(c5a_ok),
                        'C5b_obj_rel_l1': c5b_val, 'C5b_wl104': WL104_OBJ[image], 'C5b_pass': bool(abs(c5b_val - WL104_OBJ[image]) <= 0.005),
                        'S_target': st_, 'S_target_unbiased': bool(st_ok)}
    # ---------------- event table
    ev_tab = {}
    LT = comp('LAY', 'TOTAL')
    for name, (arm, refa) in PAIRS.items():
        ev_tab[name] = {}
        LTr = comp(refa, 'TOTAL')
        D_ = {e: comp(arm, e) - comp(refa, e) for e in EV + ['LOW', 'TOTAL']}
        for roi, m in M.items():
            den = lum(LTr[:, m]).sum(1)
            row = {'energy': {e: S.ratio_jk(lum(D_[e][:, m]).sum(1), den) for e in EV + ['LOW', 'TOTAL']}}
            A = {e: D_[e][:, m].reshape(R, -1) for e in EV + ['LOW', 'TOTAL']}
            msT, looT = ms_loo(A['TOTAL'], A['TOTAL'])
            terms = {}
            for i_, e in enumerate(EV):
                for f_ in EV[i_:]:
                    pt, loo = ms_loo(A[e], A[f_])
                    mult = 1 if e == f_ else 2
                    terms[f'{e}x{f_}'] = {'value': mult * pt, 'share': mult * pt / msT if msT > 0 else None,
                                          'share_ci': jk_ci(mult * pt / msT, mult * loo / looT) if msT > 0 else None}
            msL, looL = ms_loo(A['LOW'], A['LOW'])
            norm = float((LTr.mean(0)[m] ** 2).mean())
            row['ms_total'] = msT; row['ms_terms'] = terms
            row['identity_residual'] = float(msT - sum(v['value'] for v in terms.values()))
            row['low_share'] = jk_ci(msL / msT, looL / looT) if msT > 0 else None
            row['rel_rms'] = {e: S.rel_rms(S.stable_ms_jk(A[e]), norm) for e in EV + ['LOW', 'TOTAL']}
            row['rel_l1_own'] = {}
            for e in EV + ['TOTAL']:
                Dm = D_[e].mean(0)[m]; own = comp(refa, e).mean(0)[m]
                h1, h2 = D_[e][:R // 2].mean(0)[m], D_[e][R // 2:].mean(0)[m]
                row['rel_l1_own'][e] = {'value': float(np.abs(Dm).sum() / max(own.sum(), 1e-30)), 'noise_floor': float(0.5 * np.abs(h1 - h2).sum() / max(own.sum(), 1e-30))}
            ev_tab[name][roi] = row
    ln = {roi: float(0.5 * np.abs(LT[:R // 2].mean(0) - LT[R // 2:].mean(0))[m].sum() / LT.mean(0)[m].sum()) for roi, m in M.items()}
    out['events'] = ev_tab; out['lay_reference_noise_rel_l1'] = ln
    # ---------------- response (WL83 set) classes
    th = rsp['theta_deg']; nphi = rsp['n_phi']
    cls = {}
    for arm in ARMS:
        ar = np.array(rsp[arm]['albedo_ratio_luma']); es = np.array(rsp[arm]['eshape']); el = np.array(rsp[arm]['el1'])
        na = np.array(rsp['noise_albedo_ratio_luma_half']); ns = np.array(rsp['noise_half']['eshape'])
        e_sig = (np.abs(ar - 1) >= 0.01) & (np.abs(ar - 1) > 3 * na)
        s_sig = (es - ns) >= 0.02
        lab = np.where(e_sig & s_sig, 'both', np.where(e_sig, 'energy', np.where(s_sig, 'shape', 'none')))
        cls[arm] = {'labels': lab.tolist(), 'counts': {c_: int((lab == c_).sum()) for c_ in ('energy', 'shape', 'both', 'none')},
                    'mean_el1': float(el.mean()), 'mean_eshape': float(es.mean()), 'mean_abs_albedo_dev': float(np.abs(ar - 1).mean()),
                    'albedo_ratio_luma': ar.tolist(), 'eshape': es.tolist(), 'el1': el.tolist(), 'noise_eshape': ns.tolist(), 'noise_albedo': na.tolist()}
    out['response_wl83set'] = cls
    out['response_walk_depth'] = {kk: rsp[kk] for kk in ('LAY64_vs_LAY1024', 'ER_L_vs_LAY64', 'ER_DF_vs_LAY64', 'LAY64_noise_albedo_ratio_luma_half')}
    # ---------------- grids
    G = np.load(D / 'response_grids.npz')
    grid = {}
    for t_ in PR['sampling']['probe_theta_deg']:
        for p_ in PR['sampling']['probe_phi_deg']:
            key = f't{t_}_p{p_}'
            wi1 = G[f'{key}_wi'].astype(np.float64); mir = H.mirror(wi1)
            gd = {}
            pd, pw = G[f'{key}_polar_dirs'].astype(np.float64), G[f'{key}_polar_dw'].astype(np.float64)
            hd, hw = G[f'{key}_hemi_dirs'].astype(np.float64), G[f'{key}_hemi_dw'].astype(np.float64)

            def pol(v):
                L = lum(v); E = float((L * pw).sum())
                j = int(np.argmax(L)); ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
                c = (L * pw) @ pd; c = c / max(np.linalg.norm(c), 1e-30)
                rr = np.degrees(np.arccos(np.clip(pd @ c, -1, 1)))
                return {'cone_energy': E, 'peak': float(L[j]), 'peak_offset_deg': ang(pd[j], mir), 'centroid_offset_deg': ang(c, mir),
                        'rms_radius_deg': float(np.sqrt((L * pw * rr ** 2).sum() / max(E, 1e-30)))}
            la, lb = G[f'{key}_polar_LAY_A'].astype(np.float64), G[f'{key}_polar_LAY_B'].astype(np.float64)
            gd['polar'] = {'LAY': pol(0.5 * (la + lb)), 'LAY_A': pol(la), 'LAY_B': pol(lb)}
            ha, hb = G[f'{key}_hemi_LAY_A'].astype(np.float64), G[f'{key}_hemi_LAY_B'].astype(np.float64)
            hl = 0.5 * (ha + hb)
            pn = lambda v: lum(v) / max((lum(v) * hw).sum(), 1e-30)
            gd['hemi'] = {'LAY_albedo': float((lum(hl) * hw).sum()), 'LAY_tv_noise': float(0.25 * (np.abs(pn(ha) - pn(hb)) * hw).sum())}
            for arm in ARMS:
                gd['polar'][arm] = pol(G[f'{key}_polar_{arm}'].astype(np.float64))
                hv = G[f'{key}_hemi_{arm}'].astype(np.float64)
                gd['hemi'][arm] = {'albedo_ratio': float((lum(hv) * hw).sum() / max((lum(hl) * hw).sum(), 1e-30)),
                                   'tv': float(0.5 * (np.abs(pn(hv) - pn(hl)) * hw).sum()),
                                   'cone_energy_ratio': gd['polar'][arm]['cone_energy'] / max(gd['polar']['LAY']['cone_energy'], 1e-30),
                                   'peak_ratio': gd['polar'][arm]['peak'] / max(gd['polar']['LAY']['peak'], 1e-30),
                                   'rms_radius_ratio': gd['polar'][arm]['rms_radius_deg'] / max(gd['polar']['LAY']['rms_radius_deg'], 1e-30)}
            grid[key] = gd
    out['grids'] = grid
    # anisotropy of the ER closures (layered is isotropic): phi 0 vs 90
    out['er_anisotropy'] = {arm: {f't{t_}': abs(grid[f't{t_}_p0']['hemi'][arm]['albedo_ratio'] - grid[f't{t_}_p90']['hemi'][arm]['albedo_ratio'])
                                  for t_ in PR['sampling']['probe_theta_deg']} for arm in ARMS}
    # ---------------- correspondence (E1)
    er = ndimage.binary_erosion(M['objects'], structure=np.ones((3, 3)))
    vis = np.zeros(C.RES * C.RES, bool); vis[idx] = True
    sel = er.ravel() & vis
    corr = {}
    alb = {a: lum(cz[f'alb_{a}'].astype(np.float64)) for a in ARMS + ['LAY']}
    plL = lum(pl)
    VIS[image] = {'M': M}
    for arm in ARMS + ['LAY64']:
        if arm == 'LAY64':          # walk-depth correspondence: rendered LAY (depth 64) - LAY1024 vs predicted depth-64 - depth-1024
            dren = lum(comp('LAY', 'E1').mean(0) - comp('LAY1024', 'E1').mean(0)).ravel()
            dpred_full = np.zeros(C.RES * C.RES); dpred_full[idx] = lum(pl64) - plL
            en = np.zeros(C.RES * C.RES); sh = dpred_full - en
        else:                       # ER vs the data-consistent layered reference (depth 1024): rendered with the LAY1024 plugin
            dren = lum(comp(arm, 'E1').mean(0) - comp('LAY1024', 'E1').mean(0)).ravel()
            dpred_full = np.zeros(C.RES * C.RES); dpred_full[idx] = lum(cz[f'pred_{arm}'][idx]) - plL
            ratio = alb[arm] / np.maximum(alb['LAY'], 1e-30)
            en = np.zeros(C.RES * C.RES); en[idx] = (ratio - 1) * plL
            sh = dpred_full - en
        x, y = dpred_full[sel], dren[sel]
        slope = float((x * y).sum() / max((x * x).sum(), 1e-30))
        c = {'pearson_r': float(np.corrcoef(x, y)[0, 1]), 'slope_rendered_on_predicted': slope,
             'energy_ratio_rendered_over_predicted': float(y.sum() / x.sum()) if abs(x.sum()) > 1e-12 else None,
             'energy_part_share_of_predicted_ms': float((en[sel] ** 2).mean() / max((x ** 2).mean(), 1e-30)),
             'shape_part_share_of_predicted_ms': float((sh[sel] ** 2).mean() / max((x ** 2).mean(), 1e-30)),
             'energy_shape_cross_share': float(2 * (en[sel] * sh[sel]).mean() / max((x ** 2).mean(), 1e-30)),
             'r_rendered_vs_energy_part': float(np.corrcoef(en[sel], y)[0, 1]) if en[sel].std() > 0 else None,
             'r_rendered_vs_shape_part': float(np.corrcoef(sh[sel], y)[0, 1]),
             'predicted_energy_sum_split': {'energy_part': float(en[sel].sum()), 'shape_part': float(sh[sel].sum()), 'total': float(x.sum())},
             'pixels': int(sel.sum())}
        c['established'] = bool(c['pearson_r'] >= 0.8 and 0.8 <= slope <= 1.25)
        for roi in ('highlight_core', 'highlight_ring'):
            mm = M[roi].ravel() & vis
            c[f'{roi}_sum'] = {'rendered': float(dren[mm].sum()), 'predicted': float(dpred_full[mm].sum()), 'energy_part': float(en[mm].sum()), 'shape_part': float(sh[mm].sum())}
        corr[arm] = c
        VIS[image][arm] = {'dren': dren.reshape(C.RES, C.RES), 'dpred': dpred_full.reshape(C.RES, C.RES), 'en': en.reshape(C.RES, C.RES), 'sh': sh.reshape(C.RES, C.RES)}
    out['correspondence_E1'] = corr
    for e in EV + ['TOTAL']:
        for arm in ['LAY', 'LAY1024'] + ARMS:
            VIS[image][f'{arm}_{e}'] = comp(arm, e).mean(0)
    LT2 = comp('LAY1024', 'TOTAL')
    VIS[image]['LAY_noise'] = 0.5 * (LT2[:R // 2].mean(0) - LT2[R // 2:].mean(0))
    VIS[image]['grids'] = {kk: G[kk] for kk in G.files if kk.startswith(('t20_p0', 't60_p0', 't80_p0'))}
    res['images'][image] = out
    del Z
    print('analysed', image, flush=True)

# ======================================================================================================== classification
imgs = C.IMAGES
contract_fail = [im for im in imgs if not (res['images'][im]['contracts']['C1_pass'] and res['images'][im]['contracts']['C2_pass'] and res['images'][im]['contracts']['C3_pass'])]
dom = {}
for im in imgs:
    dom[im] = {}
    for roi in PR['rois']['primary']:
        row = res['images'][im]['events']['ER_L_vs_LAY1024'][roi]
        en_ = {e: abs(row['energy'][e]['point']) for e in EV}
        ms_ = {e: row['ms_terms'][f'{e}x{e}']['share'] for e in EV}
        dom[im][roi] = {'energy': max(en_, key=en_.get), 'ms': max(ms_, key=lambda e: ms_[e] or -1)}
low_ok = [im for im in imgs if res['images'][im]['events']['ER_L_vs_LAY']['objects']['low_share'] and res['images'][im]['events']['ER_L_vs_LAY']['objects']['low_share']['lo'] >= 0.5]
low_ok1024 = [im for im in imgs if res['images'][im]['events']['ER_L_vs_LAY1024']['objects']['low_share'] and res['images'][im]['events']['ER_L_vs_LAY1024']['objects']['low_share']['lo'] >= 0.5]
cmpL = {}
for im in imgs:
    r_ = res['images'][im]
    elL, elD = r_['response_wl83set']['ER_L']['mean_el1'], r_['response_wl83set']['ER_DF']['mean_el1']
    rmL, rmD = r_['events']['ER_L_vs_LAY1024']['objects']['rel_rms']['TOTAL']['point'], r_['events']['ER_DF_vs_LAY1024']['objects']['rel_rms']['TOTAL']['point']
    cmpL[im] = {'el1_L': elL, 'el1_DF': elD, 'rms_L': rmL, 'rms_DF': rmD, 'DF_clearly_better': bool(elD < 0.75 * elL and rmD < 0.75 * rmL),
                'DF_not_better': bool(elD >= 0.75 * elL and rmD >= 0.75 * rmL)}
targets_ok = all(res['images'][im]['contracts']['S_target_unbiased'] for im in imgs)
nb = sum(v['DF_clearly_better'] for v in cmpL.values()); nn = sum(v['DF_not_better'] for v in cmpL.values())
ref_lim = {}
for im in imgs:
    rw = res['images'][im]['events']['LAY_vs_LAY1024']['objects']
    lo_e, _ = S.abs_bounds(rw['energy']['TOTAL'])
    wd = res['images'][im]['contracts']['C1_render']['walk_depth_bias_E1_objects']
    ref_lim[im] = {'total_energy_sig': bool(lo_e > 0.01), 'total_rms_sig': bool(rw['rel_rms']['TOTAL']['lo'] > 0.02),
                   'E1_depth_bias_sig': bool(abs(wd['ratio_d64_over_d1024'] - 1) > 0.01 and abs(wd['z']) > 4)}
labels = []
if any(any(v.values()) for v in ref_lim.values()):
    labels.append('PHYSICAL_REFERENCE_LIMITATION_SUPPORTED (in-scene layered plugin walk depth 64 vs 1024)')
if contract_fail:
    labels.append('RENDERING_CONTRACT_ERROR_SUPPORTED')
if nb >= 2:
    labels.append('FIT_OR_TRAINING_LIMITATION_SUPPORTED')
if nn >= 2 and targets_ok and not contract_fail:
    labels.append('REPRESENTATION_LIMITATION_SUPPORTED (qualified: frozen direct-fit procedure)')
if (nb >= 1 and nn >= 1):
    labels.append('MIXED_CAUSES_SUPPORTED')
if not labels:
    labels.append('INCONCLUSIVE')
res['classification'] = {'labels': labels, 'reference_limitation': ref_lim, 'contract_failures': contract_fail, 'dominant_event_ER_L_vs_LAY1024': dom,
                         'LOW_dominant_images_vs_LAY': low_ok, 'LOW_dominant_images_vs_LAY1024': low_ok1024,
                         'LOW_dominance_supported': len(low_ok) >= 2, 'fit_vs_learned': cmpL, 'targets_unbiased_all': targets_ok,
                         'correspondence_established': {im: {a: res['images'][im]['correspondence_E1'][a]['established'] for a in ARMS + ['LAY64']} for im in imgs}}


def clean(o):
    if isinstance(o, dict):
        return {str(a): clean(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


C.write_json(OUT / 'metrics.json', clean(res))
print('CLASSIFICATION', labels, flush=True)

# ======================================================================================================== figures
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({'font.size': 8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': '#52514e'})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f3a07f', '#c4471a'])
eT = 1.0 / float(np.percentile(np.concatenate([lum(VIS[i]['LAY1024_TOTAL']).ravel() for i in imgs]), 99.5))
tone = lambda x, e: np.clip((np.clip(e * x, 0, None) / (1 + np.clip(e * x, 0, None))) ** (1 / 2.2), 0, 1)


def logdisp(M_):
    m0 = M_ / 1000.0
    return lambda x: np.sign(x) * np.log1p(np.abs(x) / m0) / np.log1p(M_ / m0)


def cbar(fig, im, axes, M_, f, label):
    vals = M_ * np.array([0, 0.001, 0.01, 0.1, 1.0]); vals = np.concatenate([-vals[::-1][:-1], vals])
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=f(vals)); cb.ax.set_yticklabels([f'{v:.3g}' for v in vals]); cb.set_label(label, fontsize=7)


# event family scale
evd = [lum(VIS[i][f'{a}_{e}'] - VIS[i][f'LAY1024_{e}']) for i in imgs for a in ARMS + ['LAY'] for e in EV + ['TOTAL']] + [lum(VIS[i]['LAY_noise']) for i in imgs]
ME = float(max(np.abs(x).max() for x in evd)); fE = logdisp(ME)
for im in imgs:
    V = VIS[im]
    fig, ax = plt.subplots(8, 4, figsize=(14, 28), constrained_layout=True)
    for j, e in enumerate(EV + ['TOTAL']):
        ee = eT if e == 'TOTAL' else 1.0 / float(np.percentile(lum(V[f'LAY1024_{e}']), 99.5) + 1e-12)
        for i_, a in enumerate(['LAY1024', 'LAY'] + ARMS):
            ax[i_, j].imshow(tone(V[f'{a}_{e}'], ee)); ax[i_, j].set_title(f'{a} {e}' + (' (own exposure)' if e != 'TOTAL' else ''), fontsize=8)
        for i_, a in enumerate(ARMS + ['LAY']):
            imh = ax[4 + i_, j].imshow(fE(lum(V[f'{a}_{e}'] - V[f'LAY1024_{e}'])), cmap=DIV, vmin=-1, vmax=1)
            ax[4 + i_, j].set_title(f'{a} - LAY1024, {e} (signed)', fontsize=8)
        if j == 3:
            ax[7, j].imshow(fE(lum(V['LAY_noise'])), cmap=DIV, vmin=-1, vmax=1); ax[7, j].set_title('LAY1024 TOTAL noise (half-split / 2)', fontsize=8)
        else:
            m_ = V['M'][['objects', 'highlight_core', 'highlight_ring'][j]]
            ax[7, j].imshow(tone(V['LAY1024_TOTAL'], eT)); ax[7, j].contour(m_, levels=[0.5], colors='#c4471a', linewidths=0.6)
            ax[7, j].set_title('ROI: ' + ['objects', 'highlight_core', 'highlight_ring'][j], fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    cbar(fig, imh, ax[4:8].ravel().tolist(), ME, fE, 'signed difference, linear HDR luminance')
    fig.suptitle(f'WL104B {im}: event-separated images (rows 1-4) and signed event differences vs the depth-1024 layered reference (rows 5-7); one scale', fontsize=9)
    fig.savefig(OUT / f'events_{im}.png', dpi=75); plt.close(fig)

# correspondence
cd_ = [VIS[i][a][k_] for i in imgs for a in ARMS for k_ in ('dren', 'dpred', 'en', 'sh')]
MC = float(max(np.abs(x).max() for x in cd_)); fC = logdisp(MC)
fig, ax = plt.subplots(3, 5, figsize=(17, 10.5), constrained_layout=True)
for i_, im in enumerate(imgs):
    V = VIS[im]['ER_L']; o = VIS[im]['M']['objects']
    for j, (k_, nm) in enumerate((('dren', 'rendered dE1 (ER_L - LAY1024)'), ('dpred', 'predicted dE1 (centre ray)'), ('en', 'energy part (albedo ratio)'), ('sh', 'shape part (remainder)'))):
        imh = ax[i_, j].imshow(fC(np.where(o, V[k_], 0)), cmap=DIV, vmin=-1, vmax=1); ax[i_, j].set_title(f'{im}: {nm}', fontsize=7)
        ax[i_, j].set_xticks([]); ax[i_, j].set_yticks([])
    c = res['images'][im]['correspondence_E1']['ER_L']
    sel = ndimage.binary_erosion(o, structure=np.ones((3, 3))) & (V['dpred'] != 0)
    ax[i_, 4].plot(V['dpred'][sel], V['dren'][sel], '.', ms=1, color='#2a78d6', alpha=0.4)
    lim_ = np.abs(V['dpred'][sel]).max(); ax[i_, 4].plot([-lim_, lim_], [-lim_, lim_], color='#52514e', lw=1)
    ax[i_, 4].set_title(f'r {c["pearson_r"]:.3f}, slope {c["slope_rendered_on_predicted"]:.3f}', fontsize=8)
    ax[i_, 4].set_xlabel('predicted dE1'); ax[i_, 4].set_ylabel('rendered dE1'); ax[i_, 4].grid(alpha=0.25)
cbar(fig, imh, ax[:, :4].ravel().tolist(), MC, fC, 'signed E1 difference, linear HDR luminance')
fig.suptitle('E1 correspondence: response-level prediction (energy + shape parts) vs rendered E1 difference, objects', fontsize=9)
fig.savefig(OUT / 'correspondence_E1.png', dpi=80); plt.close(fig)

# response metrics vs theta
fig, ax = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
for j, im in enumerate(imgs):
    r_ = res['images'][im]['response_wl83set']
    for i_, (mk, nm) in enumerate((('albedo_ratio_luma', 'albedo ratio a_X / a_LAY (energy)'), ('eshape', 'eshape (shape: normalized TV)'), ('el1', 'el1 (WL83 rel L1)'))):
        a = ax[i_, j]
        for arm, col in (('ER_L', '#2a78d6'), ('ER_DF', '#eb6834')):
            v = np.array(r_[arm][mk])
            for p_, ls in ((0, '-'), (1, '--')):
                a.plot(th, v[:, p_], ls, color=col, marker='o', ms=4, lw=2, label=f'{arm} phi set {p_}' if i_ == 0 and j == 0 else None)
        if mk == 'albedo_ratio_luma':
            a.axhspan(0.99, 1.01, color='#d9d8d4', alpha=0.5); a.axhline(1, color='#52514e', lw=0.8)
        if mk == 'eshape':
            a.plot(th, np.array(r_['ER_L']['noise_eshape']).mean(1) + 0.02, ':', color='#52514e', lw=1, label='noise + 0.02' if j == 0 else None)
        a.set_title(f'{im}: {nm}', fontsize=8); a.set_xlabel('incident theta (deg)'); a.grid(alpha=0.25)
ax[0, 0].legend(fontsize=7, frameon=False); ax[1, 0].legend(fontsize=7, frameon=False)
fig.suptitle('Response-level error on the WL83 high-precision eval set (8 theta x 2 phi per material): energy vs shape', fontsize=9)
fig.savefig(OUT / 'response_vs_theta.png', dpi=90); plt.close(fig)

# angular atlas (theta-phi maps) at theta_i 20, 60, 80, phi 0
for im in imgs:
    g = VIS[im]['grids']
    fig, ax = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True)
    allv = []
    for t_ in (20, 60, 80):
        key = f't{t_}_p0_hemi'
        lay = 0.5 * (g[f'{key}_LAY_A'] + g[f'{key}_LAY_B'])
        allv += [lum(lay), lum(g[f'{key}_ER_L']), lum(g[f'{key}_ER_DF'])]
    MA = float(max(np.abs(x).max() for x in allv)); fA = logdisp(MA)
    for i_, t_ in enumerate((20, 60, 80)):
        key = f't{t_}_p0_hemi'
        lay = lum(0.5 * (g[f'{key}_LAY_A'] + g[f'{key}_LAY_B'])).reshape(90, 180)
        for j, (nm, v) in enumerate((('LAY', lay), ('ER_L', lum(g[f'{key}_ER_L']).reshape(90, 180)), ('ER_DF', lum(g[f'{key}_ER_DF']).reshape(90, 180)),
                                     ('ER_L - LAY', lum(g[f'{key}_ER_L']).reshape(90, 180) - lay), ('ER_DF - LAY', lum(g[f'{key}_ER_DF']).reshape(90, 180) - lay))):
            imh = ax[i_, j].imshow(fA(v), cmap=DIV if j >= 3 else SEQ, vmin=-1 if j >= 3 else 0, vmax=1, aspect='auto', extent=[0, 360, 90, 0])
            ax[i_, j].set_title(f'theta_i {t_}: {nm}', fontsize=8); ax[i_, j].set_xlabel('phi_o (deg)'); ax[i_, j].set_ylabel('theta_o (deg)')
    cbar(fig, imh, ax.ravel().tolist(), MA, fA, 'eval = f cos (luminance)')
    fig.suptitle(f'{im} material: angular atlas (hemisphere 1 x 2 deg), learned-ER vs layered; mirror direction at phi_o 180', fontsize=9)
    fig.savefig(OUT / f'atlas_{im}.png', dpi=80); plt.close(fig)

# event energy bars
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True, sharey=False)
cols = {'E1': '#2a78d6', 'E2': '#1baf7a', 'HIGH': '#eda100', 'TOTAL': '#52514e'}
for a, roi in zip(ax, PR['rois']['primary']):
    for i_, im in enumerate(imgs):
        for k_, arm in enumerate(ARMS):
            for q, e in enumerate(EV + ['TOTAL']):
                c = res['images'][im]['events'][f'{arm}_vs_LAY1024'][roi]['energy'][e]
                x = i_ + (k_ - 0.5) * 0.42 + (q - 1.5) * 0.09
                a.plot([x, x], [c['lo'], c['hi']], color=cols[e], lw=2); a.plot(x, c['point'], 'o' if arm == 'ER_L' else 's', color=cols[e], ms=5,
                                                                                label=f'{e}' if (i_ == 0 and k_ == 0 and roi == 'objects') else None)
    a.axhline(0, color='#52514e', lw=0.8); a.set_xticks(range(3)); a.set_xticklabels([s.replace('test83_', '') for s in imgs])
    a.set_title(f'{roi}: signed event energy difference / LAY1024 TOTAL (o ER_L, s ER_DF)', fontsize=8); a.grid(axis='y', alpha=0.25)
ax[0].legend(fontsize=7, frameon=False)
fig.savefig(OUT / 'event_energy.png', dpi=90); plt.close(fig)

files = sorted(p.name for p in OUT.glob('*.png'))
C.write_json(OUT / 'evaluation_manifest.json', {'id': 'WL104B', 'display': {'event_scale_max': ME, 'correspondence_scale_max': MC, 'mapping': 'sign(x) log(1+|x|/m0)/log(1+M/m0), m0 = M/1000',
                                                                         'tone_exposure_total': eT}, 'files': {f: C.sha(OUT / f) for f in files + ['metrics.json']}})
html = ['<!doctype html><meta charset="utf-8"><title>WL104B ER error attribution</title><body style="font-family:sans-serif;background:#fcfcfb">',
        f'<h1>WL104B — Learned-ER material error attribution</h1><p>Classification: {"; ".join(labels)}. Numbers: metrics.json.</p>']
html += [f'<h3>{f}</h3><img src="{f}" style="max-width:100%">' for f in files]
(OUT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
print('ANALYSIS_COMPLETE', flush=True)
os._exit(0)
