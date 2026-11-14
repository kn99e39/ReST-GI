"""WL93D checkerboard addendum, revision 2 (box checker reflected by the sharp panel): comparison sheets and descriptive numbers
(post hoc, qualitative; no labels).

Full frame of an arm = the reference LOW of the same variant (physical camera-path events 1-2 = REF tot - REF h, identical for every arm)
+ the arm's rendered higher-order term hl. One exposure for every image: key from the UNTEXTURED reference median, Reinhard, gamma 2.2.
Error / difference maps: signed sqrt mapping, one absolute scale per map type over the whole addendum, colourbar in real units.
All untextured counterparts (reference and arms) are rendered in the same scene with the same seeds (common random numbers).
"""
import glob, hashlib, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARS = [('bx38', 'checker 0.3 / 0.8 (new Lambert states)'), ('bx75', 'checker 0.7 / 0.5 (existing ids 3 / 4)')]
ARMS = [('C0', 'deployed (C0)'), ('C2_src', 'M64 at source use (C2_src)'), ('C2_rcv', 'M64 at receiver use (C2_rcv)'), ('C2_both', 'M64 at both uses (C2_both)')]
SEEDS = [8601, 8602]
ALB = {3: 0.7, 4: 0.5, 5: 0.6, 6: 0.3, 7: 0.8}


def tm(x, key):
    y = x * key
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


def lum(x):
    return x.mean(-1)


def sq(x, vmax):
    return np.sign(x) * np.sqrt(np.abs(x) / max(vmax, 1e-30))


def load_ref(R, v):
    d = {}
    for f in sorted(glob.glob(os.path.join(R, 'results', 'renders', f'REF_ER_CK_{v}_*_r*.npz'))):
        z = np.load(f)
        for k in ('e1', 'e2', 'h'):
            d.setdefault(k, []).append(z[k].astype(np.float64))
    d = {k: np.stack(x) for k, x in d.items()}
    d['tot'] = d['e1'] + d['e2'] + d['h']
    return d


def load_hl(R, v, sd, a):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', f'CK_{v}_own_s{sd}_{a}_*_r*.npz')))
    return np.stack([np.load(f)['hl'].astype(np.float64) for f in fns]) if fns else None


def eh(h_reps, x_reps, tot, m):
    n, k = h_reps.shape[0], x_reps.shape[0]
    h = h_reps.mean(0); x = x_reps.mean(0)
    den = max(tot[m].sum(), 1e-30)
    o = {'E_H': float(np.abs(x[m] - h[m]).sum() / den), 'signed': float((x[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den), 'pixels': int(m.sum())}
    if n >= 2 and k >= 2:
        o['noise_bound'] = float((0.5 * np.abs(x_reps[:k // 2].mean(0)[m] - x_reps[k // 2:].mean(0)[m]).sum()
                                  + 0.5 * np.abs(h_reps[:n // 2].mean(0)[m] - h_reps[n // 2:].mean(0)[m]).sum()) / den)
    return o


def response(dref, darm, m, cells, dref_a, dref_b):
    a, b = lum(darm)[m], lum(dref)[m]
    o = {'pixels': int(m.sum()), 'ratio_sum': float(a.sum() / b.sum()) if abs(b.sum()) > 0 else None,
         'slope': float((a * b).sum() / max((b * b).sum(), 1e-30)), 'corr': float(np.corrcoef(a, b)[0, 1]) if m.sum() > 2 else None,
         'rel_err': float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30)),
         'ref_half_noise_rel': float(0.5 * np.abs(lum(dref_a)[m] - lum(dref_b)[m]).sum() / max(np.abs(b).sum(), 1e-30))}
    cs = cells[m]; u = np.unique(cs[cs >= 0])
    ca = np.array([a[cs == c].mean() for c in u]); cb = np.array([b[cs == c].mean() for c in u])
    o['cells'] = int(len(u))
    if len(u) > 2:
        o['cell_slope'] = float((ca * cb).sum() / max((cb * cb).sum(), 1e-30)); o['cell_corr'] = float(np.corrcoef(ca, cb)[0, 1])
    o['_cell_vals'] = (u.tolist(), ca.tolist(), cb.tolist())
    return o


def main(R, out):
    os.makedirs(out, exist_ok=True)
    pre = json.load(open(os.path.join(R, 'results', 'pre', 'pre.json')))
    Mz = np.load(os.path.join(R, 'results', 'pre', 'masks_ck2.npz'))
    M = {'ROI_T': Mz['ROI_T'], 'panel': Mz['panel'], 'target_direct': Mz['target_direct'], 'frame': Mz['frame']}
    mpatch = Mz['mirror_patch']; dpatch = np.where(Mz['target_direct'], Mz['pid'], -1)
    U = load_ref(R, 'none'); n = U['h'].shape[0]
    tot_u = U['tot'].mean(0); h_u = U['h'].mean(0); low_u = tot_u - h_u
    key = 0.5 / float(np.median(lum(tot_u)[lum(tot_u) > 0])); keyh = 0.5 / float(np.median(lum(h_u)[lum(h_u) > 0]))
    ys, xs = np.nonzero(M['ROI_T']); pad = 20
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    T = {v: load_ref(R, v) for v, _ in VARS}
    AT = {(v, sd, a): load_hl(R, v, sd, a) for v, _ in VARS for sd in SEEDS for a, _ in ARMS}
    AU = {(sd, a): load_hl(R, 'none', sd, a) for sd in SEEDS for a, _ in ARMS}
    num = {'exposure_key_untextured_median': key, 'reference_realizations': n, 'regions': {k: int(m.sum()) for k, m in M.items()},
           'scene': pre['scene'], 'variants': {}}
    corrmaps = {}
    for v, _ in VARS:
        tv = T[v]; tot = tv['tot'].mean(0); h = tv['h'].mean(0); low = tot - h
        pm = Mz[f'PMAT_{v}']
        mid = np.where(mpatch >= 0, pm[np.clip(mpatch, 0, len(pm) - 1)], -1)
        pred = np.where(mid >= 0, np.vectorize(lambda m: ALB.get(int(m), 0.5))(np.where(mid >= 0, mid, 4)) / 0.5, np.nan)
        obs = lum(low) / np.maximum(lum(low_u), 1e-12)
        r = M['ROI_T']
        lp, lo_ = np.log(pred[r]), np.log(obs[r])
        corr = {'pixel_corr_log_ratio': float(np.corrcoef(lp, lo_)[0, 1]), 'slope_log_ratio': float((lo_ - lo_.mean()) @ (lp - lp.mean()) / max(((lp - lp.mean()) ** 2).sum(), 1e-30))}
        ids = (3, 4) if v == 'bx75' else (6, 7)
        parity = {}
        for nm, xt, xu in (('REF_LOW', low, low_u), ('REF_h', h, h_u)):
            q = lum(xt) / np.maximum(lum(xu), 1e-12)
            parity[nm] = {f'id{i}': float(np.median(q[r & (mid == i)])) for i in ids}
        parity['ideal_mirror'] = {f'id{i}': ALB[i] / 0.5 for i in ids}
        corrmaps[v] = (pred, obs)
        dref = h - h_u
        dA = tv['h'][:n // 2].mean(0) - U['h'][:n // 2].mean(0); dB = tv['h'][n // 2:].mean(0) - U['h'][n // 2:].mean(0)
        V = {'reflection_correspondence_ROI_T (ideal-mirror predicted albedo ratio vs reference LOW ratio, log)': corr,
             'reflection_ratio_by_mirror_cell_material (median, textured / untextured)': parity, 'arms': {}}
        for sd in SEEDS:
            for a, _ in ARMS:
                xt = AT[(v, sd, a)]; xu = AU[(sd, a)]
                if xt is None or xu is None:
                    continue
                q = lum(xt.mean(0)) / np.maximum(lum(xu.mean(0)), 1e-12)
                parity[f's{sd}:{a}_hl'] = {f'id{i}': float(np.median(q[r & (mid == i)])) for i in ids}
                darm = xt.mean(0) - xu.mean(0)
                V['arms'][f's{sd}:{a}'] = {'E_H_textured': {k: eh(tv['h'], xt, tot, m) for k, m in M.items()},
                                           'E_H_untextured': {k: eh(U['h'], xu, tot_u, m) for k, m in M.items()},
                                           'texture_response': {'ROI_T': response(dref, darm, M['ROI_T'], mpatch, dA, dB),
                                                                'target_direct': response(dref, darm, M['target_direct'], dpatch, dA, dB)}}
        num['variants'][v] = V
    cellvals = {}
    for v in num['variants']:
        for k, r in num['variants'][v]['arms'].items():
            for reg in ('ROI_T', 'target_direct'):
                cellvals[(v, k, reg)] = r['texture_response'][reg].pop('_cell_vals', None)
    err_max, dif_max = 0.0, 0.0
    for v, _ in VARS:
        h = T[v]['h'].mean(0)
        dif_max = max(dif_max, float(np.abs(lum(h - h_u)[M['panel']]).max()))
        for sd in SEEDS:
            for a, _ in ARMS:
                xt = AT[(v, sd, a)]; xu = AU[(sd, a)]
                if xt is None:
                    continue
                err_max = max(err_max, float(np.abs(lum(xt.mean(0) - h)).max()))
                if xu is not None:
                    dif_max = max(dif_max, float(np.abs(lum(xt.mean(0) - xu.mean(0))[M['panel']]).max()))
    for sd in SEEDS:
        for a, _ in ARMS:
            if AU[(sd, a)] is not None:
                err_max = max(err_max, float(np.abs(lum(AU[(sd, a)].mean(0) - h_u)).max()))
    num['scales'] = {'higher_order_error_max_abs': err_max, 'texture_response_max_abs_panel_pixels': dif_max}
    json.dump(num, open(os.path.join(out, 'metrics_checker93d.json'), 'w'), indent=1)
    made = ['metrics_checker93d.json']
    # ------------------------------------------------------------------ fig 0: references and reflection correspondence
    fig, ax = plt.subplots(4, 3, figsize=(13, 17))
    cols = [('untextured reference', U, None)] + [(lab + '\nreference', T[v], v) for v, lab in VARS]
    for j, (lab, d, v) in enumerate(cols):
        tot = d['tot'].mean(0); low = tot - d['h'].mean(0)
        ax[0, j].imshow(tm(tot, key)); ax[0, j].contour(M['ROI_T'], levels=[0.5], colors='c', linewidths=0.6); ax[0, j].set_title(lab + '\nfull frame (cyan: ROI_T)', fontsize=8)
        ax[1, j].imshow(tm(low[sl], key)); ax[1, j].set_title('panel reflection close-up, events 1-2 (LOW)', fontsize=8)
        ax[2, j].imshow(tm(d['h'].mean(0)[sl], keyh)); ax[2, j].set_title('close-up, higher-order term only (events >= 3)', fontsize=8)
        if v is None:
            ax[3, j].axis('off')
            ax[3, j].text(0.02, 0.5, 'Row 4: reflection correspondence.\nLeft half of each panel: albedo ratio predicted\nby an ideal mirror (cell hit by the mirror ray).\nRight half: reference LOW, textured / untextured.\nSame log colour scale.', fontsize=9, va='center')
        else:
            pred, obs = corrmaps[v]
            c = num['variants'][v]['reflection_correspondence_ROI_T (ideal-mirror predicted albedo ratio vs reference LOW ratio, log)']
            pc, oc = np.log(pred[sl]), np.log(obs[sl])
            oc = np.where(M['ROI_T'][sl], oc, np.nan)
            w = pc.shape[1]
            both = np.concatenate([pc, np.full((pc.shape[0], 4), np.nan), oc], 1)
            im = ax[3, j].imshow(both, cmap='coolwarm', vmin=-1.0, vmax=1.0)
            ax[3, j].set_title(f'ideal-mirror prediction | reference (log ratio)\npixel corr {c["pixel_corr_log_ratio"]:.3f}, slope {c["slope_log_ratio"]:.2f}', fontsize=8)
            fig.colorbar(im, ax=ax[3, j], fraction=0.04, pad=0.01, label='log(textured / untextured)')
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('WL93D checkerboard addendum rev. 2 (POST HOC, QUALITATIVE): physical references. Checker = per-patch material ids on box A (3 x 3 patches per face), '
                 f'reflected by the sharp panel S1. REF-ER {n} x 8,192 spp, common random numbers between variants. One exposure (untextured median).', fontsize=9, wrap=True)
    fn = 'figCK0_scene_references.png'; fig.savefig(os.path.join(out, fn), dpi=110, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ------------------------------------------------------------------ fig 1: comparison sheets
    for v, vlab in VARS:
        tv = T[v]; tot = tv['tot'].mean(0); h = tv['h'].mean(0); low = tot - h
        for sd in SEEDS:
            cols = [('reference (REF-ER)', None)] + [(lab, a) for a, lab in ARMS if AT[(v, sd, a)] is not None]
            fig, ax = plt.subplots(6, len(cols), figsize=(3.4 * len(cols), 20))
            for j, (lab, a) in enumerate(cols):
                if a is None:
                    full, ho, err = tot, h, 0.5 * (tv['h'][:n // 2].mean(0) - tv['h'][n // 2:].mean(0))
                    dresp = h - h_u; t2 = 'reference half-split difference / 2 (noise)'; e_ = None
                else:
                    x = AT[(v, sd, a)].mean(0); full, ho, err = low + x, x, x - h
                    dresp = x - AU[(sd, a)].mean(0) if AU[(sd, a)] is not None else np.zeros_like(x)
                    t2 = 'arm higher-order - reference higher-order'
                    e_ = num['variants'][v]['arms'].get(f's{sd}:{a}')
                ttl = lab
                if e_ is not None:
                    e = e_['E_H_textured']; rr = e_['texture_response']['ROI_T']
                    ttl += f"\nE_H ROI_T {e['ROI_T']['E_H']:.4f} (untex. {e_['E_H_untextured']['ROI_T']['E_H']:.4f})\nROI_T response slope {rr['slope']:.2f} corr {rr['corr']:.2f}"
                ax[0, j].imshow(tm(full, key)); ax[0, j].set_title(ttl, fontsize=7)
                ax[1, j].imshow(tm(ho, keyh)); ax[1, j].set_title('higher-order term only', fontsize=7)
                ax[2, j].imshow(tm(full[sl], key)); ax[2, j].contour(M['ROI_T'][sl], levels=[0.5], colors='c', linewidths=0.6)
                ax[2, j].set_title('panel reflection close-up (cyan ROI_T)', fontsize=7)
                im = ax[3, j].imshow(sq(lum(err), err_max), cmap='RdBu_r', vmin=-1, vmax=1); ax[3, j].set_title(t2, fontsize=7)
                ax[4, j].imshow(sq(lum(err)[sl], err_max), cmap='RdBu_r', vmin=-1, vmax=1); ax[4, j].contour(M['ROI_T'][sl], levels=[0.5], colors='k', linewidths=0.5)
                ax[4, j].set_title('same, close-up', fontsize=7)
                im2 = ax[5, j].imshow(sq(lum(dresp)[sl], dif_max), cmap='PuOr_r', vmin=-1, vmax=1); ax[5, j].contour(M['ROI_T'][sl], levels=[0.5], colors='k', linewidths=0.5)
                ax[5, j].set_title('texture response of the higher-order term\n(textured - untextured, CRN), close-up', fontsize=7)
            for a_ in ax.ravel():
                a_.set_xticks([]); a_.set_yticks([])
            for row, (imx, vm) in ((3, (im, err_max)), (5, (im2, dif_max))):
                cb = fig.colorbar(imx, ax=list(ax[row]), fraction=0.02, pad=0.01)
                tk = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1]); cb.set_ticks(tk); cb.set_ticklabels([f'{np.sign(t) * t * t * vm:.3g}' for t in tk])
                cb.set_label('radiance (signed sqrt; shared over the addendum' + ('; set by panel pixels)' if row == 5 else ')'), fontsize=7)
            fig.suptitle(f'WL93D checkerboard addendum rev. 2 (POST HOC, QUALITATIVE; material-distribution test), {vlab}, WL92A own s{sd} (frozen). '
                         f'Full frame = reference LOW of this variant (events 1-2, identical for all arms) + each arm\'s higher-order term (2 x 4,096 spp). '
                         f'One exposure for the whole addendum; error and response maps on one absolute scale each.', fontsize=9, wrap=True)
            fn = f'figCK1_{v}_own_s{sd}_sheet.png'; fig.savefig(os.path.join(out, fn), dpi=100, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ------------------------------------------------------------------ fig 2: per-patch texture response
    fig, ax = plt.subplots(len(VARS), 2, figsize=(11, 5 * len(VARS)))
    mk = {'C0': 'o', 'C2_src': 's', 'C2_rcv': '^', 'C2_both': 'D'}
    for i, (v, vlab) in enumerate(VARS):
        for j, reg in enumerate(('ROI_T', 'target_direct')):
            a_ = ax[i, j]; lim = 0.0
            for sd, col in zip(SEEDS, ('tab:blue', 'tab:orange')):
                for a, _ in ARMS:
                    cv = cellvals.get((v, f's{sd}:{a}', reg))
                    if not cv:
                        continue
                    _, ca, cb = cv
                    a_.scatter(cb, ca, s=12, marker=mk[a], facecolors='none' if sd == 8602 else col, edgecolors=col, label=f'{a} s{sd}', alpha=0.8)
                    lim = max(lim, np.abs(ca).max(), np.abs(cb).max())
            a_.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7); a_.axhline(0, color='0.7', lw=0.5); a_.axvline(0, color='0.7', lw=0.5)
            a_.set_xlabel('reference texture response (patch mean, luminance)'); a_.set_ylabel('arm texture response (patch mean)')
            a_.set_title(f'{vlab}\n{reg}: ' + ('panel pixels whose mirror ray hits a checker patch (grouped by that patch)' if reg == 'ROI_T' else 'checker box seen directly (grouped by patch)'), fontsize=8)
            if i == 0 and j == 0:
                a_.legend(fontsize=6, ncol=2)
    fig.suptitle('Texture response of the higher-order term per checker patch (textured - untextured, common random numbers); dashed = identity. POST HOC, descriptive.', fontsize=9)
    fn = 'figCK2_cell_response.png'; fig.savefig(os.path.join(out, fn), dpi=110, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'class': 'WL93D POST-HOC QUALITATIVE ADDENDUM (checkerboard), revision 2; not WL93D frozen evidence', 'preflight_status': pre['status'],
           'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_local93d/addendum_checker/viz_checker93d.py (rev. 2) from results/ecro_local93d_checker renders of scene_ck2.json'}
    json.dump(man, open(os.path.join(out, 'manifest_checker93d.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)
