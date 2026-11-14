"""WL93D checkerboard addendum: comparison sheets and descriptive numbers (post hoc, qualitative; no labels).

Full frame of an arm = the TEXTURED reference LOW (physical camera-path events 1-2 = REF tot - REF h, identical for every arm) + the arm's
rendered higher-order term hl. Exposure for every image: key from the UNTEXTURED reference median (one exposure for the whole addendum),
Reinhard, gamma 2.2. Error / difference maps: signed sqrt mapping with one absolute scale per map type over the whole addendum
(max |value| over every panel of that type) and a colourbar in real units.
"""
import glob, hashlib, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARS = [('ck38', 'checker 0.3 / 0.8, 25 cm cells (new Lambert states)'), ('ck75', 'checker 0.7 / 0.5, 25 cm cells (existing ids 3 / 4)'),
        ('ck38L', 'checker 0.3 / 0.8, 1 m cells = 4 x 4 patches (post hoc A4)')]
BLOCK = {'ck38L': 4}
ARMS = [('C0', 'deployed (C0)'), ('C2_src', 'M64 at source use (C2_src)'), ('C2_rcv', 'M64 at receiver use (C2_rcv)'), ('C2_both', 'M64 at both uses (C2_both)')]
SEEDS = [8601, 8602]
RB_WALL = 512                       # first patch id of the back wall (floor 256 + ceiling 256)


def tm(x, key):
    y = x * key
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


def lum(x):
    return x.mean(-1)


def sq(x, vmax):
    return np.sign(x) * np.sqrt(np.abs(x) / max(vmax, 1e-30))


def load_ref(root, pat, reps):
    d = {}
    for r in reps:
        z = np.load(os.path.join(root, 'results', 'renders', f'{pat}_r{r}.npz'))
        for k in ('tot', 'e1', 'e2', 'h'):
            if k in z.files:
                d.setdefault(k, []).append(z[k].astype(np.float64))
    d = {k: np.stack(v) for k, v in d.items()}
    if 'tot' not in d:
        d['tot'] = d['e1'] + d['e2'] + d['h']
    return d


def load_hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
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


def response(dref, darm, m, cells=None, dref_a=None, dref_b=None):
    """Texture response (textured - untextured higher-order term), luminance; pixel level and cell level."""
    a, b = lum(darm)[m], lum(dref)[m]
    o = {'pixels': int(m.sum()), 'ratio_sum': float(a.sum() / b.sum()) if abs(b.sum()) > 0 else None,
         'slope': float((a * b).sum() / max((b * b).sum(), 1e-30)), 'corr': float(np.corrcoef(a, b)[0, 1]) if m.sum() > 2 else None,
         'rel_err': float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))}
    if dref_a is not None:
        o['ref_half_noise_rel'] = float(0.5 * np.abs(lum(dref_a)[m] - lum(dref_b)[m]).sum() / max(np.abs(b).sum(), 1e-30))
    if cells is not None:
        cs = cells[m]; u = np.unique(cs[cs >= 0])
        ca = np.array([a[cs == c].mean() for c in u]); cb = np.array([b[cs == c].mean() for c in u])
        o['cells'] = int(len(u))
        if len(u) > 2:
            o['cell_slope'] = float((ca * cb).sum() / max((cb * cb).sum(), 1e-30)); o['cell_corr'] = float(np.corrcoef(ca, cb)[0, 1])
        o['_cell_vals'] = (u.tolist(), ca.tolist(), cb.tolist())
    return o


def main(R, R86, R93D, out):
    os.makedirs(out, exist_ok=True)
    pre = json.load(open(os.path.join(R, 'results', 'pre', 'pre.json')))
    Mz = np.load(os.path.join(R86, 'results', 'masks', 'E2.npz'))
    Pk = np.load(os.path.join(R, 'results', 'pre', 'masks_ck.npz'))
    M = {'C': Mz['C'], 'obj_panel': Mz['obj_panel'], 'ROI_W': Pk['ROI_W'], 'wall_direct': (Mz['sid'] == 2) & Mz['frame'], 'stationary_all': Mz['stationary_all']}
    cell_mirror = Pk['mirror_cell']; cell_wall = np.where(Mz['sid'] == 2, Mz['pid'] - RB_WALL, -1)
    reps = sorted(int(os.path.basename(f).split('_r')[-1][:-4]) for f in glob.glob(os.path.join(R, 'results', 'renders', 'REF_ER_CK_ck38_E2_T1_r*.npz')))
    U = load_ref(R86, 'REF_ER_E2_T1', reps)
    tot_u = U['tot'].mean(0); h_u = U['h'].mean(0)
    key = 0.5 / float(np.median(lum(tot_u)[lum(tot_u) > 0])); keyh = 0.5 / float(np.median(lum(h_u)[lum(h_u) > 0]))
    ys, xs = np.nonzero(M['obj_panel']); pad = 20
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    VARS_ = [(v, l) for v, l in VARS if glob.glob(os.path.join(R, 'results', 'renders', f'REF_ER_CK_{v}_E2_T1_r*.npz'))]
    T = {v: load_ref(R, f'REF_ER_CK_{v}_E2_T1', reps) for v, _ in VARS_}
    AT, AU = {}, {}
    for v, _ in VARS_:
        for sd in SEEDS:
            for a, _ in ARMS:
                AT[(v, sd, a)] = load_hl(R, f'CK_{v}_own_s{sd}_{a}_E2_T1')
    for sd in SEEDS:
        for a, _ in ARMS:
            AU[(sd, a)] = load_hl(R93D, f'I93D_own_s{sd}_{a}_E2_T1')
    # ------------------------------------------------------------------ numbers
    num = {'exposure_key_untextured_median': key, 'reference_realizations': reps, 'regions': {k: int(m.sum()) for k, m in M.items()}, 'variants': {}}
    for v, _ in VARS_:
        tv = T[v]; tot = tv['tot'].mean(0); h = tv['h'].mean(0)
        dref = h - h_u
        n = len(reps); dA = tv['h'][:n // 2].mean(0) - U['h'][:n // 2].mean(0); dB = tv['h'][n // 2:].mean(0) - U['h'][n // 2:].mean(0)
        low_chg = {k: float(np.abs(lum((tot - h) - (tot_u - h_u)))[m].sum() / max(lum(tot_u - h_u)[m].sum(), 1e-30)) for k, m in M.items()}
        kb = BLOCK.get(v, 1); cm = cell_mirror; parm = np.where(cm >= 0, ((cm % 16) // kb + (cm // 16) // kb) % 2, -1)
        roi = M['ROI_W']
        pr = lambda xt, xu: {f'{nm}': float(np.median((lum(xt) / np.maximum(lum(xu), 1e-12))[roi & (parm == q)])) for q, nm in ((0, 'even'), (1, 'odd'))}
        parity = {'REF_e2': pr(tv['e2'].mean(0), U['e2'].mean(0)), 'REF_h': pr(h, h_u), 'REF_LOW': pr(tot - h, tot_u - h_u),
                  'ideal_mirror_albedo_ratio': {'even': 0.3 / 0.7 if v != 'ck75' else 1.0, 'odd': 0.8 / 0.7 if v != 'ck75' else 0.5 / 0.7}}
        V = {'reflection_parity_ratio_ROI_W (textured / untextured, median per checker parity of the mirror cell)': parity,
             'LOW_change_rel_L1_vs_untextured': low_chg, 'H_change_rel_L1_vs_untextured': {k: float(np.abs(lum(dref))[m].sum() / max(lum(h_u)[m].sum(), 1e-30)) for k, m in M.items()},
             'arms': {}}
        for sd in SEEDS:
            for a, _ in ARMS:
                xt = AT[(v, sd, a)]; xu = AU[(sd, a)]
                if xt is None or xu is None:
                    continue
                darm = xt.mean(0) - xu.mean(0)
                parity[f's{sd}:{a}_hl'] = pr(xt.mean(0), xu.mean(0))
                r = {'E_H_textured': {k: eh(tv['h'], xt, tot, m) for k, m in M.items()},
                     'E_H_untextured_same_pixels': {k: eh(U['h'], xu, tot_u, m) for k, m in M.items()},
                     'texture_response': {'ROI_W': response(dref, darm, M['ROI_W'], cell_mirror, dA, dB),
                                          'wall_direct': response(dref, darm, M['wall_direct'], cell_wall, dA, dB)}}
                V['arms'][f's{sd}:{a}'] = r
        num['variants'][v] = V
    cellvals = {}
    for v in num['variants']:
        for k, r in num['variants'][v]['arms'].items():
            for reg in ('ROI_W', 'wall_direct'):
                cellvals[(v, k, reg)] = r['texture_response'][reg].pop('_cell_vals', None)
    json.dump(num, open(os.path.join(out, 'metrics_checker93d.json'), 'w'), indent=1)
    made = ['metrics_checker93d.json']
    # ------------------------------------------------------------------ fig 0: the scene (references, LOW, mirror ROI)
    fig, ax = plt.subplots(3, len(VARS_) + 1, figsize=(4 * (len(VARS_) + 1), 12))
    cols = [('untextured reference (WL86 REF-ER)', U)] + [(lab + '\nreference (REF-ER)', T[v]) for v, lab in VARS_]
    for j, (lab, d) in enumerate(cols):
        tot = d['tot'].mean(0); low = tot - d['h'].mean(0)
        ax[0, j].imshow(tm(tot, key)); ax[0, j].set_title(lab + '\nfull frame', fontsize=8)
        ax[1, j].imshow(tm(tot[sl], key)); ax[1, j].contour(M['ROI_W'][sl], levels=[0.5], colors='c', linewidths=0.6)
        ax[1, j].contour(M['C'][sl], levels=[0.5], colors='y', linewidths=0.5)
        ax[1, j].set_title('panel close-up\ncyan: ROI_W (mirror ray hits back wall)\nyellow: region C', fontsize=7)
        ax[2, j].imshow(tm(d['h'].mean(0)[sl], keyh)); ax[2, j].set_title('panel close-up, higher-order term only (events >= 3)', fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'WL93D checkerboard addendum (POST HOC, QUALITATIVE), E2 T1: physical references. Back-wall checker = per-patch material ids '
                 f'(25 cm cells = patches). REF-ER {len(reps)} x 8,192 spp, seeds 86100+r (common random numbers with the untextured reference). '
                 f'One exposure (untextured reference median, Reinhard, gamma 2.2).', fontsize=9, wrap=True)
    fn = 'figCK0_scene_references.png'; fig.savefig(os.path.join(out, fn), dpi=110, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ------------------------------------------------------------------ shared absolute scales
    err_max, dif_max = 0.0, 0.0
    for v, _ in VARS_:
        h = T[v]['h'].mean(0)
        for sd in SEEDS:
            for a, _ in ARMS:
                xt = AT[(v, sd, a)]; xu = AU[(sd, a)]
                if xt is None:
                    continue
                err_max = max(err_max, float(np.abs(lum(xt.mean(0) - h)).max()))
                if xu is not None:
                    dif_max = max(dif_max, float(np.abs(lum(xt.mean(0) - xu.mean(0))[M['obj_panel']]).max()))
        dif_max = max(dif_max, float(np.abs(lum(h - h_u)[M['obj_panel']]).max()))
    num['scales'] = {'higher_order_error_max_abs': err_max, 'texture_response_max_abs_panel_pixels': dif_max}
    json.dump(num, open(os.path.join(out, 'metrics_checker93d.json'), 'w'), indent=1)
    # ------------------------------------------------------------------ fig 1: comparison sheets
    for v, vlab in VARS_:
        tv = T[v]; tot = tv['tot'].mean(0); h = tv['h'].mean(0); low = tot - h; n = len(reps)
        for sd in SEEDS:
            cols = [('reference (REF-ER)', None)] + [(lab, a) for a, lab in ARMS if AT[(v, sd, a)] is not None]
            fig, ax = plt.subplots(6, len(cols), figsize=(3.2 * len(cols), 19))
            for j, (lab, a) in enumerate(cols):
                if a is None:
                    full, ho, err = tot, h, 0.5 * (tv['h'][:n // 2].mean(0) - tv['h'][n // 2:].mean(0))
                    dresp = h - h_u; t2 = 'reference half-split difference / 2 (noise)'; eh_ = None
                else:
                    x = AT[(v, sd, a)].mean(0); full, ho, err = low + x, x, x - h
                    dresp = x - AU[(sd, a)].mean(0) if AU[(sd, a)] is not None else np.zeros_like(x)
                    t2 = 'arm higher-order - reference higher-order'
                    eh_ = num['variants'][v]['arms'][f's{sd}:{a}']
                ttl = lab
                if eh_ is not None:
                    e = eh_['E_H_textured']; rr = eh_['texture_response']['ROI_W']
                    ttl += f"\nE_H  C {e['C']['E_H']:.4f}  ROI_W {e['ROI_W']['E_H']:.4f}\nROI_W response slope {rr['slope']:.2f}  corr {rr['corr']:.2f}"
                ax[0, j].imshow(tm(full, key)); ax[0, j].set_title(ttl, fontsize=7)
                ax[1, j].imshow(tm(ho, keyh)); ax[1, j].set_title('higher-order term only', fontsize=7)
                ax[2, j].imshow(tm(full[sl], key)); ax[2, j].contour(M['ROI_W'][sl], levels=[0.5], colors='c', linewidths=0.6)
                ax[2, j].contour(M['C'][sl], levels=[0.5], colors='y', linewidths=0.5); ax[2, j].set_title('panel close-up (cyan ROI_W, yellow C)', fontsize=7)
                im = ax[3, j].imshow(sq(lum(err), err_max), cmap='RdBu_r', vmin=-1, vmax=1); ax[3, j].set_title(t2, fontsize=7)
                ax[4, j].imshow(sq(lum(err)[sl], err_max), cmap='RdBu_r', vmin=-1, vmax=1); ax[4, j].contour(M['ROI_W'][sl], levels=[0.5], colors='k', linewidths=0.5)
                ax[4, j].set_title('same, panel close-up', fontsize=7)
                im2 = ax[5, j].imshow(sq(lum(dresp)[sl], dif_max), cmap='PuOr_r', vmin=-1, vmax=1); ax[5, j].contour(M['ROI_W'][sl], levels=[0.5], colors='k', linewidths=0.5)
                ax[5, j].set_title('texture response of the higher-order term\n(textured - untextured, CRN), close-up', fontsize=7)
            for a_ in ax.ravel():
                a_.set_xticks([]); a_.set_yticks([])
            for row, (imx, vm) in ((3, (im, err_max)), (5, (im2, dif_max))):
                cb = fig.colorbar(imx, ax=list(ax[row]), fraction=0.02, pad=0.01)
                tk = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1]); cb.set_ticks(tk); cb.set_ticklabels([f'{np.sign(t) * t * t * vm:.3g}' for t in tk])
                cb.set_label('radiance (signed sqrt; shared over the addendum' + ('; set by panel pixels, wall saturates)' if row == 5 else ')'), fontsize=7)
            fig.suptitle(f'WL93D checkerboard addendum (POST HOC, QUALITATIVE; material-distribution test), E2 T1, {vlab}, WL92A own s{sd} (frozen). '
                         f'Full frame = textured reference LOW (events 1-2, identical for all arms) + each arm\'s higher-order term (2 x 4,096 spp). '
                         f'One exposure for the whole addendum; error and response maps on one absolute scale each.', fontsize=9, wrap=True)
            fn = f'figCK1_{v}_own_s{sd}_sheet.png'; fig.savefig(os.path.join(out, fn), dpi=100, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ------------------------------------------------------------------ fig 2: per-cell texture response
    fig, ax = plt.subplots(len(VARS_), 2, figsize=(11, 5 * len(VARS_)))
    mk = {'C0': 'o', 'C2_src': 's', 'C2_rcv': '^', 'C2_both': 'D'}
    for i, (v, vlab) in enumerate(VARS_):
        for j, reg in enumerate(('ROI_W', 'wall_direct')):
            a_ = ax[i, j]; lim = 0.0
            for sd, col in zip(SEEDS, ('tab:blue', 'tab:orange')):
                for a, _ in ARMS:
                    cv = cellvals.get((v, f's{sd}:{a}', reg))
                    if not cv:
                        continue
                    _, ca, cb = cv
                    a_.scatter(cb, ca, s=10, marker=mk[a], facecolors='none' if sd == 8602 else col, edgecolors=col, label=f'{a} s{sd}', alpha=0.8)
                    lim = max(lim, np.abs(ca).max(), np.abs(cb).max())
            a_.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7); a_.axhline(0, color='0.7', lw=0.5); a_.axvline(0, color='0.7', lw=0.5)
            a_.set_xlabel('reference texture response (cell mean, luminance)'); a_.set_ylabel('arm texture response (cell mean)')
            a_.set_title(f'{vlab}\n{reg}: ' + ('panel pixels whose mirror ray hits a back-wall cell' if reg == 'ROI_W' else 'back wall seen directly'), fontsize=8)
            if i == 0 and j == 0:
                a_.legend(fontsize=6, ncol=2)
    fig.suptitle('Texture response of the higher-order term per checker cell (textured - untextured, common random numbers); dashed = identity. POST HOC, descriptive.', fontsize=9)
    fn = 'figCK2_cell_response.png'; fig.savefig(os.path.join(out, fn), dpi=110, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'class': 'WL93D POST-HOC QUALITATIVE ADDENDUM (checkerboard); not WL93D frozen evidence', 'preflight_status': pre['status'],
           'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_local93d/addendum_checker/viz_checker93d.py from results/ecro_local93d_checker renders + WL86 untextured E2 T1 references r0-r3 + WL93D untextured arm renders'}
    json.dump(man, open(os.path.join(out, 'manifest_checker93d.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)
