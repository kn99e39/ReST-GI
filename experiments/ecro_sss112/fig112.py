"""WL112 reviewer figures (reads analysis outputs only; not part of the evidence computation).

  python fig112.py ROOT OUTDIR
One absolute signed scale for every higher-order residual map of the worklog (sqrt display, colorbar in linear HDR units);
one exposure for every render. Readable file names.
"""
import glob, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402

LUMW = np.array([0.2126, 0.7152, 0.0722])
lum = lambda x: x @ LUMW
NAMES = {'U': 'uniform U', 'G': 'etendue G', 'V': 'receiver V', 'S_U': '2x2 uniform', 'S_G': '2x2 etendue', 'S_V': '2x2 receiver',
         'P_v': 'camera-fitted P (diag.)', 'S_v': 'camera-fitted 2x2 (diag.)', 'A_learned': 'original ECRO (WL100)', 'UN': 'uniform U, native semantics',
         'O_U': 'retrained ECRO, U targets', 'O_G': 'retrained ECRO, G targets', 'S2': 'candidate ECRO-S2', 'oracle': 'physical oracle', 'stale': 'stale table'}


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def crop_box(m, pad=24):
    mm = m.get('mask_near', None)
    tot = (m['mask_near'] | m['mask_flank']) if 'mask_flank' in m else m['mask_floor']
    if not tot.any():
        tot = m['mask_sphere']
    ys, xs = np.nonzero(tot | m['mask_sphere_boundary'])
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, 511), max(xs.min() - pad, 0), min(xs.max() + pad, 511)
    return slice(y0, y1 + 1), slice(x0, x1 + 1)


class Disp:
    def __init__(self, ref_mean):
        self.k = 0.18 / max(ref_mean, 1e-6)

    def render(self, x):
        return np.clip((1 - np.exp(-self.k * np.maximum(x, 0))) ** (1 / 2.2), 0, 1)


class Signed:
    def __init__(self, vmax):
        self.vmax = vmax

    def show(self, ax, r, title):
        v = np.sign(r) * np.sqrt(np.abs(r) / self.vmax)
        im = ax.imshow(v, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=8); ax.axis('off')
        return im

    def bar(self, fig, im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.025)
        t = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
        cb.set_ticks(t); cb.set_ticklabels([f'{np.sign(x) * x * x * self.vmax:+.3f}' for x in t])
        cb.set_label('signed higher-order luminance error (linear HDR)', fontsize=7)


def main(root, outdir):
    root = Path(root); out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    an = root / 'analysis'
    files = []
    mB = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_B_*.npz')))}
    mD = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_D_*.npz')))}
    mE = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_E_*.npz')))}
    allres = []
    for d in list(mB.values()) + list(mD.values()) + list(mE.values()):
        for k in d:
            if k.startswith('hl_'):
                allres.append(np.abs(lum(d[k] - d['ref_h'])).max())
    S = Signed(max(allres) if allres else 1.0)
    v0 = mB.get('V0')
    D = Disp(float(lum(v0['ref_e'] + v0['ref_h']).mean()) if v0 is not None else 0.5)

    def save(fig, name):
        p = out / name; fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig); files.append(p.name)

    # 1. original reference vs original ECRO
    if v0 is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(D.render(v0['ref_e'] + v0['ref_h'])); ax[0].set_title('F0 camera V0: physical reference (contact-valid semantics), 2x4096 spp')
        ax[1].imshow(D.render(v0['ref_e'] + v0['hl_A_learned'])); ax[1].set_title('original ECRO (WL100 checkpoint), raw 4096 spp')
        im = S.show(ax[2], lum(v0['hl_A_learned'] - v0['ref_h']), 'original ECRO minus reference, higher order')
        S.bar(fig, im, ax[2]); [a.axis('off') for a in ax[:2]]
        save(fig, 'F0_V0_reference_vs_original_ECRO_render_and_higher_order_error.png')
    # 2-3. physical table arms, all views
    arms = ['U', 'G', 'S_U', 'S_G', 'P_v', 'S_v', 'A_learned']
    if mB:
        views = list(mB)
        for kind in ('full', 'crop'):
            fig, ax = plt.subplots(len(views), len(arms), figsize=(2.6 * len(arms), 2.6 * len(views)))
            for i, v in enumerate(views):
                m = mB[v]; sl = crop_box(m) if kind == 'crop' else (slice(None), slice(None))
                for j, a in enumerate(arms):
                    im = S.show(ax[i, j], lum(m[f'hl_{a}'] - m['ref_h'])[sl], f'{v}: {NAMES[a]}')
            S.bar(fig, im, ax[:, -1].tolist())
            save(fig, f'F0_physical_table_arms_signed_higher_order_error_{"contact_crops" if kind == "crop" else "full_frame"}_six_cameras.png')
    # 4. radial profiles + per-camera endpoint bars (Stage B)
    mp = an / 'stageB_metrics.json'
    if mp.exists():
        met = json.loads(mp.read_text())
        cvs = [v for v in met if met[v]['contact_valid']]
        fig, ax = plt.subplots(1, len(cvs), figsize=(4 * len(cvs), 3.4), squeeze=False)
        mids = [(a + b) / 2 for a, b in zip(C.EDGES, C.EDGES[1:])]
        for i, v in enumerate(cvs):
            for a in arms:
                ax[0, i].plot(mids, met[v]['arms'][a]['bands'], marker='o', ms=3, label=NAMES[a])
            ax[0, i].axhspan(-met[v]['band_near'], met[v]['band_near'], color='0.85'); ax[0, i].axhline(0, c='k', lw=.5)
            ax[0, i].set_title(f'{v}: floor radial HIGH excess'); ax[0, i].set_xlabel('distance from support point (m)')
        ax[0, 0].set_ylabel('signed luminance (linear HDR)'); ax[0, -1].legend(fontsize=6)
        save(fig, 'F0_floor_radial_higher_order_excess_physical_table_arms.png')
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
        for k, key in enumerate(('near', 'contrast')):
            w = 0.11
            for j, a in enumerate(arms):
                est = [met[v]['arms'][a]['endpoints'][key]['estimate'] for v in cvs]
                hw = [met[v]['arms'][a]['endpoints'][key]['half_width'] for v in cvs]
                ax[k].bar(np.arange(len(cvs)) + (j - 3) * w, est, w, yerr=hw, label=NAMES[a], capsize=2)
            bands = [met[v]['band_near' if key == 'near' else 'band_contrast'] for v in cvs]
            for i, bnd in enumerate(bands):
                ax[k].plot([i - .45, i + .45], [bnd, bnd], 'k--', lw=.8); ax[k].plot([i - .45, i + .45], [-bnd, -bnd], 'k--', lw=.8)
            ax[k].set_xticks(range(len(cvs))); ax[k].set_xticklabels(cvs); ax[k].axhline(0, c='k', lw=.5)
            ax[k].set_title('near-floor HIGH excess' if key == 'near' else 'near-minus-flank HIGH excess'); ax[k].set_ylabel('signed luminance')
        ax[1].legend(fontsize=6)
        save(fig, 'F0_contact_endpoints_per_camera_physical_table_arms.png')
    # 4b. POST HOC diagnostic: 4x4 resolution and queried-shape decomposition (labelled; not part of any decision)
    dg = an / 'diag112.json'
    if dg.exists() and mp.exists():
        dd = json.loads(dg.read_text()); met = json.loads(mp.read_text())
        vs = list(dd['b_resolution_4x4'])
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
        series = [('U', 'uniform U (1270x64)', 'B'), ('S_U', '2x2 uniform', 'B'), ('S_G', '2x2 etendue', 'B'),
                  ('S4_U', '4x4 uniform (post hoc)', 'D'), ('S4_G', '4x4 etendue (post hoc)', 'D')]
        for k, key in enumerate(('near', 'contrast')):
            w = 0.16
            for j, (a, lab, src) in enumerate(series):
                e = [(met[v]['arms'][a] if src == 'B' else dd['b_resolution_4x4'][v][a])['endpoints'][key] for v in vs]
                ax[k].bar(np.arange(len(vs)) + (j - 2) * w, [x['estimate'] for x in e], w, yerr=[x['half_width'] for x in e], capsize=2, label=lab)
            for i, v in enumerate(vs):
                bnd = met[v]['band_near' if key == 'near' else 'band_contrast']
                ax[k].plot([i - .45, i + .45], [bnd, bnd], 'k--', lw=.8); ax[k].plot([i - .45, i + .45], [-bnd, -bnd], 'k--', lw=.8)
            ax[k].set_xticks(range(len(vs))); ax[k].set_xticklabels(vs); ax[k].axhline(0, c='k', lw=.5)
            ax[k].set_title(('near-floor' if key == 'near' else 'near-minus-flank') + ' HIGH excess vs table resolution (dashed: band)')
        ax[1].legend(fontsize=7)
        save(fig, 'F0_post_hoc_resolution_2x2_vs_4x4_contact_endpoints.png')
        a = dd['a_shape_decomposition']
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
        labs = ['sphere_ring0', 'sphere_ring1', 'wall', 'ceiling', 'floor', 'boxA', 'boxB', 'panel']
        for k, r in enumerate(('near', 'flank')):
            for j, v in enumerate(vs):
                ax[k].bar(np.arange(len(labs)) + (j - 2) * 0.16, [a[v][r]['query_weight_share'].get(l, 0) for l in labs], 0.16, label=v)
            ax[k].set_xticks(range(len(labs))); ax[k].set_xticklabels(labs, rotation=30, fontsize=7)
            ax[k].set_title(f'share of {r}-floor HIGH query weight by queried shape')
        ax[1].legend(fontsize=7)
        save(fig, 'F0_queried_shape_share_of_near_and_flank_floor_higher_order.png')
    # 5. learned arms (Stage D)
    larms = [a for a in ['A_learned', 'O_U', 'O_G', 'S2', 'oracle'] if mD and f'hl_{a}' in next(iter(mD.values()))]
    if mD:
        tags = list(mD)
        fig, ax = plt.subplots(len(tags), len(larms) + 1, figsize=(2.6 * (len(larms) + 1), 2.6 * len(tags)))
        for i, t in enumerate(tags):
            m = mD[t]; sl = crop_box(m)
            ax[i, 0].imshow(D.render(m['ref_e'] + m['ref_h'])[sl]); ax[i, 0].set_title(f'{t}: reference', fontsize=8); ax[i, 0].axis('off')
            for j, a in enumerate(larms):
                im = S.show(ax[i, j + 1], lum(m[f'hl_{a}'] - m['ref_h'])[sl], f'{t}: {NAMES[a]}')
        S.bar(fig, im, ax[:, -1].tolist())
        save(fig, 'static_learned_arms_signed_higher_order_error_contact_crops_F0_F1.png')
        fig, ax = plt.subplots(len(tags), 3, figsize=(15, 4.2 * len(tags)))
        for i, t in enumerate(tags):
            m = mD[t]
            ax[i, 0].imshow(D.render(m['ref_e'] + m['ref_h'])); ax[i, 0].set_title(f'{t}: physical reference')
            ax[i, 1].imshow(D.render(m['ref_e'] + m.get('hl_S2', m['hl_O_G']))); ax[i, 1].set_title(f'{t}: learned candidate, raw 4096 spp')
            im = S.show(ax[i, 2], lum(m.get('hl_S2', m['hl_O_G']) - m['ref_h']), f'{t}: candidate minus reference, higher order')
            ax[i, 0].axis('off'); ax[i, 1].axis('off')
        S.bar(fig, im, ax[:, 2].tolist())
        save(fig, 'static_candidate_full_frame_render_and_error_per_camera.png')
    dp = an / 'stageD_metrics.json'
    if dp.exists():
        met = json.loads(dp.read_text())
        cvs = [v for v in met if met[v]['contact_valid']]
        fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
        for k, key in enumerate(('near', 'contrast')):
            w = 0.15
            for j, a in enumerate(larms):
                est = [met[v]['arms'][a]['endpoints'][key]['estimate'] for v in cvs]; hw = [met[v]['arms'][a]['endpoints'][key]['half_width'] for v in cvs]
                ax[k].bar(np.arange(len(cvs)) + (j - 2) * w, est, w, yerr=hw, label=NAMES[a], capsize=2)
            for i, v in enumerate(cvs):
                bnd = met[v]['band_near' if key == 'near' else 'band_contrast']
                ax[k].plot([i - .45, i + .45], [bnd, bnd], 'k--', lw=.8); ax[k].plot([i - .45, i + .45], [-bnd, -bnd], 'k--', lw=.8)
            ax[k].set_xticks(range(len(cvs))); ax[k].set_xticklabels(cvs, rotation=30, fontsize=7); ax[k].axhline(0, c='k', lw=.5)
            ax[k].set_title(('near-floor' if key == 'near' else 'near-minus-flank') + ' HIGH excess, learned arms')
        ax[1].legend(fontsize=7)
        save(fig, 'static_contact_endpoints_per_camera_learned_arms.png')
        fig, ax = plt.subplots(1, len(cvs), figsize=(3.6 * len(cvs), 3.2), squeeze=False)
        mids = [(a + b) / 2 for a, b in zip(C.EDGES, C.EDGES[1:])]
        for i, v in enumerate(cvs):
            for a in larms:
                ax[0, i].plot(mids, met[v]['arms'][a]['bands'], marker='o', ms=3, label=NAMES[a])
            ax[0, i].axhspan(-met[v]['band_near'], met[v]['band_near'], color='0.85'); ax[0, i].set_title(v, fontsize=8)
        ax[0, -1].legend(fontsize=6)
        save(fig, 'static_floor_radial_higher_order_excess_learned_arms.png')
    # 6. dynamic sequences
    ep = an / 'stageE_metrics.json'
    if ep.exists():
        E = json.loads(ep.read_text())
        for sname, seq in C.sequences().items():
            if sname not in E:
                continue
            v = seq['views'][0]
            n = len(seq['states']); cols = ['ref', 'S2' if 'hl_S2' in mE.get(f'{sname}_0_{v}', {}) else 'O_G', 'O_U', 'stale']
            fig, ax = plt.subplots(n, len(cols), figsize=(2.8 * len(cols), 2.6 * n))
            for st in range(n):
                m = mE[f'{sname}_{st}_{v}']; sl = crop_box(m)
                for j, a in enumerate(cols):
                    if a == 'ref':
                        ax[st, j].imshow(D.render(m['ref_e'] + m['ref_h'])[sl]); ax[st, j].set_title(f'{seq["label"][st]}: reference', fontsize=8); ax[st, j].axis('off')
                    elif f'hl_{a}' in m:
                        im = S.show(ax[st, j], lum(m[f'hl_{a}'] - m['ref_h'])[sl], f'{NAMES[a]}')
                    else:
                        ax[st, j].axis('off')
            S.bar(fig, im, ax[:, -1].tolist())
            save(fig, f'{sname}_dynamic_contact_sequence_{v}_reference_and_higher_order_error_crops.png')
            fig, ax = plt.subplots(1, len(seq['views']), figsize=(6 * len(seq['views']), 3.4), squeeze=False)
            for i, vv in enumerate(seq['views']):
                for a in ['A_learned', 'O_U', 'O_G', 'S2', 'oracle', 'stale']:
                    xs, ys, es = [], [], []
                    for st, row in enumerate(E[sname]['states']):
                        r = row['views'][vv]
                        if a in r['arms'] and 'endpoints' in r['arms'][a]:
                            xs.append(st); ys.append(r['arms'][a]['endpoints']['near']['estimate']); es.append(r['arms'][a]['endpoints']['near']['half_width'])
                    if xs:
                        ax[0, i].errorbar(xs, ys, yerr=es, marker='o', ms=3, capsize=2, label=NAMES[a])
                bn = [row['views'][vv]['band_near'] for row in E[sname]['states']]
                ax[0, i].fill_between(range(len(bn)), [-b for b in bn], bn, color='0.85')
                ax[0, i].set_xticks(range(len(bn))); ax[0, i].set_xticklabels(seq['label'], rotation=30, fontsize=7)
                ax[0, i].set_title(f'{sname} {vv}: near-floor HIGH excess per state'); ax[0, i].axhline(0, c='k', lw=.5)
            ax[0, -1].legend(fontsize=6)
            save(fig, f'{sname}_near_floor_higher_order_excess_per_state.png')
    # 7. LOW / HIGH attribution at F0 V0
    if v0 is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        ax[0].imshow(D.render(v0['ref_e'])); ax[0].set_title('physical LOW (events 1-2), shared by every arm'); ax[0].axis('off')
        ax[1].imshow(D.render(v0['ref_h'])); ax[1].set_title('physical HIGH (events 3-6)'); ax[1].axis('off')
        im = S.show(ax[2], lum(v0['hl_A_learned'] - v0['ref_h']), 'HIGH residual of the original ECRO (LOW residual is identically paired)')
        S.bar(fig, im, ax[2])
        save(fig, 'F0_V0_low_and_high_attribution.png')
    # 8. cost
    cp = an / 'cost.json'
    if cp.exists():
        c = json.loads(cp.read_text())['arms']
        names = list(c)
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
        ax[0].bar(names, [c[n]['table_mib_float32'] for n in names]); ax[0].set_title('HIGH table size (MiB float32)')
        ax[1].bar(names, [c[n]['generator']['median_ms'] for n in names]); ax[1].set_title('generator latency (ms, median)')
        ax[2].bar(names, [c[n].get('training_s', 0) / 60 for n in names]); ax[2].set_title('training time (min, 24k steps)')
        for a in ax:
            a.tick_params(axis='x', rotation=30, labelsize=7)
        save(fig, 'cost_table_size_generator_latency_training_time.png')
    # index + manifest
    rows = ''.join(f'<figure><img src="{f}" style="max-width:100%"><figcaption>{f}</figcaption></figure>\n' for f in files)
    gate = (an / 'gate.json').read_text() if (an / 'gate.json').exists() else '{}'
    (out / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>WL112 SSS near-contact gate</title>'
                                    f'<body style="font-family:sans-serif;max-width:1400px;margin:auto;padding:16px">'
                                    f'<h1>WL112 — SSS near-contact artifact resolution</h1><p>Signed error maps share one absolute scale '
                                    f'(max |error| {S.vmax:.4f} linear HDR luminance, sqrt display). Raw renders, no denoising.</p>'
                                    f'<pre>{gate}</pre>{rows}</body>', encoding='utf8')
    man = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files + ['index.html']}
    for j in ('stageA_gates.json', 'stageB_metrics.json', 'stageD_metrics.json', 'stageE_metrics.json', 'gate.json', 'cost.json'):
        if (an / j).exists():
            (out / j).write_bytes((an / j).read_bytes()); man[j] = hashlib.sha256((an / j).read_bytes()).hexdigest()
    for j in ('stageB/decision.json', 'stageC/precheck.json', 'tests112.json'):
        if (root / j).exists():
            nm = Path(j).name; (out / nm).write_bytes((root / j).read_bytes()); man[nm] = hashlib.sha256((root / j).read_bytes()).hexdigest()
    C.jd({'source_root': str(root), 'signed_scale_max': S.vmax, 'files_sha256': man}, out / 'manifest112.json')
    print('FIGURES', len(files))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
