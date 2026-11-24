"""WL113 reviewer figures (reads analysis outputs only; not part of the evidence computation).
One absolute signed scale for every higher-order residual map (sqrt display, real-unit colorbar); readable file names.

  python fig113.py ROOT OUTDIR
"""
import glob, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402

LUMW = np.array([0.2126, 0.7152, 0.0722]); lum = lambda x: x @ LUMW
NAMES = {'U': 'coarse uniform', 'S2': '2x2 uniform', 'S4': 'dense 4x4 uniform', 'L4': 'compact 4x4 (VHS+medium)', 'R0': 'lowest-ring-only 4x4 (diag.)',
         'A_learned': 'original ECRO (WL100)', 'O_U': 'retrained coarse ECRO', 'CAND': 'refined learned candidate', 'oracle': 'physical oracle (selected)',
         'oracle_U': 'physical coarse oracle', 'oracle_S4': 'physical dense 4x4 oracle', 'stale': 'stale candidate table'}


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def crop(m, pad=24):
    t = m['mask_near'] | m['mask_flank'] | m['mask_sphere_boundary']
    if not t.any():
        t = m['mask_sphere']
    ys, xs = np.nonzero(t)
    return slice(max(ys.min() - pad, 0), min(ys.max() + pad, 511) + 1), slice(max(xs.min() - pad, 0), min(xs.max() + pad, 511) + 1)


class Sg:
    def __init__(self, v):
        self.v = max(v, 1e-6)

    def show(self, ax, r, title):
        im = ax.imshow(np.sign(r) * np.sqrt(np.abs(r) / self.v), cmap='RdBu_r', vmin=-1, vmax=1); ax.set_title(title, fontsize=7); ax.axis('off'); return im

    def bar(self, fig, im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.025); t = np.array([-1, -.5, -.25, 0, .25, .5, 1]); cb.set_ticks(t)
        cb.set_ticklabels([f'{np.sign(x) * x * x * self.v:+.3f}' for x in t]); cb.set_label('signed higher-order luminance error (linear HDR)', fontsize=7)


def main(root, outdir):
    root = Path(root); out = Path(outdir); out.mkdir(parents=True, exist_ok=True); an = root / 'analysis'; files = []
    mA = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_A_*.npz')))}
    mD = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_D_*.npz')))}
    mE = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_E_*.npz')))}
    vmax = max([np.abs(lum(d[k] - d['ref_h'])).max() for d in list(mA.values()) + list(mD.values()) + list(mE.values()) for k in d if k.startswith('hl_')] or [1.0])
    S = Sg(vmax)
    ref0 = mA.get('F0_V0'); kexp = 0.18 / max(float(lum(ref0['ref_low'] + ref0['ref_h']).mean()), 1e-6) if ref0 is not None else 1.0
    render = lambda x: np.clip((1 - np.exp(-kexp * np.maximum(x, 0))) ** (1 / 2.2), 0, 1)

    def save(fig, name):
        fig.savefig(out / name, dpi=105, bbox_inches='tight'); plt.close(fig); files.append(name)

    # 1. support visualisation on the F0 views and the lift sequence
    try:
        import tracer113 as T
        for fix in ('F0', 'F1', 'F2', 'F3'):
            desc = np.load(root / 'stageA' / f'vhs_{fix}.npy'); g = T.T112.Geom112(C.fixture(fix)[0]); sup = T.support_from_desc(desc, g)
            tags = [t for t in mA if t.startswith(fix)][:3]
            fig, ax = plt.subplots(1, len(tags), figsize=(5 * len(tags), 5), squeeze=False)
            for i, t in enumerate(tags):
                gb = load(root / 'stageA' / f'gbuffer_{t}.npz'); img = render(mA[t]['ref_low'] + mA[t]['ref_h'])
                m = sup[np.clip(gb['pid'], 0, 1269)] & gb['ok']
                over = img.copy(); over[m] = 0.5 * over[m] + 0.5 * np.array([1.0, 0.55, 0.0])
                ax[0, i].imshow(over); ax[0, i].set_title(f'{t}: refined patches (orange), {int(sup.sum())} of 1270', fontsize=8); ax[0, i].axis('off')
            save(fig, f'{fix}_compact_refinement_support_overlay.png')
    except Exception as e:
        print('support figure skipped', e)
    # 2. within-patch physical HIGH variation (F0 dense 4x4, replica 0): lowest-ring sphere patch and contact floor patch
    try:
        import tracer113 as T
        import eval113 as E
        Tc, T2, T4, _ = T.tables4(E.load_acc(root / 'stageA' / 'F0_r0.npz'))
        g = T.T112.Geom112(C.fixture('F0')[0])
        ring0 = [p for p in range(1270) if g.PSID[p] == 6 and g.PSPH[p][1] == 0]
        pick = [('lowest-ring sphere patch', ring0[0]), ('lowest-ring sphere patch', ring0[3]), ('floor patch under the contact', 155), ('floor patch beside the contact', 154)]
        fig, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.6))
        for i, (lab, p) in enumerate(pick):
            v = lum(T4[p]).mean(1).reshape(4, 4); im = ax[i].imshow(v, cmap='magma', origin='lower'); fig.colorbar(im, ax=ax[i], fraction=0.045)
            ax[i].set_title(f'{lab} {p}\nmean outgoing HIGH per 4x4 sub-cell (coarse {lum(Tc[p]).mean():.3f})', fontsize=7)
        save(fig, 'F0_within_patch_physical_higher_order_variation_4x4.png')
        # learned vs physical sub-patch responses (SEQ_A state 0 = F0 contact geometry)
        tp = root / 'stageE' / 'table_SEQ_A_lift_0_CAND.npy'
        if tp.exists():
            Lt = np.load(tp)
            fig, ax = plt.subplots(2, len(pick), figsize=(4 * len(pick), 6.5))
            for i, (lab, p) in enumerate(pick):
                a = lum(T4[p]).mean(1).reshape(4, 4); b = lum(Lt[p]).mean(1).reshape(4, 4); lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
                ax[0, i].imshow(a, cmap='magma', origin='lower', vmin=lo, vmax=hi); ax[0, i].set_title(f'physical 4x4, patch {p}', fontsize=7)
                im = ax[1, i].imshow(b, cmap='magma', origin='lower', vmin=lo, vmax=hi); ax[1, i].set_title(f'learned candidate, patch {p}', fontsize=7)
                fig.colorbar(im, ax=ax[:, i].tolist(), fraction=0.045)
            save(fig, 'F0_learned_vs_physical_subpatch_higher_order_response.png')
    except Exception as e:
        print('subpatch figure skipped', e)
    # 3. oracle arms, residual crops for every Stage A view
    arms = ['U', 'S2', 'S4', 'L4', 'R0']
    if mA:
        tags = list(mA)
        fig, ax = plt.subplots(len(tags), len(arms) + 1, figsize=(2.5 * (len(arms) + 1), 2.5 * len(tags)))
        for i, t in enumerate(tags):
            m = mA[t]; sl = crop(m)
            ax[i, 0].imshow(render(m['ref_low'] + m['ref_h'])[sl]); ax[i, 0].set_title(f'{t}: reference (depth 6)', fontsize=7); ax[i, 0].axis('off')
            for j, a in enumerate(arms):
                im = S.show(ax[i, j + 1], lum(m[f'hl_{a}'] - m['ref_h'])[sl], f'{t}: {NAMES[a]}')
        S.bar(fig, im, ax[:, -1].tolist())
        save(fig, 'oracle_arms_signed_higher_order_error_contact_crops_all_views.png')
    mp = an / 'stageA_metrics.json'
    if mp.exists():
        met = json.loads(mp.read_text()); cvs = [t for t in met if met[t]['contact_valid']]
        fig, ax = plt.subplots(2, 1, figsize=(max(10, 1.1 * len(cvs)), 7))
        for k, key in enumerate(('near', 'contrast')):
            w = 0.16
            for j, a in enumerate(arms):
                e = [met[t]['arms'][a]['endpoints'][key] for t in cvs]
                ax[k].bar(np.arange(len(cvs)) + (j - 2) * w, [x['estimate'] for x in e], w, yerr=[x['half_width'] for x in e], capsize=1.5, label=NAMES[a])
            for i, t in enumerate(cvs):
                b = met[t]['band_near' if key == 'near' else 'band_contrast']; ax[k].plot([i - .45, i + .45], [b, b], 'k--', lw=.7); ax[k].plot([i - .45, i + .45], [-b, -b], 'k--', lw=.7)
            ax[k].set_xticks(range(len(cvs))); ax[k].set_xticklabels(cvs, rotation=30, fontsize=7); ax[k].axhline(0, c='k', lw=.5)
            ax[k].set_title(('near-floor' if key == 'near' else 'near-minus-flank') + ' HIGH excess (depth 6; dashed = WL112 bands)')
        ax[0].legend(fontsize=7, ncol=5)
        save(fig, 'oracle_arms_contact_endpoints_per_camera_and_configuration.png')
    # 4. learned arms (Stage D)
    larms = ['A_learned', 'O_U', 'CAND', 'oracle']
    if mD:
        tags = list(mD)
        fig, ax = plt.subplots(len(tags), len(larms) + 1, figsize=(2.6 * (len(larms) + 1), 2.5 * len(tags)))
        for i, t in enumerate(tags):
            m = mD[t]; sl = crop(m)
            ax[i, 0].imshow(render(m['ref_d6'])[sl]); ax[i, 0].set_title(f'{t}: reference (depth 6)', fontsize=7); ax[i, 0].axis('off')
            for j, a in enumerate(larms):
                if f'hl_{a}' in m:
                    im = S.show(ax[i, j + 1], lum(m[f'hl_{a}'] - m['ref_h'])[sl], f'{t}: {NAMES[a]}')
                else:
                    ax[i, j + 1].axis('off')
        S.bar(fig, im, ax[:, -1].tolist())
        save(fig, 'learned_arms_signed_higher_order_error_contact_crops_all_views.png')
        fig, ax = plt.subplots(len(tags), 3, figsize=(13, 4.2 * len(tags)))
        for i, t in enumerate(tags):
            m = mD[t]
            ax[i, 0].imshow(render(m['ref_full'])); ax[i, 0].set_title(f'{t}: full-depth physical reference')
            ax[i, 1].imshow(render(m['ro_low'] + m['hl_CAND'])); ax[i, 1].set_title(f'{t}: learned candidate, raw 4096 spp')
            im = S.show(ax[i, 2], lum(m['hl_CAND'] - m['ref_h']), f'{t}: candidate minus depth-6 reference, higher order')
            ax[i, 0].axis('off'); ax[i, 1].axis('off')
        S.bar(fig, im, ax[:, 2].tolist())
        save(fig, 'learned_candidate_full_frame_render_and_error_per_camera.png')
    dp = an / 'stageD_metrics.json'
    if dp.exists():
        D = json.loads(dp.read_text()); cvs = [t for t in D if D[t]['contact_valid']]
        fig, ax = plt.subplots(3, 1, figsize=(max(10, 1.1 * len(cvs)), 10))
        for k, (key, lab) in enumerate((('near', 'near-floor HIGH excess, depth 6 (WL112 bands)'), ('contrast', 'near-minus-flank HIGH excess, depth 6'), ('R1', 'approved 3 % R1 near bias: full depth (solid) vs depth-matched (hatched)'))):
            for j, a in enumerate(larms):
                if k < 2:
                    e = [D[t]['arms'][a]['endpoints'][key] for t in cvs]
                    ax[k].bar(np.arange(len(cvs)) + (j - 1.5) * 0.2, [x['estimate'] for x in e], 0.2, yerr=[x['half_width'] for x in e], capsize=1.5, label=NAMES[a])
                else:
                    f = [D[t]['arms'][a]['gate3_full_depth']['R1_near']['b'] for t in cvs]; dm = [D[t]['arms'][a]['gate3_depth_matched']['R1_near']['b'] for t in cvs]
                    ax[k].bar(np.arange(len(cvs)) + (j - 1.5) * 0.2, f, 0.1, label=NAMES[a]); ax[k].bar(np.arange(len(cvs)) + (j - 1.5) * 0.2 + 0.1, dm, 0.1, hatch='//', fill=False)
            for i, t in enumerate(cvs):
                b = D[t]['band_near'] if k == 0 else (D[t]['band_contrast'] if k == 1 else 0.03)
                ax[k].plot([i - .45, i + .45], [b, b], 'k--', lw=.7); ax[k].plot([i - .45, i + .45], [-b, -b], 'k--', lw=.7)
            ax[k].set_xticks(range(len(cvs))); ax[k].set_xticklabels(cvs, rotation=30, fontsize=7); ax[k].axhline(0, c='k', lw=.5); ax[k].set_title(lab, fontsize=9)
        ax[0].legend(fontsize=7, ncol=4)
        save(fig, 'learned_arms_contact_endpoints_and_3pct_near_bias_per_camera.png')
        fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.1 * len(cvs)), 3.6))
        for j, (k, lab) in enumerate((('spatial_learned_rel', 'spatial + learned (method - depth-6 ref)'), ('truncation_rel', 'truncation (depth-6 ref - full ref)'), ('method_minus_full_rel', 'total (method - full ref)'))):
            ax.bar(np.arange(len(cvs)) + (j - 1) * 0.27, [D[t]['arms']['CAND']['near_split'][k] for t in cvs], 0.27, label=lab)
        ax.axhspan(-0.03, 0.03, color='0.88'); ax.set_xticks(range(len(cvs))); ax.set_xticklabels(cvs, rotation=30, fontsize=7); ax.legend(fontsize=7)
        ax.set_title('learned candidate near-floor bias relative to the full-depth reference: spatial/learned vs depth truncation (grey: 3 %)')
        save(fig, 'learned_candidate_near_bias_spatial_vs_depth_truncation.png')
    # 5. dynamic sequences
    ep = an / 'stageE_metrics.json'
    if ep.exists():
        Em = json.loads(ep.read_text())
        for sname, seq in C.sequences().items():
            if sname not in Em:
                continue
            v = seq['views'][0]; n = len(seq['states']); cols = ['ref', 'CAND', 'O_U', 'stale', 'oracle']
            fig, ax = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.5 * n))
            for stt in range(n):
                m = mE[f'{sname}_{stt}_{v}']; sl = crop(m)
                for j, a in enumerate(cols):
                    if a == 'ref':
                        ax[stt, j].imshow(render(m['ref_d6'])[sl]); ax[stt, j].set_title(f'{seq["label"][stt]}: reference', fontsize=7); ax[stt, j].axis('off')
                    elif f'hl_{a}' in m:
                        im = S.show(ax[stt, j], lum(m[f'hl_{a}'] - m['ref_h'])[sl], NAMES[a])
                    else:
                        ax[stt, j].axis('off')
            S.bar(fig, im, ax[:, -1].tolist())
            save(fig, f'{sname}_dynamic_sequence_{v}_reference_and_higher_order_error_crops.png')
            fig, ax = plt.subplots(1, len(seq['views']), figsize=(6 * len(seq['views']), 3.4), squeeze=False)
            for i, vv in enumerate(seq['views']):
                for a in ('A_learned', 'O_U', 'CAND', 'oracle', 'stale'):
                    xs, ys, es = [], [], []
                    for stt, row in enumerate(Em[sname]['states']):
                        r = row['views'][vv]
                        if a in r['arms'] and 'endpoints' in r['arms'][a]:
                            xs.append(stt); ys.append(r['arms'][a]['endpoints']['near']['estimate']); es.append(r['arms'][a]['endpoints']['near']['half_width'])
                    if xs:
                        ax[0, i].errorbar(xs, ys, yerr=es, marker='o', ms=3, capsize=2, label=NAMES[a])
                bn = [row['views'][vv]['band_near'] for row in Em[sname]['states']]
                ax[0, i].fill_between(range(len(bn)), [-b for b in bn], bn, color='0.88'); ax[0, i].axhline(0, c='k', lw=.5)
                ax[0, i].set_xticks(range(len(bn))); ax[0, i].set_xticklabels(seq['label'], rotation=30, fontsize=7); ax[0, i].set_title(f'{sname} {vv}: near-floor HIGH excess per state')
            ax[0, -1].legend(fontsize=6)
            save(fig, f'{sname}_near_floor_higher_order_excess_per_state.png')
    # 6. cost
    cp = an / 'cost.json'
    if cp.exists():
        c = json.loads(cp.read_text())['arms']; nm = list(c)
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
        for a_, (k, lab) in zip(ax, (('table_mib', 'HIGH table (MiB)'), ('generator', 'generator latency (ms)'), ('transport_update', 'edit -> table update (ms)'), ('training_s', 'training (min)'))):
            vals = [(c[n][k]['median_ms'] if isinstance(c[n].get(k), dict) else c[n].get(k, 0) / (60 if k == 'training_s' else 1)) for n in nm]
            a_.bar(nm, vals); a_.set_title(lab)
        save(fig, 'cost_table_size_latency_training_candidate_vs_coarse.png')
    rows = ''.join(f'<figure><img src="{f}" style="max-width:100%"><figcaption>{f}</figcaption></figure>\n' for f in files)
    gate = (an / 'gate.json').read_text() if (an / 'gate.json').exists() else '{}'
    (out / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>WL113 spatial HIGH refinement</title><body style="font-family:sans-serif;max-width:1400px;margin:auto;padding:16px">'
                                    f'<h1>WL113 — spatially refined HIGH and the SSS near-contact gate</h1><p>Shared signed scale: max |error| {S.v:.4f} linear HDR luminance (sqrt display). Raw renders, no denoising.</p><pre>{gate}</pre>{rows}</body>', encoding='utf8')
    man = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files + ['index.html']}
    for j in ('stageA_qualification.json', 'stageA_metrics.json', 'stageD_metrics.json', 'stageE_metrics.json', 'stageF_truncation.json', 'gate.json', 'cost.json'):
        if (an / j).exists():
            (out / j).write_bytes((an / j).read_bytes()); man[j] = hashlib.sha256((an / j).read_bytes()).hexdigest()
    for j in ('stageB/decision.json', 'stageC/precheck.json', 'tests113.json', 'stageC/O_U/training.json', 'stageC/CAND/training.json'):
        if (root / j).exists():
            nm_ = j.replace('/', '_'); (out / nm_).write_bytes((root / j).read_bytes()); man[nm_] = hashlib.sha256((root / j).read_bytes()).hexdigest()
    C.jd({'source_root': str(root), 'signed_scale_max': S.v, 'files_sha256': man}, out / 'manifest113.json')
    print('FIGURES', len(files))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
