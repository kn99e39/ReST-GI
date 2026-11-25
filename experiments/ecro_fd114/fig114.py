"""WL114 reviewer figures (reads analysis outputs only; not part of the evidence computation). One absolute signed scale for every
residual map of the worklog (sqrt display, real-unit colorbar); readable file names; manifest with hashes.

  python fig114.py ROOT OUTDIR
"""
import glob, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402

LUMW = np.array([0.2126, 0.7152, 0.0722]); lum = lambda x: x @ LUMW
NAMES = {'S4_D6': 'dense 4x4 physical oracle, depth 6', 'S4_F': 'dense 4x4 physical oracle, full depth', 'U_D6': 'coarse physical oracle, depth 6',
         'U_F': 'coarse physical oracle, full depth', 'A': 'WL113 learned dense (depth-6 targets)', 'A1r': 'learned dense, depth-6 targets (rerun)',
         'B1': 'learned dense, full-depth targets', 'A2': 'learned dense, depth-6 targets, seed 2', 'B2': 'learned dense, full-depth targets, seed 2',
         'O_U': 'WL113 learned coarse (depth 6)', 'OB1': 'learned coarse, full-depth targets', 'stale_B1': 'stale full-depth table (previous state)',
         'stale_A1r': 'stale depth-6 table (previous state)'}


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
        cb.set_ticklabels([f'{np.sign(x) * x * x * self.v:+.3f}' for x in t]); cb.set_label('signed luminance error vs full-depth reference (linear HDR)', fontsize=7)


def main(root, outdir):
    root = Path(root); out = Path(outdir); out.mkdir(parents=True, exist_ok=True); an = root / 'analysis'; files = []

    def save(fig, name):
        fig.savefig(out / name, dpi=105, bbox_inches='tight'); plt.close(fig); files.append(name)
    qa = json.loads((an / 'stageA_qualification.json').read_text())
    # 1. missing energy versus depth (SSS Stage A views, near / full regions; LR3 GI ladder)
    s = qa['SSS']
    edges = [6, 12, 24, 48, 96]
    order = ['h', 'c7_12', 'c13_24', 'c25_48', 'c49_96', 'c97_192']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, reg in enumerate(('near', 'full')):
        for tag, v in s['views'].items():
            r = v['regions'].get(reg)
            if not r:
                continue
            b = r['buckets']; H = r['HIGH_full']
            miss = [sum(b[n] for n in order[i + 1:]) / H for i in range(len(edges))]
            ax[k].semilogy(edges, np.maximum(miss, 1e-12), marker='o', lw=1, label=tag)
        ax[k].axhline(1e-3, color='k', ls='--', lw=0.8); ax[k].set_xlabel('last camera event included (depth K)')
        ax[k].set_ylabel('missing share of full-depth higher-order energy'); ax[k].set_title(f'SSS {reg} region: energy beyond event K (deterministic, 2 x 2,048 spp)', fontsize=8)
    ax[1].legend(fontsize=6, ncol=2)
    save(fig, 'sss_missing_higher_order_energy_vs_depth.png')
    if qa['LR3'].get('curve'):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for name, cv in qa['LR3']['curve'].items():
            ds = [7, 13, 25, 49, 97, 129, 257]
            m = [np.mean(np.abs(cv[f'missing_rel_d{d}'])) for d in ds]
            ax.semilogy([d - 1 for d in ds], np.maximum(m, 1e-7), marker='o', lw=1, label=name)
        ax.axhline(1e-3, color='k', ls='--', lw=0.8); ax.axvline(64, color='gray', ls=':', lw=0.8)
        ax.set_xlabel('ECRO events (Mitsuba max_depth - 1)'); ax.set_ylabel('missing GI share vs infinite-depth roulette reference (channel mean)')
        ax.set_title('Living Room 3: missing global illumination versus path depth (Mitsuba 3.6.4, 16,384 spp, CRN)', fontsize=8); ax.legend(fontsize=6)
        save(fig, 'living_room_3_missing_gi_vs_depth.png')
    mB = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_B_*.npz')))}
    mD = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_D_*.npz')))}
    mE = {Path(p).stem[7:]: load(p) for p in sorted(glob.glob(str(an / 'maps_E_*.npz')))}
    allm = list(mB.values()) + list(mD.values()) + list(mE.values())
    if not allm:
        json.dump({'files': files}, open(out / 'manifest.json', 'w'), indent=1); return
    vmax = max([np.abs(lum(d[k] - d['ref_hf'])).max() for d in allm for k in d if k.startswith('hl_')] + [np.abs(lum(d['ref_h'] - d['ref_hf'])).max() for d in allm])
    S = Sg(vmax)
    r0 = next(iter(mB.values()))
    kexp = 0.18 / max(float(lum(r0['ref_full']).mean()), 1e-6)
    render = lambda x: np.clip((1 - np.exp(-kexp * np.maximum(x, 0))) ** (1 / 2.2), 0, 1)
    # 2. depth-6 versus full-depth physical targets (images + truncation residual), static views
    for tag in [t for t in mB if t.split('_')[0] in ('F0', 'F1', 'F3', 'N1', 'N3', 'N4')][:8]:
        m = mB[tag]; cy, cx = crop(m)
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
        ax[0].imshow(render(m['ref_d6'])); ax[0].set_title(f'{tag}: physical reference, events 1-6', fontsize=8); ax[0].axis('off')
        ax[1].imshow(render(m['ref_full'])); ax[1].set_title(f'{tag}: physical reference, full depth', fontsize=8); ax[1].axis('off')
        im = S.show(ax[2], lum(m['ref_h'] - m['ref_hf']), 'depth-6 minus full depth (truncation), whole frame')
        S.show(ax[3], lum(m['ref_h'] - m['ref_hf'])[cy, cx], 'truncation, near-contact crop'); S.bar(fig, im, ax[3])
        save(fig, f'{tag}_physical_reference_depth6_vs_full_depth.png')
    # 3. learned response comparison + shared-scale residuals (Stage D static views)
    arms = ['S4_D6', 'S4_F', 'A1r', 'B1', 'OB1']
    for tag, m in mD.items():
        cy, cx = crop(m)
        have = [a for a in arms if f'hl_{a}' in m]
        fig, ax = plt.subplots(2, len(have) + 1, figsize=(3.6 * (len(have) + 1), 7.4))
        ax[0, 0].imshow(render(m['ref_full'])[cy, cx]); ax[0, 0].set_title(f'{tag}: full-depth reference (crop)', fontsize=7); ax[0, 0].axis('off')
        ax[1, 0].imshow(render(m['ro_low'] + m['ref_hf'])[cy, cx]); ax[1, 0].axis('off'); ax[1, 0].set_title('LOW + physical full HIGH (paired readout)', fontsize=7)
        for i, a in enumerate(have):
            ax[0, i + 1].imshow(render(m['ro_low'] + m[f'hl_{a}'])[cy, cx]); ax[0, i + 1].set_title(NAMES.get(a, a), fontsize=7); ax[0, i + 1].axis('off')
            im = S.show(ax[1, i + 1], lum(m[f'hl_{a}'] - m['ref_hf'])[cy, cx], f'{NAMES.get(a, a)} minus full-depth HIGH')
        S.bar(fig, im, ax[1, -1])
        save(fig, f'{tag}_learned_and_oracle_higher_order_vs_full_depth_reference.png')
    # 4. radial ring profiles (full-depth gate R4) per static view
    try:
        dm = json.loads((an / 'stageD_metrics.json').read_text())
        rings = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]
        mids = [0.5 * (a + b) for a, b in zip(rings, rings[1:])]
        for tag, v in dm.items():
            if not v['contact_valid']:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4))
            for a in ('S4_D6', 'S4_F', 'A1r', 'B1', 'B2'):
                if a in v['arms']:
                    bs = [x['b'] for x in v['arms'][a]['gate3_full_depth']['R4_rings']]
                    ax.plot(mids, bs, marker='o', lw=1, label=NAMES.get(a, a))
            ax.axhspan(-0.03, 0.03, color='0.9'); ax.axhline(0, color='k', lw=0.6)
            ax.set_xlabel('floor distance from the support point (m)'); ax.set_ylabel('relative bias vs full-depth reference (R4 rings)')
            ax.set_title(f'{tag}: radial floor profile of the complete frame (grey = 3 % band)', fontsize=8); ax.legend(fontsize=6)
            save(fig, f'{tag}_radial_ring_bias_full_depth_gate.png')
    except Exception as e:
        print('ring figure skipped', e)
    # 5. dynamic contact sequences: near bias vs full depth over states
    try:
        em = json.loads((an / 'stageE_metrics.json').read_text())
        for sname, sq in em.items():
            vns = list(sq['states'][0]['views'])
            fig, ax = plt.subplots(1, len(vns), figsize=(6.2 * len(vns), 4), squeeze=False)
            for i, vn in enumerate(vns):
                for a in ('S4_F', 'B1', 'B2', 'A1r', 'stale_B1'):
                    ys = [r['views'][vn]['arms'][a]['b_full'] if a in r['views'][vn]['arms'] and 'b_full' in r['views'][vn]['arms'][a] else np.nan for r in sq['states']]
                    ax[0, i].plot(range(len(ys)), ys, marker='o', lw=1, label=NAMES.get(a, a))
                tr = [r['views'][vn].get('trunc_near_rel', np.nan) for r in sq['states']]
                ax[0, i].plot(range(len(tr)), tr, 'k--', lw=1, label='physical depth-6 truncation')
                ax[0, i].axhspan(-0.03, 0.03, color='0.9'); ax[0, i].set_xticks(range(len(sq['states'])))
                ax[0, i].set_xticklabels([r['label'] for r in sq['states']], rotation=30, fontsize=7)
                ax[0, i].set_ylabel('near-region relative bias vs full-depth reference'); ax[0, i].set_title(f'{sname} {vn}: near bias over states (same weights)', fontsize=8)
            ax[0, -1].legend(fontsize=6)
            save(fig, f'{sname}_dynamic_near_bias_full_depth.png')
    except Exception as e:
        print('sequence figure skipped', e)
    # 6. cost breakdown
    try:
        cj = json.loads((an / 'cost.json').read_text())
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        tg = cj['target_generation_per_light_n32']
        ax[0].bar(['depth-6 construction (WL113)', f'deep construction to K*={cj["K_star"]}'], [np.median(tg['d6_construct4_s']), np.median(tg['deep_construct_s'])])
        ax[0].set_ylabel('seconds per light (n = 32 proposals)'); ax[0].set_title('physical target generation', fontsize=8)
        names = list(cj['training_all']); ax[1].bar(names, [cj['training_all'][n] / 60 for n in names]); ax[1].set_ylabel('training minutes (24,000 steps)')
        ax[1].set_title('training wall time (contended where run in parallel)', fontsize=8)
        arms = list(cj['arms']); w = 0.25
        for j, key in enumerate(('generator', 'frame_512_1spp', 'transport_update')):
            ax[2].bar(np.arange(len(arms)) + j * w, [cj['arms'][a][key]['median_ms'] for a in arms], w, label=key)
        ax[2].set_xticks(np.arange(len(arms)) + w); ax[2].set_xticklabels(arms); ax[2].set_ylabel('milliseconds (median)'); ax[2].legend(fontsize=7)
        ax[2].set_title('inference cost: depth-6 vs full-depth arm (same architecture)', fontsize=8)
        save(fig, 'cost_breakdown_target_generation_training_inference.png')
    except Exception as e:
        print('cost figure skipped', e)
    man = {'files': {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}, 'shared_residual_scale_linear_hdr': S.v,
           'source': str(root), 'note': 'one absolute signed scale (max over all residual maps of this worklog), sqrt display'}
    json.dump(man, open(out / 'manifest.json', 'w'), indent=1)
    print('FIGURES', len(files), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
