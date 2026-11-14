"""WL93D rendered-image visuals (post-run, presentation only; no new renders): reference vs intervention arms.
  python vizrender93d.py <R> <R86> <R92A> <R92B> <out>
Full frame of an arm = reference LOW (physical camera-path events 1-2, identical for every arm: REF tot - REF h) + the arm's rendered
higher-order term hl (2 x 4,096 spp). One exposure / tone map per figure, set from the reference."""
import glob, hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402
ARMS = [('C0', 'deployed (C0)'), ('C1_both', 'other-geometry operator (C1_both)'), ('C2_src', 'M64 at source use (C2_src)'), ('C2_rcv', 'M64 at receiver use (C2_rcv)'), ('C2_both', 'M64 at both uses (C2_both)')]


def tm(x, white):
    return np.clip(x / white, 0, 1) ** (1 / 2.2)


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    F = json.load(open(os.path.join(R, 'results', 'final93d.json')))
    made = []
    for tag in ('E2_T1', 'E5k995_T1'):
        rf = None
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                rf = A.load(root, f'REF_ER_{tag}'); break
        tot = rf['tot'].mean(0); h = rf['h'].mean(0); low = tot - h
        M = AN.mask_of(R, R86, tag)
        ys, xs = np.nonzero(M['C']); pad = 25
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        for fam, sd in (('own', 8601), ('own', 8602)):
            imgs = []
            for arm, lab in ARMS:
                x = AN.hl(R, f'I93D_{fam}_s{sd}_{arm}_{tag}')
                if x is not None:
                    imgs.append((arm, lab, x['hl'].mean(0)))
            if not imgs:
                continue
            white = float(np.quantile(tot, 0.995)); whiteh = float(np.quantile(h, 0.995))
            n = len(imgs) + 1
            fig, ax = plt.subplots(3, n, figsize=(3.3 * n, 10.5))
            cols = [('reference (REF-ER 8 x 8,192 spp)', tot, h)] + [(lab, low + x, x) for arm, lab, x in imgs]
            for j, (lab, full, ho) in enumerate(cols):
                eh = None if j == 0 else F['renders'].get(tag, {}).get(f'{fam}_s{sd}:{imgs[j - 1][0]}', {}).get('C', {}).get('E_H')
                ax[0, j].imshow(tm(full, white)); ax[0, j].set_title(lab + (f'\nregion C E_H {eh:.4f}' if eh is not None else ''), fontsize=8)
                ax[1, j].imshow(tm(ho, whiteh)); ax[1, j].set_title('higher-order term only', fontsize=8)
                ax[2, j].imshow(tm(full[sl], white)); ax[2, j].contour(M['C'][sl], levels=[0.5], colors='y', linewidths=0.5)
                ax[2, j].set_title('close-up of region C (yellow contour)', fontsize=8)
            for a_ in ax.ravel():
                a_.set_xticks([]); a_.set_yticks([])
            fig.suptitle(f'{tag}, {fam} s{sd}: rendered images. Full frame = reference LOW (physical camera-path events 1-2, identical for all arms) '
                         f'+ each arm\'s higher-order term (2 x 4,096 spp). Same exposure per row (white point from the reference).', fontsize=9)
            fn = f'fig6_{tag}_{fam}_s{sd}_rendered.png'
            fig.savefig(os.path.join(out, fn), dpi=110, bbox_inches='tight'); plt.close(fig); made.append(fn)
    mf = os.path.join(out, 'manifest.json'); man = json.load(open(mf))
    for f in made:
        man['files'][f] = hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest()
    man['provenance'] += '; fig6 rendered-image panels from experiments/ecro_local93d/vizrender93d.py (post-run presentation of existing renders; no new renders)'
    json.dump(man, open(mf, 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
