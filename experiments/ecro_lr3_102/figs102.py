"""WL102 reviewer figures (numpy + matplotlib): Ref-65 / Ref-7 / TransGI / ECRO GI images and absolute LDR error maps on common
scales, plus ECRO's learned higher order vs the physical one.

  python figs102.py <ref65.npz> <ref7.npz> <ecro_eval_dir> <transgi_gi.npy> <out_dir>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r65, r7, ed, tgp, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
R65, R7 = np.load(r65), np.load(r7)
e4 = np.load(f'{ed}/ecro_spp4.npz')
h = [np.load(f'{ed}/ecro_spp512_h{i}.npz') for i in (0, 1)]
p = [np.load(f'{ed}/phys_d6_spp1024_h{i}.npz') for i in (0, 1)]
ecro_c = 0.5 * sum(x['e2'] + x['hl'] for x in h)
hl_c = 0.5 * sum(x['hl'] for x in h); h_phys = 0.5 * sum(x['h'] for x in p)
tg = np.load(tgp)
ldr = lambda x: np.clip(x, 0, 1) ** (1 / 2.2)
err = lambda a, b: np.abs(ldr(a) - ldr(b)).mean(-1)
VMAX = 0.3

imgs = [('Ref-65 GI (final reference)', ldr(R65['gi'])), ('TransGI GI, 4 spp (LeakyReLU, D_vis)', ldr(tg)),
        ('ECRO GI, 4 spp (e2 + learned hl)', ldr(e4['e2'] + e4['hl'])), ('ECRO GI, converged LOW (2 x 512 spp)', ldr(ecro_c)),
        ('Ref-7 GI (6 events)', ldr(R7['gi'])), ('|TransGI - Ref-65|', err(tg, R65['gi'])),
        ('|ECRO conv. - Ref-65|', err(ecro_c, R65['gi'])), ('|ECRO conv. - Ref-7|  (representation)', err(ecro_c, R7['gi']))]
fig, ax = plt.subplots(2, 4, figsize=(24, 7.5))
for a, (t, im) in zip(ax.ravel(), imgs):
    a.imshow(im, cmap=None if im.ndim == 3 else 'magma', vmin=0, vmax=1 if im.ndim == 3 else VMAX); a.set_title(t, fontsize=11); a.axis('off')
plt.suptitle('WL102 Living Room 3 static V0, 640x360. LDR = clip(x,0,1)^(1/2.2); error maps = mean RGB |LDR a - LDR b|, magma 0-0.3', fontsize=12)
plt.tight_layout(); plt.savefig(f'{out}/comparison_overview.png', dpi=80); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
hv = max(np.quantile(h_phys, .99), 1e-6)
for a, (t, im) in zip(ax, [('physical higher order h (events 3-6, 2 x 1,024 spp)', h_phys), ('ECRO learned hl (table lookup)', hl_c),
                            ('|hl - h| (linear)', np.abs(hl_c - h_phys))]):
    a.imshow(np.clip(im / hv, 0, 1) ** (1 / 2.2)); a.set_title(t, fontsize=11); a.axis('off')
plt.suptitle(f'ECRO higher-order component vs the physical one (linear, common scale: 99th percentile of h = {hv:.3f})', fontsize=12)
plt.tight_layout(); plt.savefig(f'{out}/ecro_higher_order_vs_physical.png', dpi=80); plt.close()

for name, im in [('transgi_gi_4spp.png', ldr(tg)), ('ecro_gi_4spp.png', ldr(e4['e2'] + e4['hl'])), ('ecro_gi_converged.png', ldr(ecro_c))]:
    plt.imsave(f'{out}/{name}', im)
for name, im in [('err_transgi_vs_ref65.png', err(tg, R65['gi'])), ('err_ecro_conv_vs_ref65.png', err(ecro_c, R65['gi'])),
                 ('err_ecro_conv_vs_ref7.png', err(ecro_c, R7['gi'])), ('err_ecro_4spp_vs_ref65.png', err(e4['e2'] + e4['hl'], R65['gi']))]:
    plt.imsave(f'{out}/{name}', im, cmap='magma', vmin=0, vmax=VMAX)
print('figures written to', out)
