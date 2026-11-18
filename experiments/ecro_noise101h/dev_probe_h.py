"""WL101H development probe BEFORE the protocol freeze. Not evidence. Computes no noise statistic.

Checks: runtime import, table vs archived WL100 table_0, repeated source/respond determinism, own center-ray gbuffer vs WL100 map_0,
WL101A / WL101C mask counts, supersampled sphere coverage, split-arm structural invariants (bitwise E1 / HL equality when only the
suffix stream changes; finiteness), frame timings. Writes an ROI-candidate overlay on the archived physical reference mean.
  python dev_probe_h.py <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import common_h as CH
import render_h as RH

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rep = {'versions': RH.versions()}
cam = RH.camera()
t0 = time.perf_counter()
rt, s, B = RH.build(split=True)
RH.R.sync(); rep['build_s'] = time.perf_counter() - t0
Bn = B.double().cpu().numpy()
arch = np.load(CH.REND100 / 'table_0.npz')['B'].astype(np.float64)
rep['table_vs_archive_rel_l1'] = float(np.abs(Bn - arch).sum() / np.abs(arch).sum())
rep['table_vs_archive_max_abs'] = float(np.abs(Bn - arch).max())
same = []
for _ in range(3):
    s2 = rt.source(); B2, _ = rt.respond(s2)
    same.append({'source_bitwise': bool((s == s2).all().item()), 'source_max_abs': float((s - s2).abs().max().item()),
                 'table_bitwise': bool((B == B2).all().item()), 'table_max_abs': float((B - B2).abs().max().item())})
rep['repeat_state'] = same

sid0, pos0 = RH.gbuffer(rt, cam, 512, 1)
with np.load(CH.ROOT100 / 'sss' / 'renders' / 'map_0.npz') as g:
    msid, mok = g['sid'], g['ok']
rep['gbuffer_sid_mismatch'] = int((np.where(mok, msid, -1) != sid0[..., 0]).sum())
m, g = CH.base_masks(CH.ROOT100 / 'sss' / 'renders' / 'map_0.npz')
rep['mask_counts_base'] = {k: int(v.sum()) for k, v in m.items()}
cov = RH.coverage(rt, cam, 512, 8)
frac, pure = cov['sphere_frac'], cov['pure_sid']
rep['silhouette_mixed'] = int(((frac > 0) & (frac < 1)).sum())
rep['interior_not_full_cover'] = int((m['sphere_interior'] & (frac < 1)).sum())
rep['sphere_boundary_mixed'] = int((m['sphere_boundary'] & (frac < 1)).sum())
np.savez_compressed(out / 'coverage_dev.npz', sphere_frac=frac, pure_sid=pure)

# split-arm structural smoke at 64 x 64 equivalent: full res, 1 spp, two renders differing only in the suffix stream
Bd = RH.R.t2d(B.view(-1))
a = RH.to_np(rt.split_frame(Bd, cam, 512, 1, 11, 12, 13))
b = RH.to_np(rt.split_frame(Bd, cam, 512, 1, 11, 12, 14))
c = RH.to_np(rt.split_frame(Bd, cam, 512, 1, 11, 15, 14))
rep['split_suffix_only'] = {'e1_bitwise': bool(np.array_equal(a[0:3], b[0:3])), 'hl_bitwise': bool(np.array_equal(a[6:9], b[6:9])),
                            'e2_differs': bool(not np.array_equal(a[3:6], b[3:6])), 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all())}
rep['split_prefix_change'] = {'hl_differs': bool(not np.array_equal(a[6:9], c[6:9])), 'e1_differs': bool(not np.array_equal(a[0:3], c[0:3]))}
a2 = RH.to_np(rt.split_frame(Bd, cam, 512, 1, 11, 12, 13))
rep['split_replay_bitwise'] = bool(np.array_equal(a, a2))

# normal-arm replay determinism + timing (descriptive)
tim = {}
for spp in (1, 4, 16):
    f1 = RH.to_np(rt.frame(Bd, cam, 512, spp, 999)); f2 = RH.to_np(rt.frame(Bd, cam, 512, spp, 999))
    ts = []
    for r in range(12):
        RH.R.sync(); t = time.perf_counter(); o = rt.frame(Bd, cam, 512, spp, 1000 + r); x = RH.to_np(o); ts.append((time.perf_counter() - t) * 1e3)
    tss = []
    for r in range(6):
        RH.R.sync(); t = time.perf_counter(); o = rt.split_frame(Bd, cam, 512, spp, 2000 + r, 3000 + r, 4000 + r); x = RH.to_np(o); tss.append((time.perf_counter() - t) * 1e3)
    tim[spp] = {'normal_replay_bitwise': bool(np.array_equal(f1, f2)), 'normal_ms_median': float(np.median(ts[2:])),
                'split_ms_median': float(np.median(tss[2:])), 'finite': bool(np.isfinite(f1).all())}
rep['frames'] = tim
RH.R.sync(); t = time.perf_counter(); hq = RH.R.render_hq(rt, Bd, cam, 512, 512, 777); rep['hq_512spp_s'] = time.perf_counter() - t
rep['table_after_frames_bitwise'] = bool((B == rt.respond(rt.source())[0]).all().item())
print(json.dumps(rep, indent=1))
CH.write_json(out / 'dev_probe.json', rep)

# ROI candidate overlay on the archived WL100 physical reference mean (pre-existing evidence; used only to place the control ROI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ref = sum(np.load(CH.REND100 / f'REF_0_{h}.npz')[k].astype(np.float64) for h in (0, 1) for k in ('e1', 'e2', 'h')) / 2
ex = 0.1938553403045695
tone = lambda x: (np.clip(ex * x, 0, None) / (1 + np.clip(ex * x, 0, None))) ** (1 / 2.2)
img = tone(ref)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(img); ax[0].set_title('WL100 physical reference mean (cfg0), sid contours')
ax[0].contour(g['sid'], levels=np.arange(-0.5, 9), colors='w', linewidths=0.4)
ov = img.copy()
for k, col in (('sphere_interior', (0.2, 0.6, 1)), ('near', (1, 0.5, 0.1)), ('flank', (0.2, 0.9, 0.9)), ('contact_shadow', (1, 1, 0))):
    mm = m[k]; ov[mm] = 0.5 * ov[mm] + 0.5 * np.array(col)
ov[(frac > 0) & (frac < 1)] = (1, 0, 1)
ax[1].imshow(ov); ax[1].set_title('candidate ROIs: interior (blue), near (orange), flank (cyan), contact/shadow (yellow), silhouette (magenta)')
for a_ in ax:
    a_.set_xticks(range(0, 513, 64)); a_.set_yticks(range(0, 513, 64)); a_.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / 'roi_candidates.png', dpi=80)
import os
os._exit(0)
