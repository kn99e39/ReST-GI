"""WL97 audit track E3: comparison of the deterministic deliveries (quad97.py deliver) with the path-traced reference e2_p (numpy only).

  python ana97.py <RES> <deliv tag> [<deliv tag 2> ...]        RES = results/audit97/E3 (holds ref_boxA.npz and deliv_<tag>.npz)

Reference reduction (derived from pt86.Ctx.dir_in_bin / pt86a.field_src, not assumed):
  a table entry T[j, b, c] is the mean outgoing radiance over directions drawn UNIFORMLY IN SOLID ANGLE inside bin b (z = cos(theta) uniform in the
  ring [iz / 4, (iz + 1) / 4], phi uniform in a 2 pi / 16 sector), so exitance_j = sum_b T[j, b] * int_bin cos dOmega,
  int_bin cos dOmega = (2 pi / 16) * ((iz + 1)^2 - iz^2) / 32   (exact; equals the ring-midpoint rule (iz + 0.5) / 4 * 2 pi / 64; sums to pi).
  Box A is Lambert, so each of the 64 bins of a patch is an independent estimate of the SAME radiance rho E / pi: the scatter across the
  64 bins x 2 halves gives a per-patch standard error that does not depend on the half split alone.
Metrics (rgb-summed exitance per patch, 54 patches): ratio = sum(arm) / sum(ref); abs rel L1 = sum|arm - ref| / sum(ref); Pearson corr;
  n0 = 0.5 sum|ref_a - ref_b| / sum(ref) (expected L1 noise of the two-half mean); chi2 = sum((arm - ref) / SE)^2.
"""
import json, math, os, sys
import numpy as np

NB, NZ, NPHI = 64, 4, 16
IZ = np.arange(NB) // NPHI
W = (2 * math.pi / NPHI) * ((IZ + 1) ** 2 - IZ ** 2) / (2 * NZ * NZ)
assert abs(W.sum() - math.pi) < 1e-12
FACES = ['-x(panel-facing)', '+x', '-y', '+y', '-z', '+z']


def ref(RES):
    z = np.load(os.path.join(RES, 'ref_boxA.npz'))
    a = z['e2_p_a'].astype(np.float64); b = z['e2_p_b'].astype(np.float64)                 # [54, 64, 3]
    ea = (a * W[None, :, None]).sum(1); eb = (b * W[None, :, None]).sum(1)                 # [54, 3] exitance per half
    # per-patch SE of the rgb-summed exitance from the scatter of the 128 (bin, half) radiance estimates (Lambert: all estimate L)
    x = np.concatenate([a.sum(2), b.sum(2)], 1)                                              # [54, 128] rgb-summed radiance per bin estimate
    w = np.concatenate([W, W]) / 2                                                           # exitance = sum_k w_k x_k
    var = x.var(1, ddof=1)                                                                   # per-estimate variance (i.i.d. across bins)
    se = np.sqrt(var * (w ** 2).sum())
    return ea, eb, se, a, b


def metrics(arm, r, ra, rb, se, mask=None):
    m = np.ones(len(r), bool) if mask is None else mask
    a, rr = arm[m], r[m]
    d = a - rr
    nz = se[m] > 0                                                                           # the +x / +y faces never see the panel: ref = arm = 0, SE = 0
    zz = d[nz] / se[m][nz]
    o = {'ratio': float(a.sum() / rr.sum()), 'abs_rel_l1': float(np.abs(d).sum() / rr.sum()),
         'corr': float(np.corrcoef(a, rr)[0, 1]) if m.sum() > 2 else float('nan'),
         'n0_half_split': float(0.5 * np.abs(ra[m] - rb[m]).sum() / rr.sum()),
         'expected_l1_from_bin_scatter': float(math.sqrt(2 / math.pi) * se[m].sum() / rr.sum()),
         'ratio_se_from_bin_scatter': float(math.sqrt((se[m] ** 2).sum()) / rr.sum()),
         'chi2': float((zz ** 2).sum()), 'dof': int(nz.sum()), 'max_abs_z': float(np.abs(zz).max()), 'median_abs_z': float(np.median(np.abs(zz))),
         'frac_abs_z_gt_2': float((np.abs(zz) > 2).mean()), 'max_abs_outside_support': float(np.abs(a[~nz]).max()) if (~nz).any() else 0.0, 'value_at_ref_peak_ratio': float(a[np.argmax(rr)] / rr.max()),
         'peak_patch_arm': int(np.argmax(a)), 'peak_patch_ref': int(np.argmax(rr))}
    return o


def main():
    RES = sys.argv[1]
    ea, eb, se, a, b = ref(RES)
    r3 = 0.5 * (ea + eb)                                                                     # [54, 3]
    r = r3.sum(1); ra = ea.sum(1); rb = eb.sum(1)
    out = {'reference': {'total_exitance_rgb': float(r.sum()), 'per_channel': r3.sum(0).tolist(), 'half_a': float(ra.sum()), 'half_b': float(rb.sum()),
                         'n0_half_split': float(0.5 * np.abs(ra - rb).sum() / r.sum()),
                         'total_se_from_bin_scatter': float(math.sqrt((se ** 2).sum())),
                         'half_diff_over_se': float((ra.sum() - rb.sum()) / (2 * math.sqrt((se ** 2).sum()))),
                         'per_patch_half_z_rms': float(np.sqrt((((ra - rb)[se > 0] / (2 * se[se > 0])) ** 2).mean())), 'patches_with_signal': int((se > 0).sum()),
                         'face_share': {FACES[f]: float(r[9 * f:9 * f + 9].sum() / r.sum()) for f in range(6)},
                         'per_patch': r.tolist(), 'per_patch_se': se.tolist()}}
    for tag in sys.argv[2:]:
        z = np.load(os.path.join(RES, f'deliv_{tag}.npz'))
        o = {'meta': json.loads(str(z['meta']))}
        for k in z.files:
            if not k.startswith('E_'):
                continue
            ex3 = 0.5 * z[k]                                                                 # exitance = rho_A * irradiance, rho_A = 0.5
            arm = ex3.sum(1)
            m = metrics(arm, r, ra, rb, se)
            m['total'] = float(arm.sum()); m['per_channel_ratio'] = (ex3.sum(0) / r3.sum(0)).tolist()
            lit = r > 0.01 * r.max()
            m['faces'] = {FACES[f]: {'arm': float(arm[9 * f:9 * f + 9].sum()), 'ref': float(r[9 * f:9 * f + 9].sum())} for f in range(6)}
            m['panel_face_only'] = metrics(arm, r, ra, rb, se, np.arange(54) < 9)
            m['per_patch'] = arm.tolist()
            m['residual_z'] = np.where(se > 0, (arm - r) / np.where(se > 0, se, 1), 0).round(2).tolist()
            o[k[2:]] = m
        out[tag] = o
        print('==', tag, o['meta'].get('hs_mm'), 'mm', o['meta'].get('nr'), 'x', o['meta'].get('nr'), os.path.basename(str(o['meta'].get('tables'))))
        for k, m in o.items():
            if k == 'meta':
                continue
            print(f"  {k:7s} total {m['total']:.4f} ratio {m['ratio']:.4f} (+-{m['ratio_se_from_bin_scatter']:.4f}) L1 {m['abs_rel_l1']:.4f} corr {m['corr']:.4f}"
                  f" chi2 {m['chi2']:.1f}/{m['dof']} max|z| {m['max_abs_z']:.2f} peak {m['peak_patch_arm']}/{m['peak_patch_ref']} @refpeak {m['value_at_ref_peak_ratio']:.3f}"
                  f" | panel face ratio {m['panel_face_only']['ratio']:.4f} L1 {m['panel_face_only']['abs_rel_l1']:.4f}")
    R = out['reference']
    print('REF total', round(R['total_exitance_rgb'], 4), 'halves', round(R['half_a'], 4), round(R['half_b'], 4), 'n0', round(R['n0_half_split'], 4),
          'expected L1 (bin scatter)', round(math.sqrt(2 / math.pi) * se.sum() / r.sum(), 4), 'total SE', round(R['total_se_from_bin_scatter'], 4),
          'per-patch half z rms', round(R['per_patch_half_z_rms'], 3))
    print('REF face share', {k: round(v, 4) for k, v in R['face_share'].items()})
    json.dump(out, open(os.path.join(RES, 'ana_' + '_'.join(sys.argv[2:]) + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
