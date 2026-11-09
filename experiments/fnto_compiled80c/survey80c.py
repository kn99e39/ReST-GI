"""WL80C Stage 0a: reference-only qualification survey (no candidate code involved).

Measures, for the Mitsuba principled reference over the candidate descriptor box:
  * texture-driven vs uniform-parameter eval identity (adapter check)
  * reciprocity of f = eval / cos_o
  * directional albedo a(wi) = int f cos dwo (defensive-mixture MC, float64 accumulation)
Writes results/fnto_compiled80c/stage0/survey.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ref80c as R

OUT = os.environ.get('C80_OUT', os.path.expanduser('~/rest_compiled80c/results/stage0'))
os.makedirs(OUT, exist_ok=True)
BOX = {'base': (0.05, 0.95), 'metallic': (0, 1), 'roughness': (0.15, 0.8), 'anisotropic': (0, 0.9),
       'clearcoat': (0, 1), 'clearcoat_gloss': (0, 0.8), 'sheen': (0, 1)}


def sample_box(rng, n):
    d = np.empty((n, 9), np.float32)
    d[:, 0:3] = rng.uniform(*BOX['base'], (n, 3))
    for j, k in zip(range(3, 9), ['metallic', 'roughness', 'anisotropic', 'clearcoat', 'clearcoat_gloss', 'sheen']):
        d[:, j] = rng.uniform(*BOX[k], n)
    return d


def main():
    t0 = time.time()
    rng = np.random.default_rng(80300)
    res = {'box': BOX, 'variant': os.environ.get('C80_MI_VARIANT', 'cuda_ad_rgb')}
    # 1. adapter identity: textured vs uniform
    d16 = sample_box(rng, 16)
    bt, n = R.textured_principled(d16)
    wi = R.uniform_dirs(rng.random((16 * 256, 2))); wo = R.uniform_dirs(rng.random((16 * 256, 2)))
    mi_idx = np.repeat(np.arange(16), 256)
    ft, pt = R.ref_eval_pdf(bt, n, mi_idx, wi, wo)
    worst = 0.0
    for k in range(16):
        bu = R.uniform_principled(d16[k])
        s = slice(k * 256, (k + 1) * 256)
        fu, pu = R.ref_eval_pdf(bu, 1, np.zeros(256), wi[s], wo[s])
        worst = max(worst, float(np.max(np.abs(fu - ft[s]) / (np.abs(fu) + 1e-6))),
                    float(np.max(np.abs(pu - pt[s]) / (np.abs(pu) + 1e-6))))
    res['texture_vs_uniform_max_rel'] = worst
    # 2. reciprocity
    nm = 2000
    d = sample_box(rng, nm)
    bsdf, n = R.textured_principled(d)
    m = np.repeat(np.arange(nm), 512)
    wi = R.uniform_dirs(rng.random((len(m), 2))); wo = R.uniform_dirs(rng.random((len(m), 2)))
    keep = (wi[:, 2] > 0.02) & (wo[:, 2] > 0.02)
    m, wi, wo = m[keep], wi[keep], wo[keep]
    f1, _ = R.ref_eval_pdf(bsdf, n, m, wi, wo)
    f2, _ = R.ref_eval_pdf(bsdf, n, m, wo, wi)
    g1 = f1 / wo[:, 2:3]; g2 = f2 / wi[:, 2:3]
    rel = np.abs(g1 - g2) / np.maximum(np.maximum(np.abs(g1), np.abs(g2)), 1e-4)
    res['reciprocity'] = {'pairs': int(len(m)), 'median_rel': float(np.median(rel)),
                          'p99_rel': float(np.quantile(rel, 0.99)), 'max_rel': float(rel.max()),
                          'frac_rel_gt_1e-3': float(np.mean(rel > 1e-3))}
    # 3. albedo survey: 2000 materials x 24 wi x 2048 samples
    n_wi, n_s = 24, 2048
    ct = np.linspace(np.cos(np.radians(88)), 1.0, n_wi)
    alb = np.zeros((nm, n_wi, 3)); alb_se = np.zeros((nm, n_wi, 3))
    for k0 in range(0, nm, 100):
        ks = np.arange(k0, min(k0 + 100, nm))
        mm = np.repeat(ks, n_wi * n_s)
        th = np.tile(np.repeat(np.arccos(ct), n_s), len(ks))
        ph = rng.uniform(0, 2 * np.pi, len(mm))
        wi = R.sph(th, ph)
        wo, q, f, use = R.mixture_dirs(bsdf, n, mm, wi, rng)
        g = np.where(use[:, None], f.astype(np.float64) / q[:, None], 0.0).reshape(len(ks), n_wi, n_s, 3)
        alb[ks] = g.mean(2)
        alb_se[ks] = g.std(2) / np.sqrt(n_s)
    lum = alb.max(2)
    z = (alb - 1.0) / np.maximum(alb_se, 1e-9)
    viol = (alb > 1.0) & (z > 4)
    res['albedo'] = {'materials': nm, 'wi_cos_theta': ct.tolist(), 'samples': n_s,
                     'max_any_channel': float(lum.max()),
                     'frac_lanes_any_channel_gt1_4sigma': float(np.mean(viol.any(2))),
                     'frac_materials_any_violation': float(np.mean(viol.any((1, 2)))),
                     'max_rel_se': float(np.max(alb_se / np.maximum(alb, 1e-6)))}
    # attribution of violations: per-parameter means of violating vs non-violating materials
    vm = viol.any((1, 2))
    res['albedo']['violating_material_param_mean'] = d[vm].mean(0).tolist() if vm.any() else None
    res['albedo']['all_material_param_mean'] = d.mean(0).tolist()
    # worst cos(theta_i) per channel
    res['albedo']['max_by_wi'] = alb.max((0, 2)).tolist()
    np.savez_compressed(os.path.join(OUT, 'survey_albedo.npz'), desc=d, alb=alb, alb_se=alb_se, ct=ct)
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, 'survey.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
