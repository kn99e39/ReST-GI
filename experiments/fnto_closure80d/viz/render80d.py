"""WL80C/80D quick renders (post hoc visual export, no new metric).
Mitsuba 3.5.1 llvm_ad_rgb (LabServer63's cuda film accumulation is known wrong), path tracer,
identical scene / lights / spp / seed per material: reference principled vs control (WL80C)
vs candidate (WL80D) compiled closures (Python BSDF plugin over the corrected Dr.Jit runtime).
White-furnace check first (Lambert-only closure, constant environment 1 -> image 1).

  C80_MI_VARIANT=llvm_ad_rgb python render80d.py <wl80c_smoke_dir> <wl80d_results_dir> <out_dir>
"""
import json, math, os, sys, time
import numpy as np
os.environ.setdefault('C80_MI_VARIANT', 'llvm_ad_rgb')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'fnto_compiled80c'))
import ref80c as R
mi = R.mi_init()
import drjit as dr
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import closure80c as C
import energy80d as E
import closure80d_dr as CD

SPP = 256
RES = 192
MATS = [0, 2, 3, 5]          # material 2 = WL80C worst case; first four otherwise
FLAGS = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.Anisotropic


class CompiledBSDF(mi.BSDF):
    def __init__(self, props, state=None):
        mi.BSDF.__init__(self, props)
        self.cl = CD.DrClosure(state)
        self.m_flags = FLAGS
        self.m_components = [FLAGS]

    def sample(self, ctx, si, sample1, sample2, active=True):
        wo, pdf, w, valid = self.cl.sample(si.wi, sample1, sample2, active)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo; bs.pdf = pdf; bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        bs.sampled_component = mi.UInt32(0)
        return bs, dr.select(valid, w, 0.0)

    def eval(self, ctx, si, wo, active=True):
        return self.cl.eval(si.wi, wo, active)

    def pdf(self, ctx, si, wo, active=True):
        return self.cl.pdf(si.wi, wo, active)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.cl.eval_pdf(si.wi, wo, active)

    def to_string(self):
        return 'CompiledBSDF'


_reg = {}


def compiled_dict(state, name):
    if name not in _reg:
        st = np.array(state, np.float32)
        mi.register_bsdf(name, lambda props, st=st: CompiledBSDF(props, st))
        _reg[name] = True
    return {'type': name}


def principled_dict(dd):
    b = {'type': 'principled', 'base_color': {'type': 'rgb', 'value': [float(x) for x in dd[0:3]]},
         'metallic': float(dd[3]), 'roughness': float(dd[4]), 'anisotropic': float(dd[5]),
         'clearcoat': float(dd[6]), 'clearcoat_gloss': float(dd[7]), 'sheen': float(dd[8])}
    b.update(R.FIXED)
    return b


def scene(bsdf, kind, rot=0.0):
    sens = {'type': 'perspective', 'fov': 30,
            'to_world': mi.ScalarTransform4f.look_at(origin=[0, -4.2, 1.6], target=[0, 0, 0.15], up=[0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': RES, 'height': RES, 'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': SPP}}
    sph = {'type': 'sphere', 'radius': 0.75, 'bsdf': bsdf,
           'to_world': mi.ScalarTransform4f.translate([0, 0, 0.75]) @ mi.ScalarTransform4f.rotate([0, 0, 1], rot) @ mi.ScalarTransform4f.rotate([1, 0, 0], 70)}
    sc = {'type': 'scene', 'integrator': {'type': 'path', 'max_depth': 6}, 'sensor': sens, 'obj': sph}
    if kind == 'furnace':
        sc['env'] = {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [1, 1, 1]}}
        sc['integrator'] = {'type': 'path', 'max_depth': 64}
    elif kind == 'highlight':
        sc['env'] = {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [0.08, 0.08, 0.1]}}
        sc['sun'] = {'type': 'directional', 'direction': [0.5, 0.75, -0.45], 'irradiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}}
    elif kind == 'indirect':
        sc['floor'] = {'type': 'rectangle', 'to_world': mi.ScalarTransform4f.scale([3, 3, 1]),
                       'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]}}}
        sc['wall'] = {'type': 'rectangle', 'to_world': mi.ScalarTransform4f.translate([0, 1.6, 1.5]) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90) @ mi.ScalarTransform4f.scale([3, 1.5, 1]),
                      'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.25, 0.2]}}}
        sc['light'] = {'type': 'rectangle', 'to_world': mi.ScalarTransform4f.translate([-1.2, -0.6, 3.0]) @ mi.ScalarTransform4f.rotate([1, 0, 0], 180) @ mi.ScalarTransform4f.scale([0.6, 0.6, 1]),
                       'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [12, 12, 12]}}}
    return mi.load_dict(sc)


def render(sc, seed):
    t = time.perf_counter()
    img = np.array(mi.render(sc, seed=seed, spp=SPP))
    return img, time.perf_counter() - t


def tonemap(x):
    x = np.clip(x, 0, None)
    return np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def main():
    c80, r80d, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    meta = {'variant': mi.variant(), 'spp': SPP, 'res': RES, 'materials': MATS}
    # furnace: Lambert-only compiled closure (lambda_0 = 1) in a constant environment of 1
    s = np.zeros(53, np.float32); s[24:38] = 0.3; s[38:45] = 1.0; s[[0, 8, 16]] = 1.0; s[45] = 1.0
    img, dt = render(scene(compiled_dict(s, 'c80d_furnace'), 'furnace'), 1)
    mask = img.sum(2) > 0
    meta['furnace'] = {'mean': float(img[mask].mean()), 'min': float(img[mask].min()), 'max': float(img[mask].max()), 'seconds': dt}
    print('furnace', meta['furnace'], flush=True)
    if abs(meta['furnace']['mean'] - 1) > 0.01:
        json.dump(meta, open(os.path.join(out, 'render_meta.json'), 'w'), indent=1)
        print('FURNACE FAILED', flush=True); return
    d = np.load(os.path.join(c80, 'materials.npz'))['test_iid']
    sP = np.load(os.path.join(c80, 'fit.npz'))['states']
    sC = np.load(os.path.join(r80d, 'candidate.npz'))['states']
    imgs = {}
    times = {}
    for kind, rot in [('highlight', 0.0), ('highlight', 40.0), ('indirect', 0.0)]:
        for k in MATS:
            for arm, bd in [('reference', principled_dict(d[k])), ('control', compiled_dict(sP[k], f'c80d_ctrl{k}')),
                            ('candidate', compiled_dict(sC[k], f'c80d_cand{k}'))]:
                img, dt = render(scene(bd, kind, rot), 7)
                imgs[(kind, rot, k, arm)] = img; times[f'{kind}|{rot}|{k}|{arm}'] = dt
                print(kind, rot, k, arm, round(dt, 1), flush=True)
    meta['render_seconds'] = times
    np.savez_compressed(os.path.join(out, 'renders.npz'), **{f'{a}_{b:g}_m{c}_{e}': v for (a, b, c, e), v in imgs.items()})
    errmax = max(np.abs(imgs[(a, b, c, e)] - imgs[(a, b, c, 'reference')]).mean(2).max()
                 for (a, b, c, e) in imgs if e != 'reference')
    meta['abs_error_scale_max'] = float(errmax)
    rel = {}
    for kind, rot in [('highlight', 0.0), ('highlight', 40.0), ('indirect', 0.0)]:
        fig, axs = plt.subplots(len(MATS), 5, figsize=(17, 3.4 * len(MATS)), constrained_layout=True)
        for i, k in enumerate(MATS):
            ref = imgs[(kind, rot, k, 'reference')]
            for j, arm in enumerate(['reference', 'control', 'candidate']):
                axs[i, j].imshow(tonemap(imgs[(kind, rot, k, arm)])); axs[i, j].set_title(f'material {k}: {arm}', fontsize=9)
                axs[i, j].axis('off')
            for j, arm in enumerate(['control', 'candidate']):
                e = np.abs(imgs[(kind, rot, k, arm)] - ref).mean(2)
                m = ref.mean(2) > 0.02
                rel[f'{kind}|{rot}|{k}|{arm}'] = float(e[m].sum() / ref.mean(2)[m].sum())
                im = axs[i, 3 + j].imshow(np.sqrt(e), cmap='inferno', vmin=0, vmax=math.sqrt(errmax))
                axs[i, 3 + j].set_title(f'|{arm} - ref| (sqrt, shared)  relL1 {rel[f"{kind}|{rot}|{k}|{arm}"]:.3f}', fontsize=8)
                axs[i, 3 + j].axis('off')
        cb = fig.colorbar(im, ax=axs[:, 3:], shrink=0.5); tk = np.linspace(0, math.sqrt(errmax), 5)
        cb.set_ticks(tk); cb.set_ticklabels([f'{x * x:.3g}' for x in tk]); cb.set_label('abs radiance error (mean RGB)')
        fig.suptitle(f'{kind} scene, sphere rotation {rot:g} deg, {SPP} spp, path depth 6, Mitsuba {mi.variant()} '
                     f'(includes MC noise of both images)', fontsize=10)
        fig.savefig(os.path.join(out, f'render_{kind}_rot{rot:g}.png'), dpi=80); plt.close(fig)
    meta['relL1_vs_reference_same_spp'] = rel
    json.dump(meta, open(os.path.join(out, 'render_meta.json'), 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)
