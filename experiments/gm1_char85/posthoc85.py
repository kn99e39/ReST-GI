"""WL85 POST-CHAIN supplements (run after the frozen chain; they change no frozen result or verdict).

  python posthoc85.py <root>
(1) Diagnostic for the failed qualification clause Q3b: the frozen enabled_reflection region overlaps pixels whose DIRECT light is also
    unblocked by the edit (the baffle's T0 shadow). Sub-region = enabled_reflection pixels whose segment to the light is clear in both
    configurations; same REF-P quantities.
(2) Repeated edits with an IN-PLACE geometry update (mesh vertex positions through mi.traverse), because the frozen chain rebuilt the scene
    for every edit, which recompiled the kernel (about 36 s per edit: a harness artefact, not a path-tracing cost).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pt85 as PT
import analyze85 as AN
mi, dr = PT.mi, PT.dr
root = sys.argv[1]; Rr = os.path.join(root, 'results')
FX = PT.FX
out = {}
# ---- (1)
gb = dict(np.load(os.path.join(Rr, 'gbuffer.npz'))); R = AN.regions(gb)
S = FX['shapes']; c0 = np.array(S['baffle']['center_T0']); h = np.array(S['baffle']['half']); c1 = c0 + np.array(FX['edit']['translation'])
Lp = np.array(FX['light']['position'])
clear = ~AN.seg_box(gb['p0'], Lp, c0 - h, c0 + h) & ~AN.seg_box(gb['p0'], Lp, c1 - h, c1 + h)
sub = R['enabled_reflection'] & clear
M = np.load(os.path.join(Rr, 'maps.npz'))
res = {'pixels_frozen_region': int(R['enabled_reflection'].sum()), 'pixels_sub_region': int(sub.sum())}
for ref in ['P', 'ER']:
    t0 = M[f'{ref}_T0'].mean(-1)[sub].mean(); d = M[f'{ref}_d'][sub].mean(); dd = M[f'{ref}_ddepth'][sub].mean(0)
    res[ref] = {'signed_change_rel_T0': float(d / t0), 'by_depth_rel_T0': (dd / t0).tolist(), 'k_ge_2_share_of_change': float(dd[1:].sum() / dd.sum())}
out['Q3b_diagnostic_direct_clear_subregion'] = res
# ---- (2)
dr.set_flag(dr.JitFlag.KernelHistory, True)
mats, scene, ptr, other, moved = PT.setup(root, 'T1')
params = mi.traverse(scene)
keys = [k for k in params.keys() if 'baffle' in k]
out['baffle_params'] = keys
vk = [k for k in keys if 'vertex_positions' in k]
cam = FX['camera']; res0 = cam['res']
acc, _, T_ = PT.render(scene, ptr, mats, 'ER', cam, res0, 256, 85600, other, moved, False)
out['warmup_256spp'] = {k: T_[k] for k in ['trace_s', 'eval_s', 'gpu_ms', 'wall_s']}
ed = []
if vk:
    v0 = np.asarray(params[vk[0]], np.float64).reshape(-1, 3).copy()
    cur = PT.baffle_center('T1')
    for fr in [0.25, 0.5, 0.75, 1.25, 0.0]:
        t0 = time.perf_counter()
        new = PT.baffle_center('T0', fr)
        params[vk[0]] = mi.Float(np.ascontiguousarray((v0 + (new - cur)).ravel().astype(np.float32)))
        params.update(); dr.sync_thread()
        t_up = time.perf_counter() - t0
        acc, _, T_ = PT.render(scene, ptr, mats, 'ER', cam, res0, 256, 85601, other, moved, False)
        img = PT.pack(acc, res0, 256)['tot']
        ed.append({'frac': fr, 'update_s': t_up, **{k: T_[k] for k in ['trace_s', 'eval_s', 'gpu_ms', 'wall_s']}, 'frame_mean': float(img.mean())})
    # correctness of the in-place update: fraction 0.0 must equal a freshly built T0 scene (same seed)
    m2, s2, p2, o2, mv2 = PT.setup(root, 'T0')
    acc2, _, _ = PT.render(s2, p2, m2, 'ER', cam, res0, 256, 85601, o2, mv2, False)
    i2 = PT.pack(acc2, res0, 256)['tot']
    out['inplace_vs_rebuilt_T0'] = {'frame_rel': float(abs(img.sum() - i2.sum()) / i2.sum()), 'per_pixel_max_rel_mean': float(np.abs(img - i2).max() / i2.mean())}
out['edits_in_place'] = ed
out['foreign'] = os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split()
json.dump(out, open(os.path.join(Rr, 'posthoc85.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
os._exit(0)
