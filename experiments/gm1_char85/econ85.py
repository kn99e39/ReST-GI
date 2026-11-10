"""WL85 Stage 4: measured full-rendering economics of REF-ER on the GM1 fixture (run ALONE on the GPU).

  python econ85.py <root> [--dev]
Everything is a measurement of path tracing with frozen ER; no reuse algorithm is implemented or estimated.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt85 as PT  # noqa: E402
mi, dr = PT.mi, PT.dr

P = PT.P
E = P['stage4_economics']
FX = PT.FX
DEV = '--dev' in sys.argv


def foreign():
    me = str(os.getpid())
    return ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)


def mem():
    s = os.popen(f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').read()
    for l in s.splitlines():
        if l.split(',')[0].strip() == str(os.getpid()):
            return l.split(',')[1].strip()
    return None


def tkeys(T):
    return {k: T[k] for k in ['trace_s', 'eval_s', 'readback_s', 'gpu_ms', 'wall_s', 'launches', 'spp']}


def main():
    root = sys.argv[1]
    Rr = os.path.join(root, 'results')
    out = os.path.join(Rr, 'econ'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    res0 = FX['camera']['res']; cam = FX['camera']
    R = {'foreign_before': foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(),
         'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__}}
    maps = np.load(os.path.join(Rr, 'maps.npz'))
    ref = maps['ER_T1'].astype(np.float64)
    masks = {k: maps['mask_' + k] for k in ['stationary_all', 'enabled_reflection', 'frame']}
    rel = lambda img, m: float(np.abs(img[m] - ref[m]).sum() / max(ref[m].sum(), 1e-30))
    # ---- (a) cold start in this fresh process: scene construction + first render (kernel compilation)
    t0 = time.perf_counter()
    mats, scene, ptr, other, moved = PT.setup(root, 'T1')
    R['scene_build_s'] = time.perf_counter() - t0
    acc, _, T0_ = PT.render(scene, ptr, mats, 'ER', cam, res0, 64, 85400, other, moved, False)
    R['cold_first_render_64spp'] = tkeys(T0_); R['mem_after_first'] = mem()
    acc, _, T1_ = PT.render(scene, ptr, mats, 'ER', cam, res0, 64, 85401, other, moved, False)
    R['warm_second_render_64spp'] = tkeys(T1_)
    # ---- (b) time-to-quality ladder (warm), 4 replicates
    lad = E['ladder_spp'] if not DEV else [1, 16, 64]
    R['ladder'] = []
    for spp in lad:
        for q in range(E['ladder_replicates'] if not DEV else 2):
            acc, _, T_ = PT.render(scene, ptr, mats, 'ER', cam, res0, spp, E['ladder_seed_base'] + 97 * spp + q, other, moved, False)
            img = PT.pack(acc, res0, spp)['tot'].astype(np.float64)
            R['ladder'].append({'spp': spp, 'rep': q, **tkeys(T_), **{'err_' + k: rel(img, m) for k, m in masks.items()}})
        print('ladder', spp, json.dumps(R['ladder'][-1]), flush=True)
    # ---- launch-size variant (same work in fewer, larger launches)
    R['launch_size'] = []
    for b in ([32, 128] if not DEV else [32]):
        acc, _, T_ = PT.render(scene, ptr, mats, 'ER', cam, res0, 1024 if not DEV else 64, 85450 + b, other, moved, False, b=b)
        R['launch_size'].append({'launch_spp': b, **tkeys(T_), 'mem': mem()})
    # ---- (c) resolution scaling, 64 spp (second of two runs)
    R['resolution'] = []
    for r in (E['scaling']['res'] if not DEV else [256]):
        for it in range(2):
            acc, _, T_ = PT.render(scene, ptr, mats, 'ER', cam, r, 64, 85460 + it, other, moved, False, b=32 if r <= 512 else 8)
        R['resolution'].append({'res': r, **tkeys(T_), 'mem': mem()})
    # ---- (d) instrumentation overhead and REF-P cost
    R['instrumented'] = []; R['ref_p'] = []
    for spp in ([64, 1024] if not DEV else [64]):
        for it in range(2):
            acc, _, Ti = PT.render(scene, ptr, mats, 'ER', cam, res0, spp, 85470 + it, other, moved, True)
        R['instrumented'].append(tkeys(Ti))
        for it in range(2):
            acc, _, Tp = PT.render(scene, ptr, mats, 'P', cam, res0, spp, 85480 + it, other, moved, False)
        R['ref_p'].append(tkeys(Tp))
    R['mem_after_instrumented'] = mem()
    # ---- (e) second view after the same edit
    R['view2'] = []
    for spp in ([64, 1024] if not DEV else [64]):
        for it in range(2):
            acc, _, T_ = PT.render(scene, ptr, mats, 'ER', FX['camera2'], res0, spp, 85490 + it, other, moved, False)
        R['view2'].append(tkeys(T_))
    # ---- (f) repeated edits: in-place update of the baffle transform, then 256 spp
    R['edits'] = []
    params = mi.traverse(scene)
    key = [k for k in params.keys() if 'baffle' in k and 'to_world' in k]
    R['edit_param'] = key
    for fr in E['repeated_edits_fracs'] if 'repeated_edits_fracs' in E else [0.25, 0.5, 0.75, 1.25]:
        t0 = time.perf_counter()
        bc = PT.baffle_center('T0', fr)
        inplace = False
        if key:
            try:
                params[key[0]] = mi.Transform4f((PT.T.translate([float(x) for x in bc]) @ PT.T.scale(FX['shapes']['baffle']['half'])).matrix)
                params.update()
                sc2, ptr2 = scene, ptr
                inplace = True
            except Exception as ex:
                R['edit_inplace_error'] = str(ex)[:300]
        if not inplace:
            sc2, ptr2 = PT.load_scene('T0', fr)
        dr.sync_thread()
        t_up = time.perf_counter() - t0
        acc, _, T_ = PT.render(sc2, ptr2, mats, 'ER', cam, res0, 256 if not DEV else 16, 85495, other, moved, False)
        R['edits'].append({'frac': fr, 'update_s': t_up, 'in_place': inplace, **tkeys(T_)})
    t0 = time.perf_counter(); sc3, _ = PT.load_scene('T1'); dr.sync_thread()
    R['scene_rebuild_s'] = time.perf_counter() - t0
    # ---- (g) Mitsuba megakernel path integrator with the cmp83 plugin (context)
    try:
        sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled83'))
        import render83 as RR
        bs = {}
        for name in PT.SHAPES:
            RR.REG[f'e85_{name}'] = ('ER', mats.states[PT.MAT_OF[name]])
            bs[name] = {'type': 'cmp83', 'key': f'e85_{name}'}
        msc = mi.load_dict(PT.scene_dict('T1', None, bs))
        integ = mi.load_dict({'type': 'path', 'max_depth': PT.D + 1, 'rr_depth': 1000})
        sampler = mi.load_dict({'type': 'independent'})
        R['mitsuba_path'] = []
        n = res0 * res0
        for spp in ([64, 64, 1024] if not DEV else [16]):
            t0 = time.perf_counter(); g = 0.0; tr = 0.0; done = 0; k = 0
            accm = np.zeros((n, 3))
            while done < spp:
                bb = min(32, spp - done)
                ta = time.perf_counter()
                dr.kernel_history()
                sampler.seed(85499 + k, n * bb)
                rng = mi.PCG32(size=n * bb, initstate=dr.opaque(mi.UInt64, 777 + k), initseq=mi.UInt64(0))
                rays, pix = PT.gen_rays(cam, res0, bb, rng)
                Lr, _, _ = integ.sample(msc, sampler, rays, None, True)
                tr += time.perf_counter() - ta
                dr.eval(Lr); dr.sync_thread()
                bufs = []
                for c in range(3):
                    buf = dr.zeros(mi.Float, n); dr.scatter_reduce(dr.ReduceOp.Add, buf, Lr[c], pix); bufs.append(buf)
                dr.eval(bufs); dr.sync_thread()
                g += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
                for c in range(3):
                    accm[:, c] += np.asarray(bufs[c], np.float64)
                done += bb; k += 1
            R['mitsuba_path'].append({'spp': spp, 'wall_s': time.perf_counter() - t0, 'gpu_ms': g, 'trace_s': tr,
                                      'err_stationary_all': rel((accm / spp).reshape(res0, res0, 3), masks['stationary_all'])})
    except Exception as ex:
        R['mitsuba_path'] = f'failed: {ex}'
    R['foreign_after'] = foreign(); R['mem_end'] = mem()
    # ---- budget table: cost to reach quality q (log-log interpolation of ladder medians)
    an = json.load(open(os.path.join(Rr, 'analysis.json')))
    gst = an['regions']['stationary_all']['gap_T1']['rel_l1']
    R['g_stationary'] = gst
    L = {}
    for row in R['ladder']:
        L.setdefault(row['spp'], []).append(row)
    med = [{'spp': s, **{k: float(np.median([r[k] for r in rows])) for k in ['gpu_ms', 'wall_s', 'trace_s', 'eval_s', 'err_stationary_all', 'err_enabled_reflection', 'err_frame']},
            'err_stationary_all_minmax': [float(min(r['err_stationary_all'] for r in rows)), float(max(r['err_stationary_all'] for r in rows))]} for s, rows in sorted(L.items())]
    R['ladder_median'] = med

    def cost(q, ek, ck):
        e = [m[ek] for m in med]; c = [m[ck] for m in med]
        if q >= e[0]:
            return {'value': c[0], 'how': 'at_or_below_first_point'}
        for i in range(len(e) - 1):
            if e[i] >= q >= e[i + 1] and e[i + 1] < e[i]:
                s = math.log(q / e[i]) / math.log(e[i + 1] / e[i])
                return {'value': float(c[i] * (c[i + 1] / c[i]) ** s), 'how': 'interpolated'}
        return {'value': None, 'how': 'not reached on the ladder (extrapolation not reported)', 'best_error': e[-1]}
    R['budget'] = {}
    for name, q in [('g_stationary', gst), ('g_stationary_half', gst / 2), ('0.05', 0.05), ('0.02', 0.02), ('0.01', 0.01)]:
        R['budget'][name] = {'q': q, 'gpu_ms': cost(q, 'err_stationary_all', 'gpu_ms'), 'wall_s': cost(q, 'err_stationary_all', 'wall_s')}
    json.dump(R, open(os.path.join(out, 'econ85.json'), 'w'), indent=1)
    print(json.dumps(R['budget']), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
