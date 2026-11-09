"""WL81 Stage E: GPU cost on the same GPU and runtime path (Dr.Jit CUDA kernels, kernel-history GPU time).

  python bench81.py <data> <evalD_dir> <fitD_dir> <out> <primary_arm> [<other_arm> ...]

Arms: REF_1 (layered reference, one walk per query; eval and sample; no closed-form pdf), CTL / CAN (compiled closures, corrected
runtime closure81_dr, seed-8101 compiled states of the first 8 test_iid materials), PRIN (Mitsuba principled sanity comparator).
REF_Neq = N_eq x REF_1 per-query GPU time at the largest size (protocol stage_E_gpu). Reused from WL80C bench80c (imported):
gpu_apps, hist_ms, time_op / run_op machinery, RefArm (principled), CmpArm (with bench80c.CD = closure81_dr).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import bench80c as BE  # noqa: E402
import closure81_dr as CD  # noqa: E402
import layered81 as L  # noqa: E402
import energy81 as E  # noqa: E402
import fit81 as F  # noqa: E402
from data81 import normalise, P  # noqa: E402

BE.CD = CD
mi = BE.mi
import drjit as dr  # noqa: E402
SD = P['seeds']
SE = P['stage_E_gpu']
dev_t = torch.device('cuda')
PRIN_DESC = [0.5, 0.4, 0.3, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5]   # fixed principled sanity material (base rgb, metallic, roughness, aniso, cc, cc_gloss, sheen)


class LayArm:
    """Layered reference, uniform material (opaque scalars), one walk per query."""

    def __init__(self, desc, seed):
        v = []
        for x in desc:
            f = mi.Float(float(x)); dr.make_opaque(f); v.append(f)
        self.m = L.Mat(v[0], v[1], v[2], mi.Color3f(v[3], v[4], v[5]), mi.Color3f(v[6], v[7], v[8]), v[9])
        self.seed = seed
        self.calls = 0

    def _rng(self, n):
        self.calls += 1
        return mi.PCG32(size=n, initstate=self.seed + self.calls, initseq=dr.arange(mi.UInt64, n))

    def eval(self, wi, wo):
        return L.layered_eval(self.m, wi, wo, self._rng(dr.width(wi))), None

    def sample(self, wi, u1, u2):
        wo, w, ok = L.layered_sample(self.m, wi, self._rng(dr.width(wi)))
        return wo, mi.Float(0.0), w


def bench_arm(arm, ops, rng, sizes_io, sizes_fu, reps, warm):
    rows = []
    for mode, sizes in [('io', sizes_io), ('fused16', sizes_fu)]:
        for n in sizes:
            wi = BE.rand_dirs(rng, n); wo = BE.rand_dirs(rng, n)
            u1 = rng.random(n).astype(np.float32); u2 = rng.random((n, 2)).astype(np.float32)
            inputs = (BE.V3(wi), BE.V3(wo), mi.Float(u1), mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1])))
            dr.eval(*inputs[:2], inputs[2], inputs[3])
            for op in ops:
                fa = BE.gpu_apps()
                r = BE.time_op(arm, op, mode, n, inputs, reps, warm)
                fb = BE.gpu_apps()
                r.update({'mode': mode, 'n': n, 'op': op, 'foreign_before': fa, 'foreign_after': fb})
                rows.append(r)
    return rows


def main():
    data, evalD, fitD, out = sys.argv[1:5]
    arms = [a for a in sys.argv[5:] if not a.startswith('--')]
    primary = arms[0]
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SD['bench'])
    mats = np.load(os.path.join(data, 'materials.npz'))
    d8 = mats['test_iid'][:8]
    s0 = P['fitting']['compiler']['seeds'][0]
    sizes_io, sizes_fu = SE['sizes_io'], SE['sizes_fused']
    if '--dev' in sys.argv:
        sizes_io, sizes_fu = [1024, 65536], [1024, 65536]
    reps, warm = SE['reps'], SE['warmup']
    res = {'gpu': torch.cuda.get_device_name(0), 'rows': {}}
    # compiled closures
    for arm in arms:
        st = np.load(os.path.join(evalD, f'states_{arm}_s{s0}_test_iid.npy'))[:8]
        res['rows'][arm] = []
        for k in range(8):
            rr = bench_arm(BE.CmpArm(st[k]), SE['ops'], rng, sizes_io, sizes_fu, reps, warm)
            for r in rr:
                r['material'] = k
            res['rows'][arm] += rr
            print(arm, k, json.dumps({f"{r['mode']}|{r['n']}|{r['op']}": r['gpu_ms_median'] for r in rr if r['n'] == max(sizes_io)}, default=str), flush=True)
    # layered reference, one walk
    res['rows']['REF_1'] = []
    for k in range(8):
        rr = bench_arm(LayArm(d8[k], SD['bench'] + 1000 * k), ['eval', 'sample'], rng, sizes_io, sizes_fu, reps, warm)
        for r in rr:
            r['material'] = k
        res['rows']['REF_1'] += rr
        print('REF_1', k, flush=True)
    # principled sanity
    res['rows']['PRIN'] = bench_arm(BE.RefArm(BE.R.uniform_principled(PRIN_DESC)), SE['ops'], rng, sizes_io, sizes_fu, reps, warm)
    # matched quality N_eq per material (primary arm, seed s0)
    nz = np.load(os.path.join(data, 'noise_test_iid.npz'))['el1']
    nwalk = json.load(open(os.path.join(data, 'noise_test_iid.json')))['walks']
    lane = np.load(os.path.join(evalD, f'lane_{primary}_s{s0}_test_iid.npz'))['el1']
    nu1 = math.sqrt(nwalk) * np.median(nz[:8].reshape(8, -1), 1)
    ecl = np.median(lane[:8].reshape(8, -1), 1)
    neq = np.clip(np.ceil((nu1 / np.maximum(ecl, 1e-12)) ** 2), 1, 65536)
    res['N_eq'] = {'nu1': nu1.tolist(), 'e_cl': ecl.tolist(), 'N_eq': neq.tolist()}
    # compile latency (compiler forward + decode + energy compile + state + runtime binding), 1 and 1024 materials
    lat = {}
    xall = torch.from_numpy(normalise(np.concatenate([mats['train'], mats['val']])[:1024])).to(dev_t)
    for arm in arms:
        net, _ = F.load_compiler(os.path.join(fitD, f'compiler_{arm}_s{s0}.pt'))
        grids = E.compile_grids(dev_t) if arm == 'candidate' else None
        u = K.strat_u(128, 128, SD['compile_M'], dev_t) if arm == 'candidate' else None
        for nb in [1, 1024]:
            ts = []
            nrep = (60 if nb == 1 else 4) if '--dev' not in sys.argv else 3
            skip = 10 if nb == 1 and '--dev' not in sys.argv else 1
            for it in range(nrep):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    p, _ = E.compiled_params(arm, net(xall[:nb]), grids, u)
                    stt = C.compiled_state(p).cpu().numpy()
                cl = CD.DrClosure81(stt, mi.UInt32(np.arange(nb, dtype=np.uint32)) if nb > 1 else None) if nb > 1 else CD.DrClosure81(stt[0])
                dr.eval(cl.lam[0][0]); dr.sync_thread()
                ts.append(time.perf_counter() - t0)
            lat[f'{arm}_{nb}'] = {'median_s': float(np.median(ts[skip:])), 'p90_s': float(np.quantile(ts[skip:], .9)), 'reps': nrep - skip}
    res['compile_latency'] = lat
    res['state_bytes'] = {'compiled_state_floats': 53, 'bytes_fp32': 212, 'runtime_table_floats': 88,
                          'compiler_params': sum(p.numel() for p in F.Compiler('control', 128, 3).parameters()), 'descriptor_floats': 10}
    res['peak_torch_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    # decision
    big_io, big_fu = max(sizes_io), max(sizes_fu)

    def t(rows, mode, n, op, k=None):
        v = [r['gpu_ms_median'] for r in rows if r['mode'] == mode and r['n'] == n and r['op'] == op and (k is None or r.get('material') == k)]
        return v[0] if v else None
    dec = {}
    for mode, n in [('io', big_io), ('fused16', big_fu)]:
        ev_ratio, sa_ratio = [], []
        for k in range(8):
            ref1 = t(res['rows']['REF_1'], mode, n, 'eval', k)
            refs = t(res['rows']['REF_1'], mode, n, 'sample', k)
            cep = t(res['rows'][primary], mode, n, 'eval_pdf', k)
            csa = t(res['rows'][primary], mode, n, 'sample', k)
            ev_ratio.append(neq[k] * ref1 / cep)
            sa_ratio.append(refs / csa)
        dec[mode] = {'eval_pdf_speedup_vs_REF_Neq': ev_ratio, 'sample_speedup_vs_REF_1': sa_ratio,
                     'eval_ok_count': int(sum(r >= SE.get('speedup_threshold', 1.10) for r in ev_ratio)),
                     'sample_ok_count': int(sum(r >= 1.10 for r in sa_ratio)),
                     'median_eval_speedup': float(np.median(ev_ratio)), 'median_sample_speedup': float(np.median(sa_ratio))}
    lin = {k: (t(res['rows']['REF_1'], 'io', big_io, 'eval', k) / t(res['rows']['REF_1'], 'io', 1 << 20, 'eval', k))
           for k in range(8)} if (1 << 20) in sizes_io else None
    res['REF_linearity_2^24_over_2^20'] = lin
    foreign = any(r['foreign_before'] or r['foreign_after'] for rows in res['rows'].values() for r in rows)
    res['foreign_gpu_process'] = bool(foreign)
    res['decision_speed_only'] = {'per_mode': dec,
                                  'PASS_speed': bool(all(v['eval_ok_count'] >= 6 and v['sample_ok_count'] >= 6 for v in dec.values()) and not foreign),
                                  'note': 'GPU_RUNTIME_UTILITY additionally requires the primary arm LEARNED_RULE_GENERALIZATION and PHYSICAL_VALIDITY PASS (decided in the verdict step)'}
    json.dump(res, open(os.path.join(out, 'stageE.json'), 'w'), indent=1, default=str)
    print(json.dumps(res['decision_speed_only'], indent=1, default=str))


if __name__ == '__main__':
    main()
    os._exit(0)
