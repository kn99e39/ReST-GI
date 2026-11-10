"""WL83 Stage D economics: Dr.Jit CUDA kernels, kernel-history GPU time (WL80C bench80c machinery, imported).

  python bench83.py <data83> <w81> <fitdir> <ckdir> <evaldir> <out> [--dev]
Arms (per material, the first 8 test83_iid materials): REF_1 (layered reference, one walk per query; eval and sample; no closed-form pdf),
C0_L / E_L / ER_L (learned closures, compiler seed 8301, closure83_dr uniform binding). Ops eval / pdf / eval_pdf / sample; modes
io (inputs resident, outputs written; memory-bound) and fused16 (16 in-kernel queries per lane; compute-bound); plus table mode
(io, 1024-material table with a random material index per lane: material-state gather bandwidth) for the closures.
REF_Neq = N_eq x REF_1 eval time, N_eq = ceil((nu_1 / e_L)^2) per material and arm (nu_1 = single-walk E_L1 noise from the split-half
reference noise, e_L = that material's learned median E_L1). Compiler / binding latency for 1 and 1024 materials; state sizes.
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch, see closure83.py)
import bench80c as BE  # noqa: E402
import bench81 as B81  # noqa: E402
import closure83_dr as CD  # noqa: E402
import compiler83 as CP  # noqa: E402
import data83 as D83  # noqa: E402
from data81 import normalise  # noqa: E402

mi = BE.mi
import drjit as dr  # noqa: E402
P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
BB = P83['bench']
SD = P83['seeds']
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv
ARMS = ['C0', 'E', 'ER']


class Arm83:
    def __init__(self, arm, st, idx=None):
        self.c = CD.DrClosure83(arm, st, idx)

    def eval_pdf(self, wi, wo):
        return self.c.eval_pdf(wi, wo)

    def eval(self, wi, wo):
        return self.c.eval(wi, wo), None

    def pdf(self, wi, wo):
        return None, self.c.pdf(wi, wo)

    def sample(self, wi, u1, u2):
        wo, p, w, _ = self.c.sample(wi, u1, u2)
        return wo, p, w


def main():
    data, w81, fitdir, ckdir, evaldir, out = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    rng = np.random.default_rng(SD['bench'])
    d = D83.descs(data, w81, 'test83_iid')
    nmat = 2 if DEV else BB['n_materials']
    s0 = P83['compiler']['seeds'][0]
    states = {}
    for arm in ARMS:
        z = torch.from_numpy(np.load(os.path.join(fitdir, f'fit_test83_iid_L_{arm}_s{s0}.npz'))['z']).to(dev_t)
        with torch.no_grad():
            states[arm] = CD.compiled_state(arm, CL.decode(arm, 'cert', z))
    sizes_io = [1 << 16, 1 << 18] if DEV else BB['sizes_io']
    sizes_fu = [1 << 16] if DEV else BB['sizes_fused']
    reps, warm = (3, 1) if DEV else (BB['reps'], BB['warmup'])
    env = {'gpu': torch.cuda.get_device_name(0), 'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__,
           'driver': os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader').read().strip(), 'apps_before': BE.gpu_apps()}
    rows = []
    for k in range(nmat):
        arms = {a: Arm83(a, states[a][k]) for a in ARMS}
        arms['REF_1'] = B81.LayArm(d[k], SD['bench'] + 1000 * k)
        for an, arm in arms.items():
            ops = ['eval', 'sample'] if an == 'REF_1' else BB['ops']
            rr = B81.bench_arm(arm, ops, rng, sizes_io, sizes_fu, reps, warm)
            for r in rr:
                r.update({'material': k, 'arm': an})
            rows += rr
        print('material', k, flush=True)
    # table mode: 1024-material table, random material per lane (closures only), io
    tab_rows = []
    zt = {a: np.load(os.path.join(fitdir, f'fit_train_eval_L_{a}_s{s0}.npz'))['z'] for a in ARMS}
    for a in ARMS:
        with torch.no_grad():
            stt = CD.compiled_state(a, CL.decode(a, 'cert', torch.from_numpy(np.resize(zt[a], (1024, CL.N_RAW))).to(dev_t)))
        for n in sizes_io:
            idx = mi.UInt32(rng.integers(0, 1024, n).astype(np.uint32))
            arm = Arm83(a, stt, idx)
            wi = BE.V3(BE.rand_dirs(rng, n)); wo = BE.V3(BE.rand_dirs(rng, n))
            u1 = mi.Float(rng.random(n).astype(np.float32)); u2 = mi.Point2f(mi.Float(rng.random(n).astype(np.float32)), mi.Float(rng.random(n).astype(np.float32)))
            dr.eval(wi, wo, u1, u2)
            for op in BB['ops']:
                r = BE.time_op(arm, op, 'io', n, (wi, wo, u1, u2), reps, warm)
                r.update({'arm': a, 'mode': 'io_table1024', 'n': n, 'op': op})
                tab_rows.append(r)
    # matched quality N_eq (learned seed s0, test83_iid)
    ev = json.load(open(os.path.join(evaldir, 'eval83_test83_iid.json')))
    nz = np.array(ev['noise_floor']['per_material_E_L1_median'])[:nmat]
    nwalk = json.load(open(os.path.join(data, 'noise83_test83_iid.json')))['walks']
    nu1 = math.sqrt(nwalk) * nz
    neq = {}
    for a in ARMS:
        el = np.array(ev['closures'][f'L_{a}_s{s0}']['per_material_E_L1_mean'])[:nmat]
        neq[a] = np.clip(np.ceil((nu1 / np.maximum(el, 1e-12)) ** 2), 1, 65536).tolist()
    # compiler forward + decode + state + binding latency
    lat = {}
    xall = torch.from_numpy(normalise(np.resize(d, (1024, 10)))).to(dev_t)
    for a in ARMS:
        net, ck = CP.load_net(ckdir, a, s0)
        for nb in [1, 1024]:
            ts, tf = [], []
            nrep = 3 if DEV else (60 if nb == 1 else 12)
            skip = 1 if DEV else (10 if nb == 1 else 2)
            for it in range(nrep):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    p = CL.decode(a, 'cert', net(xall[:nb]))
                torch.cuda.synchronize(); t1 = time.perf_counter()
                stt = CD.compiled_state(a, p)
                cl = CD.DrClosure83(a, stt, mi.UInt32(np.arange(nb, dtype=np.uint32))) if nb > 1 else CD.DrClosure83(a, stt[0])
                dr.eval(cl.lam[0][0]); dr.sync_thread()
                t2 = time.perf_counter()
                if it >= skip:
                    tf.append(t1 - t0); ts.append(t2 - t0)
            lat[f'{a}_{nb}'] = {'forward_decode_ms_median': 1e3 * float(np.median(tf)), 'end_to_end_ms_median': 1e3 * float(np.median(ts)), 'reps': nrep - skip}
    env['apps_after'] = BE.gpu_apps()

    def t(a, mode, n, op, k):
        v = [r['gpu_ms_median'] for r in rows if r['arm'] == a and r['mode'] == mode and r['n'] == n and r['op'] == op and r['material'] == k]
        return v[0] if v else None
    big = {'io': max(sizes_io), 'fused16': max(sizes_fu)}
    econ = {}
    for a in ARMS:
        per = {}
        for mode, n in big.items():
            ev_sp = [neq[a][k] * t('REF_1', mode, n, 'eval', k) / t(a, mode, n, 'eval_pdf', k) for k in range(nmat)]
            sa_sp = [t('REF_1', mode, n, 'sample', k) / t(a, mode, n, 'sample', k) for k in range(nmat)]
            per[mode] = {'eval_pdf_speedup_vs_REF_Neq': ev_sp, 'sample_speedup_vs_REF_1': sa_sp,
                         'eval_ok': int(sum(x >= 1.10 for x in ev_sp)), 'sample_ok': int(sum(x >= 1.10 for x in sa_sp)),
                         'eval_pdf_ms_median': float(np.median([t(a, mode, n, 'eval_pdf', k) for k in range(nmat)])),
                         'sample_ms_median': float(np.median([t(a, mode, n, 'sample', k) for k in range(nmat)])),
                         'vs_C0_eval_pdf_ratio_median': float(np.median([t(a, mode, n, 'eval_pdf', k) / t('C0', mode, n, 'eval_pdf', k) for k in range(nmat)])),
                         'vs_C0_sample_ratio_median': float(np.median([t(a, mode, n, 'sample', k) / t('C0', mode, n, 'sample', k) for k in range(nmat)]))}
        need = BB['speedup_min_materials'] if not DEV else 1
        econ[a] = {'per_mode': per, 'PASS_speed': bool(all(v['eval_ok'] >= need and v['sample_ok'] >= need for v in per.values()))}
    foreign = bool(env['apps_before'] or env['apps_after'])
    res = {'env': env, 'rows': rows, 'table_rows': tab_rows, 'N_eq': {'nu1': nu1.tolist(), 'per_arm': neq}, 'compile_latency': lat,
           'state': {'compiled_state_floats': 69, 'bytes_fp32': 276, 'runtime_table_floats': 104,
                     'compiler_params': sum(p.numel() for p in CP.Net('E').parameters()), 'descriptor_floats': 10},
           'economics': econ, 'foreign_gpu_process': foreign}
    json.dump(res, open(os.path.join(out, 'bench83.json'), 'w'), indent=1, default=str)
    print(json.dumps({a: {m: {k: v for k, v in x.items() if 'speedup' not in k} for m, x in e['per_mode'].items()} for a, e in econ.items()}, indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
