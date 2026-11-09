"""WL80C Stage 4: matched GPU cost. Both arms run as Dr.Jit CUDA kernels on the same
GPU through the same tracing path; the primary metric is GPU kernel execution time
from Dr.Jit's kernel history (not Python call overhead).

Arms: REF  = Mitsuba 3.5.1 principled (C++ plugin, uniform parameters)
      CMP  = compiled closure (closure80c_dr, uniform compiled state, opaque scalars)
      SIMPLE = Mitsuba roughconductor anisotropic GGX (cost sanity only)
Workloads: io (inputs resident in GPU memory, outputs written) and fused16
(16 hashed query pairs generated per lane inside the kernel, accumulated).

  python bench80c.py <data_dir> <fit_dir> <out_dir> [--dev]
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import ref80c as R
mi = R.mi_init()
import drjit as dr
import closure80c as C
import closure80c_dr as CD
from data80c import normalise, P

dev_t = torch.device('cuda')
B = P['bench']


def gpu_apps():
    """Foreign compute processes on the GPU (this process excluded)."""
    try:
        s = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f'ERR {e}'
    me = str(os.getpid())
    return '\n'.join(l for l in s.splitlines() if l.split(',')[0].strip() != me)


def hist_ms():
    h = dr.kernel_history()
    return sum(float(k.get('execution_time', 0.0)) for k in h), len(h), sum(int(bool(k.get('cache_hit', 0))) for k in h)


def rand_dirs(rng, n):
    u = rng.random((n, 2))
    z = 0.02 + 0.98 * u[:, 0]
    r = np.sqrt(1 - z * z); ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1).astype(np.float32)


def V3(a):
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


class RefArm:
    def __init__(self, bsdf):
        self.b = bsdf
        self.ctx = mi.BSDFContext()

    def _si(self, wi):
        si = dr.zeros(mi.SurfaceInteraction3f, dr.width(wi))
        si.wi = wi
        si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
        return si

    def eval_pdf(self, wi, wo):
        return self.b.eval_pdf(self.ctx, self._si(wi), wo)

    def eval(self, wi, wo):
        return self.b.eval(self.ctx, self._si(wi), wo), None

    def pdf(self, wi, wo):
        return None, self.b.pdf(self.ctx, self._si(wi), wo)

    def sample(self, wi, u1, u2):
        bs, w = self.b.sample(self.ctx, self._si(wi), u1, u2)
        return bs.wo, bs.pdf, w


class CmpArm:
    def __init__(self, state):
        self.c = CD.DrClosure(state)

    def eval_pdf(self, wi, wo):
        return self.c.eval_pdf(wi, wo)

    def eval(self, wi, wo):
        return self.c.eval(wi, wo), None

    def pdf(self, wi, wo):
        return None, self.c.pdf(wi, wo)

    def sample(self, wi, u1, u2):
        wo, p, w, _ = self.c.sample(wi, u1, u2)
        return wo, p, w


def run_op(arm, op, mode, n, inputs):
    """Trace + launch one op; returns outputs to evaluate."""
    if mode == 'io':
        wi, wo, u1, u2 = inputs
        if op == 'sample':
            r = arm.sample(wi, u1, u2)
        else:
            r = getattr(arm, op)(wi, wo)
        return [x for x in r if x is not None]
    # fused16: per-lane PCG32 stream, 16 queries accumulated in-kernel
    rng = mi.PCG32(size=n, initstate=dr.arange(mi.UInt64, n))
    acc_f = mi.Color3f(0.0); acc_p = mi.Float(0.0)
    for _ in range(B['fused_queries']):
        a, b, c, d = rng.next_float32(), rng.next_float32(), rng.next_float32(), rng.next_float32()
        z1 = 0.02 + 0.98 * a; r1 = dr.sqrt(1 - z1 * z1); s1, c1 = dr.sincos(2 * dr.pi * b)
        wi = mi.Vector3f(r1 * c1, r1 * s1, z1)
        if op == 'sample':
            wo, p, w = arm.sample(wi, c, mi.Point2f(d, rng.next_float32()))
            acc_f += w; acc_p += p
        else:
            z2 = 0.02 + 0.98 * c; r2 = dr.sqrt(1 - z2 * z2); s2, c2 = dr.sincos(2 * dr.pi * d)
            wo = mi.Vector3f(r2 * c2, r2 * s2, z2)
            f, p = getattr(arm, op)(wi, wo)
            if f is not None:
                acc_f += f
            if p is not None:
                acc_p += p
    return [acc_f, acc_p]


def time_op(arm, op, mode, n, inputs, reps, warm):
    gpu, wall, nk = [], [], []
    for i in range(warm + reps):
        dr.sync_thread(); hist_ms()
        t = time.perf_counter()
        outs = run_op(arm, op, mode, n, inputs)
        dr.eval(*outs)
        dr.sync_thread()
        w = time.perf_counter() - t
        g, k, _ = hist_ms()
        if i >= warm:
            gpu.append(g); wall.append(w * 1e3); nk.append(k)
    q = lambda x, v: float(np.quantile(x, v))
    return {'gpu_ms_median': q(gpu, .5), 'gpu_ms_p10': q(gpu, .1), 'gpu_ms_p90': q(gpu, .9),
            'wall_ms_median': q(wall, .5), 'kernels': int(np.median(nk))}


def compile_cost(fit_dir, d_all):
    T = P['training']
    s0 = P['seeds']['train'][0]
    net = C.Compiler(T['width'], T['depth']).to(dev_t)
    net.load_state_dict(torch.load(os.path.join(fit_dir, f'compiler_s{s0}.pt'), map_location=dev_t)['state_dict'])
    net.eval()
    x1 = normalise(d_all[:1])
    res = {}
    for name, xb in [('single', x1), ('batch1024', normalise(np.repeat(d_all, 1024 // len(d_all) + 1, 0)[:1024]))]:
        ts_fwd, ts_e2e = [], []
        for i in range(60):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                st = C.compiled_state(C.decode(net(torch.from_numpy(xb).to(dev_t))))
            torch.cuda.synchronize(); t1 = time.perf_counter()
            stn = st.cpu().numpy()
            cl = CD.DrClosure(stn[0])  # bind one runtime closure (opaque scalars)
            dr.eval(cl.au[0]); dr.sync_thread()
            t2 = time.perf_counter()
            if i >= 10:
                ts_fwd.append((t1 - t) * 1e3); ts_e2e.append((t2 - t) * 1e3)
        res[name] = {'forward_ms_median': float(np.median(ts_fwd)), 'end_to_end_ms_median': float(np.median(ts_e2e))}
    return res


def main():
    data, fit_dir, out = sys.argv[1:4]
    dev = '--dev' in sys.argv
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    mats = np.load(os.path.join(data, 'materials.npz'))
    d8 = mats['test_iid'][: B['n_materials']]
    T = P['training']
    s0 = P['seeds']['train'][0]
    net = C.Compiler(T['width'], T['depth']).to(dev_t)
    net.load_state_dict(torch.load(os.path.join(fit_dir, f'compiler_s{s0}.pt'), map_location=dev_t)['state_dict'])
    with torch.no_grad():
        states = C.compiled_state(C.decode(net(torch.from_numpy(normalise(d8)).to(dev_t)))).cpu().numpy()
    env = {'apps_before': gpu_apps(), 'gpu': torch.cuda.get_device_name(0),
           'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__}
    mon = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,clocks.sm,memory.used',
                            '--format=csv,noheader', '-lms', '500'], stdout=open(os.path.join(out, 'gpu_monitor.csv'), 'w'))
    rng = np.random.default_rng(P['seeds']['bench'])
    sizes_io = B['dev_sizes'] if dev else B['sizes_io']
    sizes_fu = B['dev_sizes'] if dev else B['sizes_fused']
    reps, warm = (3, 1) if dev else (B['reps'], B['warmup'])
    rows = []
    n_mat = 2 if dev else len(d8)
    for k in range(n_mat):
        arms = {'REF': RefArm(R.uniform_principled(d8[k])), 'CMP': CmpArm(states[k])}
        if k == 0:
            arms['SIMPLE'] = RefArm(mi.load_dict({'type': 'roughconductor', 'distribution': 'ggx',
                                                  'alpha_u': 0.2, 'alpha_v': 0.05}))
        for mode, sizes in [('io', sizes_io), ('fused16', sizes_fu)]:
            for n in sizes:
                if mode == 'io':
                    wi = V3(rand_dirs(rng, n)); wo = V3(rand_dirs(rng, n))
                    u1 = mi.Float(rng.random(n).astype(np.float32))
                    u2 = mi.Point2f(mi.Float(rng.random(n).astype(np.float32)), mi.Float(rng.random(n).astype(np.float32)))
                    dr.eval(wi, wo, u1, u2); dr.sync_thread()
                    inputs = (wi, wo, u1, u2)
                else:
                    inputs = None
                for op in B['ops']:
                    for an, arm in arms.items():
                        r = time_op(arm, op, mode, n, inputs, reps, warm)
                        r.update({'material': k, 'arm': an, 'mode': mode, 'n': n, 'op': op,
                                  'queries': n * (B['fused_queries'] if mode == 'fused16' else 1)})
                        rows.append(r)
                print(k, mode, n, flush=True)
                inputs = None
    mon.terminate()
    env['apps_after'] = gpu_apps()
    cc = compile_cost(fit_dir, d8)
    # aggregate speedups REF/CMP (GPU time) per mode, size, op
    agg = {}
    for mode in ['io', 'fused16']:
        for op in B['ops']:
            for n in sorted({r['n'] for r in rows if r['mode'] == mode}):
                sp = []
                for k in range(n_mat):
                    g = {r['arm']: r for r in rows if r['material'] == k and r['mode'] == mode and r['op'] == op and r['n'] == n}
                    sp.append(g['REF']['gpu_ms_median'] / max(g['CMP']['gpu_ms_median'], 1e-9))
                ref_ms = np.median([r['gpu_ms_median'] for r in rows if r['arm'] == 'REF' and r['mode'] == mode and r['op'] == op and r['n'] == n])
                cmp_ms = np.median([r['gpu_ms_median'] for r in rows if r['arm'] == 'CMP' and r['mode'] == mode and r['op'] == op and r['n'] == n])
                q = n * (B['fused_queries'] if mode == 'fused16' else 1)
                agg[f'{mode}|{op}|{n}'] = {'speedup_median': float(np.median(sp)), 'speedup_min': float(np.min(sp)),
                                           'n_materials_speedup_ge_1.10': int(np.sum(np.array(sp) >= 1.10)),
                                           'ref_ns_per_query': float(ref_ms * 1e6 / q), 'cmp_ns_per_query': float(cmp_ms * 1e6 / q)}
    # performance verdict at the largest sizes
    big = {'io': max(sizes_io), 'fused16': max(sizes_fu)}
    need = B['speedup_min_materials'] if not dev else 1
    checks = {}
    for mode, n in big.items():
        for op in ['eval_pdf', 'sample']:
            a = agg[f'{mode}|{op}|{n}']
            checks[f'{mode}|{op}'] = a['n_materials_speedup_ge_1.10'] >= need and a['speedup_median'] >= B['speedup_threshold']
    # break-even (eval_pdf, fused16, largest size): N* = compile / (t_ref - t_cmp)
    a = agg[f"fused16|eval_pdf|{big['fused16']}"]
    dt_ns = a['ref_ns_per_query'] - a['cmp_ns_per_query']
    be = (cc['single']['end_to_end_ms_median'] * 1e6 / dt_ns) if dt_ns > 0 else None
    contention = env['apps_before'] != '' or env['apps_after'] != ''
    verdict = {'checks': checks, 'PERFORMANCE_PASS': all(checks.values()) and not contention,
               'contention_detected': contention, 'break_even_queries_eval_pdf': be}
    json.dump({'env': env, 'rows': rows, 'agg': agg, 'compile_cost': cc, 'verdict': verdict,
               'seconds': time.time() - t0}, open(os.path.join(out, 'stage4.json'), 'w'), indent=1)
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
