"""WL97 audit, track E5: independent timing harness for the lifecycle-cost half of P0-4 (WL96A Stage 3).

Own harness: does NOT import w96a.py / e2e91.py. It calls the frozen production components that are being timed
(pt86.Ctx.set_geom / render / state_rays, ecro86.build_state / update_state / dirty_rays, quad87.Quad / qtable, xport87.XOp,
model86.GeoState / ECRO.factors / apply via mat94.load_frozen) from the read-only checkout <CODE> (commit a1718f7).

  python e5_time.py <R_own> <CODE> <R92A> <s> <out.json> [--quick] [--threads1] [--noprof]

<R_own>  own scratch root (nothing is read from it; passed where production code wants a root)
<s>      patch refinement (1.0 = the WL96A workload, N = 1,270)
--quick  scaling mode: components + R0 / ZERO events only
All walls are synchronised (dr.sync_thread + torch.cuda.synchronize). Compilation is excluded through the Dr.Jit kernel history
(backend_time of cache misses) and also reported; raw walls are kept.
"""
import copy, json, math, os, sys, time
import numpy as np

R_OWN, CODE, R92A, SREF, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
QUICK = '--quick' in sys.argv
THREADS1 = '--threads1' in sys.argv
NOPROF = '--noprof' in sys.argv
ARGV = list(sys.argv)
sys.argv = sys.argv[:1]                                   # production modules look at sys.argv for --dev
sys.path.insert(0, os.path.join(CODE, 'experiments', 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import fx92  # noqa: E402
import torch  # noqa: E402
import ecro86 as EC  # noqa: E402
import model86 as M86  # noqa: E402
import quad87 as Q  # noqa: E402
import xport87 as X  # noqa: E402
import scene86 as S  # noqa: E402
import pt86 as PTm  # noqa: E402

dr = PTm.dr
if THREADS1:
    torch.set_num_threads(1)

# ------------------------------------------------------------------------------------------------ kernel-history accounting
KH = {'exec': 0.0, 'codegen': 0.0, 'backend': 0.0, 'n': 0, 'miss': 0}
_orig_kh = dr.kernel_history


def _kh(*a, **k):                                         # pt86.render() drains the history itself: accumulate every drain
    h = _orig_kh(*a, **k)
    for e in h:
        KH['exec'] += float(e.get('execution_time', 0.0))
        KH['codegen'] += float(e.get('codegen_time', 0.0))
        bt = float(e.get('backend_time', 0.0))
        KH['backend'] += bt
        KH['n'] += 1
        if bt > 0.0 or e.get('cache_hit', True) is False:
            KH['miss'] += 1
    return h


dr.kernel_history = _kh
dr.set_flag(dr.JitFlag.KernelHistory, True)


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def kh_take():
    dr.kernel_history()
    r = dict(KH)
    for k in KH:
        KH[k] = 0 if k in ('n', 'miss') else 0.0
    return r


class Rec:
    """Rows of synchronised measurements per name."""

    def __init__(self):
        self.rows = {}

    def run(self, name, fn):
        sync(); kh_take()
        c0 = time.process_time(); m0 = time.thread_time(); t0 = time.perf_counter()
        out = fn()
        sync(); t1 = time.perf_counter()
        c1 = time.process_time(); m1 = time.thread_time()
        k = kh_take()
        self.rows.setdefault(name, []).append({'wall': t1 - t0 - k['backend'] / 1e3, 'raw': t1 - t0, 'cpu': c1 - c0, 'cpu_main': m1 - m0,
                                               'exec_ms': k['exec'], 'codegen_ms': k['codegen'], 'compile_ms': k['backend'], 'nk': k['n'], 'miss': k['miss']})
        return out

    def summary(self, skip=0):
        out = {}
        for name, rows in self.rows.items():
            rr = rows[skip:] if len(rows) > skip else rows
            o = {'n': len(rr)}
            for key in ('wall', 'raw', 'cpu', 'cpu_main'):
                v = np.array([r[key] for r in rr]) * 1e3
                o[key + '_ms'] = {'med': float(np.median(v)), 'q25': float(np.percentile(v, 25)), 'q75': float(np.percentile(v, 75)), 'min': float(v.min()), 'max': float(v.max())}
            for key in ('exec_ms', 'codegen_ms', 'compile_ms', 'nk', 'miss'):
                v = np.array([r[key] for r in rr], float)
                o[key] = {'med': float(np.median(v)), 'sum': float(v.sum())}
            out[name] = o
        return out


def bench(rec, name, fn, n=20, warm=1):
    for _ in range(warm):
        fn()
    for _ in range(n):
        rec.run(name, fn)


def refine(s):
    """Own copy of the econ88.refine patch-refinement rule (stationary grids, box faces, sphere bands); no material switch."""
    if s == 1.0:
        return
    o_r, o_f, o_s = S.stationary_rects, S.box_faces, (S.SPH_NPHI, S.SPH_NZ)
    S.stationary_rects = lambda: [r[:5] + (max(1, int(round(r[5] * s))), max(1, int(round(r[6] * s))), r[7]) for r in o_r()]
    n = max(1, int(round(3 * s)))
    S.box_faces = lambda box, mat: [f[:5] + (n, n, f[7]) for f in o_f(box, mat)]
    S.SPH_NPHI = max(2, int(round(o_s[0] * s))); S.SPH_NZ = max(1, int(round(o_s[1] * s)))


def shell(cmd):
    return os.popen(cmd).read().strip()


def main():
    t_start = time.perf_counter()
    env = {'uptime_before': shell('uptime'), 'apps_before': shell('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'),
           'gpu': shell('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'), 'pid': os.getpid(),
           'torch': torch.__version__, 'drjit': getattr(dr, '__version__', '?'), 'mitsuba': getattr(PTm.mi, '__version__', '?'),
           'torch_threads': torch.get_num_threads(), 'parallel_info': torch.__config__.parallel_info(),
           'env': {k: v for k, v in os.environ.items() if any(t in k for t in ('OMP', 'KMP', 'MKL', 'GOMP'))},
           'code_commit': open(os.path.join(CODE, 'COMMIT')).read().strip() if os.path.exists(os.path.join(CODE, 'COMMIT')) else '?'}
    refine(SREF)
    PT, S_, A, FX, ctx, lt = MT.ctx_ck2(R_OWN)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    E87 = fx92.E87
    N = int(ctx.N); NB = S.NB; K = S.K
    cam = FX['camera']; res0 = cam['res']
    g0 = copy.deepcopy(MT.SC['geom']); g1 = copy.deepcopy(g0); g1['A']['c'][1] += 0.15
    t0 = time.perf_counter()
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    env['quad_build_s'] = time.perf_counter() - t0
    Wt = torch.as_tensor(np.asarray(quad.W, np.float32), device=X.dev)     # resident coefficient table (own variant)
    model = MT.load_frozen(R92A, 8601)
    D = N * NB
    gen = torch.Generator(device='cpu').manual_seed(97500)
    RANKS = (24, 64) if D < 400000 else (24,)                # keep the process below ~4 GB at the largest refinement
    LAM = {r: (torch.rand(3, D, r, generator=gen).cuda(), torch.rand(3, D, r, generator=gen).cuda()) for r in RANKS}

    def lam_apply(rho, L2):
        Pm, Qm = LAM[rho]
        x = L2.reshape(D, 3)
        c = torch.einsum('cdr,dc->cr', Qm, x)
        return torch.einsum('cdr,cr->dc', Pm, c).reshape(N, NB, 3)

    # the WL96A source list rule (rng 96500, scene86.light_ok)
    srcs = []
    rng = np.random.default_rng(96500)
    while len(srcs) < 16:
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, g0):
            srcs.append(PTm.Lights([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}]))

    out = {'s': SREF, 'N': N, 'K': K, 'NB': NB, 'res': res0, 'env': env, 'P': E87['P'], 'Mq': E87['Mq'], 'M_ray': E87['M_ray']}
    state = {'st': EC.build_state(ctx)}
    state['op'] = X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W)
    qP = (E87['P'], E87['Mq'], E87['seed_q'])
    F_ = lambda B: PT.F(B.reshape(-1).contiguous())

    # correctness of the cost probe: selective update == full rebuild on the edited geometry (digest of hit / bin)
    ctx.set_geom(g1)
    st_sel, nd = EC.update_state(ctx, state['st'], ['A'])
    st_full = EC.build_state(ctx)
    out['probe_check'] = {'dirty_rays': int(nd), 'relation_rays': N * K, 'selective_equals_full': bool(st_sel.digest() == st_full.digest()),
                          'rays_changed_vs_T0': int((st_full.hit != state['st'].hit).sum() + ((st_full.hit == state['st'].hit) & (st_full.dbin != state['st'].dbin)).sum())}
    ctx.set_geom(g0)
    st_back, _ = EC.update_state(ctx, st_sel, ['A'])
    out['probe_check']['round_trip_equals_T0'] = bool(st_back.digest() == state['st'].digest())

    # ------------------------------------------------------------------------------------------------ events
    def table(arm, lt_, rec, pre):
        q = rec.run(pre + 'source_q_table', lambda: Q.qtable(ctx, lt_, *qP))

        def prefix():
            L = torch.as_tensor(np.asarray(q, np.float32), device=X.dev)
            return state['op'].apply(L)
        L2 = rec.run(pre + 'prefix_order', prefix)
        if arm == 'R0':
            def tl():
                L3 = state['op'].apply(L2); L4 = state['op'].apply(L3); L5 = state['op'].apply(L4)
                return L2 + L3 + L4 + L5
            B = rec.run(pre + 'tail_orders', tl)
        elif arm.startswith('LAM'):
            B = rec.run(pre + 'lambda_apply', lambda: L2 + lam_apply(int(arm[3:5]), L2))
        else:
            B = L2

        def ho():
            Bd = F_(B); dr.eval(Bd)
            return Bd
        return rec.run(pre + 'table_handover', ho)

    def event(kind, arm, lt_, seed, rec, geom=None, frame=(None, 64), gen_mode='prod'):
        """One synchronised event. Total wall measured independently of the part timers."""
        pre = f'{arm}|{kind}|'
        sync(); kh_take()
        c0 = time.process_time(); t0 = time.perf_counter()
        if kind == 'edit':
            rec.run(pre + 'geometry_update', lambda: ctx.set_geom(geom))

            def upd():
                state['st'], state['nd'] = EC.update_state(ctx, state['st'], ['A'])
            rec.run(pre + 'relation_update', upd)

            def coef():
                state['op'] = X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W)
            rec.run(pre + 'relation_coefficients', coef)
            if arm.startswith('LAM'):
                def g():
                    with torch.no_grad():
                        if gen_mode == 'prod':
                            gs = M86.GeoState(ctx.G, state['st'].hit, state['st'].dbin, EC.dev)
                        else:
                            gs = geostate_np(ctx.G, state['st'].hit, state['st'].dbin)
                        f = model.factors(gs)
                        del f
                rec.run(pre + 'generator', g)
        Bd = table(arm, lt_, rec, pre)
        if frame[1] > 0:
            r_ = frame[0] or res0
            rec.run(pre + 'render', lambda: ctx.render(lt_, 'ER', cam, r_, frame[1], seed, 'ecro', B=Bd, plain=True, b=min(64, frame[1])))
        sync(); t1 = time.perf_counter(); c1 = time.process_time()
        rec.rows.setdefault(pre + 'TOTAL', []).append({'wall': t1 - t0, 'raw': t1 - t0, 'cpu': c1 - c0, 'cpu_main': 0.0, 'exec_ms': 0.0, 'codegen_ms': 0.0,
                                                       'compile_ms': 0.0, 'nk': 0, 'miss': 0})

    def geostate_np(G, hit, dbin):
        """Own variant of model86.GeoState: integer conversions done by numpy (no multi-threaded torch CPU cast)."""
        gs = object.__new__(M86.GeoState)
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(np.asarray(a).astype(dt)), device=EC.dev)
        gs.N = G.N
        gs.cen, gs.nrm, gs.tan, gs.area = t(G.PC, np.float32), t(G.PN, np.float32), t(G.PT, np.float32), t(G.PAREA, np.float32)
        gs.mat = t(G.PMAT, np.int64)
        gs.hit = t(hit, np.int64).view(G.N, K); gs.dbin = t(dbin, np.int64).view(G.N, K)
        return gs

    def events(tag, arms, n_edit, n_src, frame=(None, 64), gen_mode='prod', interleave=False):
        """Warm (2 edits, 2 sources per arm, untimed) then timed events; summary skips nothing (warm events use a throw-away Rec)."""
        rec = Rec(); warm = Rec()
        ctx.set_geom(g0); state['st'] = EC.build_state(ctx); state['op'] = X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W)
        seq = []
        for arm in arms:
            for g in (g1, g0):
                event('edit', arm, lt, 1, warm, geom=g, frame=frame, gen_mode=gen_mode)
            for ls in srcs[:2]:
                event('source', arm, ls, 2, warm, frame=frame, gen_mode=gen_mode)
        if interleave:
            for i in range(n_edit):
                for arm in (arms if i % 2 == 0 else arms[::-1]):
                    seq.append(('edit', arm, i))
            for i in range(n_src):
                for arm in (arms if i % 2 == 0 else arms[::-1]):
                    seq.append(('source', arm, i))
        else:
            for arm in arms:
                seq += [('edit', arm, i) for i in range(n_edit)] + [('source', arm, i) for i in range(n_src)]
        flip = [0]
        for kind, arm, i in seq:
            if kind == 'edit':
                flip[0] += 1
                event('edit', arm, lt, 97810 + i, rec, geom=(g1 if flip[0] % 2 == 1 else g0), frame=frame, gen_mode=gen_mode)
            else:
                event('source', arm, srcs[i % 16], 97830 + i, rec, frame=frame, gen_mode=gen_mode)
        if flip[0] % 2 == 1:
            ctx.set_geom(g0); state['st'], _ = EC.update_state(ctx, state['st'], ['A'])
            state['op'] = X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W)
        out.setdefault('events', {})[tag] = {'summary': rec.summary(), 'warm_compile_ms': sum(r['compile_ms'] for rows in warm.rows.values() for r in rows),
                                              'frame': list(frame), 'gen_mode': gen_mode, 'interleave': interleave, 'threads': torch.get_num_threads()}
        json.dump(out, open(OUT, 'w'), indent=1)

    # ------------------------------------------------------------------------------------------------ run plan
    nE, nS = (8, 16) if QUICK else (12, 16)
    events('A_orig_order', ['R0', 'ZERO'] if QUICK else ['R0', 'LAM24', 'LAM64', 'ZERO'], nE, nS)
    if '--lamonly' in ARGV:
        env['uptime_after'] = shell('uptime'); json.dump(out, open(OUT, 'w'), indent=1); print('E5 DONE lamonly', flush=True)
        return
    events('B_interleaved_R0_ZERO', ['R0', 'ZERO'], nE, nS, interleave=True)
    events('C_noframe_R0_ZERO', ['R0', 'ZERO'], nE, nS, frame=(None, 0), interleave=True)
    if not QUICK:
        events('D_R0_after_LAM', ['R0'], nE, nS)
        events('E_LAM24_np_geostate', ['LAM24'], nE, 4, gen_mode='np')
        nt = torch.get_num_threads()
        torch.set_num_threads(1)
        events('F_LAM24_threads1', ['LAM24'], nE, 4)
        torch.set_num_threads(nt)
        events('G_LAM24_default_again', ['LAM24'], nE, 4)

    # ------------------------------------------------------------------------------------------------ isolated components
    rec = Rec()
    ctx.set_geom(g0); state['st'] = EC.build_state(ctx); state['op'] = X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W)
    op = state['op']
    q = Q.qtable(ctx, lt, *qP)
    L = torch.as_tensor(np.asarray(q, np.float32), device=X.dev)
    L2 = op.apply(L)
    bench(rec, 'q_table', lambda: Q.qtable(ctx, lt, *qP))
    bench(rec, 'q_table_src3', lambda: Q.qtable(ctx, srcs[3], *qP))
    bench(rec, 'q_upload', lambda: torch.as_tensor(np.asarray(q, np.float32), device=X.dev))
    bench(rec, 'one_order', lambda: op.apply(L2))
    bench(rec, 'gather_only', lambda: op.gather(L2))
    Lr = op.gather(L2)
    bench(rec, 'scatter_only', lambda: op.scatter(Lr))
    bench(rec, 'tail_3_orders', lambda: op.apply(op.apply(op.apply(L2))))
    bench(rec, 'four_orders', lambda: op.apply(op.apply(op.apply(op.apply(L)))))
    bench(rec, 'lambda24_apply', lambda: lam_apply(24, L2))
    if 64 in LAM:
        bench(rec, 'lambda64_apply', lambda: lam_apply(64, L2))
    B = L2 + op.apply(L2)

    def ho():
        Bd = F_(B); dr.eval(Bd)
        return Bd
    bench(rec, 'table_handover', ho)
    Bd = ho()
    bench(rec, 'xop_build_prod', lambda: X.XOp(ctx.G, state['st'].hit, state['st'].dbin, quad.W))
    bench(rec, 'xop_build_residentW', lambda: X.XOp(ctx.G, state['st'].hit, state['st'].dbin, Wt))
    bench(rec, 'build_state_full', lambda: EC.build_state(ctx), n=10)
    with torch.no_grad():
        bench(rec, 'geostate_prod', lambda: M86.GeoState(ctx.G, state['st'].hit, state['st'].dbin, EC.dev))
        bench(rec, 'geostate_np', lambda: geostate_np(ctx.G, state['st'].hit, state['st'].dbin))
        gs = geostate_np(ctx.G, state['st'].hit, state['st'].dbin)
        bench(rec, 'generator_factors', lambda: model.factors(gs))
        bench(rec, 'form_factors', lambda: model.form_factors(gs))
        f = model.factors(gs)
        s16 = torch.zeros(1, N, NB, 3, device=EC.dev)
        bench(rec, 'ecro_apply', lambda: model.apply(f, gs, s16))
        del f
    # geometry / relation update, alternating the two geometries, with the sub-steps of set_geom timed in an own copy
    st = state['st']
    for i in range(22):
        g = g1 if i % 2 == 0 else g0
        rec.run('set_geom', lambda: ctx.set_geom(g))
        G_old = st.G
        idx = rec.run('dirty_rays', lambda: EC.dirty_rays(st, G_old, ctx.G, ['A']))
        st = rec.run('update_state', lambda: EC.update_state(ctx, st, ['A'])[0])
        rec.run('state_rays_dirty', lambda: ctx.state_rays(idx))
    for i in range(22):
        geom = g1 if i % 2 == 0 else g0
        gG = rec.run('sg_Geom_construct', lambda: S.Geom(geom))

        def assign():
            for b, name in (('A', 'boxA'), ('B', 'boxB')):
                M = gG.box_matrix(b)
                ctx.params[f'{name}.vertex_positions'] = PTm.fl(ctx.V0[name] @ M[:3, :3].T + M[:3, 3])
                if name in ctx.N0:
                    Rm = np.stack(S.box_axes(geom[b]['yaw']), 1)
                    ctx.params[f'{name}.vertex_normals'] = PTm.fl(ctx.N0[name] @ Rm.T)
        rec.run('sg_vertex_assign', assign)
        rec.run('sg_params_update', lambda: ctx.params.update())
        rec.run('sg_set_tables', lambda: ctx.set_tables(gG))
    ctx.set_geom(g0)
    # frame conventions (the same production render; res / spp are conventions, not products)
    convs = [(512, 64), (512, 16), (512, 4), (512, 1), (256, 64), (256, 16), (256, 4), (256, 1)]
    for (r_, spp) in convs:
        bench(rec, f'render_{r_}_{spp}', lambda: ctx.render(lt, 'ER', cam, r_, spp, 7, 'ecro', B=Bd, plain=True, b=min(64, spp)), n=(10 if QUICK else 20), warm=2)
    out['components'] = rec.summary(skip=2)
    json.dump(out, open(OUT, 'w'), indent=1)

    # the same components with the kernel-history flag OFF (the flag forces a stream sync after every Dr.Jit kernel)
    dr.set_flag(dr.JitFlag.KernelHistory, False)
    rec2 = Rec()
    bench(rec2, 'q_table', lambda: Q.qtable(ctx, lt, *qP))
    bench(rec2, 'render_512_64', lambda: ctx.render(lt, 'ER', cam, 512, 64, 7, 'ecro', B=Bd, plain=True, b=64), warm=2)
    bench(rec2, 'render_512_1', lambda: ctx.render(lt, 'ER', cam, 512, 1, 7, 'ecro', B=Bd, plain=True, b=1), warm=2)
    st = state['st']
    for i in range(22):
        g = g1 if i % 2 == 0 else g0
        rec2.run('set_geom', lambda: ctx.set_geom(g))
        st = rec2.run('update_state', lambda: EC.update_state(ctx, st, ['A'])[0])
    ctx.set_geom(g0)
    out['components_kh_off'] = rec2.summary(skip=2)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    json.dump(out, open(OUT, 'w'), indent=1)

    # ------------------------------------------------------------------------------------------------ torch GPU-kernel time (profiler)
    if not NOPROF:
        try:
            from torch.profiler import profile, ProfilerActivity

            def gpu_ms(fn, n=20):
                fn(); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(n):
                        fn()
                    torch.cuda.synchronize()
                tot = 0.0; cnt = 0
                for e in prof.events():
                    dt = str(getattr(e, 'device_type', ''))
                    if 'CUDA' in dt:
                        tr = getattr(e, 'time_range', None)
                        tot += (tr.end - tr.start) if tr is not None else 0.0; cnt += 1
                return {'gpu_ms_per_call': tot / n / 1e3, 'gpu_events_per_call': cnt / n}
            op = state['op']
            with torch.no_grad():
                gs = geostate_np(ctx.G, state['st'].hit, state['st'].dbin)
                prof = {'one_order': gpu_ms(lambda: op.apply(L2)), 'tail_3_orders': gpu_ms(lambda: op.apply(op.apply(op.apply(L2)))),
                        'lambda24_apply': gpu_ms(lambda: lam_apply(24, L2)),
                        'generator_factors': gpu_ms(lambda: model.factors(gs)),
                        'xop_build_residentW': gpu_ms(lambda: X.XOp(ctx.G, state['st'].hit, state['st'].dbin, Wt)),
                        'q_upload': gpu_ms(lambda: torch.as_tensor(np.asarray(q, np.float32), device=X.dev)),
                        'dirty_rays': gpu_ms(lambda: EC.dirty_rays(state['st'], state['st'].G, S.Geom(g1), ['A']))}
            out['torch_profiler'] = prof
        except Exception as ex:                               # noqa: BLE001
            out['torch_profiler'] = {'error': repr(ex)}
    env['uptime_after'] = shell('uptime'); env['apps_after'] = shell('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader')
    env['torch_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    env['total_s'] = time.perf_counter() - t_start
    json.dump(out, open(OUT, 'w'), indent=1)
    print('E5 DONE', SREF, N, round(env['total_s'], 1), flush=True)


if __name__ == '__main__':
    main()
