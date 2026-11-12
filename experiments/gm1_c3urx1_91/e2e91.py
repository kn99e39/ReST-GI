"""WL91 end-to-end lifecycle, quality ladders and bounded scaling (run ALONE on the GPU; protocol 'economics' / 'scaling').

  python e2e91.py life <R91> [--dev]            warm end-to-end pipelines (edits, source updates, amortisation sequence), memory, ray counts
  python e2e91.py cold <R91> <arm>              cold lifecycle of one arm in a fresh process (C3URX1 | C3UR | R0 | PT)
  python e2e91.py ladder <R91> <R89> [--dev]    quality ladders (3 configurations x 3 draws for the table arms, PT)
  python e2e91.py scale <R91> [--dev]           bounded scaling: patch resolution, relation size, transport orders, source count
Timing boundary: ONE synchronised wall timer (Dr.Jit + torch) around each complete event; component timers inside; process CPU time and
Dr.Jit kernel execution time recorded. Every pipeline is warmed by untimed events of each kind first.
Kernel compilation: Dr.Jit emits allocation-history-dependent variants of the same program (identical operations, different variable order);
every new OptiX variant is compiled once (~21 s for the path-tracing kernel). Warm walls therefore EXCLUDE the measured compilation time
(backend_time of kernel-cache misses in the Dr.Jit kernel history, all arms alike); the raw wall, the compilation time and the number of
cache misses are recorded per event. Cold lifecycles keep compilation.
Ray counts: relation rays re-traced (measured), source-evaluation rays (measured: landed rays x (1 probe + lights shadow)), q-table shadow
rays (N x P x lights), render rays (nominal upper bound: samples x vertices x (1 + lights)).
"""
import glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx91  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx91.PT, fx91.Q
mi, dr, S = PT.mi, PT.dr, PT.S
F = PT.F
FX = PT.FX
P91, P89, E87 = fx91.P91, fx91.P89, fx91.E87
EC_ = P91['economics']; SC = P91['scaling']
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


# Kernel-history accumulator. pt86.Ctx.render drains the Dr.Jit history itself, so the module function is wrapped (no frozen file changes):
# every drained entry adds its execution time and, for a kernel-cache miss, its backend (compilation / load) time.
_KH = {'exec_ms': 0.0, 'compile_ms': 0.0, 'misses': 0}
_kh_orig = dr.kernel_history


def _kh(*a, **k):
    h = _kh_orig(*a, **k)
    for e in h:
        _KH['exec_ms'] += float(e.get('execution_time', 0.0))
        bt = float(e.get('backend_time', 0.0))
        if bt > 0.0:
            _KH['compile_ms'] += bt; _KH['misses'] += 1
    return h


dr.kernel_history = _kh


def drain():
    dr.kernel_history()
    r = dict(_KH)
    _KH.update(exec_ms=0.0, compile_ms=0.0, misses=0)
    return r


def foreign():
    me = str(os.getpid())
    return ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)


class Clock:
    """Event timer: wall (kernel compilation excluded; raw wall and compilation kept), CPU, Dr.Jit kernel ms and named component walls."""

    def __init__(self):
        sync(); drain()
        self.t0 = time.perf_counter(); self.c0 = time.process_time(); self.parts = {}; self.tp = self.t0
        self.comp = 0.0; self.exec_ms = 0.0; self.miss = 0

    def _take(self):
        k = drain(); self.exec_ms += k['exec_ms']; self.miss += k['misses']; self.comp += k['compile_ms'] / 1e3
        return k['compile_ms'] / 1e3

    def part(self, name):
        sync(); t = time.perf_counter(); c = self._take()
        self.parts[name] = self.parts.get(name, 0.0) + (t - self.tp) - c; self.tp = t

    def set_part(self, name, secs):
        """A component timed inside the callee; any compilation since the last mark happened inside it and is removed."""
        sync(); c = self._take()
        self.parts[name] = self.parts.get(name, 0.0) + secs - c; self.tp = time.perf_counter()

    def done(self):
        sync(); t = time.perf_counter(); self._take()
        raw = t - self.t0
        return {'wall_s': raw - self.comp, 'wall_raw_s': raw, 'compile_s': self.comp, 'kernel_cache_misses': self.miss,
                'cpu_s': time.process_time() - self.c0, 'gpu_kernel_ms': self.exec_ms,
                'parts': dict(self.parts), 'parts_sum_s': sum(self.parts.values())}


def to_drjit(B):
    return F(B.reshape(-1).contiguous()) if isinstance(B, torch.Tensor) else PT.fl(B)


def random_sources(n, seed, geom):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, geom):
            out.append([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}])
    return out


class Pipe:
    """One arm's complete pipeline on a resident scene. edit(geom, moved, lt, spp) and source(lt, spp) run every stage and return the clock."""

    def __init__(self, arm, ctx, mat, quad=None):
        import ecro86 as EC
        import xport87 as X
        import rel89 as R9
        import cand91 as C
        self.arm, self.ctx = arm, ctx
        S1 = P89['stage1']
        self.EC, self.X, self.C = EC, X, C
        cam = FX['camera']; self.cam, self.res = cam, cam['res']
        if arm == 'C3URX1':
            self.cand = C.C3URX1(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
        elif arm == 'C3UR':
            self.rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
        elif arm == 'R0':
            self.quad = quad
            self.st = EC.build_state(ctx)
            self.op = X.XOp(ctx.G, self.st.hit, self.st.dbin, quad.W)
        self.B = None

    def _table(self, lt, ck):
        a = self.arm
        if a == 'C3URX1':
            T_ = {}
            B = self.cand.table(self.ctx, lt, timing=T_)
            ck.set_part('source_exact_first_gather_eval', T_['source_eval_s'])       # the only Dr.Jit kernels of table() are in the source evaluation
            ck.parts['transport_orders'] = ck.parts.get('transport_orders', 0.0) + T_['transport_s']
        elif a == 'C3UR':
            q = Q.qtable(self.ctx, lt, E87['P'], E87['Mq'], E87['seed_q']); ck.part('source_q_table')
            B = self.C.c3ur_table(self.rc, q); ck.part('transport_orders')
        elif a == 'R0':
            q = Q.qtable(self.ctx, lt, E87['P'], E87['Mq'], E87['seed_q']); ck.part('source_q_table')
            import rel88 as RL
            B = RL.recurrence([self.op], q)[0]; ck.part('transport_orders')
        self.B = to_drjit(B); dr.eval(self.B); ck.part('table_handover')

    def _frame(self, lt, spp, seed, ck, cam=None):
        cam = cam or self.cam
        if self.arm == 'PT':
            self.ctx.render(lt, 'ER', cam, self.res, spp, seed, 'cam', plain=True, b=min(64, spp))
        else:
            self.ctx.render(lt, 'ER', cam, self.res, spp, seed, 'ecro', B=self.B, plain=True, b=min(64, spp))
        ck.part('render')

    def edit(self, geom, moved, lt, spp, seed):
        ck = Clock()
        self.ctx.set_geom(geom); ck.part('geometry_update')
        a = self.arm
        nd = 0
        if a == 'C3URX1':
            T_ = self.cand.update(self.ctx, moved); nd = T_['dirty_rays']
            ck.set_part('relation_update', T_['relation_update_s']); ck.parts['probe_buffer'] = T_['probe_buffer_s']
        elif a == 'C3UR':
            T_ = self.rc.update(self.ctx, moved); nd = T_['dirty_rays']; ck.part('relation_update')
        elif a == 'R0':
            self.st, nd = self.EC.update_state(self.ctx, self.st, moved); ck.part('relation_update')
            self.op = self.X.XOp(self.ctx.G, self.st.hit, self.st.dbin, self.quad.W); ck.part('relation_coefficients')
        if a != 'PT':
            self._table(lt, ck)
        self._frame(lt, spp, seed, ck)
        r = ck.done(); r['relation_rays_retraced'] = int(nd)
        return r

    def source(self, lt, spp, seed):
        ck = Clock()
        if self.arm != 'PT':
            self._table(lt, ck)
        self._frame(lt, spp, seed, ck)
        return ck.done()

    def frame(self, lt, spp, seed, cam=None):
        ck = Clock(); self._frame(lt, spp, seed, ck, cam); return ck.done()


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def summarize(rows):
    keys = sorted({k for r in rows for k in r['parts']})
    return {'wall_s': med(rows, 'wall_s'), 'wall_raw_s': med(rows, 'wall_raw_s'), 'events_with_compilation': int(sum(r['kernel_cache_misses'] > 0 for r in rows)),
            'compile_s_total': float(sum(r['compile_s'] for r in rows)), 'cpu_s': med(rows, 'cpu_s'), 'gpu_kernel_ms': med(rows, 'gpu_kernel_ms'),
            'parts': {k: float(np.median([r['parts'].get(k, 0.0) for r in rows])) for k in keys},
            'wall_minus_render_s': float(np.median([r['wall_s'] - r['parts'].get('render', 0.0) for r in rows])),
            'parts_sum_over_wall': float(np.median([r['parts_sum_s'] / r['wall_s'] for r in rows])), 'n': len(rows)}


def life(R):
    R8.inject_choice(R)
    fx91.fx88.setup('W1_T0')
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']])
    mat = R8.materials(ctx)
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    spp = EC_['events']['frame_spp']
    nE = 2 if DEV else EC_['events']['n_edits']; nS = 3 if DEV else EC_['events']['n_sources']
    srcs = [FX['light_sets'][k] for k in ('LMIX89', 'LSPOT89')] + random_sources(max(nS - 2, 1), EC_['events']['source_seed'], ex['T1'])
    Rr = {'foreign_before': foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(), 'frame_spp': spp, 'arms': {}}
    cam2 = FX['camera2']
    for arm in ('C3URX1', 'C3UR', 'R0', 'PT'):
        ctx.set_geom(ex['T0'])
        torch.cuda.reset_peak_memory_stats()
        p = Pipe(arm, ctx, mat, quad)
        for g in ('T1', 'T0'):                                              # warm-up of every code path (excluded)
            p.edit(ex[g], ex['moved'], lt, spp, 1)
        for ls in srcs[:2]:
            p.source(PT.Lights(ls), spp, 2)
        p.source(lt, spp, 2)
        p.frame(lt, spp, 2, cam2)
        ed = [p.edit(ex['T1' if i % 2 == 0 else 'T0'], ex['moved'], lt, spp, 91810 + i) for i in range(nE + 1)]       # ends at T1
        so = []
        dig = []
        for i, ls in enumerate(srcs):
            r = p.source(PT.Lights(ls), spp, 91830 + i); r['lights'] = len(ls)
            so.append(r)
            if arm == 'C3URX1':
                dig.append(p.cand.rel.digest())
            elif arm == 'C3UR':
                dig.append(p.rc.digest())
        fr = [p.frame(lt if arm == 'PT' else PT.Lights(srcs[-1]), spp, 91850 + i) for i in range(4)]
        # executed amortisation sequence: one edit, then S sources x V views (frames at frame_spp)
        AS = EC_['events']['amortisation_sequence']
        sync(); t0 = time.perf_counter()
        ev = lambda name, r: (name, r['wall_s'], r['compile_s'])
        sev = [ev('edit', p.edit(ex['T0'], ex['moved'], lt, spp, 91870))]
        for i in range(AS['sources']):
            l2 = PT.Lights(srcs[2 + i] if len(srcs) > 2 + i else srcs[-1])
            sev.append(ev('source', p.source(l2, spp, 91880 + i)))
            for v in range(AS['views'] - 1):
                sev.append(ev('frame_cam2' if v % 2 else 'frame', p.frame(l2, spp, 91890 + 10 * i + v, cam2 if v % 2 else None)))
        sync(); t_raw = time.perf_counter() - t0
        t_comp = sum(e[2] for e in sev)
        t_seq = t_raw - t_comp                                              # executed sequence, kernel compilation excluded
        p.edit(ex['T1'], ex['moved'], lt, spp, 91871)
        A = {'edit': summarize(ed), 'source_single_light': summarize([r for r in so if r['lights'] == 1]),
             'source_two_lights': summarize([r for r in so if r['lights'] == 2]) if any(r['lights'] == 2 for r in so) else None,
             'frame': summarize(fr), 'edit_rows': ed, 'source_rows': so,
             'amortisation_sequence_measured_s': t_seq, 'amortisation_sequence_raw_s': t_raw, 'amortisation_sequence_compile_s': t_comp, 'amortisation_events': sev, 'relation_digest_constant_across_sources': bool(len(set(dig)) <= 1),
             'relation_rays_retraced_per_edit': int(np.median([r['relation_rays_retraced'] for r in ed])),
             'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'process_gpu_mem': PT.gpu_mem()}
        e, s, f = A['edit'], A['source_single_light'], A['frame']
        A['amortisation_sequence_predicted_s'] = e['wall_s'] + AS['sources'] * s['wall_s'] + AS['sources'] * (AS['views'] - 1) * f['wall_s']
        n1 = 1
        if arm == 'C3URX1':
            import cand91 as C
            full = C.C3URX1(ctx, P89['stage1']['kappa_s'], P89['stage1']['seed_send'], mat['dense8'])
            A['selective_equals_full'] = bool(full.rel.digest() == p.cand.rel.digest() and torch.equal(full.rel.seg_t, p.cand.rel.seg_t))
            A['bytes'] = p.cand.nbytes()
            A['rays'] = {'unique_relation_rays': p.cand.rel.n_rays, 'landed_rays': p.cand.n, 'source_eval_per_light': {'probe_intersections': p.cand.n, 'shadow_rays': p.cand.n}}
            del full
        elif arm == 'C3UR':
            import rel89 as R9
            full = R9.SendRelR(ctx, P89['stage1']['kappa_s'], P89['stage1']['seed_send'], mat['dense8'])
            A['selective_equals_full'] = bool(full.digest() == p.rc.digest())
            A['bytes'] = {'relation': p.rc.nbytes(), 'q_table': int(ctx.N * NB * 3 * 4), 'scratch_incident_cells': int(ctx.N * 4096 * 3 * 4)}
            A['rays'] = {'unique_relation_rays': p.rc.n_rays, 'q_table_shadow_rays_per_light': int(ctx.N * E87['P'])}
            del full
        elif arm == 'R0':
            A['bytes'] = {'relation_state': int(p.st.hit.nbytes + p.st.dbin.nbytes + p.st.seg.nbytes), 'W': int(quad.W.astype(np.float32).nbytes), 'q_table': int(ctx.N * NB * 3 * 4)}
            A['rays'] = {'unique_relation_rays': int(ctx.N * K), 'q_table_shadow_rays_per_light': int(ctx.N * E87['P'])}
        px = FX['camera']['res'] ** 2 * spp
        A['render_rays_nominal_per_frame_one_light'] = {'intersections': px * (6 if arm == 'PT' else 2), 'shadow_rays': px * (6 if arm == 'PT' else 2)}
        Rr['arms'][arm] = A
        print('life', arm, json.dumps({'edit_ms': round(e['wall_s'] * 1000, 1), 'edit_no_render_ms': round(e['wall_minus_render_s'] * 1000, 1),
                                       'source_ms': round(s['wall_s'] * 1000, 1), 'source_no_render_ms': round(s['wall_minus_render_s'] * 1000, 1),
                                       'frame_ms': round(f['wall_s'] * 1000, 1), 'seq_s': round(t_seq, 3), 'seq_pred_s': round(A['amortisation_sequence_predicted_s'], 3)}), flush=True)
        del p
        torch.cuda.empty_cache()
    Rr['foreign_after'] = foreign()
    json.dump(Rr, open(os.path.join(out, 'life91.json'), 'w'), indent=1)
    print('LIFE_DONE', flush=True)


def cold(R, arm):
    """Cold lifecycle in a fresh process: every first call, compilation included."""
    t_all = time.perf_counter()
    R8.inject_choice(R)
    fx91.fx88.setup('W1_T0')
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    C_ = {}
    t0 = time.perf_counter(); ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']]); sync(); C_['scene_build_s'] = time.perf_counter() - t0
    quad = None; mat = None
    if arm in ('C3URX1', 'C3UR'):
        t0 = time.perf_counter(); mat = {'dense8': Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], E87['M_dense'], E87['seed_dense']).dense()}; sync(); C_['material_dense_table_s'] = time.perf_counter() - t0
    if arm == 'R0':
        t0 = time.perf_counter(); quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray']); sync(); C_['material_W_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); p = Pipe(arm, ctx, mat, quad); sync(); C_['relation_build_s'] = time.perf_counter() - t0
    r = p.source(lt, EC_['events']['frame_spp'], 91900)                    # cold: compilation INCLUDED (raw walls)
    C_['first_source_and_frame'] = {'wall_s': r['wall_raw_s'], 'compile_s': r['compile_s'], 'parts_without_compile': r['parts']}
    r2 = p.edit(ex['T1'], ex['moved'], lt, EC_['events']['frame_spp'], 91901)
    C_['first_edit_and_frame'] = {'wall_s': r2['wall_raw_s'], 'compile_s': r2['compile_s'], 'parts_without_compile': r2['parts']}
    C_['total_to_first_frame_s'] = C_['scene_build_s'] + C_.get('material_dense_table_s', 0) + C_.get('material_W_s', 0) + C_['relation_build_s'] + r['wall_raw_s']
    C_['process_wall_s'] = time.perf_counter() - t_all
    C_['process_gpu_mem'] = PT.gpu_mem()
    json.dump(C_, open(os.path.join(out, f'cold91_{arm}.json'), 'w'), indent=1)
    print('COLD', arm, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in C_.items() if not isinstance(v, dict)}), flush=True)


def ladder(R, R89):
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import cand91 as C
    R8.inject_choice(R)
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    LD = EC_['ladders']
    S1 = P89['stage1']
    seeds_all = [86001] + list(P89['stage2']['seeds'])
    Rr = {'foreign_before': foreign(), 'configs': {}}
    cfgs = LD['configs'] if not DEV else LD['configs'][:1]
    spps = LD['spp'] if not DEV else [4, 64]
    reps = LD['replicates'] if not DEV else 1
    draws = LD['draws'] if not DEV else LD['draws'][:2]
    for cfg, ls in cfgs:
        tag = cfg + ('_' + ls if ls else '')
        geom, ls0 = R8.setup(R, cfg)
        ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
        mat = R8.materials(ctx)
        q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
        tables = {}
        for rs in draws:
            ri = seeds_all.index(rs)
            us = X.realization_us(rs); X.set_realization(ctx, us)
            st = EC.build_state(ctx)
            quad = R8.quad_of(us, ri) if rs != 86001 else Q.Quad(us, E87['M_ray'], E87['seed_ray'])
            tables[f'R0_s{rs}'] = RL.recurrence([X.XOp(ctx.G, st.hit, st.dbin, quad.W)], q)[0].cpu().numpy()
            cand = C.C3URX1(ctx, S1['kappa_s'], S1['seed_send'] + (0 if rs == 86001 else 1000 * ri), mat['dense8'])
            tables[f'C3UR_s{rs}'] = C.c3ur_table(cand.rel, q).cpu().numpy()
            tables[f'C3URX1_s{rs}'] = cand.table(ctx, lt).cpu().numpy()
            del cand
        X.set_realization(ctx, X.realization_us(86001))
        torch.cuda.empty_cache()
        fns = sorted(glob.glob(os.path.join(R89, 'results', 'renders', f'REF_ER_{tag}_r*.npz')))
        ref = np.mean([np.load(f)['tot'].astype(np.float64) for f in fns], 0)
        mz = np.load(os.path.join(R89, 'results', 'masks', (cfg.split('_')[0] + ('_' + ls if ls else '')) + '.npz'))
        mk = {k: mz[k] for k in ('stationary_all', 'R_mirror', 'C_ss', 'D') if mz[k].sum() >= 30}
        err = lambda img: {k: float(np.abs(img[m] - ref[m]).sum() / max(ref[m].sum(), 1e-30)) for k, m in mk.items()}
        cam = FX['camera']; res = cam['res']
        Bd = {k: PT.fl(v) for k, v in tables.items()}
        L = {}
        ctx.render(lt, 'ER', cam, res, 4, 1, 'cam', plain=True, b=4)
        for k in Bd:
            ctx.render(lt, 'ER', cam, res, 4, 1, 'ecro', B=Bd[k], plain=True, b=4)
        for spp in spps + ([] if DEV else LD['pt_extra_spp']):
            for qr in range(reps):
                sd = LD['seed_base'] + 97 * spp + qr
                arms = ['PT'] + (list(Bd) if spp in spps else [])
                for arm in arms:
                    sync(); drain(); t0 = time.perf_counter()
                    if arm == 'PT':
                        a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd, 'cam', plain=True, b=min(64, spp))
                    else:
                        a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd + 50, 'ecro', B=Bd[arm], plain=True, b=min(64, spp))
                    sync(); wr = time.perf_counter() - t0
                    cs = drain()['compile_ms'] / 1e3; w = wr - cs                 # warm render wall (kernel compilation excluded)
                    img = PT.pack(a, res, spp)['tot'].astype(np.float64)
                    L.setdefault(arm, []).append({'spp': spp, 'rep': qr, 'wall_s': w, 'wall_raw_s': wr, 'compile_s': cs, 'gpu_ms': T_['gpu_ms'], **{'err_' + k: v for k, v in err(img).items()}})
            print('ladder', tag, spp, json.dumps({a: round(L[a][-1].get('err_R_mirror', L[a][-1]['err_stationary_all']), 4) for a in L if L[a][-1]['spp'] == spp}), flush=True)
        medl = {}
        for arm in L:
            by = {}
            for row in L[arm]:
                by.setdefault(row['spp'], []).append(row)
            medl[arm] = [{'spp': s_, **{k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ('spp', 'rep')}} for s_, rows in sorted(by.items())]
        Rr['configs'][tag] = {'ladder': L, 'ladder_median': medl, 'regions': {k: int(m.sum()) for k, m in mk.items()}, 'ref_files': len(fns)}
        del Bd
        torch.cuda.empty_cache()
    Rr['foreign_after'] = foreign()
    json.dump(Rr, open(os.path.join(out, 'ladder91.json'), 'w'), indent=1)
    print('LADDER_DONE', flush=True)


def scale(R):
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    import cand91 as C
    import econ88
    R8.inject_choice(R)
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    S1 = P89['stage1']
    rows = []
    cases = [(s, S1['kappa_s']) for s in SC['scale_factors']] + [(1.0, k) for k in SC['kappa_s'] if k != S1['kappa_s']]
    if DEV:
        cases = [(1.0, 8), (1.5, 8), (1.0, 4)]

    def tm(fn, n=3):
        fn()                                                                   # warm-up (compilation excluded)
        ts = []
        for _ in range(n):
            sync(); drain(); t0 = time.perf_counter(); fn(); sync(); w = time.perf_counter() - t0
            ts.append(w - drain()['compile_ms'] / 1e3)                         # kernel compilation of a new variant excluded
        return float(np.median(ts))
    for s, ks in cases:
        econ88.refine(s, False)
        ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']])
        d8 = Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], E87['M_dense'], E87['seed_dense']).dense()
        row = {'s': s, 'kappa_s': ks, 'N': int(ctx.N)}
        row['C3UR_build_s'] = tm(lambda: R9.SendRelR(ctx, ks, S1['seed_send'], d8), 2)
        cand = C.C3URX1(ctx, ks, S1['seed_send'], d8)
        row.update({'unique_relation_rays': cand.rel.n_rays, 'landed_rays': cand.n, 'bytes': cand.nbytes()})
        q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
        row['q_table_s'] = tm(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']))
        row['X1_source_eval_s'] = tm(lambda: cand.radiance(ctx, lt))
        rad = cand.radiance(ctx, lt)
        row['X1_first_gather_s'] = tm(lambda: cand.first_gather(rad))
        L2 = cand.first_gather(rad)
        row['apply_one_order_s'] = tm(lambda: cand.rel.apply(L2))
        for J_ in SC['orders']:
            row[f'C3URX1_table_J{J_}_s'] = tm(lambda: cand.table(ctx, lt, J_=J_))
            row[f'C3UR_table_J{J_}_s'] = tm(lambda: C.c3ur_table(cand.rel, q, J_=J_))
        # geometry edit, warm: T0 -> T1 -> T0 ... (two untimed edits first)
        def edit_cycle(geoms=('T1', 'T0')):
            ts = []
            for g in geoms:
                ctx.set_geom(ex[g])
                sync(); drain(); t0 = time.perf_counter(); cand.update(ctx, ex['moved']); sync(); w = time.perf_counter() - t0
                ts.append(w - drain()['compile_ms'] / 1e3)
            return ts
        edit_cycle()
        ts = edit_cycle() + edit_cycle()
        row['C3URX1_update_s'] = float(np.median(ts)); row['relation_rays_retraced'] = cand.counts['relation_rays_retraced_last_edit']
        if s == 1.0 and ks == S1['kappa_s']:
            src = []
            for n in SC['source_counts']:
                lts = [PT.Lights(l) for l in random_sources(n, EC_['events']['source_seed'] + n, ex['T0'])]
                src.append({'sources': n, 'C3URX1_tables_s': tm(lambda: [cand.table(ctx, l2) for l2 in lts], 2),
                            'C3UR_tables_s': tm(lambda: [C.c3ur_table(cand.rel, Q.qtable(ctx, l2, E87['P'], E87['Mq'], E87['seed_q'])) for l2 in lts], 2)})
            row['source_count'] = src
        rows.append(row)
        print('scale', json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, (dict, list))}), flush=True)
        del cand
        torch.cuda.empty_cache()
    json.dump({'rows': rows, 'foreign': foreign()}, open(os.path.join(out, 'scale91.json'), 'w'), indent=1)
    print('SCALE_DONE', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    if cmd == 'life':
        life(sys.argv[2])
    elif cmd == 'cold':
        cold(sys.argv[2], sys.argv[3])
    elif cmd == 'ladder':
        ladder(sys.argv[2], sys.argv[3])
    elif cmd == 'scale':
        scale(sys.argv[2])
    os._exit(0)
