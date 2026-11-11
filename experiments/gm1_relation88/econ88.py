"""WL88 Stage 3: lifecycle economics and scaling (run ALONE on the GPU; protocol stage3).

  python econ88.py econ <R88> [--dev]     W1 lifecycle (stages A..H), warm edits, source-only / multi-source / multi-view, quality ladders
  python econ88.py scale <R88> [--dev]    construction / application cost vs patch count, peaked coverage, transport order, source count
Arms: PT (frozen-ER path tracing), R0 (WL87 EXP), R1 (SAM), R1M (matched-ray SAM), C3 (WL87 sender-side, source-dependent),
TF<b> (path-traced table fill). Table arms share the WL86 LOW + lookup renderer. Every timed operation is executed once untimed
first (kernel compilation excluded from warm numbers; cold numbers are reported separately).
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx88  # noqa: E402
import run88  # noqa: E402
PT, Q = fx88.PT, fx88.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P88 = fx88.P88
S3 = P88['stage3']
E87 = fx88.fx87.P87['explicit']
RP = P88['relation']
DEV = '--dev' in sys.argv
NB, K = S.NB, S.K


def foreign():
    me = str(os.getpid())
    return ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)


def tk(T):
    return {k: T[k] for k in ['trace_s', 'eval_s', 'gpu_ms', 'wall_s', 'launches', 'spp'] if k in T}


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def timed(fn, warm=False):
    if warm:
        fn()
    sync(); dr.kernel_history()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return r, time.perf_counter() - t0, sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())


def ref_image(R, cfg):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', f'REF_ER_{cfg}_r*.npz')))
    acc = 0
    for fn in fns:
        acc = acc + np.load(fn)['tot'].astype(np.float64)
    return acc / max(len(fns), 1), fns


def random_lights(n, seed, geom):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, geom):
            out.append([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}])
    return out


def econ(R):
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import pt87
    run88.inject_choice(R)
    fx88.setup('W1_T0')
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cam = FX['camera']; res = cam['res']
    ex = FX['W1']; lights = FX['light_sets'][ex['light_set']]
    Rr = {'foreign_before': foreign(), 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(),
          'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__}}
    # ---------------- cold (first use in this process; includes compilation)
    C = {}
    t0 = time.perf_counter(); ctx = PT.Ctx(ex['T0']); lt = PT.Lights(lights); dr.sync_thread(); C['scene_build_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray']); C['material_W_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); d16 = RL.Dense(RP['dense_sub'], RP['M_dense'], RP['seed_dense']); C['dense16_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); d8 = Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], E87['M_dense'], E87['seed_dense']).dense(); C['dense8_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); RL.peaked(); C['peaked_rule_s'] = time.perf_counter() - t0
    (st, C['R0_state_s'], _) = timed(lambda: EC.build_state(ctx))
    T1c = {}
    (r1, C['R1_build_s'], _) = timed(lambda: RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st, timing=T1c))
    C['R1_build_parts'] = T1c
    (q, C['q_s'], _) = timed(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']))
    r0 = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
    (B0, C['R0_apply_s'], _) = timed(lambda: RL.recurrence([r0], q)[0].cpu().numpy())
    (B1, C['R1_apply_s'], _) = timed(lambda: RL.recurrence([r1], q)[0].cpu().numpy())
    _, Tc = ctx.render(lt, 'ER', cam, res, 64, 88800, 'ecro', B=PT.fl(B1), plain=True, b=64)
    _, Tp = ctx.render(lt, 'ER', cam, res, 64, 88800, 'cam', plain=True, b=64)
    C['first_table_render_64spp'] = tk(Tc); C['first_pt_render_64spp'] = tk(Tp)
    Rr['cold'] = C
    print('cold', json.dumps(C), flush=True)
    # ---------------- warm repeated edits (both relations resident; R1 updated selectively, R0 as WL87)
    nedit = S3['n_edits'] if not DEV else 1
    r1m = RL.SAM(ctx, quad, d16, RP['matched_variant']['kappa_u'], RP['matched_variant']['m'], RP['seed_area'], st=st)
    ed = []
    seq = (['T1', 'T0'] * (nedit + 1) + ['T1'])
    for ei, cfg in enumerate(seq):
        row = {'cfg': cfg}
        (_, row['A_geometry_s'], _) = timed(lambda: ctx.set_geom(ex[cfg]))
        ((st2, nd), row['B_R0_state_s'], row['B_R0_state_gpu_ms']) = timed(lambda: EC.update_state(ctx, st, ex['moved']))
        row['R0_dirty_rays'] = int(nd)
        Tu = {}
        r1.st = st                                                   # R1 shares the u state of R0 (same realization)
        (_, row['R1_update_s'], row['R1_update_gpu_ms']) = timed(lambda: r1.update(ctx, ex['moved'], timing=Tu))
        row['R1_update_parts'] = Tu
        Tm = {}
        r1m.st = st
        (_, row['R1M_update_s'], _) = timed(lambda: r1m.update(ctx, ex['moved'], timing=Tm))
        st = st2
        (q, row['D_q_s'], row['D_q_gpu_ms']) = timed(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']))
        (r0, row['C_R0_coeff_s'], _) = timed(lambda: X.XOp(ctx.G, st.hit, st.dbin, quad.W))
        (B0, row['E_R0_apply_s'], _) = timed(lambda: RL.recurrence([r0], q)[0].cpu().numpy())
        (B1, row['E_R1_apply_s'], _) = timed(lambda: RL.recurrence([r1], q)[0].cpu().numpy())
        (B1m, row['E_R1M_apply_s'], _) = timed(lambda: RL.recurrence([r1m], q)[0].cpu().numpy())
        (B1d, row['F_transfer_s'], _) = timed(lambda: PT.fl(B1))
        ((B3, _), row['C3_s'], row['C3_gpu_ms']) = timed(lambda: X.sender_side(ctx, q, d8, ctx.N * K, E87['seed_c3']))
        _, Tf1 = ctx.render(lt, 'ER', cam, res, 16, 88810, 'ecro', B=B1d, plain=True, b=16)
        _, Tf2 = ctx.render(lt, 'ER', cam, res, 16, 88811, 'ecro', B=B1d, plain=True, b=16)
        row['G_first_frame_extra_s'] = Tf1['wall_s'] - Tf2['wall_s']
        row['R1_digest'] = r1.digest()
        if ei >= 2:                                                  # first two edits = warm-up (compilation of every update path)
            ed.append(row)
        print('edit', cfg, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)}), flush=True)
    Rr['edits'] = ed
    keys = [k for k in ed[0] if isinstance(ed[0][k], float)]
    Rr['edit_median'] = {k: float(np.median([x[k] for x in ed])) for k in keys}
    full = RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st)
    Rr['selective_equals_full_digest'] = bool(full.digest() == r1.digest())
    (_, Rr['R1_full_rebuild_s'], _) = timed(lambda: RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st), warm=True)
    (_, Rr['R0_full_state_s'], _) = timed(lambda: EC.build_state(ctx), warm=True)
    pe = []
    for cfg in ['T0', 'T1'] * 2:
        (_, t_geo, _) = timed(lambda: ctx.set_geom(ex[cfg]))
        _, A_ = ctx.render(lt, 'ER', cam, res, 16, 88812, 'cam', plain=True, b=16)
        _, B_ = ctx.render(lt, 'ER', cam, res, 16, 88813, 'cam', plain=True, b=16)
        pe.append({'geometry_update_s': t_geo, 'first_frame_extra_s': A_['wall_s'] - B_['wall_s']})
    Rr['pt_edit_median'] = {k: float(np.median([x[k] for x in pe[1:]])) for k in pe[0]}
    # now at T1 with st / r1 / r1m / q / B* of T1
    st = EC.build_state(ctx)
    # ---------------- table fill cost
    TFB = {}; tf = {}
    for bi, b in enumerate(S3['tf_budgets'] if not DEV else [16]):
        (Bt, t_tf, g_tf) = timed(lambda: pt87.table_fill(ctx, lt, b, S3['tf_seed'] + bi), warm=True)
        TFB[b] = Bt; tf[b] = {'fill_s': t_tf, 'fill_gpu_ms': g_tf}
    Rr['tf_cost'] = tf
    # ---------------- source-only update and multi-source reuse (relations resident, geometry fixed)
    srcs = [FX['light_sets'][k] for k in ('LMIX88', 'LSPOT88')] + random_lights(S3['n_random_sources'] if not DEV else 2, S3['source_seed'], ex['T1'])
    rows = []
    for li, ls in enumerate(srcs):
        lt2 = PT.Lights(ls)
        row = {}
        (q2, row['q_s'], row['q_gpu_ms']) = timed(lambda: Q.qtable(ctx, lt2, E87['P'], E87['Mq'], E87['seed_q']), warm=li == 0)
        (_, row['R0_apply_s'], _) = timed(lambda: RL.recurrence([r0], q2)[0].cpu().numpy(), warm=li == 0)
        (_, row['R1_apply_s'], _) = timed(lambda: RL.recurrence([r1], q2)[0].cpu().numpy(), warm=li == 0)
        (_, row['C3_s'], _) = timed(lambda: X.sender_side(ctx, q2, d8, ctx.N * K, E87['seed_c3']), warm=li == 0)
        row['digest_R1_unchanged'] = r1.digest()
        rows.append(row)
    Rr['source_updates'] = rows
    Rr['source_update_median'] = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}
    Rr['source_reuse_digest_constant'] = bool(len(set(r['digest_R1_unchanged'] for r in rows)) == 1)
    # ---------------- memory
    Rr['memory'] = {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'process': PT.gpu_mem(),
                    'R0_state_bytes': int(st.hit.nbytes + st.dbin.nbytes + st.seg.nbytes), 'R1_relation_bytes': r1.nbytes(), 'R1M_relation_bytes': r1m.nbytes(),
                    'W_bytes': int(quad.W.astype(np.float32).nbytes), 'dense16_bytes': int(sum(t.numel() * 4 for t in {id(t): t for t in d16.T}.values())),
                    'table_bytes': int(B1.size * 4), 'R1_rays': r1.rays(), 'R1M_rays': r1m.rays(), 'R0_rays': int(ctx.N * K),
                    'R1_pairs': int(len(r1.pi)), 'R1_area_valid': r1.n_area_valid}
    # ---------------- quality ladders at T1
    tables = {'R0': B0, 'R1': B1, 'R1M': B1m, 'C3': B3}
    for b in TFB:
        tables[f'TF{b}'] = TFB[b]
    ref, fns = ref_image(R, 'W1_T1')
    mz = np.load(os.path.join(R, 'results', 'masks', 'W1.npz'))
    mk = {k: mz[k] for k in ('stationary_all', 'A', 'C_ss', 'R_mirror', 'D')}
    err = lambda img: {k: float(np.abs(img[m] - ref[m]).sum() / max(ref[m].sum(), 1e-30)) for k, m in mk.items()}
    L = {}
    arms = ['PT'] + list(tables)
    lad = S3['ladder_spp'] if not DEV else [4, 64]
    reps = S3['replicates'] if not DEV else 1
    for arm in arms:                                               # warm-up of every render kernel (excluded)
        if arm == 'PT':
            ctx.render(lt, 'ER', cam, res, 4, 1, 'cam', plain=True, b=4)
        else:
            ctx.render(lt, 'ER', cam, res, 4, 1, 'ecro', B=PT.fl(tables[arm]), plain=True, b=4)
    for spp in lad:
        for qr in range(reps):
            sd = S3['seed_base'] + 97 * spp + qr
            bsz = min(64, spp)
            for arm in arms:
                if arm == 'PT':
                    a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd, 'cam', plain=True, b=bsz)
                else:
                    a, T_ = ctx.render(lt, 'ER', cam, res, spp, sd + 50, 'ecro', B=PT.fl(tables[arm]), plain=True, b=bsz)
                img = PT.pack(a, res, spp)['tot'].astype(np.float64)
                L.setdefault(arm, []).append({'spp': spp, 'rep': qr, **tk(T_), **{'err_' + k: v for k, v in err(img).items()}})
        print('ladder', spp, json.dumps({a: round(L[a][-1]['err_stationary_all'], 4) for a in arms}), flush=True)
    Rr['ladder'] = L
    Rr['ladder_median'] = {}
    for arm in L:
        by = {}
        for row in L[arm]:
            by.setdefault(row['spp'], []).append(row)
        Rr['ladder_median'][arm] = [{'spp': s_, **{k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ('spp', 'rep', 'launches')}}
                                    for s_, rows in sorted(by.items())]
    v2 = {}
    for arm in ('PT', 'R1'):
        for rep in range(2):
            if arm == 'PT':
                _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 88820 + rep, 'cam', plain=True, b=64)
            else:
                _, Tv = ctx.render(lt, 'ER', FX['camera2'], res, 64, 88820 + rep, 'ecro', B=PT.fl(B1), plain=True, b=64)
        v2[arm] = tk(Tv)
    Rr['view2_64spp'] = v2
    Rr['ref_files'] = [os.path.basename(f) for f in fns]
    Rr['foreign_after'] = foreign()
    json.dump(Rr, open(os.path.join(out, 'econ88.json'), 'w'), indent=1)
    print('ECON_DONE', flush=True)


# ------------------------------------------------------------------------------------------------ scaling
def refine(s, glossy_B=False):
    """Patch refinement by factor s (stationary grids, box faces, sphere) and optional second glossy mover-class surface (box B = S1)."""
    if not hasattr(S, '_orig_rects'):
        S._orig_rects, S._orig_faces = S.stationary_rects, S.box_faces
        S._orig_sph = (S.SPH_NPHI, S.SPH_NZ)
    def rects():
        return [r[:5] + (max(1, int(round(r[5] * s))), max(1, int(round(r[6] * s))), r[7]) for r in S._orig_rects()]
    def faces(box, mat):
        n = max(1, int(round(3 * s)))
        return [f[:5] + (n, n, f[7]) for f in S._orig_faces(box, mat)]
    S.stationary_rects = rects; S.box_faces = faces
    S.SPH_NPHI = max(2, int(round(S._orig_sph[0] * s))); S.SPH_NZ = max(1, int(round(S._orig_sph[1] * s)))
    fam = fx88._FAM['wl88']
    fam['states'][5] = fam['states'][1] if glossy_B else fx88._ORIG['states'][5]
    fam['lambert'][5] = 0 if glossy_B else 1
    fam['albedo'][5] = 0.0 if glossy_B else fx88._ORIG['albedo'][5]
    fx88._CLASS['wl88'][5] = 0 if glossy_B else 2
    import rel88 as RL
    RL._PEAK.clear()
    fx88.use_family('wl88')


def scale(R):
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    run88.inject_choice(R)
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    rows = []
    cases = [(s, gb) for gb in (False, True) for s in S3['scale_factors']] if not DEV else [(1.0, False), (1.5, True)]
    for s, gb in cases:
        refine(s, gb)
        ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']])
        quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
        d16 = RL.Dense(RP['dense_sub'], RP['M_dense'], RP['seed_dense'])
        d8 = Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], E87['M_dense'], E87['seed_dense']).dense()
        row = {'s': s, 'glossy_B': gb, 'N': int(ctx.N), 'peaked_patches': int(RL.peaked()[0][ctx.G.PMAT].sum())}
        (st, row['R0_state_s'], row['R0_state_gpu_ms']) = timed(lambda: EC.build_state(ctx), warm=True)
        T_ = {}
        (r1, row['R1_build_s'], row['R1_build_gpu_ms']) = timed(lambda: RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st, timing=T_), warm=True)
        row['R1_parts'] = T_
        row.update({'R1_pairs': int(len(r1.pi)), 'R1_area_samples': int(len(r1.pi) * r1.m), 'R1_area_valid': r1.n_area_valid, 'R0_rays': int(ctx.N * K),
                    'R1_rays': r1.rays(), 'R0_bytes': int(st.hit.nbytes + st.dbin.nbytes + st.seg.nbytes), 'R1_bytes': r1.nbytes()})
        (q, row['q_s'], _) = timed(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']), warm=True)
        r0 = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
        Lq = torch.as_tensor(np.asarray(q, np.float32), device=RL.dev)
        for J_ in S3['orders']:
            (_, row[f'R0_apply_J{J_}_s'], _) = timed(lambda: RL.recurrence([r0], q, J_=J_)[0].cpu().numpy(), warm=True)
            (_, row[f'R1_apply_J{J_}_s'], _) = timed(lambda: RL.recurrence([r1], q, J_=J_)[0].cpu().numpy(), warm=True)
        (_, row['C3_s'], _) = timed(lambda: X.sender_side(ctx, q, d8, ctx.N * K, E87['seed_c3']), warm=True)
        # edit update (T0 -> T1) for this scale
        Tu = {}
        ctx.set_geom(ex['T1'])
        (_, row['R1_update_s'], _) = timed(lambda: r1.update(ctx, ex['moved'], timing=Tu))
        row['R1_update_parts'] = Tu
        (_, row['R0_update_s'], _) = timed(lambda: EC.update_state(ctx, st, ex['moved']))
        if s == 1.0 and not gb:
            src = []
            for nl in S3['source_counts']:
                lts = [PT.Lights(l) for l in random_lights(nl, S3['source_seed'] + nl, ex['T1'])]     # pt86 holds <= 2 lights: one q per source
                (_, tq, _) = timed(lambda: [Q.qtable(ctx, l2, E87['P'], E87['Mq'], E87['seed_q']) for l2 in lts], warm=True)
                (_, ta, _) = timed(lambda: [RL.recurrence([r1], np.zeros((ctx.N, NB, 3), np.float32))[0] for _ in lts], warm=True)
                src.append({'sources': nl, 'q_s': tq, 'R1_apply_s': ta})
            row['q_vs_lights'] = src
        rows.append(row)
        print('scale', json.dumps({k: v for k, v in row.items() if not isinstance(v, (dict, list))}), flush=True)
        del r1, st, quad, d16, ctx
        torch.cuda.empty_cache()
    json.dump({'rows': rows, 'foreign': foreign()}, open(os.path.join(out, 'scale88.json'), 'w'), indent=1)
    print('SCALE_DONE', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    {'econ': econ, 'scale': scale}[sys.argv[1]](sys.argv[2])
    os._exit(0)
