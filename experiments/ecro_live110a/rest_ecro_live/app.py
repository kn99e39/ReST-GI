"""ReST-GI ECRO live viewer, portable build (WL110A). The per-frame work is the WL110 engine (live110.Engine) over the unchanged
WL101 runtime; this module adds backend selection, an optional MPS generator path and a backend-neutral frame loop.

  python -m rest_ecro_live [--backend auto|cuda|llvm] [--generator-device auto|cpu|mps|cuda] [--case sss|near] [--res 512]
                           [--spp 1] [--fps-cap 30] [--frames 0] [--present tk|none] [--log out.json]
"""
import argparse, copy, json, os, sys, time, types


def build_engine(be, case, res, spp, period, relation='auto'):
    import live110 as L                       # imported after backend.configure: rt101 is already shimmed
    eng = L.Engine(case, res, spp, 'raw', period, 0)
    if relation == 'auto':
        relation = 'selective' if be.R.FREEZE else 'full'
    if relation == 'full':
        # rt101.RT.rebuild_state (WL101 arm B_full; equal to the selective update, WL101 + WL110 V5). Without kernel replay the
        # selective update traces a different number of rays every frame (no bucket padding when FREEZE is off), and the LLVM
        # backend then compiles a new kernel per frame (~1.2 s); the full rebuild has a fixed size and hits the kernel cache.
        rt = eng.rt
        rt.update_state = lambda g_old, moved, bucket=True: rt.rebuild_state() if moved else 0
    eng.relation_mode = relation
    if be.generator_device == 'mps':
        _attach_mps_generator(be, eng)
    return L, eng


def _attach_mps_generator(be, eng):
    """Run model86.factors / apply on MPS; the transport (Dr.Jit LLVM) stays on the CPU. The result is checked once against the
    CPU generator on the initial state; above 1e-4 relative L1 the CPU path is kept and the reason recorded."""
    import torch
    R, rt = be.R, eng.rt
    mdev = torch.device('mps')
    model = copy.deepcopy(eng.model).to(mdev)
    model.mat_states = eng.model.mat_states.to(mdev)
    cpu_respond = rt.respond

    def respond_mps(self, s, use_single=True, use_global=True):
        gs = R.M.GeoState.__new__(R.M.GeoState)
        gs.N = self.gs.N
        for k in ('cen', 'nrm', 'tan', 'area', 'mat'):
            setattr(gs, k, getattr(self.gs, k).to(mdev))
        gs.hit = self.hit.long().view(self.N, R.K).to(mdev)
        gs.dbin = self.dbin.long().view(self.N, R.K).to(mdev)
        with torch.no_grad():
            f = model.factors(gs)
            B = model.apply(f, gs, s.to(mdev), use_single=use_single, use_global=use_global)[0]
        return B.to('cpu').contiguous(), f

    s = rt.source()
    Bc, _ = cpu_respond(s)
    try:
        Bm, _ = respond_mps(rt, s)
        rel = float((Bm - Bc).abs().sum() / Bc.abs().sum())
    except Exception as e:  # noqa: BLE001
        rel, err = None, repr(e)
    else:
        err = None
    ok = rel is not None and rel <= 1e-4
    eng.generator_check = {'device': 'mps', 'table_rel_l1_vs_cpu': rel, 'accepted': ok, 'error': err}
    if ok:
        rt.respond = types.MethodType(respond_mps, rt)
    else:
        be.generator_device = 'cpu'


def overlay(be, eng, L, k, t_sim, u, rec, fps, cap, missed):
    p = rec['parts_ms'] if rec else {}
    g = lambda n: f"{p[n]:7.2f}" if n in p else '      -'
    return '\n'.join([
        'ReST-GI · ECRO live runtime',
        f"backend {be.name.upper()} ({be.variant})  gen {be.generator_device}",
        f"relation {eng.relation_mode}  kernel replay {'on' if be.R.FREEZE else 'off'}",
        f"{be.info['device_name'][:34]}",
        f"case {eng.case.upper()}  {eng.res}x{eng.res}  {eng.spp} spp",
        f"checkpoint sha256 {eng.ck_sha[:16]}…  loads {L.LOADS['torch.load']}",
        '',
        f"frame {k:6d}   sim t {t_sim:7.2f} s   u {u:5.3f}",
        f"dirty rays {rec['dirty'] if rec else 0:7d} / {eng.rt.hit.numel()}",
        f"ECRO table Σ {rec['proof']['table_sum'] if rec else 0:.6e}",
        '',
        f"PRESENTED FPS (1 s)  {fps:6.1f}   cap {cap if cap else 'none'}",
        f"compute edit→img ms {rec['compute_ms'] if rec else float('nan'):7.2f}",
        f"  geometry          {g('geometry')}",
        f"  relation          {g('relation')}",
        f"  source            {g('source')}",
        f"  neural response   {g('neural')}",
        f"  ray render        {g('render')}",
        f"  tonemap+readback  {g('display')}",
        f"present ms          {rec['present_ms'] if rec else float('nan'):7.2f}",
        f"missed deadlines    {missed}",
        '',
        'Every frame runs ECRO inference',
        '+ ray rendering (not pre-rendered).',
    ])


def run(args):
    from . import runtime, backend
    root = runtime.find_root()
    auth = runtime.authenticate(root)
    if not auth['ok']:
        raise SystemExit(f'runtime authentication failed: {json.dumps(auth, indent=1)}')
    runtime.add_import_paths(root)
    be = backend.configure(args.backend, args.generator_device, freeze=args.freeze)
    L, eng = build_engine(be, args.case, args.res, args.spp, args.period, args.relation)
    import numpy as np
    pres = L.TkPresenter(args.res, f'ReST-GI ECRO live — {args.case} · {be.name}') if args.present == 'tk' else None
    dt = 1.0 / args.anim_rate
    cap = args.fps_cap
    recs, rec, missed = [], None, 0
    t_start = t_anchor = time.perf_counter(); k_anchor = 0
    pres_ts = []
    fps = float('nan')
    k = 0
    while args.frames == 0 or k < args.frames:
        if pres is not None and pres.closed:
            break
        t_sim = k * dt
        geom, u = eng.anim.pose(t_sim)
        P = L.Parts()
        L.sync(); t0 = time.perf_counter()
        r = eng.step(geom, args.seed_base + k, P)
        compute = (time.perf_counter() - t0) * 1e3
        new = {'k': k, 'u': u, 'dirty': r['dirty'], 'parts_ms': dict(P.t), 'compute_ms': compute}
        tp = time.perf_counter()
        if pres is not None:
            pres.present(r['img'], overlay(be, eng, L, k, t_sim, u, rec, fps, cap, missed), f"LIVE · {be.name} · frame {k} · {fps:.1f} FPS")
        t2 = time.perf_counter()
        new['present_ms'] = (t2 - tp) * 1e3
        new['presented_at_s'] = t2 - t_start
        new['proof'] = eng.proof(r)
        pres_ts.append(t2)
        win = [x for x in pres_ts[-400:] if t2 - x <= 1.0]
        fps = (len(win) - 1) / (t2 - win[0]) if len(win) > 1 and t2 > win[0] else float('nan')
        if cap:
            slot_end = t_anchor + (k - k_anchor + 1) / cap
            now = time.perf_counter()
            if now > slot_end:
                new['deadline_missed'] = True
                missed += int(k >= args.warmup)
                t_anchor, k_anchor = now, k + 1
            else:
                while slot_end - time.perf_counter() > 1.5e-3:
                    time.sleep(1e-3)
                while time.perf_counter() < slot_end:
                    pass
        recs.append(new); rec = new
        if args.present == 'none' and k % 30 == 0:
            print(f"frame {k} compute {compute:.1f} ms  fps {fps:.1f}", flush=True)
        k += 1
    if pres is not None:
        pres.close()
    meas = [x for x in recs if x['k'] >= args.warmup]
    pt = [x['presented_at_s'] for x in meas]
    st = lambda v: {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95))} if len(v) else None
    summary = {'batch': 'WL110A', 'args': vars(args), 'backend': be.name, 'variant': be.variant, 'generator_device': be.generator_device,
               'generator_check': getattr(eng, 'generator_check', None), 'relation': eng.relation_mode, 'info': be.info, 'runtime': auth,
               'frames': len(recs), 'measured': len(meas),
               'presented_fps': (len(pt) - 1) / (pt[-1] - pt[0]) if len(pt) > 1 else None,
               'compute_ms': st([x['compute_ms'] for x in meas]), 'present_ms': st([x['present_ms'] for x in meas]),
               'parts_ms': {n: st([x['parts_ms'].get(n, 0.0) for x in meas]) for n in (meas[0]['parts_ms'] if meas else {})},
               'missed_deadlines': missed, 'respond_calls': eng.respond_calls, 'checkpoint_loads': L.LOADS['torch.load'],
               'parameters_unchanged': L.param_digest(eng.model) == eng.param0}
    if args.log:
        with open(args.log, 'w') as f:
            json.dump({'summary': summary, 'frames': recs}, f)
    print('SUMMARY', json.dumps({k: summary[k] for k in ('backend', 'generator_device', 'relation', 'presented_fps', 'compute_ms', 'missed_deadlines',
                                                          'respond_calls', 'checkpoint_loads', 'parameters_unchanged')}), flush=True)
    return summary


def parser():
    ap = argparse.ArgumentParser(prog='rest-ecro-live', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', default='view', choices=['view', 'selfcheck', 'info'])
    ap.add_argument('--backend', choices=['auto', 'cuda', 'llvm'], default='auto')
    ap.add_argument('--generator-device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    ap.add_argument('--freeze', choices=['auto', 'on', 'off'], default='auto', help='Dr.Jit kernel replay (auto: on for CUDA, off for LLVM)')
    ap.add_argument('--relation', choices=['auto', 'selective', 'full'], default='auto', help='relation update (auto: selective with kernel replay, full rebuild without)')
    ap.add_argument('--case', choices=['sss', 'near'], default='sss')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--spp', type=int, default=1)
    ap.add_argument('--period', type=float, default=6.0)
    ap.add_argument('--anim-rate', type=float, default=30.0)
    ap.add_argument('--fps-cap', type=float, default=30.0)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--present', choices=['tk', 'none'], default='tk')
    ap.add_argument('--seed-base', type=int, default=11000000)
    ap.add_argument('--log', default=None)
    ap.add_argument('--out', default=None, help='selfcheck: output JSON')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'view':
        run(args)
    elif args.command == 'selfcheck':
        from . import selfcheck
        selfcheck.run(args)
    else:
        from . import runtime, backend
        root = runtime.find_root()
        auth = runtime.authenticate(root)
        runtime.add_import_paths(root)
        be = backend.configure(args.backend, args.generator_device, freeze=args.freeze)
        print(json.dumps({'runtime': auth, 'backend': be.name, 'variant': be.variant, 'generator_device': be.generator_device,
                          'info': be.info}, indent=1))
    sys.stdout.flush()
    os._exit(0)
