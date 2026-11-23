"""WL110 live demonstrator: the frozen WL100 ECRO checkpoint executed every frame by the unchanged WL101 optimized runtime
(rt101.RT, arm B_sel) while both boxes follow a deterministic rigid animation. Experiment-local; no frozen source is modified.

Per presented frame (nothing is precomputed or loaded from disk):
  pose(t) -> rt.set_geom (BVH + incremental tables) -> rt.update_state (selective relation re-trace, frozen dirty rule)
  -> rt.source (shadow-ray source encoding) -> rt.respond (ECRO generator + linear application = table B)
  -> DLPack hand-off -> rt.frame (physical LOW + learned-HIGH lookup, replayed Dr.Jit kernels)
  -> compose (e1 + e2 + hl) / spp [-> optional WL101B SVGF, history reset on every geometry change]
  -> fixed display transform (WL100 exposure key, (x / (1 + x))^(1/2.2)) -> uint8 readback -> Tk window -> X round trip.

Usage (env110.sh sourced; DISPLAY set for --present tk):
  python live110.py --case sss --mode raw --fps-cap 30 --frames 540 --out <dir>
  --present none     headless (compute-only) loop, same engine
  --frames 0         run until the window is closed / Escape / SIGINT
Outputs in --out: frames.jsonl (one record per frame), summary.json, provenance in summary['provenance'].
"""
import argparse, hashlib, importlib.util, json, math, os, platform, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
sys.path.insert(0, str(EXP / 'ecro_dyn101'))
sys.path.insert(0, str(HERE))
os.environ.setdefault('FREEZE', '1')
import numpy as np  # noqa: E402
import rt101 as R  # noqa: E402  (unchanged WL101 arm-B runtime)
import anim110 as A  # noqa: E402

torch, mi, dr = R.torch, R.mi, R.dr
assert R.FREEZE, 'WL110 runs the replayed (dr.freeze) WL101 runtime only'

# WL100 per-dataset display exposure keys (evaluation/100/metrics100.json cases.*.exposure_key; WL101A display for sss)
EXPOSURE = {'sss': 0.1938553403045695, 'near': 0.4834977801154364}
WL101_CHECKPOINT_SHA = {'sss': '2d9cb5805ecff8a6aae6639ddca7a3bcd0c84b0eaa9b2a3282fe127df9284f42',
                        'near': 'b3550787e08c8a4d41c975eb485be459591d84cf14a7c010a36984d10ac62e5a'}

# ---- checkpoint-load accounting (a reload during the loop would show up here)
LOADS = {'torch.load': 0}
_torch_load = torch.load


def _counted_load(*a, **k):
    LOADS['torch.load'] += 1
    return _torch_load(*a, **k)


torch.load = _counted_load


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def param_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


class Parts:
    """Synchronised wall-clock part timer (same bracketing as WL101: full device sync before and after every part)."""

    def __init__(self):
        self.t = {}

    def __call__(self, name, fn):
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
        return r


def load_wl101b():
    """WL101B reconstruction (svgf.py + run.Runtime guide program), imported unchanged."""
    sys.path.insert(0, str(EXP / 'ecro_sss101b'))
    spec = importlib.util.spec_from_file_location('wl101b_run', EXP / 'ecro_sss101b' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Engine:
    def __init__(self, case, res=512, spp=1, mode='raw', period_s=6.0, quantize=0):
        self.case, self.res, self.spp, self.mode = case, res, spp, mode
        self.anim = A.Animation(case, period_s, quantize)
        self.ck_path = R.ROOT100 / case / 'model.pt'
        self.ck_sha = sha_file(self.ck_path)
        t0 = time.perf_counter()
        self.model = R.load_model(case)
        self.param0 = param_digest(self.model)
        g0, _ = self.anim.pose(0.0)
        self.rt = R.RT(g0, case, self.model, self.anim.light)
        sync()
        self.init_ms = (time.perf_counter() - t0) * 1e3
        self.cam = dict(R.C100.P['datasets'][case]['camera'], res=res)
        self.key = EXPOSURE[case]
        self.n = res * res
        self.respond_calls = 0
        self.revision = 0
        self.prevB = None
        nray = self.rt.hit.numel()
        self.w_hit = (torch.arange(nray, device=R.dev, dtype=torch.int64) % 65521) + 1
        self.den = self.host = None
        if mode == 'svgf':
            assert case == 'sss' and res == 512, 'WL101B reconstruction exists only for the SSS case at 512^2'
            WB = load_wl101b()
            self.WB = WB

            class GuideHost(WB.Runtime):            # reuse WL101B's guide program on this engine's RT
                def __init__(s, R_, rt):
                    s.R, s.rt, s.guides = R_, rt, {}
            self.host = GuideHost(R, self.rt)
            self.den = SVGF_ = WB.SVGF(res, R.dev, R.PT.cam_basis, WB.P['reconstruction'])
            self.recon_settings = WB.P['reconstruction']
            self.recon_protocol_sha = sha_file(EXP / 'ecro_sss101b' / 'protocol' / 'reconstruction.json')
            del SVGF_

    def compose(self, out):
        ch = torch.stack([o.torch() for o in out])          # keys sorted: e1 rgb, e2 rgb, hl rgb
        return ch.view(3, 3, self.n).sum(0) / self.spp       # [3, n] linear HDR

    def tonemap(self, hdr):
        y = hdr.clamp(min=0) * self.key
        ldr = (y / (1 + y)).pow(1 / 2.2)
        img = (ldr.T.reshape(self.res, self.res, 3) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
        return img.cpu().numpy()                             # synchronous readback (part of the measured frame)

    def step(self, geom, seed, P):
        rt = self.rt
        moved, g_old = P('geometry', lambda: rt.set_geom(geom))
        nd = P('relation', lambda: rt.update_state(g_old, moved))
        s = P('source', lambda: rt.source())
        B, _ = P('neural', lambda: rt.respond(s))
        self.respond_calls += 1
        Bd = P('handoff', lambda: R.t2d(B.view(-1)))
        out = P('render', lambda: rt.frame(Bd, self.cam, self.res, self.spp, seed))
        hdr = P('compose', lambda: self.compose(out))
        recon = None
        if moved:
            self.revision += 1
        if self.den is not None:
            guide = P('guide', lambda: self.host.guide(self.cam))
            recon, count, _ = P('reconstruct', lambda: self.den.apply(hdr, guide, self.cam, self.revision))
            shown = recon
        else:
            shown = hdr
        img = P('display', lambda: self.tonemap(shown))
        return {'moved': moved, 'dirty': int(nd), 'B': B, 'hdr': hdr, 'shown': shown, 'img': img}

    def proof(self, r):
        """Per-frame evidence that the table and image were recomputed (one device->host copy of 6 scalars)."""
        B = r['B']
        dB = (B - self.prevB).abs().sum().double() if self.prevB is not None else torch.zeros((), device=R.dev, dtype=torch.float64)
        v = torch.stack([B.double().sum(), dB, (self.rt.hit.long() * self.w_hit).sum().double(),
                         (self.rt.dbin.long() * self.w_hit).sum().double(), r['hdr'].double().sum(), r['shown'].double().sum()]).cpu().numpy()
        self.prevB = B.clone()
        return {'table_sum': float(v[0]), 'table_abs_change': float(v[1]), 'hit_checksum': int(v[2]), 'dbin_checksum': int(v[3]),
                'raw_hdr_sum': float(v[4]), 'shown_hdr_sum': float(v[5])}

    def state_digest(self):
        """WL101 run_b101 state digest (sha256 of hit int16 + dbin uint8)[:16] (host copy; verification only)."""
        hb = self.rt.hit.cpu().numpy().astype(np.int16); db = self.rt.dbin.cpu().numpy().astype(np.uint8)
        return hashlib.sha256(hb.tobytes() + db.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ presentation (Tk on X11)
class TkPresenter:
    W_PANEL = 380

    def __init__(self, res, title):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('+0+0')
        self.root.resizable(False, False)
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW', self._close)
        self.root.bind('<Escape>', lambda e: self._close())
        self.res = res
        W, H = res + self.W_PANEL + 16, res + 16
        self.c = tk.Canvas(self.root, width=W, height=H, bg='#101014', highlightthickness=0)
        self.c.pack()
        from PIL import Image, ImageTk               # ImageTk.paste: Tk photo block put (2.2 ms vs 7.7 ms for PPM data, 512^2)
        self.Image = Image
        self.photo = ImageTk.PhotoImage(Image.new('RGB', (res, res)))
        self.c.create_image(8, 8, anchor='nw', image=self.photo)
        self.txt = self.c.create_text(res + 20, 10, anchor='nw', fill='#e8e8e8', font=('DejaVu Sans Mono', 10), text='starting')
        self.badge = self.c.create_text(16, 14, anchor='nw', fill='#ffd84a', font=('DejaVu Sans Mono', 10, 'bold'), text='')
        self.root.update()

    def _close(self):
        self.closed = True

    def present(self, img, text, badge):
        self.photo.paste(self.Image.fromarray(img))
        self.c.itemconfigure(self.txt, text=text)
        self.c.itemconfigure(self.badge, text=badge)
        self.root.update_idletasks()
        self.root.update()
        self.root.winfo_pointerxy()           # X round trip: the server has processed the image upload and redraw

    def geometry(self):
        return {'x': self.root.winfo_rootx(), 'y': self.root.winfo_rooty(), 'w': self.root.winfo_width(), 'h': self.root.winfo_height()}

    def close(self):
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass


def pct(v, q):
    return float(np.percentile(np.asarray(v, np.float64), q)) if len(v) else float('nan')


def stats(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return None
    return {'n': int(v.size), 'median': float(np.median(v)), 'p95': pct(v, 95), 'p99': pct(v, 99), 'mean': float(v.mean()),
            'min': float(v.min()), 'max': float(v.max())}


def overlay(eng, k, t_sim, u, rec, hist, cap, missed, nmeas, wall_lag):
    p = rec['parts_ms'] if rec else {}
    g = lambda n: f"{p[n]:6.2f}" if n in p else '     -'
    pres = hist['present_fps_1s']
    lines = [
        'ReST-GI · ECRO live runtime (WL110)',
        f"case {eng.case.upper()}  mode {eng.mode.upper()}  {eng.res}x{eng.res}  {eng.spp} spp",
        f"checkpoint {eng.case}/model.pt",
        f"  sha256 {eng.ck_sha[:16]}…  loads {LOADS['torch.load']}",
        '',
        f"frame {k:6d}   sim t {t_sim:7.2f} s",
        f"pose u {u:5.3f}  (cfg{eng.anim.segment['from_test_config']}→cfg{eng.anim.segment['to_test_config']} segment)",
        f"dirty rays {rec['dirty'] if rec else 0:7d} / {eng.rt.hit.numel()}",
        f"ECRO table Σ {rec['proof']['table_sum'] if rec else 0:.6e}",
        f"  |ΔB| vs prev {rec['proof']['table_abs_change'] if rec else 0:.3e}",
        '',
        f"PRESENTED FPS (1 s)   {pres:6.1f}   cap {cap if cap else 'none'}",
        f"loop period      ms  {rec['loop_ms'] if rec and rec.get('loop_ms') else float('nan'):6.2f}",
        f"compute edit→img ms  {rec['compute_ms'] if rec else float('nan'):6.2f}",
        f"  geometry update    {g('geometry')}",
        f"  relation update    {g('relation')}",
        f"  source encoding    {g('source')}",
        f"  neural response    {g('neural')}",
        f"  table hand-off     {g('handoff')}",
        f"  ray render         {g('render')}",
        f"  compose            {g('compose')}",
        f"  guide+reconstruct  {(p.get('guide', 0) + p.get('reconstruct', 0)):6.2f}" if 'reconstruct' in p else '  reconstruct       off',
        f"  tonemap+readback   {g('display')}",
        f"present (Tk/X11) ms  {rec['present_ms'] if rec else float('nan'):6.2f}",
        '',
        f"last {hist['n']:4d}: compute med {hist['cmed']:5.2f} p95 {hist['cp95']:5.2f}",
        f"           period  med {hist['pmed']:5.2f} p95 {hist['pp95']:5.2f}",
        f"missed deadlines {missed} / {nmeas}   lag {wall_lag:5.2f} s",
        f"torch GPU mem {torch.cuda.memory_allocated() / 2 ** 20:7.1f} MiB",
        '',
        'Not pre-rendered: every frame runs ECRO',
        'inference + ray rendering on the GPU.',
    ]
    return '\n'.join(lines)


def gpu_snapshot():
    try:
        u = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        a = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return {'gpu': u, 'compute_apps': a, 'own_pid': os.getpid()}
    except Exception as e:  # noqa: BLE001
        return {'error': str(e)}


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snap0 = gpu_snapshot()
    free0 = torch.cuda.mem_get_info()[0]
    eng = Engine(args.case, args.res, args.spp, args.mode, args.period, args.quantize)
    pres = TkPresenter(args.res, f'ReST-GI ECRO live — {args.case} {args.mode}') if args.present == 'tk' else None
    stop = {'flag': False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    dt = 1.0 / args.anim_rate
    cap = args.fps_cap
    recs = []
    pres_ts = []
    rec = None
    missed = 0
    hist = {'n': 0, 'cmed': float('nan'), 'cp95': float('nan'), 'pmed': float('nan'), 'pp95': float('nan'), 'present_fps_1s': float('nan')}
    k = 0
    t_anchor = time.perf_counter()          # schedule anchor: frame k is due to start at t_anchor + k / cap
    k_anchor = 0
    t_start = t_anchor
    prev_present = None
    while not stop['flag'] and (args.frames == 0 or k < args.frames):
        if pres is not None and pres.closed:
            break
        t_sim = (k * dt) if args.clock == 'frame' else (time.perf_counter() - t_start)
        geom, u = eng.anim.pose(t_sim)
        P = Parts()
        sync()
        t0 = time.perf_counter()
        r = eng.step(geom, args.seed_base + k, P)
        t1 = time.perf_counter()
        compute_ms = (t1 - t0) * 1e3
        rec_new = {'k': k, 't_sim': t_sim, 'u': u, 'geom_hash': A.geom_hash(geom), 'moved': r['moved'], 'dirty': r['dirty'],
                   'parts_ms': dict(P.t), 'compute_ms': compute_ms}
        badge = f"LIVE · frame {k} · {hist['present_fps_1s']:.1f} FPS"
        if pres is not None:
            text = overlay(eng, k, t_sim, u, rec, hist, cap, missed, max(0, k - args.warmup), (time.perf_counter() - t_start) - k * dt)
            tp0 = time.perf_counter()
            pres.present(r['img'], text, badge)
            t2 = time.perf_counter()
            rec_new['present_ms'] = (t2 - tp0) * 1e3
        else:
            t2 = time.perf_counter()
            rec_new['present_ms'] = 0.0
        rec_new['presented_at_s'] = t2 - t_start
        rec_new['loop_ms'] = (t2 - prev_present) * 1e3 if prev_present is not None else None
        prev_present = t2
        pres_ts.append(t2)
        tq = time.perf_counter()
        rec_new['proof'] = eng.proof(r)
        rec_new['proof_ms'] = (time.perf_counter() - tq) * 1e3
        rec_new['respond_calls'] = eng.respond_calls
        rec_new['revision'] = eng.revision
        if args.digest_every and k % args.digest_every == 0:
            rec_new['state_digest'] = eng.state_digest()
        # ---- pacing (cap > 0): frame k owns the slot [anchor + (k - k_anchor) / cap, + 1 / cap)
        if cap:
            slot_end = t_anchor + (k - k_anchor + 1) / cap
            now = time.perf_counter()
            if now > slot_end:
                rec_new['deadline_missed'] = True
                if k >= args.warmup:
                    missed += 1
                t_anchor, k_anchor = now, k + 1            # re-anchor: no catch-up burst, no frame skipping
                rec_new['sleep_ms'] = 0.0
            else:
                rec_new['deadline_missed'] = False
                ts = time.perf_counter()
                while slot_end - time.perf_counter() > 1.5e-3:
                    time.sleep(1e-3)
                while time.perf_counter() < slot_end:
                    pass
                rec_new['sleep_ms'] = (time.perf_counter() - ts) * 1e3
        recs.append(rec_new)
        rec = rec_new
        # rolling display statistics
        win = [x for x in recs[-300:] if x['k'] >= args.warmup]
        if win:
            lp = [x['loop_ms'] for x in win if x['loop_ms'] is not None]
            hist.update(n=len(win), cmed=float(np.median([x['compute_ms'] for x in win])), cp95=pct([x['compute_ms'] for x in win], 95),
                        pmed=float(np.median(lp)) if lp else float('nan'), pp95=pct(lp, 95) if lp else float('nan'))
        now = pres_ts[-1]
        n1 = sum(1 for x in pres_ts[-200:] if now - x <= 1.0)
        span = now - min(x for x in pres_ts[-200:] if now - x <= 1.0)
        hist['present_fps_1s'] = (n1 - 1) / span if n1 > 1 and span > 0 else float('nan')
        k += 1
    t_end = time.perf_counter()
    wingeom = pres.geometry() if pres is not None else None
    # ---------------------------------------------------------------- summary
    param1 = param_digest(eng.model)
    meas = [x for x in recs if x['k'] >= args.warmup]
    lp = [x['loop_ms'] for x in meas if x['loop_ms'] is not None]
    pt = [x['presented_at_s'] for x in meas]
    pres_fps = (len(pt) - 1) / (pt[-1] - pt[0]) if len(pt) > 1 else None
    parts = sorted({n for x in meas for n in x['parts_ms']})
    moved_frames = [x for x in meas if x['moved']]
    summary = {
        'batch': 'WL110', 'args': vars(args),
        'frames_total': len(recs), 'frames_measured': len(meas), 'warmup_frames': args.warmup,
        'wall_s': t_end - t_start,
        'presented_fps': pres_fps,
        'present_interval_ms': stats(lp),
        'compute_ms': stats([x['compute_ms'] for x in meas]),
        'present_ms': stats([x['present_ms'] for x in meas]),
        'proof_ms': stats([x['proof_ms'] for x in meas]),
        'parts_ms': {n: stats([x['parts_ms'].get(n, 0.0) for x in meas]) for n in parts},
        'compute_minus_parts_ms': stats([x['compute_ms'] - sum(x['parts_ms'].values()) for x in meas]),
        'deadline': {'cap_fps': cap, 'missed': int(sum(1 for x in meas if x.get('deadline_missed'))),
                     'slot_ms': 1e3 / cap if cap else None,
                     'animation_lag_s': (recs[-1]['presented_at_s'] - recs[-1]['t_sim']) if recs and args.clock == 'frame' else None},
        'dirty_rays': stats([x['dirty'] for x in meas]),
        'integrity': {
            'checkpoint_sha256': eng.ck_sha, 'checkpoint_matches_wl101': eng.ck_sha == WL101_CHECKPOINT_SHA[args.case],
            'torch_load_calls': LOADS['torch.load'], 'param_digest_start': eng.param0, 'param_digest_end': param1,
            'parameters_unchanged': eng.param0 == param1, 'respond_calls': eng.respond_calls, 'frames': len(recs),
            'respond_every_frame': eng.respond_calls == len(recs),
            'frames_with_motion': len(moved_frames),
            'table_changed_on_every_moved_frame': all(x['proof']['table_abs_change'] > 0 for x in moved_frames if x['k'] > 0),
            'distinct_geometries': len({x['geom_hash'] for x in recs}),
        },
        'memory': {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'torch_alloc_mib': torch.cuda.memory_allocated() / 2 ** 20,
                   'device_used_delta_mib': (free0 - torch.cuda.mem_get_info()[0]) / 2 ** 20},
        'recordings': {'state': getattr(eng.rt._f_state, 'n_recordings', None), 'source': getattr(eng.rt._f_source, 'n_recordings', None),
                       'frames': {f'{a}x{b}': getattr(v, 'n_recordings', None) for (a, b), v in eng.rt._frames.items()}},
        'window': wingeom,
        'provenance': {
            'checkpoint': str(eng.ck_path.relative_to(REPO)), 'configs_sha256': sha_file(R.ROOT100 / 'configs.json'),
            'rt101_sha256': sha_file(EXP / 'ecro_dyn101' / 'rt101.py'), 'live110_sha256': sha_file(__file__),
            'anim110_sha256': sha_file(HERE / 'anim110.py'), 'animation': eng.anim.describe(),
            'camera': eng.cam, 'exposure_key': eng.key, 'seed_policy': f'frame seed = {args.seed_base} + frame index (rt101.frame)',
            'commit': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'dirty_worktree': bool(subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                                                  capture_output=True, text=True).stdout.strip()),
            'versions': {'python': platform.python_version(), 'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__,
                         'gpu': torch.cuda.get_device_name(), 'host': platform.node(), 'os': platform.platform()},
            'display': os.environ.get('DISPLAY'), 'init_ms': eng.init_ms,
            'recon': ({'protocol_sha256': eng.recon_protocol_sha, 'settings': eng.recon_settings,
                       'history_rule': 'radiance history valid only while the geometry revision is unchanged (WL101B contract)'}
                      if eng.den is not None else None),
        },
        'gpu_before': snap0, 'gpu_after': gpu_snapshot(),
    }
    with open(out / 'frames.jsonl', 'w') as f:
        for x in recs:
            f.write(json.dumps(x) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    if args.save_last:
        from PIL import Image
        Image.fromarray(r['img']).save(out / 'last_frame.png')
    if pres is not None:
        pres.close()
    print('DONE', json.dumps({'presented_fps': pres_fps, 'compute_med': summary['compute_ms']['median'] if summary['compute_ms'] else None,
                              'compute_p95': summary['compute_ms']['p95'] if summary['compute_ms'] else None,
                              'interval_med': summary['present_interval_ms']['median'] if summary['present_interval_ms'] else None,
                              'interval_p95': summary['present_interval_ms']['p95'] if summary['present_interval_ms'] else None,
                              'missed': summary['deadline']['missed'], 'integrity': summary['integrity']}), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=['sss', 'near'], default='sss')
    ap.add_argument('--mode', choices=['raw', 'svgf'], default='raw')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--spp', type=int, default=1)
    ap.add_argument('--period', type=float, default=6.0, help='animation ping-pong period in simulation seconds')
    ap.add_argument('--quantize', type=int, default=0, help='round u to k / Q (16 = WL101 V_SSS_ANIM poses)')
    ap.add_argument('--clock', choices=['frame', 'wall'], default='frame',
                    help='frame: t = k / anim-rate (deterministic); wall: t = elapsed wall time')
    ap.add_argument('--anim-rate', type=float, default=30.0, help='simulation frames per second for --clock frame')
    ap.add_argument('--fps-cap', type=float, default=30.0, help='presentation pacing; 0 = uncapped')
    ap.add_argument('--frames', type=int, default=540, help='0 = until stopped')
    ap.add_argument('--warmup', type=int, default=30, help='frames excluded from statistics (kernel recording)')
    ap.add_argument('--present', choices=['tk', 'none'], default='tk')
    ap.add_argument('--seed-base', type=int, default=11000000)
    ap.add_argument('--digest-every', type=int, default=0, help='WL101 state digest every n frames (host copy; verification only)')
    ap.add_argument('--save-last', action='store_true')
    ap.add_argument('--out', required=True)
    run(ap.parse_args())
    os._exit(0)


if __name__ == '__main__':
    main()
