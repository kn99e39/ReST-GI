"""WL112 Stage C pre-training checks (before any candidate training).

  python precheck112.py ROOT      -> ROOT/stageC/precheck.json
(1) candidate readout of the physical Stage-B table reproduces the stored Stage-B image (V0 stream 0);
(2) measure-weighted sub-cells reproduce the parent (Stage A replica 0);
(3) candidate operator: zero -> zero, superposition, non-negativity (untrained S2 at F0, two real sources);
(4) support: native query-weight share of the selected 2x2 table at every F0 view (from Stage B);
(5) 100-step budget probe (time per step, peak memory) on real D112 items.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402


def main(root):
    root = Path(root); out = {}
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    sel = {'SPATIAL_G': 'S_G', 'SPATIAL_U': 'S_U', 'PROJECTION': 'G'}[dec['outcome']]
    kind = 'coarse' if sel == 'G' else 'sub'
    import tracer112 as T
    import eval112 as E
    PT = T.PT
    geom, light_cfg = C.fixture('F0'); light = PT.Lights(light_cfg)
    tabs, native = E.stageB_tables(root)
    tab = [t for t in tabs if t[0] == f'{sel}_r0'][0]
    ctx = T.Ctx112(geom, 'C')
    seed = 11210000 + 0 * 10 + 0 + C.SO
    img, _, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, E.SPP, seed, 'ecro',
                                tables=[(tab[0], PT.fl(np.ascontiguousarray(tab[1], np.float32).ravel()), tab[2])], b=E.LAUNCH)
    with np.load(root / 'stageB' / 'stream_V0_0.npz') as z:
        stored = z[f'hl|{tab[0]}'].astype(np.float64)
    d = np.abs(img[tab[0]] - stored)
    out['readout_reproduction'] = {'table': tab[0], 'max_abs': float(d.max()), 'rel_l1': float(d.sum() / np.abs(stored).sum()),
                                   'pass': bool(d.sum() / np.abs(stored).sum() <= 1e-6)}
    del ctx
    acc = E.load_acc(root / 'stageA' / 'C_r0.npz')
    meas = 'U' if sel == 'S_U' else 'G'
    Tc, wc, _, _ = T.tables_from(acc, meas, 'coarse'); Ts, ws, _, _ = T.tables_from(acc, meas, 'sub')
    N, NS, NB = 1270, 4, 64
    rec = (Ts.reshape(N, NS, NB, 3) * ws.reshape(N, NS, NB, 1)).sum(1) / np.maximum(ws.reshape(N, NS, NB).sum(1), 1e-30)[..., None]
    m = wc.reshape(N, NB) > 0
    out['parent_child'] = {'max_rel': float(np.abs(rec[m] - Tc.reshape(N, NB, 3)[m]).max() / np.abs(Tc).max())}
    out['parent_child']['pass'] = bool(out['parent_child']['max_rel'] <= 1e-6)
    # operator contracts
    import torch
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model112 as M2
    torch.manual_seed(0)
    ctx = T.Ctx112(geom, 'C')
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    s1 = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
    s2 = torch.as_tensor(ctx.source(PT.Lights(C.wl100_test(1)['lights'][0])), device='cuda').unsqueeze(0)
    model = (M2.ECROS2() if kind == 'sub' else M.ECRO()).cuda()
    model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    with torch.no_grad():
        f = model.factors(gs)
        z0 = model.apply(f, gs, torch.zeros_like(s1)); b1 = model.apply(f, gs, s1); b2 = model.apply(f, gs, s2); b12 = model.apply(f, gs, s1 + s2)
    out['operator'] = {'zero_max': float(z0.abs().max()), 'superposition_rel': float((b12 - b1 - b2).abs().sum() / b12.abs().sum()),
                       'min_value': float(min(b1.min(), b2.min())), 'params': int(sum(x.numel() for x in model.parameters())),
                       'params_original': int(sum(x.numel() for x in M.ECRO().parameters()))}
    out['operator']['pass'] = bool(out['operator']['zero_max'] == 0 and out['operator']['superposition_rel'] <= 1e-5 and out['operator']['min_value'] >= 0)
    met = json.loads((root / 'analysis' / 'stageB_metrics.json').read_text())
    out['support'] = {v: met[v]['arms'][sel]['coverage'] for v in met}
    # budget probe on real items
    import train112 as TR
    key = 'Gs' if sel == 'S_G' else ('Us' if sel == 'S_U' else 'Gc')
    files = sorted((root / 'stageC' / 'data' / 'train').glob('g0000*.npz'))[:2]
    items = [TR.Item(f, key) for f in files if '.tmp' not in f.name]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(100):
        g_, s_, t2, th, mm = items[i % len(items)].tensors(); f = model.factors(g_)
        a = model.apply(f, g_, s_, use_global=False); b = model.apply(f, g_, s_, use_single=False)
        loss = sum(TR.own_losses(a, b, t2, th, mm)); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    out['budget_probe'] = {'arm': 'S2' if kind == 'sub' else 'O_G', 'seconds_per_step': (time.perf_counter() - t0) / 100,
                           'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
                           'projected_train_minutes': (time.perf_counter() - t0) / 100 * C.P['stage_C']['training']['steps'] / 60}
    out['pass'] = bool(out['readout_reproduction']['pass'] and out['parent_child']['pass'] and out['operator']['pass'])
    C.jd(out, root / 'stageC' / 'precheck.json')
    print('PRECHECK', json.dumps({k: v.get('pass') if isinstance(v, dict) else v for k, v in out.items()}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
