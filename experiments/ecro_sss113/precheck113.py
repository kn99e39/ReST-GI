"""WL113 Stage B validation before training.

  python precheck113.py ROOT   -> ROOT/stageC/precheck.json (and, for a provisional dense S4 selection, the budget verdict)
(1) parent / child aggregation of the 4x4 targets; (2) compact indexing: compact readout table == dense expansion and the
compact (coarse + refined-slot) storage reproduces it; (3) readout consistency: the candidate readout of the physical Stage A
table reproduces the stored Stage A image; (4) candidate operator zero / superposition / non-negativity; (5) coverage (Stage A);
(6) camera independence and (7) support recomputation: identical tables / supports for repeated construction at one state;
(8) 100-step training probe (time, peak memory) -> dense budget (training <= 90 min, peak <= 13 GB, table <= 16 MiB).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402


def main(root):
    root = Path(root); out = {}
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    import tracer113 as T
    import eval113 as E
    PT = T.PT
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    ctx = T.Ctx113(geom, 'C')
    acc = E.load_acc(root / 'stageA' / 'F0_r0.npz')
    Tc, T2, T4, nat = T.tables4(acc)
    w = acc['U_w'].reshape(1270, 16, 64)
    rec = (T4 * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-30)[..., None]
    out['parent_child_max_rel'] = float(np.abs(rec - Tc)[nat['c']].max() / np.abs(Tc).max())
    desc = ctx.vhs(); sup = T.support_from_desc(desc, ctx.G)
    dense = T.to_dense16(T4, 'compact', sup, Tc)
    slots = -np.ones(1270, int); slots[sup] = np.arange(sup.sum())
    compact_store = (Tc, T4[sup])
    rebuilt = np.repeat(compact_store[0][:, None], 16, 1); rebuilt[sup] = compact_store[1][slots[sup]]
    out['compact_vs_dense_max_abs'] = float(np.abs(rebuilt - dense).max())
    out['compact_floats'] = int(Tc.size + T4[sup].size); out['dense_floats'] = int(T4.size); out['coarse_floats'] = int(Tc.size)
    name = 'L4_r0' if dec['selected'] == 'L4' else 'S4_r0'
    tab = T4 if name == 'S4_r0' else dense
    img, _, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, E.SPP, 11310000 + C.SO, 'ecro',
                                tables=[(name, PT.fl(np.ascontiguousarray(tab, np.float32).ravel()), 'sub')], b=E.LAUNCH)
    with np.load(root / 'stageA' / 'stream_F0_V0_0.npz') as z:
        st = z[f'hl|{name}'].astype(np.float64)
    out['readout_reproduction_rel_l1'] = float(np.abs(img[name] - st).sum() / np.abs(st).sum())
    desc2 = T.Ctx113(geom, 'C').vhs()
    out['support_recompute_identical'] = bool(np.array_equal(desc, desc2))
    import torch
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    hit, bn, _ = ctx.state_rays()
    gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    use = sup if dec['selected'] == 'L4' else np.ones(1270, bool)
    gs.sup = torch.as_tensor(np.flatnonzero(use), device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
    s1 = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
    s2 = torch.as_tensor(ctx.source(PT.Lights(C.C112.wl100_test(1)['lights'][0])), device='cuda').unsqueeze(0)
    torch.manual_seed(0)
    model = M3.ECROL4().cuda(); model.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    with torch.no_grad():
        f = model.factors(gs)
        z0 = model.outputs(f, gs, torch.zeros_like(s1)); b1 = model.outputs(f, gs, s1); b2 = model.outputs(f, gs, s2); b12 = model.outputs(f, gs, s1 + s2)
    out['operator'] = {'zero_max': float(max(z0[0].abs().max(), z0[1].abs().max())),
                       'superposition_rel': float(((b12[1] - b1[1] - b2[1]).abs().sum() / b12[1].abs().sum())),
                       'min_value': float(min(b1[1].min(), b2[1].min(), b1[0].min())), 'params': int(sum(x.numel() for x in model.parameters())),
                       'params_original': int(sum(x.numel() for x in M.ECRO().parameters()))}
    # camera independence: the table is computed without any camera; two constructions -> identical
    with torch.no_grad():
        again = model.outputs(model.factors(gs), gs, s1)
    out['table_repeat_identical'] = bool(torch.equal(again[1], b1[1]))
    met = json.loads((root / 'analysis' / 'stageA_metrics.json').read_text())
    arm = 'L4' if dec['selected'] == 'L4' else 'S4'
    out['coverage_min'] = float(min(min(v['arms'][arm]['coverage'].values()) for v in met.values()))
    # budget probe on real items
    import train113 as TR
    files = sorted((root / 'stageC' / 'data' / 'train').glob('g0000*.npz'))[:2]
    items = [TR.Item(f, True) for f in files if '.tmp' not in f.name]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(100):
        g_, s_, c, sb = items[i % len(items)].tensors()
        a, b, sa, sbb = TR.forward(model, 'CAND', g_, s_)
        loss = sum(TR.own(a, b, *c)) + sum(TR.own(sa, sbb, *sb)); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); sps = (time.perf_counter() - t0) / 100
    out['budget_probe'] = {'seconds_per_step': sps, 'projected_train_minutes': sps * 24000 / 60, 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
                           'table_mib': out['compact_floats' if arm == 'L4' else 'dense_floats'] * 4 / 2 ** 20}
    budget_ok = out['budget_probe']['projected_train_minutes'] <= 90 and out['budget_probe']['peak_mib'] <= 13 * 1024 and out['budget_probe']['table_mib'] <= 16
    out['checks_pass'] = bool(out['parent_child_max_rel'] <= 1e-6 and out['compact_vs_dense_max_abs'] == 0 and out['readout_reproduction_rel_l1'] <= 1e-6
                              and out['support_recompute_identical'] and out['operator']['zero_max'] == 0 and out['operator']['superposition_rel'] <= 1e-5
                              and out['operator']['min_value'] >= 0 and out['table_repeat_identical'])
    out['budget_ok'] = bool(budget_ok)
    out['pass'] = bool(out['checks_pass'] and budget_ok)
    C.jd(out, root / 'stageC' / 'precheck.json')
    print('PRECHECK', json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
