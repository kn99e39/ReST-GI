"""WL86 ECRO v0 runtime: geometry state (full / selective), source encoding, learned response application, LOW + lookup rendering.

  python ecro86.py render <root> <cfg> <model_tag|oracle:<field_tag>> <variant> <seed> <spp> <tag> [--ls=] [--plain] [--cam=2]
      variant: ecro | stale | nog (no global term) | nos (no single-flight term) | oracle
  python ecro86.py contracts <root> <model_tag>
Runtime inputs: current geometry, lights, frozen ER states, frozen learned parameters. No reference, target or test-image data.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt86 as PT  # noqa: E402
import model86 as M  # noqa: E402
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
NB, K = S.NB, S.K
dev = torch.device('cuda')


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def load_model(root, tag):
    m = M.ECRO().to(dev)
    m.load_state_dict(torch.load(os.path.join(root, 'results', 'models', f'{tag}.pt'), map_location=dev))
    return m.eval()


class State:
    """Geometry-dependent runtime state: patch tables + state rays (+ the response factors derived from them)."""

    def __init__(self, G, hit, dbin, seg):
        self.G, self.hit, self.dbin, self.seg = G, hit, dbin, seg

    def digest(self):
        import hashlib
        return hashlib.sha256(self.hit.tobytes() + self.dbin.tobytes()).hexdigest()[:16]


def build_state(ctx):
    hit, bh, seg = ctx.state_rays()
    return State(ctx.G, hit, bh, seg)


def _seg_obb_t(o, e, obb):
    c, ax, h = (torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev) for v in obb)
    lo = (o - c) @ ax.T; d = (e - c) @ ax.T - lo
    d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t0 = (-h - lo) / d; t1 = (h - lo) / d
    tmin = torch.minimum(t0, t1).amax(1); tmax = torch.maximum(t0, t1).amin(1)
    return (tmax >= tmin.clamp(min=0)) & (tmin <= 1)


def dirty_rays(st, G_old, G_new, moved):
    """Ray ids whose record can change: origin on a moved box, or segment through the old or new volume of a moved box (1 mm pad).
    Evaluated on the GPU (the numpy version cost more than re-tracing every ray)."""
    seg = torch.as_tensor(st.seg, device=dev)
    o, e = seg[:, :3], seg[:, 3:]
    d = torch.zeros(seg.shape[0], dtype=torch.bool, device=dev)
    for b in moved:
        d |= _seg_obb_t(o, e, G_old.obb(b)) | _seg_obb_t(o, e, G_new.obb(b))
        sid = S.SID_BOXA if b == 'A' else S.SID_BOXB
        d |= torch.as_tensor(G_new.PSID == sid, device=dev).repeat_interleave(K)
    return torch.nonzero(d)[:, 0].cpu().numpy()


def update_state(ctx, st, moved):
    """Selective update after ctx.set_geom(new geometry): re-trace only the dirty rays."""
    idx = dirty_rays(st, st.G, ctx.G, moved)
    if len(idx) == 0:
        return State(ctx.G, st.hit, st.dbin, st.seg), 0
    hit, bh, seg = ctx.state_rays(idx)
    h2, b2, s2 = st.hit.copy(), st.dbin.copy(), st.seg.copy()
    h2[idx] = hit; b2[idx] = bh; s2[idx] = seg
    return State(ctx.G, h2, b2, s2), len(idx)


@torch.no_grad()
def respond(model, st, s, use_single=True, use_global=True, factors=None):
    """-> (B numpy [N, NB, 3], factors)."""
    gs = M.GeoState(st.G, st.hit, st.dbin, dev)
    f = factors if factors is not None else model.factors(gs)
    B = model.apply(f, gs, torch.as_tensor(s, device=dev).view(1, st.G.N, NB, 3), use_single=use_single, use_global=use_global)[0]
    return B.cpu().numpy(), f


def render_cfg(root, cfg, model_tag, variant, seed, spp, ls=None, plain=False, cam=None, res=None):
    geom, ls0 = PT.config(cfg)
    ls = ls or ls0
    lights = FX['light_sets'][ls]
    cam = cam or FX['camera']; res = res or cam['res']
    T_ = {}
    ctx = PT.Ctx(geom)
    lt = PT.Lights(lights)
    t0 = time.perf_counter()
    if variant == 'oracle':
        z = np.load(os.path.join(root, 'results', 'fields', f'{model_tag.split(":")[1]}.npz'))
        B = 0.5 * (z['fa'] + z['fb'])
    else:
        model = load_model(root, model_tag)
        s = ctx.source(lt)
        if variant == 'stale':
            e = cfg.split('_')[0]
            c0 = PT.Ctx(FX[e]['T0'])
            st = build_state(c0)                      # T0 response state (rays, patch tables); only the source field is current
        else:
            st = build_state(ctx)
        B, _ = respond(model, st, s, use_single=variant != 'nos', use_global=variant != 'nog')
    T_['prepare_s'] = time.perf_counter() - t0
    acc, Tr = ctx.render(lt, 'ER', cam, res, spp, seed, 'ecro', B=PT.fl(B), plain=plain, b=128 if plain else 32)
    T_.update(Tr)
    return PT.pack(acc, res, spp), B, T_


def contracts(root, tag):
    """Synthetic operator contracts on the trained operator (E1_T1 geometry unless stated)."""
    out = {}
    model = load_model(root, tag)
    ctx = PT.Ctx(FX['E1']['T1'])
    st = build_state(ctx)
    L1 = FX['light_sets']['L1']; LM = FX['light_sets']['LMIX']
    s1 = ctx.source(PT.Lights(L1))
    B1, f = respond(model, st, s1)
    # zero input, non-negativity, determinism
    B0, _ = respond(model, st, np.zeros_like(s1), factors=f)
    out['zero_input_max_abs'] = float(np.abs(B0).max())
    out['min_response'] = float(B1.min())
    B1b, _ = respond(model, st, s1)
    out['determinism_max_abs_diff'] = float(np.abs(B1 - B1b).max())
    # superposition and scaling
    sa = ctx.source(PT.Lights([LM[0]])); sb = ctx.source(PT.Lights([LM[1]])); sab = ctx.source(PT.Lights(LM))
    Ba, _ = respond(model, st, sa, factors=f); Bb, _ = respond(model, st, sb, factors=f); Bab, _ = respond(model, st, sab, factors=f)
    out['source_encoding_superposition_rel'] = float(np.abs(sab - sa - sb).max() / sab.max())
    out['superposition_rel'] = float(np.abs(Bab - Ba - Bb).max() / Bab.max())
    B3, _ = respond(model, st, 3.0 * s1, factors=f)
    out['scaling_rel'] = float(np.abs(B3 - 3.0 * B1).max() / B3.max())
    # directional distinction: two narrow spots aimed at the panel centre from different directions, equal panel-centre irradiance
    pc = np.array(FX['panel']['center'])
    spots = []
    for pos in ([0.5, -1.0, 2.5], [0.8, 1.2, 0.6]):
        d2 = float(((np.array(pos) - pc) ** 2).sum()); cs = (pos[0] - pc[0]) / d2 ** 0.5
        spots.append([{'pos': pos, 'rgb': [5.0 * d2 / cs] * 3, 'aim': pc.tolist(), 'cone_deg': 12.0}])
    Bt, Bm, ss = [], [], []
    for sp in spots:
        lt = PT.Lights(sp)
        fa, _ = ctx.field(lt, 512, 86700 + len(Bt)); Bt.append(fa)
        ss.append(ctx.source(lt)); Bm.append(respond(model, st, ss[-1], factors=f)[0])
    rl = lambda a, b: float(np.abs(a - b).sum() / b.sum())
    out['directional'] = {'encoded_irradiance_ratio': float(ss[0].sum() / ss[1].sum()),
                          'truth_distance': rl(Bt[0], Bt[1]), 'model_distance': rl(Bm[0], Bm[1]),
                          'matched_error': [rl(Bm[0], Bt[0]), rl(Bm[1], Bt[1])], 'swapped_error': [rl(Bm[1], Bt[0]), rl(Bm[0], Bt[1])]}
    # energy: outgoing higher-order flux of the model vs the oracle field (same binning), E1_T1 / L1
    fo, _ = ctx.field(PT.Lights(L1), 512, 86710)
    w = (ctx.G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None] * S.DOMEGA)[..., None]      # area x cos x solid angle
    out['flux_ratio_model_over_oracle_rgb'] = ((B1 * w).sum((0, 1)) / (fo * w).sum((0, 1))).tolist()
    out['field_rel_l1_E1_T1'] = rl(B1, fo)
    # state lifecycle: selective update == full rebuild; move-and-return; null
    c2 = PT.Ctx(FX['E1']['T0'])
    st0 = build_state(c2)
    c2.set_geom(FX['E1']['T1'])
    st_sel, nd = update_state(c2, st0, FX['E1']['moved'])
    st_full = build_state(c2)
    out['selective_equals_full'] = bool(np.array_equal(st_sel.hit, st_full.hit) and np.array_equal(st_sel.dbin, st_full.dbin))
    out['selective_equals_fresh_scene'] = bool(np.array_equal(st_sel.hit, st.hit) and np.array_equal(st_sel.dbin, st.dbin))
    out['dirty_ray_fraction'] = nd / st0.hit.shape[0]
    c2.set_geom(FX['E1']['T0'])
    st_ret, _ = update_state(c2, st_sel, FX['E1']['moved'])
    out['move_and_return_state_identical'] = bool(np.array_equal(st_ret.hit, st0.hit) and np.array_equal(st_ret.dbin, st0.dbin))
    st_null, n0 = update_state(c2, st_ret, [])
    out['null_edit_dirty_rays'] = int(n0)
    Bs, _ = respond(model, st_sel, s1); Bf, _ = respond(model, st_full, s1)
    out['selective_vs_full_response_max_abs'] = float(np.abs(Bs - Bf).max())
    d = out['directional']
    out['PASS'] = bool(out['zero_input_max_abs'] == 0.0 and out['min_response'] >= 0.0 and out['determinism_max_abs_diff'] <= 1e-6 * B1.max()
                       and out['superposition_rel'] <= 1e-4 and out['scaling_rel'] <= 1e-4 and out['selective_equals_full']
                       and out['move_and_return_state_identical'] and out['null_edit_dirty_rays'] == 0 and out['selective_vs_full_response_max_abs'] == 0.0
                       and d['model_distance'] >= 0.5 * d['truth_distance'] and max(d['matched_error']) < min(d['swapped_error']))
    os.makedirs(os.path.join(root, 'results', 'contracts'), exist_ok=True)
    json.dump(out, open(os.path.join(root, 'results', 'contracts', f'{tag}.json'), 'w'), indent=1)
    print('CONTRACTS', 'PASS' if out['PASS'] else 'FAIL', json.dumps(out), flush=True)


def main():
    cmd, root = sys.argv[1], sys.argv[2]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if a.startswith('--') and '=' in a}
    args = [a for a in sys.argv[3:] if not a.startswith('--')]
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    if cmd == 'render':
        cfg, mtag, variant, seed, spp, tag = args[0], args[1], args[2], int(args[3]), int(args[4]), args[5]
        o, B, T_ = render_cfg(root, cfg, mtag, variant, seed, spp, ls=opt.get('--ls'), plain='--plain' in sys.argv,
                              cam=FX['camera2'] if opt.get('--cam') == '2' else None)
        out = os.path.join(root, 'results', 'renders'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), B=B.astype(np.float32), **o)
        json.dump({'cfg': cfg, 'model': mtag, 'variant': variant, 'seed': seed, 'spp': spp, 'timing': T_, 'gpu_mem_mib': PT.gpu_mem()},
                  open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
        print(tag, json.dumps(T_), flush=True)
    elif cmd == 'contracts':
        contracts(root, args[0])


if __name__ == '__main__':
    main()
    os._exit(0)
