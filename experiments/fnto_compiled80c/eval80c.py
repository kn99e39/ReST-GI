"""WL80C Stage 2 evaluation: representation (oracle-P) and learned-rule (compiler
seeds) metrics on fresh directions, highlight geometry, and the frozen gates.

  python eval80c.py <data_dir> <fit_dir> <out_dir> [--dev]
"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import closure80c as C
import ref80c as R
from data80c import normalise, P

dev_t = torch.device('cuda')
A = P['angular']
TH = np.array(A['eval_theta_deg'])


def compiled_params(arm, setname, mats, fit_dir):
    if arm.startswith('s'):
        T = P['training']
        net = C.Compiler(T['width'], T['depth']).to(dev_t)
        net.load_state_dict(torch.load(os.path.join(fit_dir, f'compiler_{arm}.pt'), map_location=dev_t)['state_dict'])
        with torch.no_grad():
            return C.decode(net(torch.from_numpy(normalise(mats[setname])).to(dev_t)))
    z = np.load(os.path.join(fit_dir, f'oracle_{setname}.npz'))
    if arm == 'oracleP':
        return C.decode(torch.from_numpy(z['zP']).to(dev_t), 'passive')
    return C.decode(torch.from_numpy(z['zU']).to(dev_t), 'nonneg')


def lane_metrics(p, ev):
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    G = nm * nth * nph
    s_abs = torch.zeros(G, dtype=torch.float64, device=dev_t)
    s_ref = torch.zeros(G, dtype=torch.float64, device=dev_t)
    a_ref = torch.zeros(G, 3, dtype=torch.float64, device=dev_t)
    a_cl = torch.zeros(G, 3, dtype=torch.float64, device=dev_t)
    n = len(ev['m'])
    for s in range(0, n, 1 << 21):
        e = slice(s, min(n, s + (1 << 21)))
        m = torch.from_numpy(ev['m'][e]).to(dev_t).long()
        wi = torch.from_numpy(ev['wi'][e]).to(dev_t); wo = torch.from_numpy(ev['wo'][e]).to(dev_t)
        f = torch.from_numpy(ev['f'][e]).to(dev_t).double()
        w = (torch.from_numpy(ev['use'][e]).to(dev_t).double() / torch.from_numpy(ev['q'][e]).to(dev_t).double())[:, None]
        with torch.no_grad():
            pr = C.eval_closure(C.gather(p, m), wi, wo).double()
        g = torch.arange(s, e.stop, device=dev_t) // nwo
        s_abs.index_add_(0, g, ((pr - f).abs() * w).sum(1))
        s_ref.index_add_(0, g, (f * w).sum(1))
        a_ref.index_add_(0, g, f * w); a_cl.index_add_(0, g, pr * w)
    el1 = (s_abs / s_ref.clamp_min(1e-12)).view(nm, nth, nph)
    ealb = ((a_cl - a_ref).abs().sum(1) / a_ref.sum(1).clamp_min(1e-12)).view(nm, nth, nph)
    alb_ref = (a_ref / nwo).view(nm, nth, nph, 3)
    return el1.cpu().numpy(), ealb.cpu().numpy(), alb_ref.cpu().numpy()


def q(x, v):
    return float(np.quantile(x, v)) if len(x) else float('nan')


def summarise(el1, ealb):
    out = {'E_L1': {'median': q(el1.ravel(), .5), 'p90': q(el1.ravel(), .9), 'p99': q(el1.ravel(), .99),
                    'max': float(el1.max())},
           'E_alb': {'median': q(ealb.ravel(), .5), 'p95': q(ealb.ravel(), .95), 'max': float(ealb.max())},
           'E_L1_by_theta_median': {int(t): q(el1[:, i].ravel(), .5) for i, t in enumerate(TH)},
           'E_L1_by_theta_p90': {int(t): q(el1[:, i].ravel(), .9) for i, t in enumerate(TH)}}
    gi = [i for i, t in enumerate(TH) if t >= A['grazing_min_deg']]
    bi = [i for i, t in enumerate(TH) if A['heldout_incident_band_deg'][0] <= t <= A['heldout_incident_band_deg'][1]]
    ni = [i for i, t in enumerate(TH) if t in A['band_neighbour_theta_deg']]
    out['grazing_median'] = q(el1[:, gi].ravel(), .5)
    out['band_median'] = q(el1[:, bi].ravel(), .5)
    out['band_neighbour_median'] = q(el1[:, ni].ravel(), .5)
    mat_mean = el1.mean((1, 2))
    out['worst_material_E_L1_mean'] = float(mat_mean.max())
    out['worst10_materials'] = np.argsort(-mat_mean)[:10].tolist()
    return out


def descriptor_strata(el1, d):
    """Median E_L1 per descriptor-tercile (worst strata report)."""
    names = ['base_r', 'base_g', 'base_b', 'metallic', 'roughness', 'anisotropic', 'clearcoat', 'clearcoat_gloss', 'sheen']
    mm = el1.mean((1, 2))
    res = {}
    for j, nme in enumerate(names):
        qs = np.quantile(d[:, j], [1 / 3, 2 / 3])
        b = np.digitize(d[:, j], qs)
        res[nme] = [q(mm[b == k], .5) for k in range(3)]
    return res


# ---------------- highlight geometry ----------------
def principled_alphas(d):
    aspect = np.sqrt(1 - 0.9 * d[:, 5])
    r2 = d[:, 4] ** 2
    return np.maximum(1e-4, r2 / aspect), np.maximum(1e-4, r2 * aspect)


def grid_dirs(wi, smax, n):
    s = np.linspace(-1, 1, n)
    sx, sy = np.meshgrid(s, s, indexing='ij')
    sx = sx.ravel()[None] * smax[:, None]; sy = sy.ravel()[None] * smax[:, None]
    h = np.stack([-sx, -sy, np.ones_like(sx)], -1)
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    wo = 2 * (h * wi[:, None, :]).sum(-1, keepdims=True) * h - wi[:, None, :]
    return wo, sx, sy


def highlight_stats(v, wo, sx, sy, n):
    """v [G, n*n] luminance values (invalid -> -1). Returns peak dir, half-max region stats."""
    G = v.shape[0]
    pk = v.argmax(1)
    vp = v[np.arange(G), pk]
    H = v >= 0.5 * vp[:, None]
    Hg = H.reshape(G, n, n)
    touches = Hg[:, 0, :].any(1) | Hg[:, -1, :].any(1) | Hg[:, :, 0].any(1) | Hg[:, :, -1].any(1)
    area = H.sum(1)
    mx = (sx * H).sum(1) / area; my = (sy * H).sum(1) / area
    vx = ((sx - mx[:, None]) ** 2 * H).sum(1) / area; vy = ((sy - my[:, None]) ** 2 * H).sum(1) / area
    return {'peak_wo': wo[np.arange(G), pk], 'vpeak': vp, 'area': area, 'vx': vx, 'vy': vy, 'touch': touches}


def highlight_eval(arms_p, setname, d, ev, rng):
    hl = A['highlight']
    n = hl['grid']
    ths = np.radians(np.array(hl['theta_deg']))
    nm = len(d)
    phis = ev['phis'][:, 0]
    bsdf, nb = R.textured_principled(d)
    ax, ay = principled_alphas(d)
    acc = {a: {'peak_err': [], 'logw': [], 'logwx': [], 'logwy': [], 'pron': [], 'res': []} for a in arms_p}
    chunk = max(1, (1 << 22) // (n * n))
    pairs = [(k, t) for k in range(nm) for t in ths]
    for c0 in range(0, len(pairs), chunk):
        pc = pairs[c0:c0 + chunk]
        mk = np.array([k for k, _ in pc]); tk = np.array([t for _, t in pc])
        wi = R.sph(tk, phis[mk])
        smax = np.clip(hl['extent_alpha_mult'] * np.maximum(ax[mk], ay[mk]), hl['extent_min'], hl['extent_max'])
        wo, sx, sy = grid_dirs(wi, smax, n)
        G = len(pc)
        ok = wo[..., 2] > 1e-4
        wo_c = wo.copy(); wo_c[..., 2] = np.maximum(wo_c[..., 2], 1e-4)
        wo_c /= np.linalg.norm(wo_c, axis=-1, keepdims=True)
        mm = np.repeat(mk, n * n)
        wiL = np.repeat(wi, n * n, 0)
        fr, _ = R.ref_eval_pdf(bsdf, nb, mm, wiL, wo_c.reshape(-1, 3))
        vr = np.where(ok, fr.mean(1).reshape(G, -1), -1.0)
        # reference albedo for the 'pronounced' criterion
        nsa = hl['albedo_samples']
        mA = np.repeat(mk, nsa); wiA = np.repeat(wi, nsa, 0)
        woA, qA, fA, useA = R.mixture_dirs(bsdf, nb, mA, wiA, rng)
        aref = (np.where(useA, fA.mean(1) / qA, 0).reshape(G, nsa)).mean(1)
        sr = highlight_stats(vr, wo, sx, sy, n)
        cos_pk = np.clip(sr['peak_wo'][:, 2], 0, 1)
        pron = sr['vpeak'] >= hl['pronounced_factor'] * aref * cos_pk / math.pi
        for a, p in arms_p.items():
            with torch.no_grad():
                pr = C.eval_closure(C.gather(p, torch.from_numpy(mm).to(dev_t)),
                                    torch.from_numpy(wiL.astype(np.float32)).to(dev_t),
                                    torch.from_numpy(wo_c.reshape(-1, 3).astype(np.float32)).to(dev_t))
            vc = np.where(ok, pr.mean(1).cpu().numpy().reshape(G, -1), -1.0)
            sc = highlight_stats(vc, wo, sx, sy, n)
            dots = np.clip((sr['peak_wo'] * sc['peak_wo']).sum(1), -1, 1)
            acc[a]['peak_err'].append(np.degrees(np.arccos(dots)))
            acc[a]['logw'].append(0.5 * np.log(sc['area'] / sr['area']))
            acc[a]['logwx'].append(0.5 * np.log(np.maximum(sc['vx'], 1e-20) / np.maximum(sr['vx'], 1e-20)))
            acc[a]['logwy'].append(0.5 * np.log(np.maximum(sc['vy'], 1e-20) / np.maximum(sr['vy'], 1e-20)))
            acc[a]['pron'].append(pron)
            acc[a]['res'].append(~(sr['touch'] | sc['touch']))
    out = {}
    for a in arms_p:
        z = {k: np.concatenate(v) for k, v in acc[a].items()}
        sel = z['pron'] & z['res']
        out[a] = {'n_cases': int(len(z['pron'])), 'n_pronounced': int(z['pron'].sum()),
                  'n_pronounced_resolved': int(sel.sum()),
                  'peak_err_deg_median': q(z['peak_err'][z['pron']], .5),
                  'peak_err_deg_p90': q(z['peak_err'][z['pron']], .9),
                  'abs_logw_median': q(np.abs(z['logw'][sel]), .5),
                  'abs_logw_p90': q(np.abs(z['logw'][sel]), .9),
                  'logw_median_signed': q(z['logw'][sel], .5),
                  'abs_logwx_median': q(np.abs(z['logwx'][sel]), .5),
                  'abs_logwy_median': q(np.abs(z['logwy'][sel]), .5)}
    return out


def gates(S, H):
    G = P['gates']
    v = {}
    # representation (oracle-P on test_iid)
    o = S['oracleP']['test_iid']; oh = H['test_iid']['oracleP']
    g = G['representation']
    r = {'R1': o['E_L1']['median'] <= g['E_L1_median'] and o['E_L1']['p90'] <= g['E_L1_p90'],
         'R2': o['E_alb']['median'] <= g['E_alb_median'] and o['E_alb']['p95'] <= g['E_alb_p95'],
         'R3': oh['peak_err_deg_median'] <= g['peak_err_deg_median'] and oh['abs_logw_median'] <= math.log(g['width_ratio_max'])}
    r['clear_fail'] = o['E_L1']['median'] > g['clear_fail_E_L1_median'] or o['E_alb']['median'] > g['clear_fail_E_alb_median']
    r['PASS'] = r['R1'] and r['R2'] and r['R3']
    v['REPRESENTATION'] = r
    g = G['learned_rule']
    lr = {}
    for s in P['seeds']['train']:
        a = f's{s}'
        ti, tb, tr = S[a]['test_iid'], S[a]['test_block'], S[a]['train_eval']
        hh = H['test_iid'][a]
        x = {'L1': ti['E_L1']['median'] <= g['iid_E_L1_median'] and
                   ti['E_L1']['median'] <= max(g['iid_vs_oracle_ratio'] * o['E_L1']['median'], o['E_L1']['median'] + g['iid_vs_oracle_abs'])}
        x['L2'] = tb['E_L1']['median'] <= g['block_E_L1_median'] and tb['E_L1']['p90'] <= g['block_E_L1_p90']
        x['L3'] = ti['E_L1']['median'] <= g['gen_gap_ratio'] * tr['E_L1']['median']
        x['L4'] = ti['E_alb']['median'] <= g['iid_E_alb_median'] and tb['E_alb']['median'] <= g['block_E_alb_median']
        x['L5'] = hh['peak_err_deg_median'] <= g['peak_err_deg_median'] and hh['abs_logw_median'] <= math.log(g['width_ratio_max'])
        x['L6'] = ti['band_median'] <= g['band_vs_neighbour_ratio'] * ti['band_neighbour_median'] and \
                  ti['grazing_median'] <= g['grazing_vs_all_ratio'] * ti['E_L1']['median']
        x['PASS'] = all(x[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
        lr[a] = x
    lr['PASS'] = all(lr[f's{s}']['PASS'] for s in P['seeds']['train'])
    v['LEARNED_RULE'] = lr
    return v


def main():
    data, fit_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    mats = np.load(os.path.join(data, 'materials.npz'))
    rng = np.random.default_rng(P['seeds']['eval'])
    arms = [f's{s}' for s in P['seeds']['train']] + ['oracleP', 'oracleU']
    S, H, strata = {}, {}, {}
    for setname in ['test_iid', 'test_block', 'train_eval']:
        ev = dict(np.load(os.path.join(data, f'eval_{setname}.npz')))
        dset = mats['train'][:len(ev['phis'])] if setname == 'train_eval' else mats[setname]
        ps = {}
        for a in arms:
            if setname == 'train_eval' and a.startswith('oracle'):
                continue
            p = compiled_params(a, setname, mats, fit_dir) if setname != 'train_eval' else \
                compiled_params(a, 'train', {'train': dset}, fit_dir)
            ps[a] = p
            el1, ealb, alb_ref = lane_metrics(p, ev)
            S.setdefault(a, {})[setname] = summarise(el1, ealb)
            strata.setdefault(a, {})[setname] = descriptor_strata(el1, dset)
            np.savez_compressed(os.path.join(out, f'lane_{a}_{setname}.npz'), el1=el1, ealb=ealb)
            if a == arms[0]:
                np.savez_compressed(os.path.join(out, f'albedo_ref_{setname}.npz'), alb_ref=alb_ref)
            print(setname, a, json.dumps(S[a][setname]['E_L1']), flush=True)
        if setname != 'train_eval':
            H[setname] = highlight_eval(ps, setname, dset, ev, rng)
            print(setname, 'highlight', json.dumps(H[setname]), flush=True)
    V = gates(S, H)
    # compiled-state / model sizes
    T = P['training']
    nparam = sum(p.numel() for p in C.Compiler(T['width'], T['depth']).parameters())
    size = {'compiler_params': nparam, 'compiler_bytes_fp32': 4 * nparam,
            'compiled_state_independent_floats': 45, 'compiled_state_floats_incl_probs': 53,
            'runtime_table_floats': 88, 'reference_param_floats': 13}
    json.dump({'summary': S, 'highlight': H, 'strata': strata, 'verdict': V, 'sizes': size,
               'seconds': time.time() - t0}, open(os.path.join(out, 'stage2.json'), 'w'), indent=1)
    print(json.dumps(V, indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
