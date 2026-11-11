"""WL86A Stages A / B: baseline verification and error localisation of the frozen WL86 ECRO v0 (nothing is trained or changed).

  python diag86a.py baseline <root86> <out>      Stage A: recompute the WL86 Stage 1 / 3 numbers from the stored renders with independent code
  python diag86a.py localize <root86> <out>      Stage B: field-level and rendered error of the frozen tables, importance decomposition
Stage B identity (field -> render): for a region R the camera reads sum_{pixels in R} hl = sum_{patch, bin} W_R[patch, bin] . B[patch, bin],
with W_R the accumulated path throughput of the LOW bounce x1 -> x2 (identical camera paths for every table). So the signed rendered
error of a table is sum W_R . (B_hat - B_oracle), which decomposes exactly over sender patches, outgoing bins and materials.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
SEEDS = ['ecro_s8601', 'ecro_s8602']
REGS = ['stationary_all', 'A', 'C', 'D', 'E', 'obj_floor', 'obj_ceiling', 'obj_walls', 'obj_panel', 'obj_sphere', 'obj_boxes_static']
SHAPES = ['floor', 'ceiling', 'wall_back', 'wall_left', 'wall_right', 'panel', 'sphere', 'boxA', 'boxB']


def load(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    assert fns, pat
    d = {}
    for fn in fns:
        z = np.load(fn)
        for k in z.files:
            d.setdefault(k, []).append(z[k].astype(np.float64))
    d = {k: np.stack(v) for k, v in d.items()}
    if 'tot' not in d:
        d['tot'] = d['e1'] + d['e2'] + (d['h'] if 'h' in d else d['hl'])
    return d


def masks(root, e):
    z = np.load(os.path.join(root, 'results', 'masks', f'{e}.npz'))
    return {k: z[k] for k in z.files if z[k].dtype == bool}


def baseline(root, out):
    """Independent recomputation (per-pixel loops over regions, no shared code with analyze86.py) and comparison with final.json / stage1.json."""
    fj = json.load(open(os.path.join(root, 'results', 'final.json'))); sj = json.load(open(os.path.join(root, 'results', 'stage1.json')))
    res = {'max_abs_diff_vs_wl86': 0.0, 'table': {}}
    for e in ('E1', 'E2'):
        M = masks(root, e)
        for t in ('T0', 'T1'):
            ref = load(root, f'REF_ER_{e}_{t}')
            n = ref['h'].shape[0]
            h = ref['h'].mean(0); tot = ref['tot'].mean(0); ha = ref['h'][:n // 2].mean(0); hb = ref['h'][n // 2:].mean(0)
            arms = {'oracle': load(root, f'ORA_{e}_{t}')}
            for sd in SEEDS:
                arms[sd] = load(root, f'EC_{sd}_ecro_{e}_{t}')
            for k in REGS:
                m = M[k]
                if m.sum() < 30:
                    continue
                den = tot[m].sum()
                row = {'pixels': int(m.sum()), 'H_share': float(h[m].sum() / den), 'ref_h_noise': float(0.5 * np.abs(ha[m] - hb[m]).sum() / den)}
                for name, a in arms.items():
                    hl = a['hl'].mean(0)
                    eh = float(np.abs(hl[m] - h[m]).sum() / den)
                    row[name] = {'E_H': eh, 'signed': float((hl[m].sum() - h[m].sum()) / den), 'hl_over_h': float(hl[m].sum() / h[m].sum()),
                                 'hl_noise': float(0.5 * np.abs(a['hl'][0][m] - a['hl'][1][m]).sum() / den),
                                 'excess_over_oracle': None}
                    w86 = (sj[e]['oracle'][t].get(k, {}).get('E_H') if name == 'oracle' else fj['seeds'][name]['fidelity'][e][t].get(k, {}).get('E_H'))
                    if w86 is not None:
                        res['max_abs_diff_vs_wl86'] = max(res['max_abs_diff_vs_wl86'], abs(w86 - eh))
                for sd in SEEDS:
                    row[sd]['excess_over_oracle'] = row[sd]['E_H'] - row['oracle']['E_H']
                    # learned vs oracle directly (same LOW paths are NOT shared between these renders; this is |hl_learned - hl_oracle| / tot)
                    row[sd]['vs_oracle'] = float(np.abs(arms[sd]['hl'].mean(0)[m] - arms['oracle']['hl'].mean(0)[m]).sum() / den)
                res['table'][f'{e}_{t}:{k}'] = row
    res['reproduced'] = bool(res['max_abs_diff_vs_wl86'] <= 1e-9)
    json.dump(res, open(os.path.join(out, 'baseline.json'), 'w'), indent=1)
    print('reproduced', res['reproduced'], res['max_abs_diff_vs_wl86'])
    for key in ('E2_T1:C', 'E2_T0:C', 'E1_T1:C', 'E2_T1:obj_panel', 'E1_T1:obj_panel', 'E2_T1:obj_sphere', 'E2_T1:D', 'E2_T1:stationary_all'):
        r = res['table'][key]
        print(key, 'Hshare', round(r['H_share'], 3), 'ref noise', round(r['ref_h_noise'], 4), {n: (round(r[n]['E_H'], 4), round(r[n]['signed'], 4), round(r[n]['hl_over_h'], 3), round(r[n]['hl_noise'], 4)) for n in ['oracle'] + SEEDS})


def walk(PT, ctx, cam, res, b, seed, launch):
    """The LOW bounce of the ecro estimator (identical random-number use): pixel, table slot read at x2, throughput, validity, x2 shape."""
    mi, dr, CD = PT.mi, PT.dr, PT.CD
    n = res * res; N = n * b
    sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
    rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = PT.gen_rays(cam, res, b, rng)
    si = ctx.scene.ray_intersect(ray)
    act = si.is_valid()
    mid = dr.gather(PT.U, ctx.SMAT, ctx.shape_id(si))
    wi = si.to_local(-ray.d)
    act &= wi.z > 1e-6
    wi = dr.select(act, wi, mi.Vector3f(0, 0, 1))
    cl = CD.DrClosure83('ER', ctx.mats.states, mid)
    u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    wo, pdf, w, valid = cl.sample(wi, u1, u2, act)
    act &= valid
    thr = dr.select(act, w, 0.0)
    r2 = si.spawn_ray(si.to_world(wo))
    si2 = ctx.scene.ray_intersect(r2, act)
    act &= si2.is_valid()
    sid2 = ctx.shape_id(si2)
    act &= si2.to_local(-r2.d).z > 1e-6
    pid, bn = ctx.patch_bin(si2, sid2, -r2.d)
    return pix, pid * PT.NB + bn, thr, act


def importance(PT, ctx, cam, res, spp, seed, mks, b=32):
    """W[mask][slot, 3]: mean (per sample) accumulated throughput of paths from the mask's pixels that read each table slot."""
    mi, dr = PT.mi, PT.dr
    ns = ctx.N * PT.NB
    names = list(mks)
    W = {k: np.zeros((ns, 3)) for k in names}
    mk = {k: PT.fl(mks[k].ravel().astype(np.float32)) for k in names}
    for launch in range(spp // b):
        pix, slot, thr, act = walk(PT, ctx, cam, res, b, seed, launch)
        vals = []
        for k in names:
            g = dr.select(act, dr.gather(PT.F, mk[k], pix), 0.0)
            vals += [thr.x * g, thr.y * g, thr.z * g]
        slot = dr.select(act, slot, 0)
        dr.eval(vals, slot); dr.sync_thread()
        for i, v in enumerate(vals):
            buf = dr.zeros(PT.F, ns)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, slot)
            W[names[i // 3]][:, i % 3] += np.asarray(buf, np.float64)
    return {k: v / spp for k, v in W.items()}


def localize(root, out):
    import pt86 as PT
    S = PT.S; FX = PT.FX; NB = PT.NB
    res = {}
    cam = FX['camera']; R = cam['res']
    ring = np.arange(NB) // S.NPHI
    for cfg in ('E2_T1', 'E2_T0', 'E1_T1'):
        e = cfg.split('_')[0]
        geom, ls = PT.config(cfg)
        ctx = PT.Ctx(geom)
        G = ctx.G
        M = masks(root, e)
        mks = {k: M[k] for k in ('C', 'D', 'obj_panel', 'obj_sphere', 'stationary_all')}
        W = importance(PT, ctx, cam, R, 1024, 86600, mks)                     # seed base of the WL86 ECRO renders (replica 0)
        z = np.load(os.path.join(root, 'results', 'fields', f'F_{cfg}.npz'))
        Bo = 0.5 * (z['fa'] + z['fb']).astype(np.float64); Ba, Bb = z['fa'].astype(np.float64), z['fb'].astype(np.float64)
        ref = load(root, f'REF_ER_{cfg}'); ora = load(root, f'ORA_{cfg}')
        psid = G.PSID; pmat = G.PMAT
        o = {'field': {}, 'render': {}}
        tabs = {'oracle_half_a': Ba}
        for sd in SEEDS:
            tabs[sd] = np.load(os.path.join(root, 'results', 'renders', f'EC_{sd}_ecro_{cfg}_r0.npz'))['B'].astype(np.float64)
        st = np.load(os.path.join(root, 'results', 'renders', f'EC_{SEEDS[0]}_stale_{cfg}_r0.npz'))['B'].astype(np.float64) if cfg.endswith('T1') else None
        if st is not None:
            tabs['stale'] = st
        # ---- field level, by sender shape (unweighted and flux-weighted)
        fw = (G.PAREA[:, None] * ((ring + 0.5) / S.NZ)[None])[..., None]
        for name, B in tabs.items():
            f = {}
            ref_t = Bb if name == 'oracle_half_a' else Bo
            for si_, sn in enumerate(SHAPES):
                p = psid == si_
                f[sn] = {'rel_l1': float(np.abs(B[p] - ref_t[p]).sum() / ref_t[p].sum()), 'flux_ratio': float((B[p] * fw[p]).sum() / (ref_t[p] * fw[p]).sum())}
            f['all'] = {'rel_l1': float(np.abs(B - ref_t).sum() / ref_t.sum()), 'flux_ratio': float((B * fw).sum() / (ref_t * fw).sum())}
            o['field'][name] = f
        # ---- rendered, per region: exact decomposition of the signed error
        for rk, Wr in W.items():
            Wr = Wr.reshape(G.N, NB, 3)
            m = mks[rk]
            tot = ref['tot'].mean(0)[m].sum() / m.size * 1.0      # region sum per image / pixels of the image (W is per sample, summed over pixels)
            tot_sum = ref['tot'].mean(0)[m].sum()
            h_sum = ref['h'].mean(0)[m].sum()
            base = (Wr * Bo).sum()
            r = {'recon_oracle_render_ratio': float(base / ora['hl'].mean(0)[m].sum()), 'recon_h_ratio': float(base / h_sum), 'H_share': float(h_sum / tot_sum),
                 'oracle_noise_signed': float((Wr * (Ba - Bb)).sum() * 0.5 / tot_sum), 'importance_by_shape': {}, 'importance_by_ring': []}
            for si_, sn in enumerate(SHAPES):
                r['importance_by_shape'][sn] = float((Wr[psid == si_] * Bo[psid == si_]).sum() / base)
            r['importance_by_ring'] = [float((Wr[:, ring == q] * Bo[:, ring == q]).sum() / base) for q in range(S.NZ)]
            for name, B in tabs.items():
                if name == 'oracle_half_a':
                    continue
                d = Wr * (B - Bo)                                             # [N, NB, 3] signed contributions
                t = {'signed_error_over_tot': float(d.sum() / tot_sum), 'signed_error_over_h': float(d.sum() / base),
                     'abs_slot_error_over_tot': float(np.abs(d.sum(-1)).sum() / tot_sum),
                     'by_shape': {sn: float(d[psid == si_].sum() / tot_sum) for si_, sn in enumerate(SHAPES)},
                     'by_ring': [float(d[:, ring == q].sum() / tot_sum) for q in range(S.NZ)]}
                # magnitude vs directional split per sender patch: alpha_p = importance-weighted table ratio of the patch
                num = (Wr * B).sum((1, 2)); den = (Wr * Bo).sum((1, 2))
                alpha = np.where(den > 0, num / np.maximum(den, 1e-30), 1.0)
                mag = (Wr * Bo).sum((1, 2)) * (alpha - 1.0)                    # equals the patch's signed contribution (by construction)
                # unweighted patch ratio (does the patch's whole table have the wrong level?) vs redistribution over bins
                beta = B.sum((1, 2)) / np.maximum(Bo.sum((1, 2)), 1e-30)
                lvl = ((Wr * Bo).sum((1, 2)) * (beta - 1.0)).sum()
                t['level_part_over_tot'] = float(lvl / tot_sum); t['directional_part_over_tot'] = float((d.sum() - lvl) / tot_sum)
                pc = d.sum((1, 2))
                top = np.argsort(-np.abs(pc))[:12]
                t['top_patches'] = [{'pid': int(p), 'shape': SHAPES[psid[p]], 'centroid': [round(float(v), 2) for v in G.PC[p]], 'contrib_over_tot': float(pc[p] / tot_sum),
                                     'alpha_weighted': float(alpha[p]), 'beta_patch_level': float(beta[p]), 'importance_share': float(den[p] / base)} for p in top]
                t['abs_patch_sum_over_tot'] = float(np.abs(pc).sum() / tot_sum)
                r[name] = t
            o['render'][rk] = r
        res[cfg] = o
        np.savez_compressed(os.path.join(out, f'importance_{cfg}.npz'), **{k: v.astype(np.float32) for k, v in W.items()})
        print(cfg, 'recon', {k: (round(v['recon_oracle_render_ratio'], 4), round(v['recon_h_ratio'], 4)) for k, v in o['render'].items()}, flush=True)
        for rk in ('C', 'obj_panel', 'obj_sphere', 'D'):
            r = o['render'][rk]
            print(' ', rk, 'imp by shape', {k: round(v, 3) for k, v in r['importance_by_shape'].items()}, 'rings', [round(v, 3) for v in r['importance_by_ring']])
            for name in SEEDS + (['stale'] if 'stale' in r else []):
                t = r[name]
                print('    ', name, 'signed/tot', round(t['signed_error_over_tot'], 4), 'signed/h', round(t['signed_error_over_h'], 4), 'level', round(t['level_part_over_tot'], 4), 'dir', round(t['directional_part_over_tot'], 4),
                      'by shape', {k: round(v, 4) for k, v in t['by_shape'].items()}, 'by ring', [round(v, 4) for v in t['by_ring']])
        print('  field', {name: {k: (round(v['rel_l1'], 3), round(v['flux_ratio'], 3)) for k, v in f.items()} for name, f in o['field'].items()}, flush=True)
    json.dump(res, open(os.path.join(out, 'localize.json'), 'w'), indent=1)


if __name__ == '__main__':
    os.makedirs(sys.argv[3], exist_ok=True)
    {'baseline': baseline, 'localize': localize}[sys.argv[1]](sys.argv[2], sys.argv[3])
    os._exit(0)
