"""WL110 focused validation of the live engine (headless, one process, same Engine.step as the window loop).

  python verify110.py <out_dir> [--wl101 ~/rest_dyn101/results/ecro_dyn101/evidence]

V1 checkpoint identity        sha256 of both WL100 checkpoints == the WL101-recorded hashes (and the archived timing JSON)
V2 WL101 V_SSS_ANIM replay    the archived 48-event pose sequence through Engine.step: per-event state digest and dirty-ray
                              count == the archived WL101 B_sel runs (r1, r2); compute-only timing beside the archived medians
V3 archived WL101 output      the live animation reaches its segment end points (u = 0 / 1 = WL100 test configs exactly) ->
                              table B vs archived PRED_B_<cfg>.npz and 4096-spp images (rt101.render_hq, same seeds) vs archive
V4 fresh single-frame render  at intermediate live poses: a NEW RT built directly at that pose (full state) + same seed ->
                              composed HDR and table vs the live frame
V5 selective == full          every 5th live frame: resident selective state == full state rebuild
V6 determinism                two independent engines over the same 180-frame fixed-clock animation: per-frame geometry, state
                              checksums, table and image sums
V7 loop execution             respond calls == frames, one checkpoint load, parameter digest unchanged, table changes every edit
V8 timing synchronisation     per-part synchronised sum vs end-to-end wall; same frames with a single end synchronisation
V9 reconstruction history     WL101B SVGF on the live animation: reconstructed == same-frame forced-reset control on every edited
                              frame; history accumulates only on held (unchanged) frames; silhouette / contact luminance ratios
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live110 as L  # noqa: E402
import anim110 as A  # noqa: E402

R, torch, np, dr, mi = L.R, L.torch, L.np, L.dr, L.mi
Q = A.Q


def rel_l1(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))


def rel_max(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def free(eng):
    del eng.rt, eng.model
    torch.cuda.empty_cache()


def t_of(k, rate=30.0):
    return k / rate


def v2_replay(arch):
    case, start, ev = A.v_sss_anim_poses()
    eng = L.Engine('sss')
    assert A.geom_hash(start[0]) == A.geom_hash(eng.anim.pose(0.0)[0])
    runs = {r: json.loads((arch / 'timing' / f'V_SSS_ANIM_B_sel_{r}.json').read_text()) for r in ('r1', 'r2')}
    rows, comp = [], []
    for i, (g, Lt, sid) in enumerate(ev):
        P = L.Parts()
        L.sync(); t0 = time.perf_counter()
        r = eng.step(g, 9000 + i, P)
        tot = (time.perf_counter() - t0) * 1e3
        dg = eng.state_digest()
        a1, a2 = runs['r1']['events'][i], runs['r2']['events'][i]
        assert a1['state'] == sid
        thru_render = sum(v for k, v in P.t.items() if k in ('geometry', 'relation', 'source', 'neural', 'handoff', 'render'))
        rows.append({'i': i, 'pose': sid, 'digest': dg, 'digest_eq_r1': dg == a1['state_digest'], 'digest_eq_r2': dg == a2['state_digest'],
                     'dirty': r['dirty'], 'dirty_eq_r1': r['dirty'] == a1['dirty_rays'], 'thru_render_ms': thru_render, 'compute_ms': tot})
        if i >= Q.WARM:
            comp.append(thru_render)
    arch_med = {rr: runs[rr]['summary']['dynamic_frame_ms']['512x1']['median'] for rr in runs}
    out = {'events': len(rows), 'state_digest_equal_r1': sum(x['digest_eq_r1'] for x in rows), 'state_digest_equal_r2': sum(x['digest_eq_r2'] for x in rows),
           'dirty_equal_r1': sum(x['dirty_eq_r1'] for x in rows),
           'live_engine_geometry_to_render_ms_median': float(np.median(comp)), 'live_engine_geometry_to_render_ms_p95': float(np.percentile(comp, 95)),
           'wl101_archived_dynamic_frame_ms_median': arch_med,
           'note': 'live engine sums the same synchronised parts (geometry, relation, source, generator, hand-off, frame) as WL101 total_ms '
                   'minus its trivial per-event set_light; compose / tonemap / readback / presentation are excluded here (they are WL110 additions).',
           'rows': rows}
    free(eng)
    return out


def hq_compare(eng, sid, arch_case_dir):
    z = np.load(arch_case_dir / f'PRED_B_{sid}.npz')
    B = eng.rt.respond(eng.rt.source())[0]
    Bn = B.cpu().numpy()
    acc = R.render_hq(eng.rt, R.t2d(B.contiguous().view(-1)), eng.cam, 512, 4096, 101500 + 10 * sid)
    keys = ['e10', 'e11', 'e12', 'e20', 'e21', 'e22', 'hl0', 'hl1', 'hl2']
    res = {'state': sid, 'table_rel_l1': rel_l1(Bn, z['B']), 'table_rel_max': rel_max(Bn, z['B'])}
    for name in ('e1', 'e2', 'hl'):
        img = np.stack([(acc[keys.index(f'{name}{c}')] / 4096).reshape(512, 512) for c in range(3)], -1)
        res[f'{name}_rel_l1'] = rel_l1(img, z[name])
    tot = sum(np.stack([(acc[keys.index(f'{n}{c}')] / 4096).reshape(512, 512) for c in range(3)], -1) for n in ('e1', 'e2', 'hl'))
    res['total_rel_l1'] = rel_l1(tot, z['e1'] + z['e2'] + z['hl'])
    return res


def v3_archived(arch, case, nframes=91):
    """Live fixed-clock animation (period 6 s, 30 Hz): k = 0 is u = 0 (cfg_from), k = 90 is u = 1 (cfg_to), exactly."""
    eng = L.Engine(case)
    a, b = A.SEGMENTS[case]
    out = {'case': case, 'segment': [a, b]}
    for k in range(nframes):
        g, u = eng.anim.pose(t_of(k))
        eng.step(g, 11000000 + k, L.Parts())
        if k in (0, 90):
            sid = a if k == 0 else b
            assert u == (0.0 if k == 0 else 1.0)
            target = Q._rows(case)[sid]['geom']
            assert A.geom_hash(g) == A.geom_hash(target), (k, g, target)
            out[f'frame{k}'] = hq_compare(eng, sid, arch / 'quality' / case)
    free(eng)
    return out


def v4_v5_v6_v8(nframes=180, sample=(15, 45, 60, 75, 120, 150)):
    eng = L.Engine('sss')
    loads0 = L.LOADS['torch.load']
    rec, saved, selfull, parts_gap = [], {}, [], []
    for k in range(nframes):
        g, u = eng.anim.pose(t_of(k))
        P = L.Parts()
        L.sync(); t0 = time.perf_counter()
        r = eng.step(g, 11000000 + k, P)
        tot = (time.perf_counter() - t0) * 1e3
        pr = eng.proof(r)
        rec.append({'k': k, 'geom': A.geom_hash(g), **pr})
        parts_gap.append(tot - sum(P.t.values()))
        if k in sample:
            saved[k] = {'hdr': r['hdr'].clone(), 'B': r['B'].clone(), 'g': copy.deepcopy(g)}
        if k % 5 == 0 and r['moved']:
            h0, d0, s0 = eng.rt.hit.clone(), eng.rt.dbin.clone(), eng.rt.seg.clone()
            eng.rt.rebuild_state()
            selfull.append(bool(torch.equal(h0, eng.rt.hit) and torch.equal(d0, eng.rt.dbin)))
            eng.rt.hit, eng.rt.dbin, eng.rt.seg = h0, d0, s0
    # V8: same engine, same frames again with ONE synchronisation at the end (tonemap readback) instead of per-part sync
    class NoSync:
        def __init__(s):
            s.t = {}

        def __call__(s, name, fn):
            return fn()
    nos, syn = [], []
    for rep in range(2):
        for k in range(nframes):
            g, _ = eng.anim.pose(t_of(k))
            L.sync(); t0 = time.perf_counter()
            if rep == 0:
                eng.step(g, 11000000 + k, NoSync()); L.sync(); nos.append((time.perf_counter() - t0) * 1e3)
            else:
                eng.step(g, 11000000 + k, L.Parts()); L.sync(); syn.append((time.perf_counter() - t0) * 1e3)
    integ = {'respond_calls': eng.respond_calls, 'expected': 3 * nframes, 'checkpoint_loads_during_loops': L.LOADS['torch.load'] - loads0,
             'params_unchanged': L.param_digest(eng.model) == eng.param0}
    free(eng)
    # V4: fresh renderer per sampled pose
    v4 = []
    for k, s in saved.items():
        model = R.load_model('sss')
        rt = R.RT(s['g'], 'sss', model, A.Animation('sss').light)
        B = rt.respond(rt.source())[0]
        out = rt.frame(R.t2d(B.contiguous().view(-1)), dict(R.C100.P['datasets']['sss']['camera'], res=512), 512, 1, 11000000 + k)
        hdr = torch.stack([o.torch() for o in out]).view(3, 3, -1).sum(0)
        v4.append({'k': k, 'table_rel_l1': rel_l1(B.cpu().numpy(), s['B'].cpu().numpy()), 'table_bitwise': bool(torch.equal(B, s['B'])),
                   'hdr_mean_rel': float((hdr - s['hdr']).abs().mean() / s['hdr'].abs().mean()), 'hdr_bitwise': bool(torch.equal(hdr, s['hdr'])),
                   'hdr_max_abs': float((hdr - s['hdr']).abs().max())})
        del rt, model
        torch.cuda.empty_cache()
    # V6: second independent engine, same sequence
    eng2 = L.Engine('sss')
    rec2 = []
    for k in range(nframes):
        g, u = eng2.anim.pose(t_of(k))
        r = eng2.step(g, 11000000 + k, L.Parts())
        rec2.append({'k': k, 'geom': A.geom_hash(g), **eng2.proof(r)})
    free(eng2)
    keys = ['geom', 'hit_checksum', 'dbin_checksum']
    v6 = {'frames': nframes, 'geometry_and_state_identical': all(all(a[x] == b[x] for x in keys) for a, b in zip(rec, rec2)),
          'table_sum_max_rel': max(abs(a['table_sum'] - b['table_sum']) / abs(b['table_sum']) for a, b in zip(rec, rec2)),
          'table_sum_identical_frames': sum(a['table_sum'] == b['table_sum'] for a, b in zip(rec, rec2)),
          'raw_hdr_sum_max_rel': max(abs(a['raw_hdr_sum'] - b['raw_hdr_sum']) / abs(b['raw_hdr_sum']) for a, b in zip(rec, rec2)),
          'raw_hdr_sum_identical_frames': sum(a['raw_hdr_sum'] == b['raw_hdr_sum'] for a, b in zip(rec, rec2))}
    v8 = {'per_part_sync_minus_parts_ms': {'median': float(np.median(parts_gap[30:])), 'max': float(np.max(parts_gap[30:]))},
          'end_to_end_single_sync_ms': {'median': float(np.median(nos[30:])), 'p95': float(np.percentile(nos[30:], 95))},
          'end_to_end_per_part_sync_ms': {'median': float(np.median(syn[30:])), 'p95': float(np.percentile(syn[30:], 95))},
          'note': 'single-sync run: no device synchronisation between parts; the tonemap readback (.cpu()) and a final sync close the frame. '
                  'Some parts synchronise internally regardless (update_state nonzero count, source/respond host-side bookkeeping).'}
    v5 = {'checked': len(selfull), 'equal': sum(selfull)}
    v7 = integ
    return v4, v5, v6, v7, v8


def v9_recon(nframes=120, hold=(60, 8)):
    """SVGF mode: frames follow the animation; at frame hold[0] the pose is held for hold[1] frames (no edit) to show that
    history accumulates only without an edit. Same-frame forced-reset control from a separate SVGF instance."""
    eng = L.Engine('sss', mode='svgf')
    cold = eng.WB.SVGF(512, R.dev, R.PT.cam_basis, eng.WB.P['reconstruction'])
    rows = []
    seq = []
    for k in range(nframes):
        seq.append(t_of(k))
        if k == hold[0]:
            seq += [t_of(k)] * hold[1]
    sph_ratio_b, sph_ratio_i, contact = [], [], []
    for j, t in enumerate(seq):
        g, u = eng.anim.pose(t)
        r = eng.step(g, 11000000 + j, L.Parts())
        guide = eng.host.guide(eng.cam)
        ctl, _, _ = cold.apply(r['hdr'], guide, eng.cam, 0, reset=True)
        eq = bool(torch.equal(ctl, r['shown']))
        age = float(eng.den.history[5].max())
        rows.append({'j': j, 'moved': bool(r['moved']), 'recon_equals_reset_control': eq, 'history_age_max': age})
        if r['moved'] and j >= 30:
            mp = torch.nn.functional.max_pool2d
            sid = guide[6].view(512, 512)
            sph = (sid == R.S.SID_SPHERE).float()[None, None]
            er = (-mp(-sph, 9, 1, 4))[0, 0] > 0.5                 # 4 px erosion
            dil = mp(sph, 25, 1, 12)[0, 0] > 0.5                  # 12 px dilation
            sphb = sph[0, 0] > 0.5
            band = sphb & ~er
            lum = lambda x: (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).view(512, 512)
            lr, lc = lum(r['hdr']), lum(r['shown'])
            sph_ratio_b.append(float(lc[band].mean() / lr[band].mean()))
            sph_ratio_i.append(float(lc[er].mean() / lr[er].mean()))
            ring = dil & ~sphb & (sid >= 0) & (sid != R.S.SID_SPHERE)  # non-sphere surfaces within 12 px of the silhouette (contact / shadow)
            contact.append(float(lc[ring].mean() / lr[ring].mean()))
    moved = [x for x in rows if x['moved']]
    held = [x for x in rows if not x['moved'] and x['j'] > 0]
    out = {'frames': len(rows), 'edited_frames': len(moved), 'edited_equal_reset': sum(x['recon_equals_reset_control'] for x in moved),
           'held_frames': len(held), 'held_equal_reset': sum(x['recon_equals_reset_control'] for x in held),
           'held_history_age_max': [x['history_age_max'] for x in held],
           'sphere_boundary_band_recon_over_raw_luminance': {'median': float(np.median(sph_ratio_b)), 'min': float(np.min(sph_ratio_b))},
           'sphere_interior_recon_over_raw_luminance': {'median': float(np.median(sph_ratio_i))},
           'silhouette_ring_12px_recon_over_raw_luminance': {'median': float(np.median(contact))},
           'note': 'ratios are reconstruction vs the same frame raw mean (no reference); band = 4 px inner sphere boundary from the '
                   'center-ray shape id. A ratio < 1 on the band with ~1 inside is the WL101B silhouette darkening signature.'}
    free(eng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--wl101', default=str(Path.home() / 'rest_dyn101/results/ecro_dyn101/evidence'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    arch = Path(a.wl101)
    res = {'batch': 'WL110 verification', 'snapshot_before': L.gpu_snapshot()}
    t0 = time.time()
    ck = {c: L.sha_file(R.ROOT100 / c / 'model.pt') for c in ('sss', 'near')}
    archived_ck = json.loads((arch / 'timing' / 'V_SSS_ANIM_B_sel_r1.json').read_text())['checkpoint_sha256']
    res['V1_checkpoint'] = {'sha256': ck, 'equals_wl101_recorded': {c: ck[c] == L.WL101_CHECKPOINT_SHA[c] for c in ck},
                            'sss_equals_archived_timing_json': ck['sss'] == archived_ck}
    print('V1', res['V1_checkpoint'], flush=True)
    res['V2_v_sss_anim_replay'] = v2_replay(arch)
    print('V2', {k: v for k, v in res['V2_v_sss_anim_replay'].items() if k != 'rows'}, flush=True)
    res['V3_archived_wl101_output'] = {c: v3_archived(arch, c) for c in ('sss', 'near')}
    print('V3', json.dumps(res['V3_archived_wl101_output']), flush=True)
    v4, v5, v6, v7, v8 = v4_v5_v6_v8()
    res.update(V4_fresh_single_frame=v4, V5_selective_equals_full=v5, V6_determinism=v6, V7_loop_integrity=v7, V8_timing_sync=v8)
    print('V4', v4, '\nV5', v5, '\nV6', v6, '\nV7', v7, '\nV8', v8, flush=True)
    res['V9_reconstruction_history'] = v9_recon()
    print('V9', {k: v for k, v in res['V9_reconstruction_history'].items() if k != 'held_history_age_max'}, flush=True)
    res['wall_s'] = time.time() - t0
    res['snapshot_after'] = L.gpu_snapshot()
    res['versions'] = {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    (out / 'verify.json').write_text(json.dumps(res, indent=1))
    print('VERIFY DONE', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
