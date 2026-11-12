"""WL89 post-freeze correctness audit of the R1 (WL88 SAM) selective-update STOP (amendment 1 evidence; no candidate-quality output).

  python audit89.py <R89>  -> results/amend1/audit89.json, last line AUDIT PASS|FAIL

On the qualified W1 (V3), with the frozen evidence parameters (quadrature M_ray, dense table M_dense, seeds) of tests88:
  1. exact strict-digest mismatching entries (selective T0 -> T1 vs full T1), their validity, provenance (receiver / sender cosines, whether the
     lane was traced at all) and the first hit of each lane at T0 and T1
  2. operational state equality, bitwise: pair list, validity masks, valid entries (dbin, cell, w), u state (hit, dbin, segments), MIS factors,
     and every tensor the transport reads (uop index / mask / W, a_recv, a_idx, a_w, a_cell, Lambert and ER groups)
  3. transport output equality, bitwise, under a deterministic CPU evaluation of the unchanged operator (same tensors, serial index_add) for a
     random field, the W1 once-scattered table q and the 4-order recurrence; the GPU outputs are reported against their own repeat floor
  4. null edit and move-and-return under the operational predicate
  5. invalid-lane perturbation: invalid entries' dbin -> random / 255, cell -> random / max, w -> NaN; operator tensors, CPU outputs (finite)
     and a subsequent selective edit (T1 -> T0) must equal the unperturbed path bitwise
"""
import hashlib, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx89  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx89.PT, fx89.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
E87 = fx89.fx88.fx87.P87['explicit']
RP = fx89.fx88.P88['relation']
NB, K = S.NB, S.K
cpu = torch.device('cpu')


def op_digest(rel):
    """Operational digest (amendment 1 predicate): everything the transport or later updates read; invalid lanes' dbin / cell / w excluded."""
    h = hashlib.sha256()
    for a in (rel.st.hit, rel.st.dbin, rel.st.seg, rel.mr.cpu().numpy(), rel.pi, rel.pp):
        h.update(np.ascontiguousarray(a).tobytes())
    if rel.m:
        v = rel.A['valid']
        h.update(np.ascontiguousarray(v).tobytes())
        for k in ('dbin', 'cell', 'w'):
            h.update(np.ascontiguousarray(rel.A[k][v]).tobytes())
        h.update(rel.seg_t.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def op_tensors(rel):
    u = rel.uop
    t = {'uop_idx': u.idx, 'uop_ok': u.ok, 'uop_W': u.W, 'a_recv': rel.a_recv, 'a_idx': rel.a_idx, 'a_w': rel.a_w, 'a_cell': rel.a_cell, 'a_lam': rel.a_lam,
         'lam_rho': rel.lam_rho, 'mr': rel.mr}
    for m, ix in u.groups:
        t[f'uop_group{m}'] = ix
    for m, ix in rel.a_er:
        t[f'a_er{m}'] = ix
    return {k: v.detach().to(cpu) for k, v in t.items()}


def tensors_equal(a, b):
    keys = sorted(set(a) | set(b))
    return {k: bool(k in a and k in b and a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k])) for k in keys}


def cpu_apply(rel, L):
    """The unchanged SAM.apply on CPU copies of the operator tensors (serial, deterministic)."""
    u = rel.uop
    N = rel.G.N
    L = L.to(cpu)
    Lr = L.reshape(-1, 3)[u.idx.to(cpu)].view(N, K, 3) * u.ok.to(cpu)
    out = torch.zeros(N, NB, 3)
    W = u.W.to(cpu)
    for m, ix in u.groups:
        ixc = ix.to(cpu)
        out[ixc] = torch.einsum('nkc,kbc->nbc', Lr[ixc], W[m])
    if rel.m == 0:
        return out
    e = L.reshape(-1, 3)[rel.a_idx.to(cpu)] * rel.a_w.to(cpu)[:, None]
    recv = rel.a_recv.to(cpu)
    lam = rel.a_lam.to(cpu)
    if len(lam):
        s = torch.zeros(N, 3)
        s.index_add_(0, recv[lam], e[lam])
        out = out + (s * rel.lam_rho.to(cpu)[:, None])[:, None, :]
    for mm, ix in rel.a_er:
        ixc = ix.to(cpu)
        out.index_add_(0, recv[ixc], rel.dense.T[mm].to(cpu)[rel.a_cell.to(cpu)[ixc]] * e[ixc][:, None, :])
    return out


def cpu_rec(rel, q, J=4):
    L = torch.as_tensor(np.asarray(q, np.float32))
    acc = 0
    for _ in range(J):
        L = cpu_apply(rel, L); acc = acc + L
    return acc


def main():
    import ecro86 as EC
    import rel88 as RL
    R = sys.argv[1]
    out = os.path.join(R, 'results', 'amend1'); os.makedirs(out, exist_ok=True)
    R8.inject_choice(R)
    fx89.fx88.setup('W1_T0')
    ex = FX['W1']; g0, g1 = ex['T0'], ex['T1']
    lt = PT.Lights(FX['light_sets'][ex['light_set']])
    Rr = {'w1': json.load(open(os.path.join(R, 'results', 'w_choice.json')))['chosen']}
    ctx = PT.Ctx(g0)
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    d16 = RL.Dense(RP['dense_sub'], RP['M_dense'], RP['seed_dense'])
    st0 = EC.build_state(ctx)
    rs = RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st0)
    d_t0 = op_digest(rs); strict_t0 = rs.digest()
    ctx.set_geom(g1)
    rs.update(ctx, ex['moved'])
    rf = RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'])
    # 1. strict mismatches and provenance
    G = ctx.G
    same_pairs = bool(np.array_equal(rs.pi, rf.pi) and np.array_equal(rs.pp, rf.pp))
    mism = {}
    for k in ('valid', 'dbin', 'cell', 'w'):
        mism[k] = np.argwhere(rs.A[k] != rf.A[k]).tolist() if same_pairs else None
    rows = []
    seg = rs.seg_t.cpu().numpy()
    bad = sorted({tuple(x) for k in mism for x in (mism[k] or [])})
    for pr, j in bad:
        i, p = int(rs.pi[pr]), int(rs.pp[pr])
        x, y = seg[pr, j, :3].astype(np.float64), seg[pr, j, 3:].astype(np.float64)
        w = (y - x) / np.linalg.norm(y - x)
        ny = G.PN[p] if G.PSID[p] != S.SID_SPHERE else (y - G.SC) / G.SR
        cx, cy = float(G.PN[i] @ w), float(-(ny @ w))
        lane = {}
        for tn, gg in (('T0', g0), ('T1', g1)):
            c2 = PT.Ctx(gg)
            o = mi.Point3f(*[float(v) for v in x + G.PN[i] * 1e-4]); d = mi.Vector3f(*[float(v) for v in w])
            si = c2.scene.ray_intersect(mi.Ray3f(o, d))
            sid = c2.shape_id(si)
            ph, bh = c2.patch_bin(si, sid, -d)
            dr.eval(si.t, sid, ph, bh); dr.sync_thread()
            lane[tn] = {'hit_valid': bool(si.is_valid()[0]), 't': float(si.t[0]), 'shape': int(sid[0]),
                        'hit_patch': int(ph[0]), 'bin_if_traced': int(bh[0])}
        rows.append({'pair': pr, 'sample': j, 'receiver': i, 'sender': p, 'receiver_shape': int(G.PSID[i]), 'sender_shape': int(G.PSID[p]),
                     'valid_sel': bool(rs.A['valid'][pr, j]), 'valid_full': bool(rf.A['valid'][pr, j]),
                     'dbin_sel': int(rs.A['dbin'][pr, j]), 'dbin_full': int(rf.A['dbin'][pr, j]), 'cell_sel': int(rs.A['cell'][pr, j]), 'cell_full': int(rf.A['cell'][pr, j]),
                     'w_sel': float(rs.A['w'][pr, j]), 'w_full': float(rf.A['w'][pr, j]), 'cos_receiver': cx, 'cos_sender': cy,
                     'lane_traced': bool(cx > 1e-3 and cy > 1e-6), 'segment_equal': bool(np.array_equal(seg[pr, j], rf.seg_t.cpu().numpy()[pr, j])),
                     'first_hit_if_traced': lane})
    Rr['strict'] = {'digest_selective': rs.digest(), 'digest_full': rf.digest(), 'equal': rs.digest() == rf.digest(), 'same_pairs': same_pairs,
                    'mismatch_counts': {k: (len(v) if v is not None else None) for k, v in mism.items()}, 'entries': rows,
                    'all_mismatches_invalid_in_both': bool(rows) and all(not r['valid_sel'] and not r['valid_full'] for r in rows)}
    # 2. operational state
    ts, tf = op_tensors(rs), op_tensors(rf)
    teq = tensors_equal(ts, tf)
    st_eq = bool(np.array_equal(rs.st.hit, rf.st.hit) and np.array_equal(rs.st.dbin, rf.st.dbin) and np.array_equal(rs.st.seg, rf.st.seg))
    Rr['operational'] = {'op_digest_selective': op_digest(rs), 'op_digest_full': op_digest(rf), 'op_digest_equal': op_digest(rs) == op_digest(rf),
                         'operator_tensors_equal': teq, 'all_tensors_equal': bool(all(teq.values())), 'u_state_equal': st_eq,
                         'valid_mask_equal': bool(np.array_equal(rs.A['valid'], rf.A['valid'])), 'segments_equal': bool(torch.equal(rs.seg_t, rf.seg_t))}
    # 3. transport outputs
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    rnd = torch.as_tensor(np.random.default_rng(89001).random((G.N, NB, 3)).astype(np.float32))
    o = {}
    for name, fs in (('random', lambda r: cpu_apply(r, rnd)), ('q_single', lambda r: cpu_apply(r, torch.as_tensor(np.asarray(q, np.float32)))),
                     ('q_recurrence_J4', lambda r: cpu_rec(r, q))):
        a, b, a2 = fs(rs), fs(rf), fs(rs)
        o[name] = {'cpu_bitwise_equal': bool(torch.equal(a, b)), 'cpu_self_repeat_equal': bool(torch.equal(a, a2)), 'finite': bool(torch.isfinite(a).all()),
                   'max_abs': float((a - b).abs().max())}
    gs = RL.recurrence([rs], q)[0]; gf = RL.recurrence([rf], q)[0]; gs2 = RL.recurrence([rs], q)[0]
    o['gpu_recurrence_J4'] = {'sel_vs_full_maxrel': float((gs - gf).abs().max() / gf.abs().max()), 'self_repeat_maxrel': float((gs - gs2).abs().max() / gs.abs().max()),
                              'note': 'GPU index_add_ is atomic (non-deterministic summation order); reported against its own repeat floor'}
    Rr['transport_outputs'] = o
    # 4. null edit and move-and-return (operational predicate) on a fresh relation
    c3 = PT.Ctx(g0)
    rn = RL.SAM(c3, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=EC.build_state(c3))
    dn0, sn0 = op_digest(rn), rn.digest()
    rn.update(c3, ex['moved'])
    null_op, null_strict = op_digest(rn) == dn0, rn.digest() == sn0
    c3.set_geom(g1); rn.update(c3, ex['moved']); c3.set_geom(g0); rn.update(c3, ex['moved'])
    Rr['null_and_return'] = {'null_edit_op_equal': null_op, 'null_edit_strict_equal': null_strict, 'move_and_return_op_equal': op_digest(rn) == d_t0,
                             'move_and_return_strict_equal': rn.digest() == strict_t0}
    # 5. invalid-lane perturbation
    rng = np.random.default_rng(89002)
    pert = {}
    for name, fn in (('dbin_random', lambda A, v: A['dbin'].__setitem__(~v, rng.integers(0, NB, (~v).sum()).astype(np.uint8))),
                     ('dbin_255_cell_max', lambda A, v: (A['dbin'].__setitem__(~v, 255), A['cell'].__setitem__(~v, np.iinfo(np.int32).max))),
                     ('w_nan_cell_random', lambda A, v: (A['w'].__setitem__(~v, np.nan), A['cell'].__setitem__(~v, rng.integers(0, 1 << 30, (~v).sum()))))):
        ctxp = PT.Ctx(g1)
        rp = RL.SAM(ctxp, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'])                  # full T1 relation on a fresh scene
        rq = RL.SAM(ctxp, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'])                  # unperturbed twin
        # baseline = the unperturbed twin built on the SAME scene object (a fresh Ctx(T1) and a set_geom(T0 -> T1) scene differ in mesh
        # vertex bits, so cross-context comparisons are not identity tests)
        base_tensors = op_tensors(rq); base_out = cpu_rec(rq, q)
        v = rp.A['valid']
        fn(rp.A, v)
        rp._finish()
        tp = op_tensors(rp)
        eq_t = all(tensors_equal(tp, base_tensors).values())
        outp = cpu_rec(rp, q)
        eq_o = bool(torch.equal(outp, base_out)) and bool(torch.isfinite(outp).all())
        ctxp.set_geom(g0)
        rp.update(ctxp, ex['moved']); rq.update(ctxp, ex['moved'])
        eq_u = op_digest(rp) == op_digest(rq) and all(tensors_equal(op_tensors(rp), op_tensors(rq)).values())
        pert[name] = {'perturbed_invalid_entries': int((~v).sum()), 'operator_tensors_unchanged': bool(eq_t), 'cpu_outputs_bitwise_unchanged_and_finite': eq_o,
                      'later_selective_update_uncontaminated': bool(eq_u)}
    Rr['perturbation'] = pert
    ok = (Rr['strict']['all_mismatches_invalid_in_both'] and Rr['operational']['op_digest_equal'] and Rr['operational']['all_tensors_equal']
          and Rr['operational']['u_state_equal'] and Rr['operational']['valid_mask_equal'] and Rr['operational']['segments_equal']
          and all(v['cpu_bitwise_equal'] and v['cpu_self_repeat_equal'] and v['finite'] for k, v in o.items() if k != 'gpu_recurrence_J4')
          and Rr['null_and_return']['null_edit_op_equal'] and Rr['null_and_return']['move_and_return_op_equal']
          and all(all(v.values()) for v in [{k: x for k, x in p.items() if k != 'perturbed_invalid_entries'} for p in pert.values()]))
    Rr['AUDIT_PASS'] = bool(ok)
    json.dump(Rr, open(os.path.join(out, 'audit89.json'), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print(json.dumps({'strict': Rr['strict']['mismatch_counts'], 'all_invalid': Rr['strict']['all_mismatches_invalid_in_both'],
                      'op_equal': Rr['operational']['op_digest_equal'], 'tensors': Rr['operational']['all_tensors_equal'],
                      'outputs': {k: v.get('cpu_bitwise_equal') for k, v in o.items()}, 'null_return': Rr['null_and_return'], 'perturbation': pert}), flush=True)
    print('AUDIT', 'PASS' if ok else 'FAIL', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
