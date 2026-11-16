"""WL97 audit, track E1, add-on: is the learned S1 sender wrong where the deployed relation actually READS it?
The learned once-scattered table Bq = a (c^T s) is an internal intermediate of model86.apply (never an output, never supervised
directly: own92a.objective supervises the single-flight OUTPUT and the global output). Entries (S1 patch, outgoing bin) that no state ray
reads are unconstrained by any loss. This script restricts / re-weights the sender comparison to the measure of the deployed relation:
  w[n, d] = sum over state rays (receiver i, ray k) that hit S1 patch n in outgoing bin d of rw[k] * area[i]   (power per unit radiance).
  python e1_read.py     (env as run_e1.sh; reads results/sender_e1.npz of the sender stage and WL95A senders.npz, writes results/read.json)
"""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_audit as E  # noqa: E402  (own module; builds the CK2 context)
np = E.np
N, K, NB, S1, BOXA, WT = E.N, E.K, E.NB, E.S1, E.BOXA, E.WT
G = E.G
hit, dbin, seg = E.ctx.state_rays()
rs = np.random.default_rng(E.S.P['representation'].get('table_seed', 86001))
us = rs.random((4, 1, K), np.float32)
assert np.array_equal(us, E.ctx.u_state_np)
rw = E.ray_w(us)
h = hit.astype(np.int64).reshape(N, K); db = dbin.astype(np.int64).reshape(N, K)
area = np.asarray(G.PAREA, np.float64)
z = np.load(os.path.join(E.OUT, 'sender_e1.npz'))
orig = np.load(os.path.join(E.ORIG, 'senders.npz'))
q = 0.5 * (z['qa'] + z['qb'])
snd = {'A0_s8601': z['Bq_s8601'], 'A0_s8602': z['Bq_s8602'], 'M64_s16': orig['A4'].astype(np.float64), 'stored_q256': orig['q256'].astype(np.float64),
       'qa_vs_qb_noise': z['qa']}
loc = np.full(N, -1); loc[S1] = np.arange(len(S1))
is_s1 = (h >= 0) & (loc[np.clip(h, 0, None)] >= 0)


def wtab(recv_mask):
    w = np.zeros((len(S1), NB)); cnt = np.zeros((len(S1), NB))
    m = is_s1 & recv_mask[:, None]
    ii, kk = np.nonzero(m)
    np.add.at(w, (loc[h[ii, kk]], db[ii, kk]), rw[kk] * area[ii])
    np.add.at(cnt, (loc[h[ii, kk]], db[ii, kk]), 1.0)
    return w, cnt


def met(X, Y, w):
    W = w[:, :, None]
    return {'rel_l1': float((np.abs(X - Y) * W).sum() / (Y * W).sum()), 'energy': float((X * W).sum() / (Y * W).sum())}


o = {}
allr = np.ones(N, bool); allr[S1] = False
ba = np.zeros(N, bool); ba[BOXA] = True
sid = np.asarray(G.PSID)
groups = {'all_receivers': allr, 'boxA': ba, 'not_boxA': allr & ~ba}
for gn, gm in groups.items():
    w, cnt = wtab(gm)
    read = cnt > 0
    d = {'rays': int(cnt.sum()), 'entries_read': int(read.sum()), 'entries_total': int(read.size),
         'q_WT_energy_share_on_read_entries': float((q * (WT[None, :] * read)[:, :, None]).sum() / (q * WT[None, :, None]).sum()),
         'read_entries_by_outgoing_ring': read.reshape(len(S1), 4, 16).sum((0, 2)).tolist(),
         'read_power_weight_share_by_outgoing_ring': (w.reshape(len(S1), 4, 16).sum((0, 2)) / w.sum()).tolist(),
         'q_delivered_power_share_by_outgoing_ring': ((q.sum(2) * w).reshape(len(S1), 4, 16).sum((0, 2)) / (q.sum(2) * w).sum()).tolist()}
    for k_, X in snd.items():
        Y = q if k_ != 'qa_vs_qb_noise' else z['qb']
        d[k_] = {'read_power_weighted': met(X, Y, w), 'WT_on_read_entries_only': met(X, Y, WT[None, :] * read), 'WT_on_unread_entries_only': met(X, Y, WT[None, :] * (~read)) if (~read).any() else None}
    o[gn] = d
# where does the deployed relation send S1 light: receivers by surface id
w_all, _ = wtab(allr)
pw = {}
for s_ in np.unique(sid):
    gm = (sid == s_) & allr
    if gm.any():
        w, cnt = wtab(gm)
        pw[int(s_)] = {'rays': int(cnt.sum()), 'q_power': float((q.sum(2) * w).sum()), 'A0_s8601_power': float((snd['A0_s8601'].sum(2) * w).sum()),
                       'A0_s8602_power': float((snd['A0_s8602'].sum(2) * w).sum()), 'M64_power': float((snd['M64_s16'].sum(2) * w).sum())}
o['delivered_power_by_receiver_surface_id'] = pw
json.dump(o, open(os.path.join(E.OUT, 'read.json'), 'w'), indent=1)
print(json.dumps(o, indent=1))
print('E1_DONE read')
os._exit(0)
