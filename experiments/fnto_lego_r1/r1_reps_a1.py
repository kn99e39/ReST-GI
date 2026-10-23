"""WL42 protocol fnto_r1 amendment 1 - band / mid representatives rebuilt (geometry only; runs before any r1 training).

Gate 2b failed with the a0 representatives (one uniformly drawn source per equal-count distance stratum x the stratum
area, K 48 / 64). Attribution (r1_gate2_attrib.py, r1_gate2_ksweep.py): aggregation 3-7 %, pick bias <= 8 %, the
dominant part is the single-sample variance (25-40 %). a1 keeps the sources (1.5 cm / 8 cm aggregates), the radii and
the consumer code, and changes only the selection:
  near_exact_ori : the K/2 nearest aggregates exactly with their own area; the rest ordered by (8 distance shells,
                   dominant normal axis, distance), split into K/2 equal-count strata, one representative per stratum
                   drawn with probability proportional to its area, carrying the stratum area (unbiased)
  K_BAND 96, K_MID 128 = the smallest swept K whose sampling error (median vs all aggregates) on the adverse synthetic
  field (cos x 4 cm checker) is <= 0.075, half the Gate 2b threshold (ksweep: band 0.075, mid 0.049).

    python r1_reps_a1.py        # anchors.npz -> anchors_a0.npz (kept), anchors.npz rewritten with a1 nbr/nbrA/mid/midA
"""
import json
import time

import numpy as np
from scipy.spatial import cKDTree

import r1_common as C


def axis_key(n):
    a = np.argmax(np.abs(n), 1)
    return (a * 2 + (n[np.arange(len(n)), a] > 0)).astype(np.int64)


def near_exact_ori(query_pos, src_pos, src_area, src_ax, R, K, rng, rows=4096):
    tree = cKDTree(src_pos)
    Q = len(query_pos); h = K // 2
    rep = np.full((Q, K), -1, np.int32); Arep = np.zeros((Q, K), np.float32); nball = np.zeros(Q, np.int32)
    kq = 256; todo = np.arange(Q)
    while len(todo):
        d, i = tree.query(query_pos[todo], k=min(kq, len(src_pos)), distance_upper_bound=R, workers=-1)
        d = d.reshape(len(todo), -1); i = i.reshape(len(todo), -1)
        full = np.isfinite(d[:, -1]) & (kq < len(src_pos))
        done = todo[~full]; dd = d[~full]; ii = i[~full]
        cnt = np.isfinite(dd).sum(1); nball[done] = cnt
        for c in np.unique(cnt):
            if c == 0:
                continue
            sel = np.flatnonzero(cnt == c)
            for s0 in range(0, len(sel), rows):
                ss = sel[s0:s0 + rows]; q_ = done[ss]; ii_ = ii[ss][:, :c]; dd_ = dd[ss][:, :c]
                if c <= K:
                    rep[q_, :c] = ii_; Arep[q_, :c] = src_area[ii_]
                    continue
                rep[q_, :h] = ii_[:, :h]; Arep[q_, :h] = src_area[ii_[:, :h]]
                fi = ii_[:, h:]; fd = dd_[:, h:]; nf = c - h
                shell = np.minimum((fd / R * 8).astype(np.int64), 7)
                key = (shell * 8 + src_ax[fi]) * nf + np.arange(nf)[None]          # columns are already distance-sorted
                o = np.argsort(key, 1); fi = np.take_along_axis(fi, o, 1); a = src_area[fi].astype(np.float64)
                cs = np.concatenate([np.zeros((len(ss), 1)), np.cumsum(a, 1)], 1)
                e = np.linspace(0, nf, K - h + 1).astype(np.int64)
                lo, hi = cs[:, e[:-1]], cs[:, e[1:]]
                t = lo + rng.random(lo.shape) * (hi - lo)
                off = (np.arange(len(ss)) * (cs[:, -1].max() + 1.0))[:, None]   # rows made disjoint for one flat search
                j = np.searchsorted((cs + off).ravel(), (t + off).ravel(), side="right").reshape(t.shape) - 1 - np.arange(len(ss))[:, None] * cs.shape[1]
                j = np.clip(j, e[:-1][None], e[1:][None] - 1)
                rep[q_, h:] = np.take_along_axis(fi, j, 1); Arep[q_, h:] = (hi - lo).astype(np.float32)
        todo = todo[full]; kq *= 4
    return rep, Arep, nball


def main():
    t0 = time.time(); src = C.OUT1 / "anchors.npz"; a0 = C.OUT1 / "anchors_a0.npz"
    if not a0.exists():
        src.rename(a0)
    Z = dict(np.load(a0)); rng = np.random.default_rng(1)
    obj, pos = Z["obj"], Z["pos"].astype(np.float64)
    N = len(obj)
    nbr = np.full((N, C.K_BAND), -1, np.int32); nbrA = np.zeros((N, C.K_BAND), np.float32)
    mid = np.full((N, C.K_MID), -1, np.int32); midA = np.zeros((N, C.K_MID), np.float32)
    bax, cax = axis_key(Z["bnrm"].astype(np.float64)), axis_key(Z["cnrm"].astype(np.float64))
    rec = {"amendment": 1, "anchors_a0_sha16": C.sha16(a0), "K_band": C.K_BAND, "K_mid": C.K_MID, "scheme": "near_exact_ori", "seed": 1}
    for o in (0, 1):
        g = np.flatnonzero(obj == o); gb = np.flatnonzero(Z["bobj"] == o); gc = np.flatnonzero(Z["cobj"] == o)
        r_, A_, n_ = near_exact_ori(pos[g], Z["bpos"][gb].astype(np.float64), Z["barea"][gb], bax[gb], C.R_LOCAL, C.K_BAND, rng)
        nbr[g] = np.where(r_ >= 0, gb[np.clip(r_, 0, None)], -1); nbrA[g] = A_
        rm, Am, nm_ = near_exact_ori(pos[g], Z["cpos"][gc].astype(np.float64), Z["carea"][gc], cax[gc], C.R_MID, C.K_MID, rng)
        mid[g] = np.where(rm >= 0, gc[np.clip(rm, 0, None)], -1); midA[g] = Am
        rec[f"domain{o}"] = {"band_ball_count_q50": float(np.median(n_)), "mid_ball_count_q50": float(np.median(nm_)),
                             "band_slots_used_q50": float(np.median((nbr[g] >= 0).sum(1))), "mid_slots_used_q50": float(np.median((mid[g] >= 0).sum(1)))}
        print(o, rec[f"domain{o}"], f"{time.time() - t0:.0f}s", flush=True)
    Z.update(nbr=nbr, nbrA=nbrA, mid=mid, midA=midA)
    np.savez(src, **Z)
    rec.update({"anchors_sha16": C.sha16(src), "seconds": time.time() - t0})
    (C.OUT1 / "anchors_a1.json").write_text(json.dumps(rec, indent=1)); print(rec)


if __name__ == "__main__":
    main()
