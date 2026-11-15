"""WL96 D4 scratch 2: noise-free carrier representability (each carrier substituted for L1 inside the dense deterministic area quadrature)
and the cost / accuracy of a deterministic sender-cell x receiver-point delivery with lobe culling. Same simplified CK2 model as ck2_model."""
import json, os, sys, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
TAG = sys.argv[1] if len(sys.argv) > 1 else 'ck2'
sys.argv = [sys.argv[0]] + ([TAG] if TAG != 'ck2' else [])
import ck2_model as M
import er_np as ER

ys = M.recv_points(4)                                   # [9, 16, 3]


def E_dense_with(Lfun, h=0.004):
    xs, dA = M.panel_grid(h)
    out = np.zeros((9, 3))
    for p in range(9):
        acc = np.zeros(3)
        for y in ys[p]:
            d = y - xs; r2 = (d ** 2).sum(1); w = d / np.sqrt(r2)[:, None]
            cx = w[:, 0]; cy = -(w @ M.FN); ok = (cx > 0) & (cy > 0)
            acc += (Lfun(xs[ok], w[ok]) * (cx[ok] * cy[ok] / r2[ok] * dA)[:, None]).sum(0)
        out[p] = acc / len(ys[p])
    return out


def L_q(q):
    def f(x, w):
        return q[M.patch_of(x), M.pbin(M.to_panel_local(w))]
    return f


def L_x1c(x, w):
    return M.L1(M.panel_centre(M.patch_of(x)), w)


def L_cp(g):
    return lambda x, w: M.L1_cp(x, w, g)


def L_obin(x, w):
    """exact position, outgoing direction snapped to its bin centre (outgoing-angular projection only, no averaging)."""
    wl = M.to_panel_local(w); bn = M.pbin(wl)
    c = M.bins_dir(None, bn, np.full(len(bn), 0.5), np.full(len(bn), 0.5))
    ww = np.stack([c[:, 2], c[:, 0], c[:, 1]], 1)
    return M.L1(x, ww)


def sender_cells(g):
    h = M.PS / g
    xs, dA = M.panel_grid(h)
    return xs, dA


def deliver_cells(g, n_r, cull):
    """deterministic: sender cell centres (exact ER at the exact direction from the cell centre to each receiver point) x n_r x n_r receiver
    points per patch; pairs whose lobe value is below cull * (max over pairs of that receiver patch) are dropped (count = visibility rays)."""
    xs, dA = sender_cells(g)
    yy = M.recv_points(n_r)
    out = np.zeros((9, 3)); pairs = 0; total = 0
    for p in range(9):
        acc = np.zeros(3); vals = []
        for y in yy[p]:
            d = y - xs; r2 = (d ** 2).sum(1); w = d / np.sqrt(r2)[:, None]
            cx = w[:, 0]; cy = -(w @ M.FN); ok = (cx > 0) & (cy > 0)
            c = np.zeros((len(xs), 3)); c[ok] = M.L1(xs[ok], w[ok]) * (cx[ok] * cy[ok] / r2[ok] * dA)[:, None]
            vals.append(c)
        V = np.array(vals); lv = V.sum(-1); thr = cull * lv.max()
        keep = lv >= thr
        pairs += int(keep.sum()); total += lv.size
        out[p] = (V * keep[..., None]).sum((0, 1)) / len(yy[p])
    return out, pairs, total


if __name__ == '__main__':
    T = {}; t0 = time.time()
    Eref = E_dense_with(M.L1); T['ref_s'] = time.time() - t0
    q = M.q_table()
    res = {'light': M.LIGHT.tolist()}
    arms = {'binned_q(patch x bin mean)': L_q(q), 'X1c(patch centre, exact dir)': L_x1c, 'outgoing bin-centre dir, exact pos': L_obin}
    for g in (1, 2, 4, 8):
        arms[f'CP{g}(incident state at {g}x{g} ctrl pts, exact out dir)'] = L_cp(g)
    for k, f in arms.items():
        t1 = time.time(); E = E_dense_with(f); res[k] = M.metr(E, Eref); res[k]['s'] = round(time.time() - t1, 1)
    dc = {}
    for g in (2, 4, 8, 16):
        for n_r in (2, 4, 8):
            for cull in (0.0, 1e-3, 1e-2):
                E, pr, tot = deliver_cells(g, n_r, cull)
                m = M.metr(E, Eref); m.update(vis_pairs=pr, all_pairs=tot)
                dc[f'g{g}_r{n_r}_cull{cull}'] = m
    res['deterministic_sender_cells'] = dc
    res['timing_s'] = T
    json.dump(res, open(os.path.join(H, f'ck2_model2_{TAG}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
