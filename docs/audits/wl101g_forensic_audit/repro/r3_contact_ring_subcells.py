"""WL101G audit, read-only: per-subcell visible / non-visible split of the stored construction packets
(results/ecro_sss101g/primary_a2/construction_{0,1,2}.npz), floor contact patch 155 and its neighbours.

The ring fraction 0.0273 of subcell 3 is the a-priori spawn-offset model (r2_tangency_model.py):
pi * 2R * mag / (0.25^2 / 4) with mag = (1 + 0.9) * 1500 * 2^-24 m. Run from the repository root.
"""
import numpy as np

LUMA = np.array([.2126, .7152, .0722])
RING_FRACTION_SUBCELL3 = 0.0273
root = 'results/ecro_sss101g/primary_a2/'
for r in range(3):
    f = np.load(root + f'construction_{r}.npz')
    n, v, L, vL = (f[k] for k in ('n', 'v', 'L', 'vL'))   # (slots, 4, 4[, 3])

    def patch(pid):
        sl = slice(pid * 64, (pid + 1) * 64)
        return (n[sl].sum((0, 2)), v[sl].sum((0, 2)),
                (L[sl] @ LUMA).sum((0, 2)), (vL[sl] @ LUMA).sum((0, 2)))

    for pid, label in ((155, 'floor contact patch'), (154, 'floor x-1'), (139, 'floor y-1'),
                       (171, 'floor y+1'), (156, 'floor x+1')):
        ns, vs, Ls, vLs = patch(pid)
        nv = ns - vs
        print(f"r{r} {label:20s} pid {pid}: nonvis frac per subcell {np.round(nv / ns, 4)}  "
              f"nonvis mean L {np.round((Ls - vLs) / np.maximum(nv, 1), 3)}  vis mean L {np.round(vLs / vs, 3)}  "
              f"Un {np.round(Ls / ns, 4)}")
    ns, vs, Ls, vLs = patch(155)
    fnv = (ns[3] - vs[3]) / ns[3]
    m_nv = (Ls[3] - vLs[3]) / (ns[3] - vs[3])
    m_v = vLs[3] / vs[3]
    ring_mean = (m_nv * fnv - (fnv - RING_FRACTION_SUBCELL3) * m_v) / RING_FRACTION_SUBCELL3
    print(f"r{r} patch155 slot Un {Ls.sum() / ns.sum():.4f}  V {vLs.sum() / vs.sum():.4f}  "
          f"Un-V {Ls.sum() / ns.sum() - vLs.sum() / vs.sum():+.4f}; subcell3 Un {Ls[3] / ns[3]:.4f}  V {m_v:.4f}  "
          f"Un-V {Ls[3] / ns[3] - m_v:+.4f}")
    print(f"r{r} implied mean luminance of the modelled ring samples (assuming the other non-visible samples "
          f"have the visible mean): {ring_mean:.3f}")
