"""Read-only: per-bin non-visible fraction and non-visible mean luminance in floor patch 155 (primary_a2 packets)."""
import numpy as np
luma=np.array([.2126,.7152,.0722])
for r in range(3):
    f=np.load(f'results/ecro_sss101g/primary_a2/construction_{r}.npz')
    sl=slice(155*64,156*64)
    n=f['n'][sl].sum(2); v=f['v'][sl].sum(2); L=(f['L'][sl]@luma).sum(2); vL=(f['vL'][sl]@luma).sum(2)  # (64 bins, 4 subcells)
    nv=(n-v)/n
    print(f'r{r} subcell3 per-bin nonvis fraction: min {nv[:,3].min():.4f} median {np.median(nv[:,3]):.4f} max {nv[:,3].max():.4f}; '
          f'bins with nonvis>=2%: {(nv[:,3]>=0.02).sum()}/64 ; subcell0 min {nv[:,0].min():.4f} median {np.median(nv[:,0]):.4f}')
    nvm=np.where(n-v>0,(L-vL)/np.maximum(n-v,1),np.nan); vm=vL/np.maximum(v,1)
    print(f'   subcell3 nonvis-mean / vis-mean per bin: median {np.nanmedian(nvm[:,3]/vm[:,3]):.2f}  (subcell0 {np.nanmedian(nvm[:,0]/vm[:,0]):.2f})')
    print(f'   expected ring fraction of subcell 3 from spawn-offset model: 0.0273')
