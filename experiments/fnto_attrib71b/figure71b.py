"""WL71B reviewer figure: single-group substitution effects (degrade = Early with group from Late; recover = Late with group from Early) as fractions of the drift, arm R."""
import json,sys
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;import numpy as np
L=json.load(open(sys.argv[1]))['attribution']['R'];out=sys.argv[2]
mets=[('m1_H1_coarse','H1 coarse (pooled)'),('m2_H1_contrast','H1 contrast (pooled)'),('s5_coarse_rest','slot 5 coarse'),('s6_coarse_rest','slot 6 coarse'),('s6_flux_rest','slot 6 flux <1 cm')]
grp=['W','U','V','GLOBAL','M'];fig,ax=plt.subplots(1,len(mets),figsize=(20,4.2),sharey=True);x=np.arange(len(grp))
for k,(m,t) in enumerate(mets):
    r=L[m];D=r['drift'];deg=[r['shares'][g]['degrade']/D for g in grp];rec=[r['shares'][g]['recover']/D for g in grp]
    ax[k].bar(x-.2,deg,.4,label='degrade: Early + group from Late');ax[k].bar(x+.2,rec,.4,label='recover: Late − (Late with group from Early)')
    ax[k].axhline(.5,ls='--',c='k',lw=.7);ax[k].axhline(0,c='k',lw=.5);ax[k].set_xticks(x);ax[k].set_xticklabels(['W','U','V','U+V','mass']);ax[k].set_title(f'{t}\ndrift {r["values"]["EEEE"]:.3f} → {r["values"]["LLLL"]:.3f}',fontsize=9)
ax[0].set_ylabel('fraction of drift (Early R@2000 → Late R@20000)');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out+'.png',dpi=130);fig.savefig(out+'.pdf')
