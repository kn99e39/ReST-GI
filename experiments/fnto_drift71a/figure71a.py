"""WL71A reviewer figure: checkpoint trajectories (training J, FIT vs H1/H3 worst-channel upper bounds, slot-6 flux) for arms R and N."""
import json,sys
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
e=json.load(open(sys.argv[1]));R=e['results'];out=sys.argv[2];ks=[0,500,2000,6000,10000,20000]
w=lambda c,g,m:max(x['hi'] for x in R[c]['groups'][g]['pooled'][m])
name=lambda a,k:'init' if k==0 else f'{a}@{k}'
fig,ax=plt.subplots(1,4,figsize=(20,4.6));xs=[max(k,100) for k in ks]
for i,(m,th) in enumerate((('abs_coarse',.10),('abs_fine16',.15),('con_coarse',.25))):
    for a,ls in (('R','-'),('N','--')):
        for g,c in (('FIT_rest','tab:blue'),('H1_rest','tab:red'),('H1_pt0','tab:orange'),('H3_cf3','tab:purple')):
            ax[i].plot(xs,[w(name(a,k),g,m) for k in ks],ls,color=c,marker='o',ms=3,label=f'{g} {a}')
    for c,col in (('smokeA','k'),('frozen_P_5902','gray')):ax[i].axhline(w(c,'H1_rest',m),color=col,lw=.7,ls=':')
    ax[i].axhline(th,color='k',lw=.8,ls='-.');ax[i].set_xscale('log');ax[i].set_yscale('log');ax[i].set_title(f'{m} (upper 95 %, worst ch.)');ax[i].set_xlabel('updates (0 plotted at 100)')
ax[0].legend(fontsize=6,ncol=2)
for a,ls in (('R','-'),('N','--')):
    for s,c in ((0,'tab:blue'),(5,'tab:orange'),(6,'tab:red'),(7,'tab:green')):
        ax[3].plot(xs,[R[name(a,k)]['flux'][f'rest_s{s}'][1] for k in ks],ls,color=c,marker='o',ms=3,label=f'slot {s} {a}')
ax[3].axhline(e['physical_flux']['rest_s6'][1],color='k',lw=.8,ls='-.');ax[3].set_xscale('log');ax[3].set_title('flux within 1 cm of source (rest); dash-dot = physical');ax[3].legend(fontsize=6,ncol=2)
fig.tight_layout();fig.savefig(out+'.png',dpi=130);fig.savefig(out+'.pdf')
