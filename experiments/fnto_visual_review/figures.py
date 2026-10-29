"""Report-only WL58/59 figures from completed JSON and stored boundary readouts."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS=['#4477aa','#cc6677'];RGB=['#cc6677','#228833','#4477aa']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def load(path):return json.loads(Path(path).read_text())
def save(fig,dest,name):
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def plots58(repo,dest):
    r=load(repo/'results/fnto_reuse58/preflight/out/angular.json')
    cases=r['cases'];fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    ax=axs[0];x=np.arange(len(cases));vals=[100*v['quadratures'][-1]['best_output_relative_l2'] for v in cases]
    ax.bar(x,vals,color=[COLORS[1] if v['alpha']==.04 else COLORS[0] for v in cases]);ax.axhline(10,color='black',ls='--',label='10% preregistered bound')
    ax.set(xticks=x,xticklabels=[v['label'].replace('theta','t').replace('phi','p') for v in cases],ylabel='Optimal output projection error (%)',title='WL58: every declared output case')
    ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.legend(fontsize=8)
    ax=axs[1];pairs=r['source_swap_pairs'];vals=[100*p['best_common_prediction_error_lower_bound'] for p in pairs]
    ax.bar(range(4),vals,color=[COLORS[1],COLORS[1],COLORS[0],COLORS[0]])
    for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.1f}%',ha='center')
    ax.axhline(25,color='black',ls='--',label='25% pair bound');ax.set(xticks=range(4),xticklabels=[f"alpha={p['alpha']}\ninput cell {p['declared_input_bin']}" for p in pairs],ylim=(0,70),ylabel='Necessary worst-pair relative error (%)',title='Identical encoded inputs, distinct physical outputs');ax.legend(fontsize=8)
    fig.suptitle('WL58 angular preflight only — no learned F/S execution',fontsize=13)
    save(fig,dest,'angular_gate')
    # Analytic visualization using the frozen evaluator, not a new acceptance run.
    sys.path.insert(0,str(repo/'experiments/fnto_reuse58'))
    import angular_preflight as A
    grid=512;u=(np.arange(grid)+.5)/grid;uu,vv=np.meshgrid(u,u);wo=A.square_direction(uu,vv)
    maps=[]
    for off in (.2,.8):
        wi=A.square_direction((18%8+off)/8,(18//8+.5)/8);z=A.ggx(wi,wo,.04)
        means=z.reshape(16,grid//16,16,grid//16).mean((1,3));approx=np.repeat(np.repeat(means,grid//16,0),grid//16,1)
        maps.extend([z,approx])
    maxv=max(z.max() for z in maps);fig,axs=plt.subplots(1,4,figsize=(13,3.8),layout='constrained')
    for ax,z,title in zip(axs,maps,['Source X: analytic GGX','X: best 256-bin projection','Source Y: analytic GGX','Y: best 256-bin projection']):
        im=ax.imshow(np.log10(1+z),origin='lower',extent=(0,1,0,1),vmin=0,vmax=np.log10(1+maxv),cmap='magma');ax.set(title=title,xlabel='Concentric-square u',ylabel='v')
    fig.colorbar(im,ax=axs,label='log10(1 + radiance)',shrink=.75)
    fig.suptitle('WL58 lobe illustration: alpha=0.04; X/Y both encode to input cell18',fontsize=12)
    save(fig,dest,'angular_lobes')

def plots59(repo,dest):
    root=repo/'results/fnto_persist59/final/out';h=load(root/'heldout.json');c=load(root/'cost.json');tags=list(h)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axs[0,0];x=np.arange(3)
    for i,tag in enumerate(tags):
        ax.bar(x+(i-.5)*.32,100*np.array(h[tag]['poses']['pt0']['metrics']['F']['coarse']['mean']),.32,color=COLORS[i],label=tag)
    ax.axhline(10,color='black',ls='--',label='10% bound');ax.set(xticks=x,xticklabels=list('RGB'),ylim=(0,12),ylabel='Absolute-response mean relative L2 (%)',title='Absolute action norms pass; F and S overlap');ax.legend(fontsize=8)
    ax=axs[0,1]
    for i,tag in enumerate(tags):
        ax.bar(x+(i-.5)*.32,[100*v['error'] for v in h[tag]['poses']['pt0']['dynamic']['F']],.32,color=COLORS[i],label=tag)
    ax.axhline(35,color='black',ls='--',label='35% bound');ax.set(xticks=x,xticklabels=list('RGB'),ylim=(0,50),ylabel='Signed deformation relative L2 (%)',title='Blue deformation response fails both policies');ax.legend(fontsize=8)
    ax=axs[1,0];poses=['pt0','cf0','cf1','cf2','cf3']
    for i,tag in enumerate(tags):ax.plot(poses,[h[tag]['poses'][p]['F_S']['max_relative_L1'] for p in poses],'o-',color=COLORS[i],label=tag)
    ax.axhline(1e-5,color='black',ls='--',label='F/S bound');ax.set(yscale='log',ylim=(1e-8,3e-5),ylabel='Maximum relative L1',title='Actual F/S equivalence; no oracle used for invalidation');ax.legend(fontsize=8)
    ax=axs[1,1]
    for i,tag in enumerate(tags):
        cs=[v for v in c['comparisons'] if v['edit']=='dent' and v['seed']==h[tag]['seed']]
        ys=np.array([v['ratio'] for v in cs]);lo=np.array([v['bootstrap95'][0] for v in cs]);hi=np.array([v['bootstrap95'][1] for v in cs])
        ax.errorbar(range(4),ys,yerr=[ys-lo,hi-ys],fmt='o-',color=COLORS[i],capsize=4,label=tag)
    ax.axhline(1,color='black',ls='--');ax.set(xticks=range(4),xticklabels=['1','4','16','64'],xlabel='Queries after dent',ylabel='S / F measured intrinsic lifecycle',title='Small raw cost saving; physical quality is not eligible');ax.legend(fontsize=8)
    fig.suptitle('WL59: computed state reuse works; required physical fidelity still fails',fontsize=13)
    save(fig,dest,'summary')
    fig,axs=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    for ax,tag in zip(axs,tags):
        d=h[tag]['poses']['pt0']['directional_pairs']['F'];z=100*np.array(d['errors'])
        im=ax.imshow(z,cmap='YlOrRd',vmin=0,vmax=55,aspect='auto')
        for i in range(8):
            for j in range(3):ax.text(j,i,f'{z[i,j]:.1f}'+(' x' if z[i,j]>25 else ''),ha='center',va='center',color='white' if z[i,j]>40 else 'black')
        ax.set(xticks=range(3),xticklabels=list('RGB'),yticks=range(8),yticklabels=[f'pair{i}: {48+2*i}/{49+2*i}' for i in range(8)],title=tag+' — F and S agree')
    fig.colorbar(im,ax=axs,label='Directional-contrast relative L2 (%); x >25% bound',shrink=.8)
    fig.suptitle('All primary source-swap pairs/channels; every cell is reference-qualified',fontsize=12)
    save(fig,dest,'directional_all')
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,tag in enumerate(tags):
        t=load(root/'training'/f'{tag}.json');history=t['history'];steps=[r['step'] for r in history]
        for ax,key,title in zip(axs,['action','delta','val'],['Sampled training action loss','Sampled training delta loss','Validation action loss']):
            ax.plot(steps,[v[key] for v in history],color=COLORS[i],label=tag);ax.set(xlabel='Optimizer step',ylabel=key,title=title)
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('WL59 fixed 6000-step fits — unsmoothed recorded losses',fontsize=12)
    save(fig,dest,'training')
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    labels=[];geo=[];learn=[];persistent=[];peak=[]
    for key,w in c['workers'].items():
        if w['seed']!=5901:continue
        labels.append(w['edit']+' '+w['policy']);rs=w['records'];geo.append(np.median([r['geometry_and_B_seconds'] for r in rs])*1000)
        learn.append(np.median([r['update_seconds']-r['geometry_and_B_seconds'] for r in rs])*1000)
        persistent.append(max(r['persistent_GPU_bytes'] for r in rs)/1e6);peak.append(max(r['peak_GPU_allocated'] for r in rs)/1e6)
    axs[0].bar(labels,geo,color='#bbbbbb',label='geometry + B');axs[0].bar(labels,learn,bottom=geo,color=COLORS[0],label='raw state / normalization / detection');axs[0].set(ylabel='Median component time (ms)',title='Shared geometry work dominates');axs[0].legend(fontsize=7)
    axs[1].bar(np.arange(4)-.18,persistent,.36,label='persistent state');axs[1].bar(np.arange(4)+.18,peak,.36,label='update/query allocated peak');axs[1].set(xticks=range(4),xticklabels=labels,ylabel='GPU memory (decimal MB)',title='State and transient memory are distinct');axs[1].legend(fontsize=7)
    audit=h[tags[0]]['poses']['pt0']['invalidation'];fractions=np.array(audit['node_updates'])/audit['node_total']
    axs[2].bar(['encoder0','encoder1','encoder2','raw edges','normalized','B'],np.r_[fractions,audit['edge_updates']/audit['edge_total'],1,1]*100,color=COLORS[1]);axs[2].set(ylim=(0,110),ylabel='Recomputed entries (%)',title='Raw selectivity is not total-state selectivity')
    for ax in axs:ax.tick_params(axis='x',rotation=30,labelsize=8)
    fig.suptitle('WL59 cost decomposition: seed5901, actual prototype; K-only lifecycle',fontsize=12)
    fig.supxlabel('Component medians need not add to the median total. Update/query peaks reset after initial rest-cache setup.',fontsize=9)
    save(fig,dest,'cost_and_state')

def poly(ax,tri,values,cmap=None,vmin=None,vmax=None):
    coll=PolyCollection(tri[:,:,:2]*1000,edgecolors='none',linewidths=0,antialiaseds=False,rasterized=True)
    if cmap is None:coll.set_facecolors(values)
    else:coll.set_array(values);coll.set_cmap(cmap);coll.set_clim(vmin,vmax)
    ax.add_collection(coll);ax.set(xlim=(-50,50),ylim=(-50,50),aspect='equal',xlabel='Material x (mm)',ylabel='Material y (mm)')
    ax.tick_params(labelsize=7);return coll

def readouts(npz,meta,dest):
    a=np.load(npz);manifest=load(meta);fields=a['fields'].tolist();strong=manifest['field_selection']['strongest_blue_field']
    jobs=[(5901,f,v) for f in fields for v in ('top','bottom')]+[(5902,strong,'top'),(5902,strong,'bottom')]
    for seed,field,view in jobs:
        k=fields.index(field);tri=a[f'pt0_{view}_triangles'];ref=a[f'pt0_{view}_oracle'][k]
        assert np.allclose(tri[:,:,:2],a[f'rest_{view}_triangles'][:,:,:2]),'Material triangle correspondence is required for delta images'
        full=a[f'pt0_{view}_F_{seed}'][k];selective=a[f'pt0_{view}_S_{seed}'][k];stale=a[f'pt0_{view}_stale_physical'][k]
        # Shared exposure based ONLY on the current reference; disclose clipping.
        exposure=max(float(np.percentile(ref,99)),1e-30)
        display=lambda z:np.clip(np.log1p(np.maximum(z,0)*10/exposure)/np.log(11),0,1)
        dr=ref[:,2]-a[f'rest_{view}_oracle'][k,:,2]
        df=full[:,2]-a[f'rest_{view}_F_{seed}'][k,:,2]
        ds=selective[:,2]-a[f'rest_{view}_S_{seed}'][k,:,2]
        limit=max(float(np.abs(np.r_[dr,df,ds]).max()),1e-30)
        fig,axs=plt.subplots(2,4,figsize=(14,8),layout='constrained')
        src=a[f'pt0_{view}_incident'][k,:,2];source=poly(axs[0,0],tri,src,'Greys',0,max(float(src.max()),1e-30))
        axs[0,0].set_title(f'Incident blue irradiance\n0 to {src.max():.3g} flux / area (own scale)',fontsize=9)
        for ax,z,title in [(axs[0,1],ref,'Current physical oracle'),(axs[0,2],full,'Full rebuilt learned state F'),(axs[0,3],selective,'Selective resident state S'),(axs[1,0],stale,'Stale physical K, current readout')]:
            poly(ax,tri,display(z));clip=100*np.mean(z>exposure);ax.set_title(title+f'\n{clip:.2f}% RGB components clipped',fontsize=9)
        for ax,z,title in zip(axs[1,1:],[dr,df,ds],['Oracle geometry delta — blue','F geometry delta — blue','S geometry delta — blue']):
            im=poly(ax,tri,z,'RdBu_r',-limit,limit);ax.set_title(title,fontsize=10)
        fig.colorbar(im,ax=list(axs[1,1:]),shrink=.65,label='Signed blue outgoing-bin radiance change')
        err=float(np.abs(full-selective).sum()/max(np.abs(full).sum(),1e-30))
        fig.suptitle(f'WL59 K-only boundary readout | seed{seed} | field{field:02} | {view} | face-readout F/S L1 {err:.2e}',fontsize=12)
        fig.supxlabel('Stored first-exit flux / (current cell area x projected-bin angle). Not Cycles or coupled GI.\nAll RGB panels share log display and reference p99 exposure; delta panels share a symmetric scale. Post hoc views.',fontsize=9)
        save(fig,dest,f'K_only_seed{seed}_field{field:02}_{view}')
    # Actual geometry is separate from response colours; no artificial lighting.
    fig=plt.figure(figsize=(11,4.5),layout='constrained')
    for i,name in enumerate(('rest','pt0')):
        ax=fig.add_subplot(1,2,i+1,projection='3d');tr=a[name+'_triangles']*1000
        if name=='pt0':values=(a['pt0_triangles']-a['rest_triangles'])[:,:,2].mean(1)*1000
        else:values=np.zeros(len(tr))
        col=Poly3DCollection(tr,linewidths=.06,edgecolors='#555555',rasterized=True);col.set_facecolor(plt.get_cmap('viridis')((values+5.75)/5.75));ax.add_collection3d(col)
        ax.set(xlim=(-50,50),ylim=(-50,50),zlim=(tr[:,:,2].min(),a['rest_triangles'][:,:,2].max()*1000),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title=name+' — displacement colour, no lighting')
        ax.set_box_aspect((1,1,.22));ax.view_init(25,-65)
    fig.suptitle('Actual held-out local dent: depth5.75mm, radius22.5mm, centre(5,-5)mm',fontsize=12)
    save(fig,dest,'geometry')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--readouts',type=Path);p.add_argument('--readout-manifest',type=Path);args=p.parse_args()
    d58=args.repo/'results/evaluation/58';d59=args.repo/'results/evaluation/59'
    plots58(args.repo,d58);plots59(args.repo,d59)
    if args.readouts:readouts(args.readouts,args.readout_manifest,d59)
    for dest in (d58,d59):
        records={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.iterdir() if p.is_file() and p.suffix in ('.png','.pdf')}
        (dest/'visual_manifest.json').write_text(json.dumps(dict(kind='POST_HOC_VISUALIZATION',changes_primary_verdict=False,artifacts=records),indent=2)+'\n')

if __name__=='__main__':main()
