"""Report-only export of Gate 2 and (if executed) A/B evidence; every row retained; never feeds training/selection."""
import json,csv,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
def table(name,rows):
    if not rows:return
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');w.writeheader();w.writerows(rows)

q=read(ROOT/'qualification.json');s=q['summary']
for key in ('absolute','directional','deformation','geometry_delta','historical_discrepancy','phaseA_topup_consistency','bank_change_vs_phaseA'):table(f'gate2_{key}_all.csv',q[key])
table('label_noise_all.csv',read(ROOT/'label_noise.json')['rows'])
summary={'gate2':s}
poses=['rest']+[f'tr{i:02d}' for i in range(12)];fields=list(range(80,96))
pa=read(Path.home()/'rest_clean65/analysis.json')['deformation']
def grid(rows):
    z=np.zeros((12,16))
    for i,p in enumerate(poses[1:]):
        for k,f in enumerate(fields):z[i,k]=max(v['loss_scale_SE'] for v in rows if v['pose']==p and v['field']==f and v['resolution']=='coarse')
    return z
before,after=grid(pa),grid(q['deformation'])
comparison=(ROOT/'comparison.json').exists()
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,3 if comparison else 1,figsize=(17 if comparison else 7,9),constrained_layout=True,squeeze=False)
for a,z,t in ((ax[0,0],before,'Phase A (128M/action)'),(ax[1,0],after,'Combined bank (84-87: 512M/action)')):
    im=a.imshow(z,vmin=0,vmax=.5,cmap='magma',aspect='auto');a.set(title=f'Delta SE / loss scale, max RGB — {t}\ngate <=0.25; failing cells: {int((z>.25).sum())}',xticks=range(16),xticklabels=fields,yticks=range(12),yticklabels=poses[1:],xlabel='Train field')
    for i,j in zip(*np.where(z>.25)):a.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec='#39d0ff',lw=1.2))
    fig.colorbar(im,ax=a,shrink=.8)
if comparison:
    c=read(ROOT/'comparison.json');summary['comparison']={k:v for k,v in c.items() if k!='models'}
    rows=[];comps=[];absolute=[];deltas=[];history=[];training=[];costs=[];fs=[]
    for key,row in c['models'].items():
        arm,seed=key.split('_');o=row['original'];h=row['training']
        training.append(dict(arm=arm,seed=seed,seconds=h['seconds'],peak_GPU_bytes=h['peak_GPU_bytes'],parameters=h['parameters'],steps=h['steps']))
        for e in h['history']:history.append(dict(arm=arm,seed=seed,**e))
        for pose,p in o['poses'].items():
            for policy in ('F','S'):
                fs.append(dict(arm=arm,seed=seed,pose=pose,policy=policy,F_S_max_relative_L1=p['F_S']['max_relative_L1'],F_S_passed=p['F_S_passed'],physics_passed=p['physics']['passed'],static_useful=p['static_useful'][policy],**{f'physics_{k}':v for k,v in p['physics'].items() if k!='passed'}))
                for res in ('coarse','fine16','fine64'):
                    for ch,l in enumerate('RGB'):absolute.append(dict(arm=arm,seed=seed,pose=pose,primary=p['primary'],policy=policy,resolution=res,channel=l,mean=p['metrics'][policy][res]['mean'][ch],p90=p['metrics'][policy][res]['p90'][ch]))
                if 'directional_pairs' in p:
                    dp=p['directional_pairs'][policy]
                    for pair in range(8):
                        for ch,l in enumerate('RGB'):rows.append(dict(arm=arm,seed=seed,pose=pose,policy=policy,pair=pair,channel=l,error=dp['errors'][pair][ch],qualified=dp['detectable'][pair][ch],historically_inspected=True))
                    for ch,e in enumerate(p['dynamic'][policy]):deltas.append(dict(arm=arm,seed=seed,pose=pose,policy=policy,channel='RGB'[ch],**e))
        for pop,r in row['populations'].items():
            for comp in ('T','B','Scat'):
                for pair in range(len(r[comp]['error'])):
                    for ch,l in enumerate('RGB'):comps.append(dict(arm=arm,seed=seed,population=pop,component=comp,pair=pair,channel=l,error=r[comp]['error'][pair][ch],amplitude=r[comp]['amplitude'][pair][ch],correlation=r[comp]['correlation'][pair][ch]))
        for v in read(ROOT/'arms'/arm/'out/cost.json')['comparisons']:
            if str(v['seed'])==seed:costs.append(dict(arm=arm,**v))
    for name,rr in (('directional_original_all.csv',rows),('directional_populations_all.csv',comps),('absolute_all.csv',absolute),('deformation_all.csv',deltas),('FS_physics_all.csv',fs),('training_history.csv',history),('training_cost.csv',training),('cache_lifecycle_cost.csv',costs)):table(name,rr)
    # Report-only U65 reference-replica sensitivity (all four even/odd assignments).
    rep=np.load(ROOT/'fresh/independent_reference.npz')['replicas'];pred=np.load(ROOT/'comparison_components.npz');sens={}
    norm=lambda z:np.linalg.norm(z.reshape(len(z),-1,3),axis=1)
    for seed in ('5901','5902'):
        aa=pred[f'A_{seed}_U65_uninspected'][:,0];bb=pred[f'B_{seed}_U65_uninspected'][:,0];v=[]
        for ra in range(2):
            for rb in range(2):
                t=rep[ra,::2,0]-rep[rb,1::2,0];ea=norm(aa[::2]-aa[1::2]-t)/np.maximum(norm(t),1e-30);eb=norm(bb[::2]-bb[1::2]-t)/np.maximum(norm(t),1e-30)
                v.append(dict(even=ra,odd=rb,mean_A=float(ea.mean()),mean_B=float(eb.mean()),ratio=float(eb.mean()/ea.mean())))
        sens[seed]=v
    summary['U65_replica_sensitivity']=sens;summary['training_cost']=training
    if (ROOT/'trained_invariants.json').exists():summary['trained_invariants']=read(ROOT/'trained_invariants.json')
    col={'A':'#406b9c','B':'#d46d38'};d=c['decisions']
    a=ax[0,1]
    for k,(arm,seed) in enumerate([(x,y) for y in ('5901','5902') for x in ('A','B')]):
        e=np.array(c['models'][f'{arm}_{seed}']['original']['poses']['pt0']['directional_pairs']['F']['errors']).reshape(-1)
        a.scatter(np.arange(24)+(k-1.5)*.15,e,s=14,color=col[arm],marker='o' if seed=='5901' else '^',label=f'{arm} {seed}')
    a.axhline(.25,color='k',lw=.8,ls='--');a.set(title='pt0 primary directional error (pair x RGB), F policy',xlabel='pair*3+channel',ylabel='relative L2');a.legend(fontsize=8)
    a=ax[1,1];labels=['primary pt0','old pairs 0-6','U65 uninspected','WL64 (inspected)','Pair7 (inspected)'];x=np.arange(len(labels))
    for k,seed in enumerate(('5901','5902')):
        v=[d[seed]['primary_ratio'],d[seed]['old_pairs0_6_ratio'],d[seed]['U65_uninspected']['ratio'],d[seed]['WL64_historically_inspected']['ratio'],d[seed]['Pair7_historically_inspected']['ratio']]
        a.bar(x+(k-.5)*.35,v,.35,label=f'seed {seed}',color=['#7a9cc6','#c67a4a'][k])
    a.axhline(1,color='k',lw=.8);a.axhline(.9,color='g',lw=.8,ls='--');a.axhline(1.1,color='r',lw=.8,ls='--');a.set(xticks=x,xticklabels=labels,ylabel='B / A mean total directional error',title='Directional error ratio (<0.90 improvement, >1.10 regression)');a.tick_params(axis='x',rotation=20);a.legend(fontsize=8)
    a=ax[0,2]
    for k,(arm,seed) in enumerate([(x,y) for y in ('5901','5902') for x in ('A','B')]):
        dyn=c['models'][f'{arm}_{seed}']['original']['poses']['pt0']['dynamic']['F'];a.bar(np.arange(3)+(k-1.5)*.2,[v.get('error',np.nan) for v in dyn],.2,color=col[arm],alpha=1 if seed=='5901' else .55,label=f'{arm} {seed}')
    a.axhline(.35,color='k',lw=.8,ls='--');a.set(xticks=range(3),xticklabels=list('RGB'),title='pt0 signed deformation error (gate <=0.35)',ylabel='relative L2');a.legend(fontsize=8)
    a=ax[1,2]
    for arm in ('A','B'):
        for seed,ls in (('5901','-'),('5902',':')):
            hh=c['models'][f'{arm}_{seed}']['training']['history'];a.plot([e['step'] for e in hh],[e['val'] for e in hh],ls,color=col[arm],label=f'{arm} {seed} val (original noisy labels)')
    a.set(yscale='log',xlabel='step',ylabel='validation action loss',title='Original validation loss (noisy labels; not physical fidelity)');a.legend(fontsize=7)
    fig.suptitle(f"WL65 extension: Gate 2 {s['status']}; outcome {c['outcome']}{' ('+c['outcome_B_sublabel']+')' if c['outcome_B_sublabel'] else ''}",fontsize=12)
else:
    fig.suptitle(f"WL65 extension Gate 2: {s['status']} (no training)",fontsize=12)
fig.savefig(OUT/'topup_ab.png',dpi=160);fig.savefig(OUT/'topup_ab.pdf');plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()},indent=2)+'\n')
