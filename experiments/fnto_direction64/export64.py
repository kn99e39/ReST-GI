"""Report-only export; never used by training or checkpoint selection."""
import json,csv,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(p.read_text())
c=read(ROOT/'comparison.json');cal=read(ROOT/'pretraining/calibration.json');prec=read(ROOT/'independent_complete.json')

def table(name,fields,rows):
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
rows=[];components=[];absolute=[];deltas=[];history=[];costs=[];training=[]
for arm in ('A','B'):
 for seed in (5901,5902):
    row=c['models'][f'{arm}_{seed}'];orig=row['original'];h=read(ROOT/'arms'/arm/'out/training'/f'P_{seed}.json')
    training.append({'arm':arm,'seed':seed,'seconds':h['seconds'],'peak_GPU_bytes':h['peak_GPU_bytes'],'parameters':h['parameters'],'steps':h['steps']})
    for e in h['history']:history.append({'arm':arm,'seed':seed,**e})
    for pose,p in orig['poses'].items():
        for policy in ('F','S'):
            for res in ('coarse','fine16','fine64'):
                for ch,label in enumerate('RGB'):
                    absolute.append(dict(arm=arm,seed=seed,pose=pose,primary=p['primary'],policy=policy,resolution=res,channel=label,mean=p['metrics'][policy][res]['mean'][ch],p90=p['metrics'][policy][res]['p90'][ch]))
            if 'directional_pairs' in p:
                dp=p['directional_pairs'][policy]
                for pair in range(8):
                    for ch,label in enumerate('RGB'):
                        error=dp['errors'][pair][ch];qualified=dp['detectable'][pair][ch]
                        rows.append(dict(arm=arm,seed=seed,population='original',pose=pose,policy=policy,pair=pair,channel=label,error=error,qualified=qualified,passed=qualified and error<=.25,historically_inspected_pair7=pair==7))
                for ch,e in enumerate(p['dynamic'][policy]):deltas.append(dict(arm=arm,seed=seed,pose=pose,policy=policy,channel='RGB'[ch],**e))
    for pop,key in [('fresh','fresh'),('historical_pair7','pair7_historically_inspected')]:
        for comp,r in row[key].items():
            for pair in range(len(r['error'])):
                for ch,label in enumerate('RGB'):
                    q=bool(prec['SNR_pairs_components_RGB'][pair][['T','B','Scat'].index(comp)][ch]>=3) if pop=='fresh' else True
                    components.append(dict(arm=arm,seed=seed,population=pop,component=comp,pair=pair if pop=='fresh' else 7,channel=label,error=r['error'][pair][ch],amplitude=r['amplitude'][pair][ch],correlation=r['correlation'][pair][ch],qualified=q))
    for v in read(ROOT/'arms'/arm/'out/cost.json')['comparisons']:costs.append(dict(arm=arm,**v))
table('directional_original_all.csv',list(rows[0]),rows)
table('directional_components_all.csv',list(components[0]),components)
table('absolute_all.csv',list(absolute[0]),absolute)
table('deformation_all.csv',sorted(set().union(*(r.keys() for r in deltas))),deltas)
table('training_history.csv',sorted(set().union(*(r.keys() for r in history))),history)
table('training_cost.csv',list(training[0]),training)
table('cache_lifecycle_cost.csv',list(costs[0]),costs)
# Original primary summary, all original thresholds retained.
summary={'adjudication':{k:v for k,v in c.items() if k!='models'},'precision':{'fresh_all24_qualified':prec['total_all24_qualified'],'fresh_min_total_SNR':float(np.asarray(prec['SNR_pairs_components_RGB'])[:,0].min()),'rest_original_training_noise_to_contrast_coarse_range':list(np.asarray(cal['precision']['True']['noise_to_contrast']).reshape(-1)[[np.argmin(np.asarray(cal['precision']['True']['noise_to_contrast'])),np.argmax(np.asarray(cal['precision']['True']['noise_to_contrast']))]])},'training_cost':training,'primary':{}}
for key,r in c['models'].items():
    o=r['original'];p=o['poses']['pt0'];rr=o['poses']['rest'];dp=np.array(p['directional_pairs']['F']['errors']);cp=r['fresh']['Scat'];p7=r['pair7_historically_inspected']['Scat']
    summary['primary'][key]=dict(static_primary_all=all(o['poses'][g]['static_useful']['F'] for g in ('rest','pt0')),physics_primary_all=all(o['poses'][g]['physics']['passed'] for g in ('rest','pt0')),FS_primary_all=all(o['poses'][g]['F_S_passed'] for g in ('rest','pt0')),pt0_absolute_coarse_mean=p['metrics']['F']['coarse']['mean'],pt0_absolute_fine16_mean=p['metrics']['F']['fine16']['mean'],rest_absolute_coarse_mean=rr['metrics']['F']['coarse']['mean'],delta=p['dynamic']['F'],direction_mean=float(dp.mean()),direction_failures=int((dp>.25).sum()),direction_pair7=dp[7].tolist(),fresh_scattered_mean=float(np.mean(cp['error'])),pair7_scattered_error=p7['error'][0],pair7_scattered_amplitude=p7['amplitude'][0],pair7_scattered_correlation=p7['correlation'][0],historical_baseline_direction_max_abs_drift=float(np.abs(r['historical_direction_error_change']).max()),max_primary_FS_error=max(o['poses'][g]['F_S']['max_relative_L1'] for g in ('rest','pt0')))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Four panels expose both gains and regressions, across both fixed seeds.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);colors={'A':'#406b9c','B':'#d46d38'}
for arm in ('A','B'):
    for seed,ls in ((5901,'-'),(5902,'--')):
        r=c['models'][f'{arm}_{seed}'];e=np.asarray(r['original']['poses']['pt0']['directional_pairs']['F']['errors']);fresh=np.asarray(r['fresh']['T']['error'])
        ax[0,0].plot(range(8),e.mean(1),ls,marker='o',color=colors[arm],alpha=.8,label=f'{arm} seed{seed}')
        ax[0,1].plot(range(8),fresh.mean(1),ls,marker='o',color=colors[arm],alpha=.8)
        h=read(ROOT/'arms'/arm/'out/training'/f'P_{seed}.json')['history'];ax[1,1].plot([v['step'] for v in h],[v['val'] for v in h],ls,color=colors[arm],alpha=.8)
for a,title in ((ax[0,0],'Original primary pairs: mean RGB error'),(ax[0,1],'Fresh sources on pt0: mean RGB error')):
    a.axhline(.25,color='gray',ls=':',label='Original per-channel limit .25');a.set(xlabel='Directional pair',ylabel='Relative L2',title=title,xticks=range(8));a.grid(alpha=.2)
ax[0,0].legend(fontsize=8);ax[0,0].axvspan(6.7,7.3,color='gray',alpha=.12)
xx=np.arange(3)
for i,(arm,seed) in enumerate((('A',5901),('A',5902),('B',5901),('B',5902))):
    d=c['models'][f'{arm}_{seed}']['original']['poses']['pt0']['dynamic']['F'];ax[1,0].bar(xx+(i-1.5)*.18,[z['error'] for z in d],.17,color=colors[arm],alpha=1 if seed==5901 else .5,label=f'{arm}/{seed}')
ax[1,0].axhline(.35,color='gray',ls=':');ax[1,0].set(xticks=xx,xticklabels=list('RGB'),ylabel='Relative L2',title='Original primary signed deformation');ax[1,0].legend(fontsize=8);ax[1,0].grid(axis='y',alpha=.2)
ax[1,1].set(xlabel='Optimizer step',ylabel='Original absolute validation loss',title='Validation: unchanged absolute-action metric');ax[1,1].grid(alpha=.2)
fig.suptitle('WL64: fixed representation and budget; one contrast-loss intervention\nA original objective; B adds fixed contrast loss; Pair7 historically inspected',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
(OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},indent=2)+'\n')
print(json.dumps(summary,indent=2))
