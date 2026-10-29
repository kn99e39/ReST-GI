"""POST HOC report-only extraction/plots from completed files. No model execution.
Keep the machine verdicts untouched; clarify absolute/dynamic/directional scopes.
"""
import os,json,csv,hashlib
from pathlib import Path

ROOT=Path(os.environ.get('WL57',Path.home()/'rest_wl57_recovery'));OUT=ROOT/'out';DEST=OUT/'review'
def main():
    DEST.mkdir(exist_ok=True);h=json.load(open(OUT/'heldout.json'));cost=json.load(open(OUT/'cost.json'));tags=['D_5701','D_5702','R_5701','R_5702'];summary={};dynamic=[];angular=[];lifetimes=[]
    for tag in tags:
        z=h[tag];poses=z['poses'];ds=[];af=an=0
        for pose,p in poses.items():
            for c,q in enumerate(p.get('dynamic',[])):
                dynamic.append(dict(model=tag,pose=pose,channel='RGB'[c],**q));ds.append(q)
            if 'directional_pairs' in p:
                q=p['directional_pairs']
                for i in range(8):
                    for c in range(3):
                        detect=q['detectable'][i][c];error=q['error'][i][c];passed=(not detect) or error<=.25
                        angular.append(dict(model=tag,pose=pose,pair=i,channel='RGB'[c],detectable=detect,relative_L2=error,passed=passed));an+=int(detect);af+=int(detect and not passed)
        t=json.load(open(OUT/'training'/f'{tag}.json'))
        summary[tag]=dict(parameters=z['parameters'],static_norms_pass=all(p['static_useful'] for p in poses.values()),hard_physics_pass=all(p['physics']['passed'] for p in poses.values()),
            max_coarse_mean=max(max(p['metrics']['coarse']['mean']) for p in poses.values()),max_coarse_p90=max(max(p['metrics']['coarse']['p90']) for p in poses.values()),max_fine16_mean=max(max(p['metrics']['fine16']['mean']) for p in poses.values()),max_fine64_mean=max(max(p['metrics']['fine64']['mean']) for p in poses.values()),
            max_reference_noise_coarse_mean=max(max(p['oracle_seed_noise']['coarse']['mean']) for p in poses.values()),max_stale_coarse_mean=max(max(p['stale_rest_control']['coarse']['mean']) for p in poses.values() if 'stale_rest_control' in p),
            dynamic_pass=sum(q['pass_dynamic'] for q in ds),dynamic_total=len(ds),max_dynamic_error=max(q['relative_L2'] for q in ds),min_dynamic_correlation=min(q['corr'] for q in ds),
            angular_failed=af,angular_detectable=an,eligible=z['useful_and_physical'],steps=t['steps'],training_seconds=t['seconds'],training_peak_GPU_bytes=t['peak_GPU_bytes'],first_validation=t['history'][0]['val'],last_validation=t['history'][-1]['val'])
    for pose,rs in cost.items():
        for seed in (5701,5702):
            d,r=rs[f'D_{seed}'],rs[f'R_{seed}']
            for N in ('1','4','16','64'):
                a,b=d['queries'][N],r['queries'][N]
                lifetimes.append(dict(pose=pose,seed=seed,queries=int(N),D_seconds=a['best'],R_seconds=b['best'],ratio=b['best']/a['best'],D_policy=a['selected'],R_policy=b['selected'],
                    common_geometry_seconds=d['geometry']['median'],D_learned_lifecycle=a[a['selected']+'_lifecycle']['median'],R_learned_lifecycle=b[b['selected']+'_lifecycle']['median'],
                    D_cache_bytes=d['cache_policy_total_bytes'][a['selected']],R_cache_bytes=r['cache_policy_total_bytes'][b['selected']],uncontended=d['uncontended'] and r['uncontended']))
    for name,rows in [('dynamic_all',dynamic),('directional_all',angular),('cost_all',lifetimes),('model_summary',[dict(model=k,**v) for k,v in summary.items()])]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with open(DEST/f'{name}.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    verdict=dict(absolute_action_norm_contract='SUPPORTED',directional_contrast_contract='NOT SUPPORTED',deformation_response_contract='NOT SUPPORTED',hard_zero_linearity_positivity_passivity_cache_constraints='SUPPORTED',
        both_fixed_recipes_eligible='NOT SUPPORTED',R_advantage_under_declared_rule='NOT SUPPORTED',quality_matched_D_vs_R_runtime_winner='INCONCLUSIVE',general_abstract_Rule_State_hypothesis='INCONCLUSIVE',
        reason='Both frozen recipes fail dynamic and angular gates. Raw cost ratio is measured but neither qualifies as a useful winner. Common angular/spatial basis, finite ballistic quadrature, noisy reused training supervision and optimization are not causally separated. No new training, architecture or threshold is authorised.')
    (DEST/'review_summary.json').write_text(json.dumps(dict(kind='POST_HOC_REPORT_ONLY',primary_source='e809c15',models=summary,verdict=verdict),indent=1)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(4);ax=axs[0,0]
    ax.bar(x-.18,[100*summary[t]['max_coarse_mean'] for t in tags],.36,label='1cm /16out',color='#4477aa');ax.bar(x+.18,[100*summary[t]['max_fine16_mean'] for t in tags],.36,label='2.5mm /16out',color='#eeaa33')
    ax.axhline(10,color='#4477aa',ls='--',lw=1);ax.axhline(15,color='#cc8811',ls='--',lw=1);ax.set(xticks=x,xticklabels=tags,ylim=(0,18),ylabel='Worst mean relative L2 (%)',title='Absolute K-action norm bounds pass');ax.legend(fontsize=8)
    ax=axs[0,1];counts=[summary[t]['angular_failed'] for t in tags];ax.bar(tags,counts,color=['#4477aa']*2+['#cc6677']*2)
    for i,n in enumerate(counts):ax.text(i,n+1,str(n),ha='center')
    ax.set(ylim=(0,100),ylabel='Failed pair/channel cases (of96)',title='Directional contrast: every model fails');ax.text(.02,.94,'0 failures required; same physical fixtures repeat across poses',transform=ax.transAxes,fontsize=8,va='top')
    ax=axs[1,0];poses=['cf0','cf1','cf2','cf3'];values=[];failed=[]
    for arm in ('D','R'):
        for p in poses:
            values.append([max(h[f'{arm}_{seed}']['poses'][p]['dynamic'][c]['relative_L2'] for seed in (5701,5702)) for c in range(3)])
            failed.append([not all(h[f'{arm}_{seed}']['poses'][p]['dynamic'][c]['pass_dynamic'] for seed in (5701,5702)) for c in range(3)])
    im=ax.imshow(values,vmin=0,vmax=.7,cmap='RdYlGn_r',aspect='auto')
    for i in range(8):
        for c in range(3):ax.text(c,i,f'{values[i][c]:.2f}'+(' x' if failed[i][c] else ''),ha='center',va='center',fontsize=8)
    ax.set(xticks=range(3),xticklabels=list('RGB'),yticks=range(8),yticklabels=[a+' '+p for a in ('D','R') for p in poses],title='Delta-action error (worst fixed seed)')
    fig.colorbar(im,ax=ax,label='Relative L2 of deformation response; x = fail')
    ax=axs[1,1];groups=[[r['ratio'] for r in lifetimes if r['queries']==n] for n in (1,4,16,64)]
    ax.boxplot(groups,labels=['1','4','16','64'],whis=(0,100));ax.axhline(.75,color='#cc6677',ls='--',label='25% R speed benefit');ax.axhline(1,color='#666666',lw=1)
    ax.set(ylim=(.70,1.04),xlabel='Queries after geometry edit',ylabel='R / D raw lifecycle time',title='No measured cost gain; no qualified winner');ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('WL57 A2/A3: reference recovered; static norms pass, dynamic/angular fidelity fails',fontsize=12)
    fig.savefig(DEST/'summary.png',dpi=180);fig.savefig(DEST/'summary.pdf');plt.close(fig)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
if __name__=='__main__':main()
