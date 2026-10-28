"""Post-run WL56 reporting only. Preserve analysis.json; no primary criterion changes.

Cost sensitivity A1 separates new-package construction from cache overhead; the
original illustrative projection double-counted the dirty package identity hash.
"""
import sys,json,csv,hashlib,shutil,math
from pathlib import Path
import numpy as np
import s56 as S
import s56_fixtures as F

def median(v):return float(np.median(list(v)))

def report():
    a=json.load(open(S.OUT/'analysis.json')); costs=a['cost']; summary={}
    f=[v for k,v in costs.items() if k in ('dent_F_r0','dent_F_r1')]
    s=[v for k,v in costs.items() if k in ('dent_S_r0','dent_S_r1')]
    def component(v,key):return sum(p.get(key,0) for p in v['packages'].values() if p['recomputed'])
    def pkg(v):return component(v,'complete_build_wall')+sum(p['identity_check_seconds'] for p in v['packages'].values() if p['recomputed'])
    hF=median(v['total_update_seconds']-pkg(v) for v in f);hS=median(v['total_update_seconds']-pkg(v) for v in s)
    summary['cost_A1']=dict(label='POST HOC ACCOUNTING FIX, primary measured timings unchanged; analysis.json retained',
       reason='Original sensitivity included dirty post-build identity hashing in both the hypothetical new-package k and H_S. Use disjoint components here.',
       new_package_definition='Complete build wall + post-build identity check, per recomputed domain. Includes construction, setup, serialization, hashing.',
       fixed_raw_cache_model=dict(H_F=hF,H_S=hS,break_even_20pct=max(0,(1.25*hS-hF)/.75),
          curve=[dict(k=k,F=2*k+hF,S=k+hS,speedup=(2*k+hF)/(k+hS)) for k in (.001,.01,.1,1,10,100)],
          caveat='Holding 1.68 GB raw-cache access fixed is not a prediction for a smaller learned K. Millisecond new-package creation is incompatible with current serialization/hash bytes.'),
       remaining_TR=median(component(v,'TR_seconds') for v in s),
       remaining_complete_new_package=median(pkg(v) for v in s),
       TR_fraction_of_S=median(component(v,'TR_seconds')/v['total_update_seconds'] for v in s),
       new_package_fraction_of_S=median(pkg(v)/v['total_update_seconds'] for v in s),
       residual_if_TR_free=median(v['total_update_seconds']-component(v,'TR_seconds') for v in s))
    H=median(r['meta']['update_shared_seconds'] for r in a['resident_rigid']['records'])
    summary['cost_A1']['conditional_compact_resident_model']=dict(H=H,origin='Measured RIGID resident update, used only as illustrative H; dent compact/learned model unmeasured',
      formula='F=2k+H, S=k+H+O; >=20% saving iff k >= (0.25H+1.25O)/0.75',
      assumed_O=0,illustrative_break_even_20pct=H/3,
      curve=[dict(k=k,speedup=(2*k+H)/(k+H)) for k in (.001,.01,.1,1,10,100)])
    summary['primary_counts']={}
    for state,cams in a['fidelity'].items():
        summary['primary_counts'][state]={}
        for unit in ('amended','declared'):
            for arm in ('F','S','T'):
                for mode in ('nee_ball','bin'):
                    rows=[x for cam,us in cams.items() for x in us[unit] if x['primary'] and x['arm']==arm and x['mode']==mode]
                    if rows:summary['primary_counts'][state][f'{unit}|{arm}|{mode}']=dict(passed=sum(sum(x['passed']) for x in rows),total=sum(sum(x['ruled']) for x in rows))
    summary['cost_summary']=a['cost_summary'];summary['shared_F_S_all_passed']=all(v['passed'] for v in a['equivalence'].values())
    summary['max_shared_relative_L1']=max(v['relative_L1'] for r in a['equivalence'].values() for v in r['fields'].values())
    summary['independent_clean_all_passed']=all(all(all(q) for q in r['passed']) for r in a['independent_clean'])
    summary['historical_tests_note']='WL54 full runner reported 10/10 including one skip; 9 actually performed, then the skipped mask check ran separately and passed. Keep original records.'
    S.dump(summary,S.OUT/'report_summary.json')
    with open(S.OUT/'all_groups.csv','w') as ff:
        w=csv.writer(ff);w.writerow(['state','camera','unit','arm','readout','group','channel','n_ports','primary','ruled','passed','bias','SE','abs_radiance_error','relative_L2','p99_abs_error'])
        for st,cams in a['fidelity'].items():
            for cam,units in cams.items():
                for unit,rows in units.items():
                    for r in rows:
                        for c,ch in enumerate('RGB'):w.writerow([st,cam,unit,r['arm'],r['mode'],r['group'],ch,r['n_ports'],r['primary'],r['ruled'][c],r['passed'][c],r['relative_bias'][c],r['se_total'][c],r['mean_absolute_radiance_error'][c],r['relative_L2'][c],r['absolute_error_p99'][c]])
    figdir=S.OUT/'figures';figdir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for i,(arm,vs) in enumerate((('Full',f),('Selective',s))):
        tr=median(component(v,'TR_seconds') for v in vs);tt=median(v['total_update_seconds'] for v in vs)
        axes[0].bar(i,tr,color='#2a6fbb',label='TR construction' if i==0 else None)
        axes[0].bar(i,tt-tr,bottom=tr,color='#edb458',label='All other update work' if i==0 else None)
        axes[0].text(i,tt+5,f'{tt:.1f} s',ha='center')
    axes[0].set(xticks=[0,1],xticklabels=['Full','Selective'],ylabel='Actual measured update (s)',title='Local dent: same representation')
    axes[0].set_ylim(0,420);axes[0].legend(fontsize=8)
    ks=np.logspace(-3,3,200)
    axes[1].semilogx(ks,(2*ks+hF)/(ks+hS),label='Fixed raw-cache model (A1)')
    axes[1].semilogx(ks,(2*ks+H)/(ks+H),label='Compact resident illustration')
    axes[1].axhline(1.25,color='gray',ls='--',lw=1);axes[1].axhline(1,color='gray',lw=.5)
    axes[1].set(xlabel='Hypothetical complete K package cost / domain (s)',ylabel='Full / Selective time',title='Analytic sensitivity; no learned model measured')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'cost_and_headroom.png');plt.close(fig)
    # Coarse boundary response, fixed receiver, green channel. Reference is independently rendered.
    ref0=S.extract('rest')['gapdown']['amended'];ref1=S.extract('dent')['gapdown']['amended'];lab=S.CM.labels_of(F.scene('dent'))
    mask=ref0['sel']&ref1['sel']&(lab=='lower:top');m=F.intrinsic('rest',0)[0];xy=m['port_center'];sel=mask[:240]
    zz={}
    for arm,state in (('S','dent'),('T','dent'),('rest','rest')):
        ar='S' if arm=='rest' else arm
        preds=[]
        for r in (0,1):
            z=np.load(S.OUT/'runs'/f'{state}_{ar}_r{r}.npz');preds.append(z['top|bin'][:,1])
        zz[arm]=np.mean(preds,0)
    base=ref0['mean'][:,1].mean();delta_ref=ref1['mean'][:,1]-ref0['mean'][:,1]
    vals=[delta_ref[:240][sel],(zz['S']-zz['rest'])[:240][sel],(zz['T']-zz['rest'])[:240][sel]]
    vmax=max(float(abs(v).max()) for v in vals);fig,axs=plt.subplots(1,3,figsize=(10,3.5),sharex=True,sharey=True)
    for ax,v,title in zip(axs,vals,['Cycles signed change','Selective signed change','Stale signed change']):
        im=ax.scatter(xy[sel,0]*100,xy[sel,1]*100,c=v,cmap='RdBu_r',vmin=-vmax,vmax=vmax,s=170,marker='s')
        ax.set(title=title,xlabel='Receiver x (cm)',aspect='equal')
    axs[0].set_ylabel('Receiver y (cm)');fig.colorbar(im,ax=axs,label='Green radiance change',shrink=.7)
    fig.suptitle('Unchanged lower slab: complete K reused, radiance solved again');fig.subplots_adjust(top=.82,right=.84,wspace=.3)
    fig.savefig(figdir/'stationary_receiver_change.png',bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary['cost_A1'],indent=1))

def export():
    dst=S.OUT/'export';dst.mkdir(exist_ok=True)
    for name in ('analysis.json','report_summary.json','all_groups.csv','reference_gate.json','initial_cache.json','environment.json','resident_rigid.json'):
        shutil.copy2(S.OUT/name,dst/name)
    for p in (S.OUT/'figures').glob('*.png'):shutil.copy2(p,dst/p.name)
    pred=dst/'coarse_predictions';pred.mkdir(exist_ok=True)
    for p in sorted((S.OUT/'runs').glob('*.npz')):
        z=np.load(p);np.savez(pred/p.name,**{k:z[k] for k in z.files if not k.startswith('phi_')})
    # Compact physical-reference boundary arrays preserve every seed and both masks.
    for state in ('rest','dent','rigid'):
        data={}
        for cam,units in S.extract(state).items():
            for unit,u in units.items():
                for k in ('seeds','mean','sel','cnt','cov'):data[f'{cam}|{unit}|{k}']=u[k]
        np.savez(dst/f'boundary_reference_{state}.npz',**data)
    rows=[]
    for root in (S.OPS,S.CACHE,S.OUT/'cycles',S.OUT/'runs'):
        for p in sorted(root.rglob('*')):
            if p.is_file():rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=S.hash_file(p)))
    S.dump(dict(server='LabServer69',files=rows,code_commit=(S.ROOT/'repo_commit.txt').read_text().strip()),dst/'raw_artifact_manifest.json')
    print('EXPORTED',len(rows))

if __name__=='__main__':{'report':report,'export':export}[sys.argv[1]]();sys.stdout.flush();__import__('os')._exit(0)
