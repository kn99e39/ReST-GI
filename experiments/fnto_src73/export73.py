"""WL73 export: final/summary.json, group_metrics.csv, per_source.csv (with context and distances to the training populations),
matched S16 vs S16-R figure, Stage 0/1 records, reference manifests, manifest."""
import sys,json,csv,shutil,hashlib
from common73 import ROOT,write_new

def main():
    a=json.loads((ROOT/'eval/analysis.json').read_text());pj=json.loads((ROOT/'ports.json').read_text());F=ROOT/'final';F.mkdir(exist_ok=True);rows=[];ps=[]
    info={}
    for part in ('new','confirmation'):
        for s,v in pj[part].items():info[int(s)]=v
    for cand,r in a['results'].items():
        for g,v in r['groups'].items():
            if v.get('unmeasured'):rows.append(dict(model=cand,group=g,metric='UNMEASURED',channel='',rho='',hi='',lo='',signal_over_SE='',Q=''));continue
            for m,ch in v['pooled'].items():
                for c,x in enumerate(ch):rows.append(dict(model=cand,group=g,metric=m,channel='RGB'[c],rho=x['rho'],hi=x['hi'],lo=x['lo'],signal_over_SE=x['signal_over_SE'],Q=v['Q']['passed']))
        for k,v in r['per_matrix'].items():
            s=int(k.split('_s')[1]);d=info.get(s,{})
            for m,ch in v.items():
                for c,x in enumerate(ch):ps.append(dict(model=cand,matrix=k,slot=s,context=d.get('context',''),stratum=d.get('spec',{}).get('stratum',''),
                    d_S16R_actual_mm=d.get('d_to_S16R_actual_mm',''),d_S16R_images_mm=d.get('d_to_S16R_images_mm',''),d_S16_actual_mm=d.get('d_to_S16_actual_mm',''),
                    d_S16_images_mm=d.get('d_to_S16_images_mm',''),metric=m,channel='RGB'[c],hi=x,flux_5_10_20_40mm=json.dumps(r['flux'][k])))
    for name,data in (('group_metrics.csv',rows),('per_source.csv',ps)):
        with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    arch={'D1':(['D1_5901','D1_5902'],['D1R_5901','D1R_5902']),'A':(['A_5901','A_5902'],['AR_5901','AR_5902']),'B':(['B_7201','B_7202'],['BR_7201','BR_7202'])}
    groups=['TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_REMOVED_rest','TRAIN_REMOVED_pt0','BND_NEAR','BND_FAR','UNS','INT_ALL','REM_NEAR','FP_te0'];mets=['con_coarse','abs_coarse','abs_fine16']
    fig,ax=plt.subplots(3,3,figsize=(20,13))
    for i,(X,(old,new)) in enumerate(arch.items()):
        for k,m in enumerate(mets):
            for j,(lab,ns,mk) in enumerate((('S16',old,'o'),('S16-R',new,'s'))):
                for n in ns:
                    ys=[max(x['hi'] for x in a['results'][n]['groups'][g]['pooled'][m]) if g in a['results'][n]['groups'] and not a['results'][n]['groups'][g].get('unmeasured') else float('nan') for g in groups]
                    ax[i,k].scatter([t+(.15 if j else -.15) for t in range(len(groups))],ys,marker=mk,s=26,label=f'{lab} {n}')
            ax[i,k].axhline({'abs_coarse':.1,'abs_fine16':.15,'con_coarse':.25}[m],ls='--',c='k',lw=.8);ax[i,k].set_yscale('log');ax[i,k].set_xticks(range(len(groups)))
            ax[i,k].set_xticklabels(groups,rotation=40,fontsize=7,ha='right');ax[i,k].set_title(f'{X}: {m} (upper 95 %, worst channel)',fontsize=9)
        ax[i,0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(F/'src73.png',dpi=130);fig.savefig(F/'src73.pdf')
    for p,n in (('ports.json','ports.json'),('refs/complete_train.json','refs_complete_train.json'),('refs/complete_conf.json','refs_complete_conf.json'),
                ('refs/manifest.json','refs_manifest.json'),('stage1/summary.json','stage1_summary.json')):shutil.copy(ROOT/p,F/n)
    write_new(F/'summary.json',dict(decision=a['decision'],oracle=a['oracle_fixed_AO'],precision=a['precision'],reference_flux_profile=a['reference_flux_profile'],
        cost={n:r.get('cost') for n,r in a['results'].items()},runs={n:r.get('run') for n,r in a['results'].items()},
        deformation={n:{k:(v if v.get('unmeasured') else dict(Q=v['Q'],corr_slope=v['corr_slope'],abs_coarse=[x['hi'] for x in v['pooled']['abs_coarse']])) for k,v in r['deformation'].items()} for n,r in a['results'].items()}))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
