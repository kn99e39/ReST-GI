"""WL75 export: final/summary.json (gate, Stage 3 decision, cost, run records, analysis summaries), group_metrics.csv (every model / stage /
group / metric / channel), per_matrix.csv (Stage 3 per matrix incl. oracles), figure, copies of the stage records, manifest."""
import sys,os,json,csv,shutil,hashlib
from common75 import CFG,ROOT,write_new

def main():
    F=ROOT/'final';F.mkdir(exist_ok=True);s2=json.loads((ROOT/'stage2/analysis.json').read_text())
    p3=ROOT/'stage3/analysis.json';s3=json.loads(p3.read_text()) if p3.exists() else None
    an=json.loads((ROOT/'analysis/analysis.json').read_text());cost=json.loads((ROOT/'cost/cost.json').read_text());rows=[];pm=[]
    for st,data in (('stage2',s2),('stage3',s3)):
        if not data:continue
        for n,r in data['results'].items():
            for g,v in r['groups'].items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],Q=v['Q']['passed']))
        for b,o in data['oracle'].items():
            for g,v in o['groups'].items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=f'oracle_{b}',group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],Q=None))
    if s3:
        for n,r in s3['results'].items():
            for split,key in (('r0123','per_matrix'),('r123','per_matrix_r123')):
                for k,v in r.get(key,{}).items():
                    for m,ch in v.items():
                        for c,x in enumerate(ch):pm.append(dict(model=n,replicas=split,matrix=k,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
        for b,o in s3['oracle'].items():
            for k,v in o['per_matrix'].items():
                for m,ch in v.items():
                    for c,x in enumerate(ch):pm.append(dict(model=f'oracle_{b}',replicas='r123',matrix=k,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
    for name,data in (('group_metrics.csv',rows),('per_matrix.csv',pm)):
        if data:
            with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    W=lambda r,g,m:max(x['hi'] for x in r['groups'][g]['pooled'][m])
    panels=[('stage2',g) for g in ('TRAIN_NEW_rest','TRAIN_KEEP_rest','BND_NEAR','BND_FAR','UNS','INT_ALL')]+([('stage3',g) for g in ('BND4_NEAR','BND4_FAR','INT4',f"FP4_{CFG['stage3']['pose']}")] if s3 else [])
    models=['AR_5901','AR_5902','A75_5901','A75_5902'];fig,ax=plt.subplots(1,3,figsize=(21,5.5))
    for k,m in enumerate(('abs_coarse','abs_fine16','con_coarse')):
        for j,n in enumerate(models+['oracle_G0_r0',f"oracle_{CFG['basis']['selected']}"]):
            ys=[];xs=[]
            for t,(st,g) in enumerate(panels):
                d=s2 if st=='stage2' else s3
                r=d['results'].get(n) if not n.startswith('oracle_') else d['oracle'].get(n[7:])
                if r and g in r['groups'] and not r['groups'][g].get('unmeasured'):ys.append(W(r,g,m));xs.append(t+(j-3)*.1)
            ax[k].scatter(xs,ys,s=20,label=n,marker='o' if n.startswith('AR') else '^' if n.startswith('A75') else 'x')
        ax[k].axhline({'abs_coarse':.1,'abs_fine16':.15,'con_coarse':.25}[m],ls='--',c='k',lw=.8);ax[k].set_yscale('log');ax[k].set_xticks(range(len(panels)))
        ax[k].set_xticklabels([f'{g}' for _,g in panels],rotation=35,ha='right',fontsize=8);ax[k].set_title(f'{m} (upper 95 %, worst channel)',fontsize=9)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'rank75.png',dpi=130);fig.savefig(F/'rank75.pdf')
    for p in ('stage2/gate.json','ports4.json','probe/lanes.json','cost/cost.json','tests/tests.json'):
        if (ROOT/p).exists():shutil.copy(ROOT/p,F/p.replace('/','_'))
    if (ROOT/'refs/complete.json').exists():shutil.copy(ROOT/'refs/complete.json',F/'refs_complete.json')
    runs={n:s2['results'][n].get('run') for n in ('A75_5901','A75_5902')}
    write_new(F/'summary.json',json.loads(json.dumps(dict(gate=s2['gate'],stage3=s3['decision'] if s3 else None,cost=dict(ratio=cost['ratio_rank16_over_rank8'],cost_ratio_exceeds=cost['cost_ratio_exceeds']),
        runs=runs,orientation=an.get('orientation',{}).get('summary'),by_subcell=an.get('by_subcell'),by_face=an.get('by_face'),by_context=an.get('by_context'),
        oracle_vs_learned_groups=an.get('oracle_vs_learned_groups'),drift=an.get('drift'),stage2_seconds=s2['seconds'],stage3_seconds=s3['seconds'] if s3 else None),default=float)))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
