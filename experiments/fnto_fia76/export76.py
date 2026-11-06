"""WL76 export: final/summary.json, group_metrics.csv, per_matrix.csv (Stage 4), figure, copies of the stage records, manifest."""
import sys,os,json,csv,shutil,hashlib
from common76 import CFG,ROOT,write_new

def main():
    F=ROOT/'final';F.mkdir(exist_ok=True);s1=json.loads(__import__('pathlib').Path(os.environ.get('R76_STAGE1',str(ROOT/'stage1/physics.json'))).read_text());s3=json.loads((ROOT/'stage3/analysis.json').read_text())   # s1: the dev smoke reads the evidence Stage 1 record (R76_STAGE1)
    p4=ROOT/'stage4/analysis.json';s4=json.loads(p4.read_text()) if p4.exists() else None;rows=[];pm=[]
    for st,data in (('stage3',s3),('stage4',s4)):
        if not data:continue
        for n,r in list(data['results'].items())+[(f'oracle_{b}',o) for b,o in data['oracle'].items()]:
            for g,v in r['groups'].items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
    if s4:
        for n,r in list(s4['results'].items())+[(f'oracle_{b}',o) for b,o in s4['oracle'].items()]:
            for key in ('per_matrix','per_matrix_r123'):
                for k,v in r.get(key,{}).items():
                    for m,ch in v.items():
                        for c,x in enumerate(ch):pm.append(dict(model=n,replicas='r0123' if key=='per_matrix' and not n.startswith('oracle') else 'r123',matrix=k,metric=m,channel='RGB'[c],hi=x['hi'],rho=x['rho']))
    for name,data in (('group_metrics.csv',rows),('per_matrix.csv',pm)):
        if data:
            with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    Wm=lambda r,g,m:max(x['hi'] for x in r['groups'][g]['pooled'][m])
    panels=[('stage3',g) for g in ('TRAIN_NEW_rest','TRAIN_KEEP_rest','BND_NEAR','BND_FAR','INT_ALL')]+([('stage4',g) for g in ('BND_NEAR','BND_FAR','INT','FP_rest',f"FP_{CFG['stage4']['pose']}")] if s4 else [])
    models=['AR_5901','AR_5902','A75_5901','A75_5902']+list(CFG['runs'])+['oracle_G0_r0','oracle_G2_r0'];fig,ax=plt.subplots(1,3,figsize=(22,5.5))
    for k,m in enumerate(('abs_coarse','abs_fine16','con_coarse')):
        for j,n in enumerate(models):
            xs=[];ys=[]
            for t,(st,g) in enumerate(panels):
                d=s3 if st=='stage3' else s4;r=d['oracle'].get(n[7:]) if n.startswith('oracle_') else d['results'].get(n)
                if r and g in r['groups'] and not r['groups'][g].get('unmeasured'):xs.append(t+(j-4)*.08);ys.append(Wm(r,g,m))
            ax[k].scatter(xs,ys,s=20,label=n,marker='o' if n.startswith('AR') else '^' if n.startswith('A75') else 's' if n.startswith('F76') else 'x')
        ax[k].axhline({'abs_coarse':.1,'abs_fine16':.15,'con_coarse':.25}[m],ls='--',c='k',lw=.8);ax[k].set_yscale('log');ax[k].set_xticks(range(len(panels)))
        ax[k].set_xticklabels([f'{st[-1]}:{g}' for st,g in panels],rotation=35,ha='right',fontsize=8);ax[k].set_title(f'{m} (upper 95 %, worst channel)',fontsize=9)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'fia76.png',dpi=130);fig.savefig(F/'fia76.pdf')
    for p in ('stage1/physics.json','stage3/gate.json','ports76.json','probe/lanes.json','tests/tests.json','refs/complete.json'):
        if (ROOT/p).exists():shutil.copy(ROOT/p,F/p.replace('/','_'))
    write_new(F/'summary.json',json.loads(json.dumps(dict(stage1=s1['gate'],stage3_gate=s3['gate'],cost=s3['cost'],stage4=s4['decision'] if s4 else None,mechanism=s4['mechanism'] if s4 else None,
        runs={n:s3['results'][n].get('run') for n in CFG['runs']}),default=float)))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
