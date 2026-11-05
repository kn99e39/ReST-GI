"""WL74 export: final/summary.json, bases.csv (Stage 1/2 group bounds per basis), per_matrix.csv, figure, manifest."""
import sys,json,csv,shutil,hashlib
from common74 import ROOT,write_new
import ato74 as AT

def main():
    F=ROOT/'final';F.mkdir(exist_ok=True);s1=json.loads((ROOT/'stage1/scores.json').read_text());ids=AT.basis_ids()
    s2={b:json.loads((ROOT/'stage2'/f'{b}.json').read_text()) for b in ids};dec=json.loads((ROOT/'decision.json').read_text());rows=[];pm=[]
    for b in ids:
        for src,grs in (('stage1',s1[b]['groups']),('stage2',s2[b]['groups'])):
            for g,v in grs.items():
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(basis=b,stage=src,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],Q=v['Q']['passed']))
        for src,per in (('stage1',s1[b]['per_matrix']),('stage2',s2[b]['per_matrix'])):
            for k,v in per.items():
                for m,ch in v.items():
                    for c,x in enumerate(ch):pm.append(dict(basis=b,stage=src,matrix=k,metric=m,channel='RGB'[c],hi=x))
    for name,data in (('bases.csv',rows),('per_matrix.csv',pm)):
        with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    groups=[('stage1','TRAIN_NEW_rest'),('stage1','TRAIN_NEW_pt0'),('stage1','TRAIN_KEEP_rest'),('stage2','BND_NEAR'),('stage2','BND_FAR'),('stage2','UNS'),('stage2','INT_DED'),('stage2','REM_NEAR'),('stage2','FP_te0')]
    fig,ax=plt.subplots(1,3,figsize=(21,5.5))
    for k,m in enumerate(('abs_coarse','abs_fine16','con_coarse')):
        for j,b in enumerate(ids):
            ys=[max(x['hi'] for x in (s1 if st=='stage1' else s2)[b]['groups'][g]['pooled'][m]) for st,g in groups]
            ax[k].scatter([t+(j-len(ids)/2)*.08 for t in range(len(groups))],ys,s=18,label=b,marker='o' if b.startswith('G0') else 's' if b.startswith('G1') else '^')
        ax[k].axhline({'abs_coarse':.1,'abs_fine16':.15,'con_coarse':.25}[m],ls='--',c='k',lw=.8);ax[k].set_yscale('log');ax[k].set_xticks(range(len(groups)))
        ax[k].set_xticklabels([g for _,g in groups],rotation=35,ha='right',fontsize=8);ax[k].set_title(f'oracle {m} (upper 95 %, worst channel)',fontsize=9)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'ato74.png',dpi=130);fig.savefig(F/'ato74.pdf')
    e4=json.loads((ROOT/'eval4/analysis.json').read_text())['decision'] if (ROOT/'eval4/analysis.json').exists() else None
    for p in ('ports4.json','decision.json','stage1/scores.json'):shutil.copy(ROOT/p,F/p.replace('/','_'))
    if (ROOT/'refs/complete.json').exists():shutil.copy(ROOT/'refs/complete.json',F/'refs4_complete.json')
    write_new(F/'summary.json',dict(decision=dec,stage4=e4,bases={b:dict(J_final=s1[b]['J_final'],seconds=s1[b]['seconds'],parameters=s1[b]['parameters'],passivity=s1[b]['passivity'],
        convergence_G0=s1[b]['convergence_G0'],stage2_convergence=s2[b]['convergence']) for b in ids}))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
