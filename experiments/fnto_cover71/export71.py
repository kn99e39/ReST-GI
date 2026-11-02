"""WL71 export: final/summary.json, group_metrics.csv, teacher.csv, per-matrix CSV, figure, copies of gate records."""
import sys,json,csv,shutil,hashlib
from pathlib import Path
from common71 import ROOT,write_new

def main():
    a=json.loads((ROOT/'eval/analysis.json').read_text());F=ROOT/'final';F.mkdir(exist_ok=True);rows=[]
    for cand,r in a['results'].items():
        for g,v in r['groups'].items():
            for m,ch in v['pooled'].items():
                for c,x in enumerate(ch):rows.append(dict(candidate=cand,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],signal_over_SE=x['signal_over_SE'],
                    Q_passed=v['Q']['passed'] if v.get('Q') else '',corr=v['corr_slope'][c]['corr'],slope=v['corr_slope'][c]['slope']))
    with open(F/'group_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    trow=[]
    for rid,t in a['teacher'].items():
        for st,per in t['errors'].items():
            for k,v in per.items():
                for m,vals in v.items():
                    for c,x in enumerate(vals):trow.append(dict(run=rid,stage=st,set=k,metric=m,channel='RGB'[c],value=x))
    with open(F/'teacher.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(trow[0]));w.writeheader();w.writerows(trow)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    groups=['S16_rest','S16_pt0','S16not4_rest','FIN_rest','FIN_cf2','DEF_cf2'];cands=list(a['results']);mets=['abs_coarse','abs_fine16','con_coarse']
    fig,ax=plt.subplots(1,3,figsize=(20,5.5))
    for k,m in enumerate(mets):
        for j,g in enumerate(groups):
            ys=[max(c['hi'] for c in a['results'][n]['groups'][g]['pooled'][m]) if g in a['results'][n]['groups'] and m in a['results'][n]['groups'][g]['pooled'] else float('nan') for n in cands]
            ax[k].scatter(range(len(cands)),ys,label=g,s=22)
        th={'abs_coarse':.10,'abs_fine16':.15,'con_coarse':.25}[m];ax[k].axhline(th,ls='--',c='k',lw=.8)
        ax[k].set_yscale('log');ax[k].set_xticks(range(len(cands)));ax[k].set_xticklabels(cands,rotation=90,fontsize=7);ax[k].set_title(f'{m} (upper 95 %, worst channel)')
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'cover71.png',dpi=130);fig.savefig(F/'cover71.pdf')
    for p in ('ports.json','refs/budget71.json','refs/complete.json','refs/manifest.json'):shutil.copy(ROOT/p,F/Path(p).name)
    runs={r:{k:v for k,v in json.loads(json.dumps(x,default=str)).items() if k in ('J_final','converged','seconds','peak_GPU_allocated','concurrent')} for r,x in a['results'].items()}
    write_new(F/'summary.json',dict(decision=a['decision'],precision=a['precision'],teacher={k:{x:v[x] for x in v if x!='errors'} for k,v in a['teacher'].items()},runs=runs,groups=a['groups']))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
