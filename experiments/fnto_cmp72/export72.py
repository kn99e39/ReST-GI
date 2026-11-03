"""WL72 export: final/summary.json, group_metrics.csv, per_source.csv, comparison figure, gate records, manifest."""
import sys,json,csv,shutil,hashlib
from pathlib import Path
from common72 import ROOT,write_new

def main():
    a=json.loads((ROOT/'eval/analysis.json').read_text());F=ROOT/'final';F.mkdir(exist_ok=True);rows=[];ps=[]
    for cand,r in a['results'].items():
        for g,v in r['groups'].items():
            if v.get('unmeasured'):continue
            for m,ch in v['pooled'].items():
                for c,x in enumerate(ch):rows.append(dict(candidate=cand,group=g,metric=m,channel='RGB'[c],rho=x['rho'],hi=x['hi'],signal_over_SE=x['signal_over_SE'],Q=v['Q']['passed']))
        for k,v in r['per_matrix'].items():
            for m,ch in v.items():
                for c,x in enumerate(ch):ps.append(dict(candidate=cand,matrix=k,metric=m,channel='RGB'[c],hi=x,flux_1cm=r['flux'][k]))
    for name,data in (('group_metrics.csv',rows),('per_source.csv',ps)):
        with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cands=[c for c in a['results'] if '@' not in c];groups=['S16_rest','FIN71_rest','FIN71_cf2','H3_cf3','CONF_rest','CONF_te2'];mets=['abs_coarse','abs_fine16','con_coarse']
    fig,ax=plt.subplots(1,3,figsize=(20,5))
    for k,m in enumerate(mets):
        for g in groups:
            ys=[max(x['hi'] for x in a['results'][n]['groups'][g]['pooled'][m]) if not a['results'][n]['groups'][g].get('unmeasured') else float('nan') for n in cands]
            ax[k].scatter(range(len(cands)),ys,label=g,s=22)
        ax[k].axhline({'abs_coarse':.1,'abs_fine16':.15,'con_coarse':.25}[m],ls='--',c='k',lw=.8);ax[k].set_yscale('log');ax[k].set_xticks(range(len(cands)));ax[k].set_xticklabels(cands,rotation=90,fontsize=7);ax[k].set_title(f'{m} (upper 95 %, worst channel)')
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'cmp72.png',dpi=130);fig.savefig(F/'cmp72.pdf')
    for p,n in (('ports.json','ports.json'),('refs/complete.json','refs_complete.json'),('refs/manifest.json','refs_manifest.json'),('geom/variants.json','geom_variants.json')):shutil.copy(ROOT/p,F/n)
    write_new(F/'summary.json',dict(decision=a['decision'],oracle=a['oracle_fixed_AO'],precision=a['precision'],cost={n:r.get('cost') for n,r in a['results'].items()},
        runs={n:{k:r.get(k) for k in ('J_final','loss_final','L_sym_final','converged','seconds','peak_GPU_allocated','trainable')} for n,r in a['results'].items()}))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
