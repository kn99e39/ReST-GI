"""Report-only export of eval/analysis.json: CSV tables, a figure, summary.json and a file manifest under final/."""
import sys,os,json,csv,shutil
from pathlib import Path
from common68 import CFG,ROOT,sha,write_new

def main():
    a=json.loads((ROOT/'eval/analysis.json').read_text());F=ROOT/'final';F.mkdir(exist_ok=True);R=a['results']
    rows=[]
    for cand,r in R.items():
        for part in ('fit','heldout'):
            for grp,v in r[part].items():
                for met,chs in v['pooled'].items():
                    for c,x in enumerate(chs):rows.append(dict(candidate=cand,kind=r['kind'],population=part,group=grp,metric=met,channel='RGB'[c],
                        rho=x['rho'],lo=x['lo'],hi=x['hi'],signal_over_SE=x['signal_over_SE'],Q_passed=None if v['Q'] is None else v['Q']['passed']))
    with open(F/'pooled_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for cand,r in R.items():
        for key,v in r.get('bases',{}).items():
            for c,x in v.items():
                rows.append(dict(candidate=cand,group=key,channel='RGB'[int(c)],L7=x['L7']['rho'],F_A=x.get('F_A',{}).get('rho'),F_A_hi=x.get('F_A',{}).get('hi'),
                                 PAOc=x.get('PAOc',{}).get('rho'),PAOc_hi=x.get('PAOc',{}).get('hi')))
    with open(F/'basis_floors.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for t,v in a['teacher'].items():
        for st in ('initial','final'):
            for g,m in v['errors'][st].items():
                for met,chs in m.items():
                    for c,x in enumerate(chs):rows.append(dict(control=t,stage=st,geometry=g,metric=met,channel='RGB'[c],rel_error=x,passed=v['passed'],informative=v['informative']))
    with open(F/'teacher_controls.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for cand,r in R.items():
        for k,v in r.get('deformation',{}).items():
            for met,chs in v['pooled'].items():
                for c,x in enumerate(chs):rows.append(dict(candidate=cand,matrix=k,metric=met,channel='RGB'[c],rho=x['rho'],hi=x['hi'],corr=v['corr_slope'][c]['corr'],slope=v['corr_slope'][c]['slope']))
    if rows:
        with open(F/'deformation.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ext={c:dict(eligible=r['extended_fields']['F_path_eligible'],coarse_mean={g:r['extended_fields']['poses'][g]['metrics']['coarse']['mean'] for g in r['extended_fields']['poses']},
              pairs=r['extended_fields']['directional_pairs']['passed'],dynamic=[d['passed'] for d in r['extended_fields']['dynamic']],
              max_column_sum=max(p['max_colsum'] for p in r['passivity'].values()),physics_passed=all(p['passed'] for p in r['passivity'].values()))
         for c,r in R.items() if r['kind']=='net'}
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        cands=[c for c in R if R[c]['kind'] in ('net','coef') or c.startswith('cvx_J')];groups=['fit/rest','fit/pt0']+[f'heldout/{g}' for g in CFG['populations']['heldout']['groups']]
        fig,axes=plt.subplots(1,3,figsize=(18,5),sharey=True)
        for c,ax in enumerate(axes):
            for gi,g in enumerate(groups):
                part,grp=g.split('/');ys=[R[k][part].get(grp,{}).get('pooled',{}).get('con_coarse',[{}]*3)[c].get('hi',float('nan')) for k in cands]
                ax.plot(range(len(cands)),ys,'o',label=g,alpha=.8)
            ax.axhline(CFG['criteria']['quality']['con_coarse_hi'],color='k',lw=.8,ls='--');ax.set_xticks(range(len(cands)));ax.set_xticklabels(cands,rotation=90,fontsize=7)
            ax.set_title(f'coarse contrast rho (upper 95%), channel {"RGB"[c]}');ax.set_yscale('log')
        axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'repr68.png',dpi=130);fig.savefig(F/'repr68.pdf');plt.close(fig)
    except Exception as e:print('FIGURE_SKIPPED',repr(e),flush=True)
    for f in ('wl66_reproduction.json','heldout/cross_machine.json','heldout/ports.json','heldout/budget.json','heldout/preflight.json','heldout/complete.json','eval/analysis.json'):
        if (ROOT/f).exists():shutil.copy2(ROOT/f,F/Path(f).name)
    runs={}
    for p in sorted((ROOT/'runs').glob('*.pt')):
        import torch
        x=torch.load(p,map_location='cpu',weights_only=False);runs[p.stem]=dict(sha256=sha(p),seconds=x.get('seconds'),peak_GPU_allocated=x.get('peak_GPU_allocated'),
            J_initial=x.get('J_initial'),J_final=x.get('J_final'),converged=x.get('converged'),last_decile_relative_change=x.get('last_decile_relative_change'),trainable=x.get('trainable'),
            history=[(h['step'],h['J']) for h in x.get('history',[])][::8])
    write_new(F/'runs.json',runs)
    write_new(F/'summary.json',dict(decision=a['decision'],extended_and_passivity=ext,precision=a['precision'],seconds_eval=a['seconds']))
    write_new(F/'manifest.json',{p.name:sha(p) for p in sorted(F.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
