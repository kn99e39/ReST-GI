"""Reviewer tables/figure from analysis.json (no new measurement)."""
import sys,json,csv,os
import numpy as np
from common66 import CFG,ROOT,GEOMS,sha,write_new,ports

def rows_csv(path,header,rows):
    with open(path,'w',newline='\n') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(rows)

def main():
    a=json.loads((ROOT/'analysis.json').read_text());out=ROOT/'review';out.mkdir(exist_ok=True);pr=ports();cks=a['checkpoints']
    names=['S_C','S_Z','L7_rel','U7_rel','L7_absrel','U7_absrel','OGC_L8_rel','OGC_U8_rel']+[f'{ck}/{s}' for ck in cks for s in ('F_A_rel','PAOc_rel','POc_rel','E_Mc_rel','F_A_absZ_rel','PAOz_rel','POz_rel','E_Mz_rel','E_raw_scatter_rel','E_raw_ball_rel','E_raw_total_rel','FA_over_EM','FAO_over_EM')]
    def flat(prefix,d):return [prefix+[n,d[n]['est'],d[n]['se'],d[n]['lo'],d[n]['hi']] for n in names if n in d]
    rows=[];[rows.extend(flat(k.split('/'),v)) for k,v in a['pooled'].items()]
    rows_csv(out/'pooled.csv',['space','channel','statistic','estimate','jackknife_SE','lo95','hi95'],rows)
    rows=[];[rows.extend(flat(k.split('/'),v)) for k,v in a['pooled_by_geometry'].items()]
    rows_csv(out/'pooled_by_geometry.csv',['space','channel','geometry','statistic','estimate','jackknife_SE','lo95','hi95'],rows)
    rows=[]
    for k,v in a['per_matrix'].items():
        g,s,c,sp=k.split('/');rows.extend(flat([g,s,pr['ports'][int(s)]['port'],'RGB'[int(c)],sp],v))
    rows_csv(out/'per_matrix.csv',['geometry','slot','port','channel','space','statistic','estimate','jackknife_SE','lo95','hi95'],rows)
    rows=[]
    for src,dd in (('pooled',a['pooled']),('matrix',a['per_matrix'])):
        for k,v in dd.items():
            if 'coarse' not in k:continue
            for m in range(16):e=v[f'eig{m}'];rows.append([src,k,m,e['est'],e['se'],e['est']>3*e['se']])
    rows_csv(out/'contrast_spectrum_coarse.csv',['population','key','index','eigenvalue','jackknife_SE','resolved_gt_3SE'],rows)
    rows=[]
    for src,dd in (('pooled',a['pooled']),('matrix',a['per_matrix'])):
        for k,v in dd.items():
            if 'coarse' not in k:continue
            for ck in a['primary_checkpoints']:
                for n in v:
                    if n.startswith(ck+'/pair') and n.endswith('_phys'):
                        base=n[:-5];ph=v[n]['est'];pa=base.split('pair')[1]
                        rows.append([src,k,ck,pa,ph,v[base+'_FA']['est']/max(ph,1e-300),v[base+'_EM']['est']/max(ph,1e-300),pa=='8_13'])
    rows_csv(out/'bin_pair_contrasts_coarse.csv',['population','key','checkpoint','bins','phys_contrast_energy','frozenA_rel_energy','model_rel_energy','Pair7_bins_historically_inspected'],rows)
    rows=[]
    for k,v in a['invariants'].items():
        for b in range(16):rows.append([k,b,v['zero_collision_fraction_phys'][b],v['beta_model'][b],*v['collided_energy_RGB'][b],*v['source_cell_collided_share_RGB'][b]])
    rows_csv(out/'probe_invariants.csv',['geometry_slot','bin','zero_collision_fraction_phys','beta_model','collided_R','collided_G','collided_B','source_cell_share_R','source_cell_share_G','source_cell_share_B'],rows)
    rows=[[k,np.mean(v['packet_rows_mean']),np.mean(v['sample_rows']),v['packet_C_mean'],v['sample_C']] for k,v in a['noise'].items()]
    rows_csv(out/'noise_accounting.csv',['matrix','packet_variance_row_mean','replica_sample_variance_row_mean','packet_contrast_noise','replica_contrast_noise'],rows)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
    for c,ch in enumerate('RGB'):
        p=a['pooled'][f'coarse/{c}'];sq=lambda x:np.sqrt(max(x,0));labels=[];lo=[];hi=[];est=[]
        labels.append('relaxed shared\nrank-8 [L,U]');lo.append(sq(p['L7_rel']['lo']));hi.append(sq(p['U7_rel']['hi']));est.append(sq(.5*(p['L7_rel']['est']+p['U7_rel']['est'])))
        for ck in a['primary_checkpoints']:
            for s,lab in (('F_A_rel','frozen A'),('PAOc_rel','frozen A+O'),('E_Mc_rel','model')):
                labels.append(f'{lab}\n{ck[-4:]}');lo.append(sq(p[f'{ck}/{s}']['lo']));hi.append(sq(p[f'{ck}/{s}']['hi']));est.append(sq(p[f'{ck}/{s}']['est']))
        x=np.arange(len(labels));ax[c].bar(x,est,color=['#4c78a8']+['#f58518','#54a24b','#b279a2']*2)
        ax[c].errorbar(x,est,yerr=[np.array(est)-np.array(lo),np.array(hi)-np.array(est)],fmt='none',ecolor='k',capsize=3)
        for th,ls in ((.10,'--'),(.05,':')):ax[c].axhline(th,color='gray',ls=ls,lw=1)
        ax[c].set_xticks(x);ax[c].set_xticklabels(labels,fontsize=7);ax[c].set_title(f'{ch}: pooled contrast floors, 1 cm/16-out ({a["per_channel"].get(ch,{}).get("category","-") if isinstance(a["per_channel"],dict) else "-"})')
    ax[0].set_ylabel('sqrt(residual energy / physical contrast energy)');fig.suptitle(f'WL66 incoming rank feasibility — outcome {a["outcome"]} (95% jackknife intervals)')
    fig.tight_layout();fig.savefig(out/'rank66.png',dpi=140);fig.savefig(out/'rank66.pdf');plt.close(fig)
    summary=dict(outcome=a['outcome'],per_channel=a['per_channel'],precision=a['precision'],replicas=a['replicas'],t_quantile=a['t_quantile'],
        frozen_basis=a['frozen_basis'],principal_cosines=a['principal_cosines_TA_vs_physical_top7'],sharing=a['sharing'],
        model_span_sanity_max=max(max(v.values()) for v in a['model_span_sanity'].values()),analysis_sha256=sha(ROOT/'analysis.json'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_new(out/'manifest.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.name!='manifest.json'})
    print('EXPORT',a['outcome'],flush=True)

if __name__=='__main__':main()
