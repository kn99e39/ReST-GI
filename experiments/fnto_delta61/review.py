"""Report-only CSV/figure export of completed fixed-population WL61 analysis."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=json.loads(a.analysis.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Do not replace completed reviewer evidence')
    rows=[];absolute=[]
    for seed,m in r['models'].items():
        for pose,d in m['dynamic'].items():
            ref=r['dynamic'][pose]
            for c,ch in enumerate('RGB'):
                rows.append(dict(seed=seed,pose=pose,channel=ch,qualified=ref['qualified_RGB'][c],conservative_delta_SNR=ref['conservative_pooled_SNR'][c],clean_mean_SE_over_delta=ref['mean_reference_SE_norm_ratio_RGB'][c],prediction_vs_old=d['prediction_vs_old'][c],prediction_vs_clean=d['prediction_vs_clean'][c],prediction_vs_old_common_clean_denominator=d['prediction_vs_old_common_clean_denominator'][c],old_vs_clean_delta=ref['historical_vs_clean_delta'][c],clean_correlation=d['clean_alignment'][c]['correlation'],clean_slope=d['clean_alignment'][c]['slope'],clean_error_replica_min=min(v['error'][c] for v in d['replica_sensitivity']),clean_error_replica_max=max(v['error'][c] for v in d['replica_sensitivity'])))
        for pose,d in m['absolute'].items():
            for i,field in enumerate(r['fields']):
                for c,ch in enumerate('RGB'):
                    absolute.append(dict(seed=seed,pose=pose,field=field,channel=ch,prediction_vs_old=d['prediction_vs_old_coarse_per_field'][i][c],prediction_vs_clean=d['prediction_vs_clean_coarse_per_field'][i][c],old_vs_clean=r['absolute'][pose]['old_vs_clean_coarse_per_field'][i][c]))
    table(out/'paired_delta_all.csv',rows);table(out/'absolute_all.csv',absolute)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.9),sharey=True)
    colors=['#c0772a','#336f9c'];series=[('5901','old'),('5901','clean'),('5902','old'),('5902','clean')]
    for c,(ch,ax) in enumerate(zip('RGB',axes)):
        for k,(seed,target) in enumerate(series):
            yy=[r['models'][seed]['dynamic'][p]['prediction_vs_'+target][c] for p in ('tr11','va1')]
            x=[i+(k-1.5)*.18 for i in range(2)]
            ax.bar(x,yy,width=.17,color=colors[int(seed)-5901],alpha=.45 if target=='old' else 1.,label=f'{seed}: '+('historical r0' if target=='old' else 'independent mean'))
        
        if not r['dynamic']['va1']['qualified_RGB'][c]:
            ax.text(.5,.97,'va1: delta precision unqualified',transform=ax.transAxes,ha='center',va='top',fontsize=8,color='#993333')
            for bar in ax.patches:
                if bar.get_x()>.5:bar.set_hatch('//')
        ax.axhline(.35,color='#777777',lw=.9,ls='--');ax.set_title(ch);ax.set_xticks([0,1],['trained tr11','validation va1']);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Paired delta relative L2 error')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.025),fontsize=8)
    fig.suptitle('Frozen-K clean paired delta audit — identical train fields 8/9',fontsize=12)
    fig.text(.5,.005,'Dashed line: historical .35 error criterion for context. Bars are estimates, not confidence intervals. Fixed two-field scope.',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.18,1,.93]);fig.savefig(out/'paired_delta_comparison.png',dpi=180);fig.savefig(out/'paired_delta_comparison.pdf');plt.close(fig)
    manifest=dict(scope='Report-only derived artifacts; not a new primary population or architecture verdict',analysis_sha256=sha(a.analysis),script_sha256=sha(Path(__file__)),artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
