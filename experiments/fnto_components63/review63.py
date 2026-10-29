"""Export complete component-attribution tables and static scientific figure."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Preserve reviewer artifacts')
    data=json.loads((root/'analysis.json').read_text());rows=[];rep=[]
    for seed,m in data['models'].items():
        for r in m['rows']:rows.append(dict(seed=seed,**r))
        for variant in m['replica_variants']:
            for r in variant['rows']:rep.append(dict(seed=seed,field62_replica=variant['field62_replica'],field63_replica=variant['field63_replica'],**r))
    table(out/'component_attribution_all.csv',rows);table(out/'replica_attribution_all.csv',rep)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharey=True)
    for ax,seed in zip(axes,['5901','5902']):
        rr=data['models'][seed]['rows'];x=list(range(3));b=[r['B_squared_over_total_signal'] for r in rr];s=[r['Scat_squared_over_total_signal'] for r in rr];cross=[r['cross_over_total_signal'] for r in rr]
        ax.bar(x,b,color='#c18c43',label='Ballistic error squared');ax.bar(x,s,bottom=b,color='#3a759e',label='Collided error squared');ax.bar(x,cross,bottom=[bb+ss for bb,ss in zip(b,s)],color='#b64f58',label='Signed cross term')
        ax.set_xticks(x,list('RGB'));ax.set_title('Frozen checkpoint '+seed);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,r in enumerate(rr):ax.text(i,r['total_squared_over_total_signal']+.004,f"total L2 {r['total_relative_L2']:.3f}",ha='center',fontsize=8)
    axes[0].set_ylabel('Squared contrast error / physical total contrast squared')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.5,.035),frameon=False,fontsize=8)
    fig.suptitle('Physical first-exit component attribution — pt0, test pair 7 (62−63)',fontsize=12)
    fig.text(.5,.018,'All components independently tagged; cross term retained. Same total-reference normalization. No training or B change.',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.17,1,.94]);fig.savefig(out/'component_error_decomposition.png',dpi=180);fig.savefig(out/'component_error_decomposition.pdf');plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(analysis_sha256=sha(root/'analysis.json'),script_sha256=sha(__file__),artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
if __name__=='__main__':main()
