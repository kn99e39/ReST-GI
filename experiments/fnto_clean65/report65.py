"""Reference-only reviewer exports; every weak/unqualified row retained."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=a.root;o=a.out;o.mkdir(parents=True,exist_ok=False)
    data=json.loads((r/'analysis.json').read_text());s=data['summary'];cfg=json.loads((r/'source/experiments/fnto_clean65/protocol/clean_v1.json').read_text())
    for key in ('absolute','directional','deformation','geometry_delta','historical_discrepancy'):
        rows=data[key]
        with open(o/f'{key}_all.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(v.keys() for v in rows))),lineterminator='\n');w.writeheader();w.writerows(rows)
    (o/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    poses=cfg['poses'];fields=cfg['fields'];d=np.zeros((13,8));delta=np.zeros((12,16));fine=np.zeros((13,16));old=np.zeros((13,8))
    for i,p in enumerate(poses):
        for k in range(8):
            d[i,k]=min(v['SNR'] for v in data['directional'] if v['pose']==p and v['pair']==k and v['resolution']=='coarse')
            old[i,k]=max(v['relative_L2'] for v in data['historical_discrepancy'] if v['pose']==p and v['kind']=='directional' and v['field_or_pair']==k and v['resolution']=='coarse')
        for k,f in enumerate(fields):
            fine[i,k]=max(v['mean_SE_ratio'] for v in data['absolute'] if v['pose']==p and v['field']==f and v['resolution']=='fine16')
            if i:delta[i-1,k]=max(v['loss_scale_SE'] for v in data['deformation'] if v['pose']==p and v['field']==f and v['resolution']=='coarse')
    plt.rcParams.update({'font.size':9});fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    panels=[(d,'Coarse directional SNR: minimum RGB (gate >=3)',poses,range(8),'Pair'),(delta,'Delta SE / original loss scale: maximum RGB (gate <=0.25)',poses[1:],fields,'Train field'),(fine,'Fine16 absolute mean-SE ratio: maximum RGB (gate <=0.05)',poses,fields,'Train field'),(old,'Original-label / new directional discrepancy: maximum RGB',poses,range(8),'Pair')]
    for ax,(z,title,ys,xs,xlab) in zip(axs.flat,panels):
        z=np.maximum(z,1e-8);im=ax.imshow(z,aspect='auto',norm=LogNorm(vmin=max(float(z.min()),1e-8),vmax=max(float(z.max()),float(z.min())*1.01)),cmap='viridis');ax.set(title=title,xlabel=xlab,xticks=range(len(xs)),xticklabels=list(xs),yticks=range(len(ys)),yticklabels=ys);fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('WL65 Phase A: '+s['status']+'\nFixed208 actions /26.624B photons; precision only, no model trained',fontsize=12)
    fig.savefig(o/'reference_precision.png',dpi=170);fig.savefig(o/'reference_precision.pdf');plt.close(fig)
    (o/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in o.iterdir() if p.is_file()},indent=2)+'\n')
if __name__=='__main__':main()
