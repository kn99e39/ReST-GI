"""Report-only same-population action/contrast detail; no new sampling or model evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    if (root/'component_detail.json').exists():raise FileExistsError('Preserve report')
    z=np.load(root/'coarse_components.npz');ref=z['reference'];pred=z['predictions'];rows=[]
    for si,seed in enumerate([5901,5902]):
        for ki,name in enumerate(['T','B','Scat']):
            r=ref[:,ki];p=pred[si,ki];rd=r[0]-r[1];pd=p[0]-p[1]
            for c in range(3):
                xx=rd[...,c].reshape(-1);yy=pd[...,c].reshape(-1);xc=xx-xx.mean();yc=yy-yy.mean();rr=np.linalg.norm(xx);pp=np.linalg.norm(yy)
                rows.append(dict(seed=seed,component=name,channel=c,physical_action_power=[float(v[...,c].sum()) for v in r],model_action_power=[float(v[...,c].sum()) for v in p],absolute_action_relative_L2=[float(np.linalg.norm((p[i]-r[i])[...,c])/np.linalg.norm(r[i,...,c])) for i in range(2)],predicted_contrast_norm_over_physical=float(pp/rr),prediction_contrast_cosine=float(xx@yy/max(rr*pp,1e-300)),contrast_centered_slope=float(xc@yc/max(xc@xc,1e-300)),contrast_centered_correlation=float(xc@yc/max(np.linalg.norm(xc)*np.linalg.norm(yc),1e-300))))
    report=dict(post_hoc=True,changes_primary_gates=False,script_sha256=sha(__file__),input_sha256=sha(root/'coarse_components.npz'),rows=rows,limitation='Component error localization does not identify why learned weights/geometry/budget/objective produced it. No fitting or quadrature change.')
    (root/'component_detail.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
