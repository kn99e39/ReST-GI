"""POST HOC squared-error noise review; unchanged data, population and readiness gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def cross_squared(pred,d0,d1):
    return float(np.sum((pred-d0)*(pred-d1))),float(np.sum(d0*d1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    output=root/'noise_review.json'
    if output.exists():raise FileExistsError('Preserve completed review')
    z=np.load(root/'coarse_evidence.npz');r=json.loads((root/'analysis.json').read_text());h=z['half_actions'];v=z['reference_mean_variance'];means=z['reference'];old=z['historical_r0'];preds=z['predictions'];rows=[]
    for pi,pose in [(1,'tr11'),(2,'va1')]:
        d=means[pi]-means[0];d0=h[pi,:,0].mean(1)-h[0,:,0].mean(1);d1=h[pi,:,1].mean(1)-h[0,:,1].mean(1);dv=v[pi]+v[0]
        for c,ch in enumerate('RGB'):
            for si,subject in [(None,'historical_r0'),(0,'P_5901'),(1,'P_5902')]:
                yy=(old[pi]-old[0]) if si is None else (preds[si,pi]-preds[si,0])
                yy=yy[...,c];dd=d[...,c];variance=float(dv[...,c].sum());den=float(np.sum(dd*dd));err=float(np.sum((yy-dd)**2));e2,s2=cross_squared(yy,d0[...,c],d1[...,c])
                mden=den-variance;merr=err-variance
                rows.append(dict(pose=pose,channel=ch,subject=subject,qualified=r['dynamic'][pose]['qualified_RGB'][c],raw_relative_squared=err/den,mean_variance_trace=variance,moment_corrected_error_squared=merr,moment_corrected_signal_squared=mden,moment_corrected_relative_squared=merr/mden if mden>0 else None,cross_error_squared=e2,cross_signal_squared=s2,cross_relative_squared=e2/s2 if s2>0 else None))
    obj=dict(post_hoc=True,changes_readiness_or_verdict=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['analysis.json','coarse_evidence.npz']},method='Subtract propagated mean-reference variance trace from squared residual and squared signal; separately cross independent replica residuals. Squared numerators/denominators are expectation-corrected under independent unbiased reference errors; their ratio and square root are NOT unbiased and have no confidence interval. Negative estimates are retained. Not a repair of unqualified channels.',rows=rows)
    output.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
