"""Independent analytic GGX checks; no training, targets or render selection."""
import json
from pathlib import Path
import sys
import numpy as np
import render93e as R


def main(root):
    PT, S, ctx, lights, cam, original, _ = R.setup(root,'S1','bx38',128)
    mi,dr=PT.mi,PT.dr
    rng=np.random.default_rng(931482)
    def hemisphere(n):
        z=rng.uniform(.05,1,n);phi=rng.uniform(-np.pi,np.pi,n)
        return np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],axis=-1)
    a,b=hemisphere(100000),hemisphere(100000)
    wi=mi.Vector3f(*[PT.fl(a[:,i]) for i in range(3)])
    wo=mi.Vector3f(*[PT.fl(b[:,i]) for i in range(3)])
    result={}
    for name,alpha in R.ALPHA.items():
        cl=PT.CD.DrClosure83('ER',R.ggx_state(alpha))
        x=cl.eval(wi,wo);rev=cl.eval(wo,wi)
        pcg=mi.PCG32(size=len(a),initstate=mi.UInt64(931483))
        sw,pdf,weight,ok=cl.sample(wi,pcg.next_float32(),mi.Point2f(pcg.next_float32(),pcg.next_float32()))
        dr.eval(x,rev,weight,ok)
        val=np.asarray(x)[:,0].astype(np.float64)
        reverse=np.asarray(rev)[:,0].astype(np.float64)
        half=a+b;half/=np.linalg.norm(half,axis=-1,keepdims=True)
        D=alpha**2/(np.pi*((half[:,0]**2+half[:,1]**2)+alpha**2*half[:,2]**2)**2)
        # Height-correlated Smith masking: G2 = 2/(sqrt(1+a²tan²i)
        # +sqrt(1+a²tan²o)). eval is BRDF times outgoing cosine.
        G2=2/(np.sqrt(1+alpha**2*(1-a[:,2]**2)/a[:,2]**2)+np.sqrt(1+alpha**2*(1-b[:,2]**2)/b[:,2]**2))
        expected=.85*D*G2/(4*a[:,2])
        relative=np.sqrt(np.mean((val-expected)**2))/np.sqrt(np.mean(expected**2))
        reciprocity=np.max(np.abs(val/b[:,2]-reverse/a[:,2]))/max(np.max(val/b[:,2]),1e-20)
        w=np.asarray(weight)[:,0]
        result[name]={'numpy_analytic_eval_relative_rms':float(relative),
                      'reciprocity_relative_max':float(reciprocity),
                      'sample_weight_max':float(w.max()),
                      'nonnegative':bool(np.min(val)>=0 and np.min(w)>=0),
                      'pass':bool(relative<.001 and reciprocity<1e-4 and w.max()<.8501 and w.min()>=0)}
    result['all_pass']=all(d['pass'] for d in result.values())
    result['semantics']='Neutral .85 single-GGX with F0=1, height-correlated Smith; alpha is direct NDF slope. This is an idealized passive reflector, not the original coated-conductor P material.'
    result['mirror_scope']='True camera-side delta sampled exactly. Point-light next-event sampling cannot estimate a delta at the light-side end of a path (caustic connector). Ideal-mirror e2 on the primary panel is valid; ideal-mirror full and h images omit that family and are restricted diagnostics. Do not use the latter as complete physical ground truth.'
    R.write(result,Path(root)/'results/analytic_material_checks.json')
    print(json.dumps(result),flush=True)
    assert result['all_pass']


if __name__=='__main__':main(Path(sys.argv[1]))
