"""Post hoc CPU-only independent replay of frozen Gate1. No target generation or model fitting.
Run: run.sh r57_gate_audit.py [audit_output.json]
"""
import sys,os,json
from pathlib import Path
import numpy as np
import torch

def audit(root):
    torch.set_num_threads(4)
    g=torch.load(root/'data/geometry/rest.pt',map_location='cpu')
    assert np.array_equal(g['parent'].numpy(),np.repeat(np.arange(240),16));del g
    def coarse(name,rep):
        z=torch.load(root/'data/targets'/f'{name}_r{rep}.pt',map_location='cpu')['test'].numpy()
        assert z.shape==(64,3840,64,3)
        return z.reshape(64,240,16,4,2,4,2,3).sum(axis=(2,4,6),dtype=np.float64).reshape(64,240,16,3)
    def norm(x):return np.sqrt((x*x).sum(axis=(1,2)))
    a,b=coarse('rest',0),coarse('rest',1);base=(a+b)/2;bn=(a-b)/np.sqrt(2)
    old=json.load(open(root/'out/gate1.json'));rows=[]
    for ix,name in enumerate(['te0','te1','te2','te3']):
        c,d=coarse(name,0),coarse(name,1);delta=(c+d)/2-base;noise=(c-d)/np.sqrt(2)
        sn=norm(delta)/np.maximum(np.hypot(norm(noise),norm(bn)),1e-30)
        frac=(sn[:16]>=3).mean(0);ok=frac>=.25
        error=float(np.max(np.abs(frac-old['signal'][ix]['concentrated_fraction_ge3'])));assert error==0
        rows.append(dict(pose=name,independent_fraction_ge3=frac.tolist(),focused_count_ge3=(sn[:16]>=3).sum(0).tolist(),
            maximum_focused_SNR=sn[:16].max(0).tolist(),meaningful_channels=ok.tolist(),replay_fraction_error=error))
    passed=sum(sum(x['meaningful_channels'])>=2 for x in rows)>=3
    assert bool(passed)==bool(old['proceed'])
    return dict(method='Independent NumPy float64 nested geometry/angular sums; no production aggregation/gate call',rows=rows,
        signal_gate_passed=bool(passed),status='REPRODUCED_STOP; no evidence of aggregation/count implementation defect in Gate1' if not passed else 'REPRODUCED_PASS')

if __name__=='__main__':
    result=audit(Path(os.environ.get('WL57',Path.home()/'rest_wl57')));text=json.dumps(result,indent=1)+'\n'
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(text)
    print(text,flush=True);os._exit(0)
