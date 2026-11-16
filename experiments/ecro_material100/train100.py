"""Unchanged ECRO architecture + original WL92A OWN loss, fresh fixed fits."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
import common100 as C
import model86 as M

torch.set_num_threads(4)
DEV=torch.device('cuda')


class Item:
    def __init__(self,path):
        z=np.load(path)
        self.z={k:z[k] for k in ['pc','pn','pt','parea','pmat','hit','dbin','s','f2','fh']}
        if 'f2b' in z.files:
            self.z['f2']=.5*(self.z['f2']+z['f2b']);self.z['fh']=.5*(self.z['fh']+z['fhb'])
    def tensors(self):
        z=self.z;g=type('G',(),{})();g.N=len(z['pc'])
        g.PC,g.PN,g.PT,g.PAREA,g.PMAT=[z[k] for k in ['pc','pn','pt','parea','pmat']]
        gs=M.GeoState(g,z['hit'],z['dbin'],DEV)
        return gs,*[torch.as_tensor(z[k],device=DEV) for k in ['s','f2','fh']]


@torch.no_grad()
def evaluate(model,items):
    errors=np.zeros(3);truth=np.zeros(3)
    for item in items:
        gs,s,t2,th=item.tensors();f=model.factors(gs)
        a=model.apply(f,gs,s,use_global=False);b=model.apply(f,gs,s,use_single=False)
        for i,(p,t) in enumerate([(a+b,t2+th),(a,t2),(b,th)]):
            errors[i]+=float((p-t).abs().sum());truth[i]+=float(t.sum())
    return dict(zip(['whole_rel_l1','single_rel_l1','global_rel_l1'],(errors/np.maximum(truth,1e-20)).tolist()))


def main(root,case,steps=None):
    root=Path(root);p=C.P['training'];steps=int(steps or p['steps'])
    seed=p['seed'];torch.manual_seed(seed);np.random.seed(seed)
    train=[Item(f) for f in sorted((root/case/'data/train').glob('g*.npz'))]
    val=[Item(f) for f in sorted((root/case/'data/val').glob('g*.npz'))]
    test=[Item(f) for f in sorted((root/case/'data/test').glob('g*.npz'))]
    assert len(train)==128 and len(val)==16 and len(test)==8
    model=M.ECRO().to(DEV);model.mat_states=torch.as_tensor(C.materials(case),device=DEV)
    count=sum(x.numel() for x in model.parameters());assert count==735664
    opt=torch.optim.Adam(model.parameters(),lr=p['lr']);rng=np.random.default_rng(seed);order=[];log=[];running=[]
    t0=time.perf_counter();print('TRAIN_START',case,count,len(train),flush=True)
    for step in range(steps):
        if not order:order=rng.permutation(len(train)).tolist()
        item=train[order.pop()]
        lr=p['lr']*(step+1)/p['warmup'] if step<p['warmup'] else p['lr_final']+.5*(p['lr']-p['lr_final'])*(1+math.cos(math.pi*(step-p['warmup'])/max(steps-p['warmup'],1)))
        for group in opt.param_groups:group['lr']=lr
        gs,s,t2,th=item.tensors();f=model.factors(gs)
        a=model.apply(f,gs,s,use_global=False);b=model.apply(f,gs,s,use_single=False)
        t=t2+th;den=t.mean(2,keepdim=True)+.05*t.mean(dim=(1,2,3),keepdim=True)
        losses=[(((a+b-t)/den)**2).mean(),(((a-t2)/den)**2).mean(),(((b-th)/den)**2).mean()]
        loss=sum(losses);assert torch.isfinite(loss),'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        assert torch.isfinite(norm),'STOP_NONFINITE_GRADIENT'
        opt.step();running.append([float(x.detach()) for x in losses])
        if (step+1)%500==0 or step+1==steps:
            row={'step':step+1,'losses':np.mean(running,axis=0).tolist(),'seconds':time.perf_counter()-t0};running=[]
            if (step+1)%4000==0 or step+1==steps:row['val']=evaluate(model,val)
            log.append(row);print(json.dumps(row),flush=True)
    torch.cuda.synchronize();train_s=time.perf_counter()-t0
    ck=root/case/'model.pt';torch.save(model.state_dict(),ck)
    result={'case':case,'seed':seed,'steps':steps,'params':count,'train_s':train_s,
            'train_items':len(train),'val_items':len(val),'test_items':len(test),
            'log':log,'validation':evaluate(model,val),'test':evaluate(model,test),
            'checkpoint_sha256':C.sha(ck),'model86_source_sha256':C.sha(C.BASE/'model86.py'),
            'architecture_changed':False,'objective':'WL92A OWN (1,1,1)','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0),'peak_mib':torch.cuda.max_memory_allocated()/2**20}
    C.jd(result,root/case/'training.json');print('TRAIN_COMPLETE',case,train_s,result['test'],flush=True)


if __name__=='__main__':main(*sys.argv[1:])
