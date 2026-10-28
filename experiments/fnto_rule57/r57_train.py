"""Matched training only on declared train poses/lights; final checkpoint, no test selection."""
import sys,os,time,json,math
from pathlib import Path
import numpy as np
import torch
import r57_geometry as RG
from r57_model import Transport,parameter_count
from r57_data import ROOT,OUT,DATA,dump,log,aggregate

def loss_metric(y,t,parent):
    vals=[]
    for coarse in (True,False):
        a,b=aggregate(y,parent,coarse),aggregate(t,parent,coarse)
        vals.append(((a-b).flatten(1,2).square().sum(1)/(b.flatten(1,2).square().sum(1)+1e-20)).mean())
    return .5*(vals[0]+vals[1])

def validation(model,geoms,lights,targets):
    model.eval();v=[]
    with torch.no_grad():
        for name in RG.VAL:
            g=geoms[name];co=model.coefficients(g)
            for i in range(0,32,4):
                xx=lights['validation'][i:i+4].to('cuda');tt=targets[name]['validation'][i:i+4].to('cuda')
                v.append(float(loss_metric(model.apply(g,xx,co),tt,g['parent'])))
    model.train();return float(np.mean(v))

def train(arm,seed,steps=6000):
    assert json.load(open(OUT/'gate1.json'))['proceed'];assert arm in ('D','R')
    pdir=ROOT/'checkpoints';pdir.mkdir(exist_ok=True);path=pdir/f'{arm}_{seed}.pt'
    if path.exists():raise FileExistsError('Preserve completed training; do not silently retrain/replace checkpoint')
    torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Transport(arm).cuda();assert abs(parameter_count('D')/parameter_count('R')-1)<.05
    geoms={n:RG.device(torch.load(DATA/'geometry'/f'{n}.pt')) for n in RG.TRAIN+RG.VAL}
    # Explicit lists, not directory discovery: held-out geometry/lighting targets cannot enter the optimizer.
    targets={n:torch.load(DATA/'targets'/f'{n}_r0.pt') for n in RG.TRAIN+RG.VAL};lights=torch.load(DATA/'lights.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-5)
    rng=np.random.default_rng(574001);history=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        name=RG.TRAIN[1+step%(len(RG.TRAIN)-1)];ids=rng.integers(0,96,2)
        x=lights['train'][ids].to('cuda');g=geoms[name];rest=geoms['rest']
        t=targets[name]['train'][ids].to('cuda');tr=targets['rest']['train'][ids].to('cuda')
        y=model(g,x);yr=model(rest,x)
        L=.5*(loss_metric(y,t,g['parent'])+loss_metric(yr,tr,rest['parent']))
        dy,dt=aggregate(y-yr,g['parent']),aggregate(t-tr,g['parent']);base=aggregate(tr,rest['parent'])
        denom=dt.flatten(1,2).square().sum(1)+.0001*base.flatten(1,2).square().sum(1)+1e-20
        delta=((dy-dt).flatten(1,2).square().sum(1)/denom).mean();loss=L+.25*delta
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss: preserve failed result, no automatic architecture change')
        opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        lr=.00003+.5*(.003-.00003)*(1+math.cos(math.pi*(step+1)/steps))
        for pg in opt.param_groups:pg['lr']=lr
        if (step+1)%250==0 or step==0:
            val=validation(model,geoms,lights,targets);rec=dict(step=step+1,train=float(loss),action=float(L),delta=float(delta),grad_norm=grad,val=val,seconds=time.perf_counter()-t0)
            history.append(rec);dump(dict(arm=arm,seed=seed,history=history),OUT/'training'/f'{arm}_{seed}.json');log('TRAIN',arm,seed,rec)
        if (step+1)%1000==0:
            torch.save(dict(model=model.state_dict(),step=step+1,optimizer=opt.state_dict(),arm=arm,seed=seed),pdir/f'{arm}_{seed}_progress.pt')
    torch.cuda.synchronize()
    torch.save(dict(model=model.state_dict(),arm=arm,seed=seed,steps=steps),path)
    dump(dict(arm=arm,seed=seed,steps=steps,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-t0,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),history=history,training_poses=RG.TRAIN,validation_poses=RG.VAL,
        primary_commit=(ROOT/'repo_commit.txt').read_text().strip()),OUT/'training'/f'{arm}_{seed}.json')
    log('TRAIN_DONE',arm,seed,time.perf_counter()-t0)

if __name__=='__main__':
    train(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 6000);sys.stdout.flush();os._exit(0)
