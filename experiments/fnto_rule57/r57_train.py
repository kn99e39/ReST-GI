"""Matched training only on declared train poses/lights; final checkpoint, no test selection."""
import sys,os,time,json,math
from pathlib import Path
import numpy as np
import torch
import r57_geometry as RG
from r57_model import Transport,parameter_count
from r57_data import ROOT,OUT,DATA,dump,log,aggregate,save

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
            g=RG.device(geoms[name]);co=model.coefficients(g);batch=int(os.environ.get('WL57_EVAL_BATCH','4'))
            for i in range(0,32,batch):
                xx=lights['validation'][i:i+batch].to('cuda');tt=targets[name]['validation'][i:i+batch].to('cuda')
                v.append(float(loss_metric(model.apply(g,xx,co),tt,g['parent'])))
    model.train();return float(np.mean(v))

def train(arm,seed,steps=6000):
    assert json.load(open(OUT/'gate1.json'))['proceed'];assert arm in ('D','R')
    pdir=ROOT/'checkpoints';pdir.mkdir(exist_ok=True);path=pdir/f'{arm}_{seed}.pt'
    if path.exists():raise FileExistsError('Preserve completed training; do not silently retrain/replace checkpoint')
    torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Transport(arm).cuda();assert abs(parameter_count('D')/parameter_count('R')-1)<.05
    streamed=os.environ.get('WL57_STREAM_GEOMETRY','0')=='1'
    geoms={n:torch.load(DATA/'geometry'/f'{n}.pt') for n in RG.TRAIN+RG.VAL}
    if not streamed:geoms={n:RG.device(g) for n,g in geoms.items()}
    # Explicit lists, not directory discovery: held-out geometry/lighting targets cannot enter the optimizer.
    target_dir=Path(os.environ.get('WL57_TRAIN_TARGETS',DATA/'targets'))
    targets={n:torch.load(target_dir/f'{n}_r0.pt') for n in RG.TRAIN+RG.VAL};lights=torch.load(DATA/'lights.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-5)
    rng=np.random.default_rng(574001);history=[];start=0;prior_seconds=0.;progress=pdir/f'{arm}_{seed}_progress.pt'
    if progress.exists():
        ck=torch.load(progress,map_location='cpu');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);history=ck['history'];prior_seconds=ck['seconds'];log('RESUME',arm,seed,start)
    t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    micro=int(os.environ.get('WL57_MICROBATCH','2'));assert micro in (1,2)
    for step in range(start,steps):
        name=RG.TRAIN[1+step%(len(RG.TRAIN)-1)];ids=rng.integers(0,96,2)
        x=lights['train'][ids].to('cuda');g=RG.device(geoms[name]) if streamed else geoms[name];rest=RG.device(geoms['rest']) if streamed else geoms['rest']
        t=targets[name]['train'][ids].to('cuda');tr=targets['rest']['train'][ids].to('cuda')
        opt.zero_grad(set_to_none=True);Lv=dv=lv=0.
        for j in range(0,2,micro):
            sl=slice(j,j+micro);y=model(g,x[sl]);yr=model(rest,x[sl])
            L=.5*(loss_metric(y,t[sl],g['parent'])+loss_metric(yr,tr[sl],rest['parent']))
            dy,dt=aggregate(y-yr,g['parent']),aggregate(t[sl]-tr[sl],g['parent']);base=aggregate(tr[sl],rest['parent'])
            denom=dt.flatten(1,2).square().sum(1)+.0001*base.flatten(1,2).square().sum(1)+1e-20
            delta=((dy-dt).flatten(1,2).square().sum(1)/denom).mean();loss=L+.25*delta
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss: preserve failed result, no automatic architecture change')
            (loss*(micro/2)).backward();Lv+=float(L)*micro/2;dv+=float(delta)*micro/2;lv+=float(loss)*micro/2
            del y,yr,L,delta,loss,dy,dt,base,denom
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        lr=.00003+.5*(.003-.00003)*(1+math.cos(math.pi*(step+1)/steps))
        for pg in opt.param_groups:pg['lr']=lr
        if (step+1)%250==0 or step==0:
            val=validation(model,geoms,lights,targets);rec=dict(step=step+1,train=lv,action=Lv,delta=dv,grad_norm=grad,val=val,seconds=prior_seconds+time.perf_counter()-t0)
            history.append(rec);dump(dict(arm=arm,seed=seed,history=history),OUT/'training'/f'{arm}_{seed}.json');log('TRAIN',arm,seed,rec)
        if (step+1)%100==0 or step==0:
            save(dict(model=model.state_dict(),step=step+1,optimizer=opt.state_dict(),arm=arm,seed=seed,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,seconds=prior_seconds+time.perf_counter()-t0),progress)
        if streamed:del g,rest
    torch.cuda.synchronize()
    save(dict(model=model.state_dict(),arm=arm,seed=seed,steps=steps),path)
    dump(dict(arm=arm,seed=seed,steps=steps,parameters=sum(p.numel() for p in model.parameters()),seconds=prior_seconds+time.perf_counter()-t0,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),history=history,training_poses=RG.TRAIN,validation_poses=RG.VAL,
        primary_commit=(ROOT/'repo_commit.txt').read_text().strip()),OUT/'training'/f'{arm}_{seed}.json')
    log('TRAIN_DONE',arm,seed,time.perf_counter()-t0)

if __name__=='__main__':
    train(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 6000);sys.stdout.flush();os._exit(0)
