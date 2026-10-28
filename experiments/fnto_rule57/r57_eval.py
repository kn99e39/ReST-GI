"""Frozen primary held-out evaluation, physics and fair cache-lifecycle costs."""
import sys,os,json,time,resource
import numpy as np
import torch
import r57_geometry as RG
from r57_model import Transport
from r57_data import ROOT,OUT,DATA,dump,log,aggregate

def model_of(arm,seed):
    ck=torch.load(ROOT/'checkpoints'/f'{arm}_{seed}.pt',map_location='cpu');m=Transport(arm).cuda();m.load_state_dict(ck['model']);m.eval();return m
def rel(y,t):return (y-t).flatten(1,2).norm(dim=1)/t.flatten(1,2).norm(dim=1).clamp(min=1e-20)
def scores(y,t,g):
    out={}
    for label,coarse in (('coarse',True),('fine16',False)):
        rr=rel(aggregate(y,g['parent'],coarse),aggregate(t,g['parent'],coarse))
        out[label]=dict(mean=rr.mean(0).tolist(),p90=torch.quantile(rr,.9,dim=0).tolist(),per_field=rr.cpu().tolist())
    rr=rel(y,t);out['fine64']=dict(mean=rr.mean(0).tolist(),p90=torch.quantile(rr,.9,dim=0).tolist())
    out['signed_energy_bias']=((y.sum((1,2))-t.sum((1,2)))/t.sum((1,2)).clamp(min=1e-20)).mean(0).tolist()
    return out

def physics(m,g,x,co):
    y=m.apply(g,x,co);z=m.apply(g,x*0,co);s=m.apply(g,x[:1]+x[1:2],co)
    lin=float((s-y[:2].sum(0,keepdim=True)).abs().sum()/s.abs().sum().clamp(min=1e-30));cache=m(g,x,m.state(g))
    ce=float((cache-y).abs().sum()/y.abs().sum().clamp(min=1e-30));col=float(m.column_sums(g,co).max())
    return dict(zero=float(z.abs().max()),minimum=float(y.min()),linearity=lin,max_colsum=col,cache_error=ce,
        passed=bool(float(z.abs().max())<=1e-8 and float(y.min())>=-1e-8 and lin<=1e-5 and col<=1+1e-6 and ce<=1e-6))

def evaluate():
    lights=torch.load(DATA/'lights.pt');res={};coarse_outputs={}
    for arm in ('D','R'):
        for seed in (5701,5702):
            m=model_of(arm,seed);tag=f'{arm}_{seed}';rows={};all_useful=True
            with torch.no_grad():
                predictions={};references={};noises={};gs={}
                for name in ['rest']+RG.TEST:
                    g=RG.device(torch.load(DATA/'geometry'/f'{name}.pt'));gs[name]=g
                    a=torch.load(DATA/'targets'/f'{name}_r0.pt')['test'];b=torch.load(DATA/'targets'/f'{name}_r1.pt')['test'];ref=.5*(a+b);noise=(a-b)/2**.5
                    co=m.coefficients(g);pr=[]
                    batch=int(os.environ.get('WL57_EVAL_BATCH','4'))
                    for i in range(0,64,batch):pr.append(m.apply(g,lights['test'][i:i+batch].to('cuda'),co).cpu())
                    y=torch.cat(pr);predictions[name]=y;references[name]=ref;noises[name]=noise
                    metric=scores(y.cuda(),ref.cuda(),g);ph=physics(m,g,lights['test'][:2].to('cuda'),co)
                    useful=all(v<=.10 for v in metric['coarse']['mean']) and all(v<=.20 for v in metric['coarse']['p90']) and all(v<=.15 for v in metric['fine16']['mean'])
                    rows[name]=dict(metrics=metric,physics=ph,static_useful=useful,geometry_signature=g['signature'])
                    rows[name]['oracle_seed_noise']=scores(noise.cuda()+ref.cuda(),ref.cuda(),g)
                    all_useful &= useful and ph['passed']
                    del co
                rest=predictions['rest'];restref=references['rest'];parent=gs['rest']['parent']
                for name in RG.TEST:
                    rows[name]['stale_rest_control']=scores(restref.cuda(),references[name].cuda(),gs[name])
                    yd=aggregate((predictions[name]-rest).cuda(),parent);td=aggregate((references[name]-restref).cuda(),parent)
                    noise=torch.hypot(aggregate(noises[name].cuda(),parent),aggregate(noises['rest'].cuda(),parent))
                    sn=td.flatten(1,2).norm(dim=1)/noise.flatten(1,2).norm(dim=1).clamp(min=1e-30);dynamic=[]
                    for c in range(3):
                        mask=(sn[:16,c]>=3).nonzero().flatten();xx=td[:16,:,:,c][mask].flatten();yy=yd[:16,:,:,c][mask].flatten()
                        if len(mask)==0:dynamic.append(dict(detectable=False,pass_dynamic=None));continue
                        er=float((xx-yy).norm()/xx.norm().clamp(min=1e-30));xc=xx-xx.mean();yc=yy-yy.mean();corr=float((xc@yc)/(xc.norm()*yc.norm()).clamp(min=1e-30));slope=float((xc@yc)/(xc@xc).clamp(min=1e-30))
                        passed=er<=.35 and corr>=.90;dynamic.append(dict(detectable=True,n_fields=len(mask),relative_L2=er,corr=corr,slope=slope,pass_dynamic=passed));all_useful &= passed
                    rows[name]['dynamic']=dynamic
                    # Direction-sensitive equal-irradiance controls, all eight fixed pairs.
                    yp=aggregate(predictions[name][48::2].cuda()-predictions[name][49::2].cuda(),parent)
                    tp=aggregate(references[name][48::2].cuda()-references[name][49::2].cuda(),parent)
                    pn=aggregate(noises[name][48::2].cuda()-noises[name][49::2].cuda(),parent)
                    dd=rel(yp,tp);detect=tp.flatten(1,2).norm(dim=1)>3*pn.flatten(1,2).norm(dim=1)
                    rows[name]['directional_pairs']=dict(error=dd.cpu().tolist(),detectable=detect.cpu().tolist(),passed=bool(((dd<=.25)|~detect).all()))
                    all_useful &= rows[name]['directional_pairs']['passed']
                for name,y in predictions.items():
                    coarse_outputs[f'{tag}|{name}']=aggregate(y.cuda(),parent).cpu().numpy()
            res[tag]=dict(arm=arm,seed=seed,useful_and_physical=all_useful,poses=rows,parameters=sum(p.numel() for p in m.parameters()))
            dump(res,OUT/'heldout.json');log('HELDOUT',tag,'useful',all_useful,{k:v['metrics']['coarse']['mean'] for k,v in rows.items()})
            del m,gs;torch.cuda.empty_cache()
    np.savez(OUT/'coarse_predictions.npz',**coarse_outputs)

def byte_count(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size() if not x.is_sparse else x._indices().numel()*8+x._values().numel()*x._values().element_size()
    if isinstance(x,dict):return sum(byte_count(v) for v in x.values())
    return 0
def timed(fn,n=1):
    ts=[];r=None
    for _ in range(n):
        torch.cuda.synchronize();t=time.perf_counter();r=fn();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
    return r,dict(median=float(np.median(ts)),p10=float(np.quantile(ts,.1)),p90=float(np.quantile(ts,.9)),all=ts)

def benchmark():
    res={};lights=torch.load(DATA/'lights.pt')['test'].to('cuda')
    with torch.no_grad():
        for name in RG.TEST:
            g,geo_time=timed(lambda:RG.geometry(name),3)
            # Geometry trace setup/cells/B construction is charged to both arms, never an oracle table.
            pose={}
            for arm in ('D','R'):
                for seed in (5701,5702):
                    m=model_of(arm,seed);st,prep=timed(lambda:m.state(g),20);co,build=timed(lambda:m.coefficients(g,st),20)
                    for _ in range(5):m.apply(g,lights[:1],co)
                    y,full=timed(lambda:m.apply(g,lights[:1],co),20)
                    yy,compact=timed(lambda:m(g,lights[:1],st),20)
                    assert float((y-yy).abs().sum()/y.abs().sum())<=1e-6
                    record=dict(geometry=geo_time,state_prepare=prep,coefficient_prepare=build,query_full_cache=full,query_compact=compact,
                        geometry_bytes=byte_count(RG.serial(g)),compact_state_bytes=byte_count(st),coefficient_bytes=byte_count(co),parameters=sum(p.numel() for p in m.parameters()),
                        peak_GPU_bytes=torch.cuda.max_memory_allocated(),CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                    record['cache_eligible']={p:(record['geometry_bytes']+record['compact_state_bytes']+(record['coefficient_bytes'] if p=='full' else 0)<=512_000_000) for p in ('full','compact')}
                    choices={}
                    for N in (1,4,16,64):
                        def life(policy):
                            st0=m.state(g);co0=m.coefficients(g,st0) if policy=='full' else None
                            for j in range(N):q=m.apply(g,lights[j:j+1],co0) if policy=='full' else m(g,lights[j:j+1],st0)
                            return q
                        _,lf=timed(lambda:life('full'),3);_,lc=timed(lambda:life('compact'),3)
                        f=geo_time['median']+lf['median'];c=geo_time['median']+lc['median']
                        eligible={p:v for p,v in [('full',f),('compact',c)] if record['cache_eligible'][p]}
                        assert eligible,'No cache policy meets preregistered ceiling'
                        policy=min(eligible,key=eligible.get)
                        choices[N]=dict(full=f,compact=c,best=eligible[policy],selected=policy,full_lifecycle=lf,compact_lifecycle=lc,
                            note='Directly measured cache-build+N distinct light queries, plus separately measured common geometry build. Not TR-to-neural speedup.')
                    record['queries']=choices;pose[f'{arm}_{seed}']=record;del m,st,co;torch.cuda.empty_cache()
            res[name]=pose;dump(res,OUT/'cost.json');log('COST',name,{k:v['queries'] for k,v in pose.items()})

if __name__=='__main__':
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    {'evaluate':evaluate,'cost':benchmark}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
