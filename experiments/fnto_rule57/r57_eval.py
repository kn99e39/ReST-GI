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

def benchmark():
    from r57_cost import benchmark as measured
    measured()

if __name__=='__main__':
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    {'evaluate':evaluate,'cost':benchmark}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
