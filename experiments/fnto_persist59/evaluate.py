import os,sys
import torch
import numpy as np
from common import *
from runtime import Transport, full, selective, apply
from r57_eval import scores, physics, rel

def model_of(arm,seed):
    ck=torch.load(ROOT/'checkpoints'/f'P_{seed}.pt',map_location='cpu')
    m=Transport().cuda();m.load_state_dict(ck['model']);m.eval();return m

def actions(m,c,x):
    ys=[];batch=int(os.environ.get('WL57_EVAL_BATCH','4'))
    for i in range(0,len(x),batch):ys.append(apply(m,c,x[i:i+batch].cuda()).cpu())
    return torch.cat(ys)

def agg(y,g,coarse=True):
    # Fixed shape metadata and bin map avoid reading a GPU max into Python.
    bm=torch.as_tensor(RG.G.bin_parent(8,4),device=y.device)
    z=torch.zeros(y.shape[0],y.shape[1],16,3,device=y.device).index_add_(2,bm,y)
    if not coarse:return z
    return torch.zeros(y.shape[0],240,16,3,device=y.device).index_add_(1,g['parent'],z)

def main():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.load(DATA/'lights.pt')['test'];results={}
    for seed in CFG['training']['seeds']:
        m=model_of('P',seed);rest=loaded_geometry('rest');preds={};refs={};noise={};rows={}
        with torch.no_grad():
            rc=full(m,rest);preds['rest']=dict(F=actions(m,rc,x));preds['rest']['S']=preds['rest']['F'].clone()
            for name in ['rest']+PRIMARY+DIAGNOSTIC:
                g=rest if name=='rest' else loaded_geometry(name)
                a=torch.load(DATA/'targets'/f'{name}_r0.pt')['test'];b=torch.load(DATA/'targets'/f'{name}_r1.pt')['test']
                refs[name]=.5*(a+b);noise[name]=(a-b)/(2**.5)
                audit={};equivalence=dict(max_relative_L1=0.,retained_raw_bitwise=True,retained_head_bitwise=True)
                if name!='rest':
                    fc=full(m,g);sc=full(m,rest)
                    oldraw=sc['raw'].clone();oldhead=sc['head'].clone()
                    sc,audit=selective(m,sc,g);masks=audit.pop('masks')
                    equivalence['retained_raw_bitwise']=bool(torch.equal(oldraw[~masks['edges']],sc['raw'][~masks['edges']]))
                    equivalence['retained_head_bitwise']=bool(torch.equal(oldhead[~masks['nodes']],sc['head'][~masks['nodes']]))
                    preds[name]=dict(F=actions(m,fc,x),S=actions(m,sc,x))
                    yy,zz=preds[name]['S'],preds[name]['F']
                    er=(yy-zz).abs().sum((1,2))/zz.abs().sum((1,2)).clamp(min=1e-30)
                    equivalence.update(max_relative_L1=float(er.max()),per_field_channel=er.tolist())
                    co=fc['co']
                else:co=rc['co']
                fs=equivalence['max_relative_L1']<=1e-5 and equivalence['retained_raw_bitwise'] and equivalence['retained_head_bitwise']
                ph=physics(m,g,x[:2].cuda(),co)
                metrics={p:scores(preds[name][p].cuda(),refs[name].cuda(),g) for p in ('F','S')}
                static={p:all(v<=.10 for v in metrics[p]['coarse']['mean']) and all(v<=.20 for v in metrics[p]['coarse']['p90']) and all(v<=.15 for v in metrics[p]['fine16']['mean']) for p in ('F','S')}
                rows[name]=dict(primary=name in PRIMARY or name=='rest',metrics=metrics,physics=ph,static_useful=static,
                                F_S=equivalence,F_S_passed=fs,invalidation=audit,
                                reference_seed_noise=scores((refs[name]+noise[name]).cuda(),refs[name].cuda(),g))
                if name!='rest':del fc,sc,oldraw,oldhead
                dump(dict(partial=True,seed=seed,poses=rows),OUT/f'heldout_{seed}_partial.json')
                torch.cuda.empty_cache()
            for name in PRIMARY+DIAGNOSTIC:
                g=rest;td=agg((refs[name]-refs['rest']).cuda(),g)
                nd=torch.hypot(agg(noise[name].cuda(),g),agg(noise['rest'].cuda(),g))
                snr=td.flatten(1,2).norm(dim=1)/nd.flatten(1,2).norm(dim=1).clamp(min=1e-30)
                rows[name]['stale_physical_K']=scores(refs['rest'].cuda(),refs[name].cuda(),g)
                rows[name]['dynamic']={};rows[name]['directional_pairs']={}
                for policy in ('F','S'):
                    yd=agg((preds[name][policy]-preds['rest'][policy]).cuda(),g);dynamic=[]
                    for channel in range(3):
                        ids=(snr[:16,channel]>=3).nonzero().flatten()
                        if not len(ids):dynamic.append(dict(detectable=False,passed=None));continue
                        xx=td[:16,:,:,channel][ids].flatten();yy=yd[:16,:,:,channel][ids].flatten()
                        er=float((xx-yy).norm()/xx.norm().clamp(min=1e-30));xc=xx-xx.mean();yc=yy-yy.mean()
                        corr=float(xc@yc/(xc.norm()*yc.norm()).clamp(min=1e-30));slope=float(xc@yc/(xc@xc).clamp(min=1e-30))
                        dynamic.append(dict(detectable=True,n_fields=len(ids),error=er,corr=corr,slope=slope,passed=er<=.35 and corr>=.90))
                    rows[name]['dynamic'][policy]=dynamic
                    yp=agg((preds[name][policy][48::2]-preds[name][policy][49::2]).cuda(),g)
                    tp=agg((refs[name][48::2]-refs[name][49::2]).cuda(),g)
                    pn=agg((noise[name][48::2]-noise[name][49::2]).cuda(),g)
                    errors=rel(yp,tp);det=tp.flatten(1,2).norm(dim=1)>3*pn.flatten(1,2).norm(dim=1)
                    rows[name]['directional_pairs'][policy]=dict(errors=errors.cpu().tolist(),detectable=det.cpu().tolist(),
                        qualified=int(det.sum()),passed=bool(det.any() and ((errors<=.25)|~det).all()))
            ref_ok=all(v<=.02 for n in ['rest']+PRIMARY for v in rows[n]['reference_seed_noise']['coarse']['mean'])
            eligible={}
            for p in ('F','S'):
                eligible[p]=ref_ok and all(rows[n]['static_useful'][p] and rows[n]['physics']['passed'] and rows[n]['F_S_passed'] for n in ['rest']+PRIMARY)
                eligible[p] &= all(all(d['detectable'] and d['passed'] for d in rows[n]['dynamic'][p]) and rows[n]['directional_pairs'][p]['passed'] for n in PRIMARY)
            results[f'P_{seed}']=dict(seed=seed,reference_precision_passed=ref_ok,poses=rows,eligible=eligible,
                                    useful_and_physical=all(eligible.values()),parameters=sum(v.numel() for v in m.parameters()))
            dump(results,OUT/'heldout.json')
            # All fields retained, including errors and unchanged/stale controls; no best-case selection.
            save(dict(predictions=preds,reference=refs,noise=noise),OUT/f'actions_{seed}.pt')
            del m,rc,rest,preds,refs,noise
            torch.cuda.empty_cache()

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
