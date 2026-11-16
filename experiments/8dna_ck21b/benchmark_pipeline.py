"""Paired execution-only correctness/profiling on a paused native checkpoint."""
import copy,gc,json,os,sys,time
from pathlib import Path
import numpy as np
BASE=Path.home()/'rest_8dna_ck21b'
UP=Path.home()/'NeuralShaderDemo'/'external'/'8dna26'
sys.path.insert(0,str(UP)); os.chdir(UP)
import torch
torch.set_num_threads(4)
import bridge as B
import types
module=types.ModuleType('scenes.ck21b_bench'); module.get_scene=B.training_scene
sys.modules[module.__name__]=module
from utils.dataset import PathSamplingDataset
from models.eight_dna import EightDNA
from packed_pipeline import CUDAPrefetch,FIELDS,unpack
from omegaconf import OmegaConf

def save(result): (BASE/'results'/'pipeline_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
def make_model(checkpoint):
    model=EightDNA(**OmegaConf.create(checkpoint['hyper_parameters']).model).cuda().train()
    model.load_state_dict({k[len('model.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')})
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_states'][0]))
    return model,optimizer

def step(model,optimizer,batch,keep_grad=False):
    mask=batch['mask']
    if not mask.any(): raise ValueError('Empty benchmark batch')
    logpdf,albedo=model(batch['xi'][mask],batch['wi'][mask],batch['xo'][mask],batch['wo'][mask])
    target=batch['throughput'][mask]
    loss_albedo=torch.nn.functional.mse_loss(target,albedo)
    loss_pdf=-(target*logpdf).mean(); loss=loss_albedo+loss_pdf
    if loss.isnan() or loss.isinf(): raise ValueError('Nonfinite loss')
    optimizer.zero_grad(); loss.backward()
    grad=torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).clone() if keep_grad else None
    torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()
    return float(loss.detach()),grad

def main():
    pause=json.loads((BASE/'pause_boundary.json').read_text())
    anchor=torch.load(pause['checkpoint'],map_location='cpu')
    result={'anchor':pause,'anchor_epoch':anchor['epoch'],'anchor_step':anchor['global_step'],'status':'running','model_and_batch':'Unchanged FP32 8DNA, Adam state, clipping=5, batch32768','scope':'Timing fixture only: real native small-buffer tuples tiled to the full 128^4 host address footprint. Never used for the main training fit.'}
    save(result)
    B.dr.set_flag(B.dr.JitFlag.LoopRecord,True); B.dr.set_flag(B.dr.JitFlag.VCallRecord,True)
    torch.manual_seed(211700); torch.cuda.manual_seed_all(211700)
    fixture=PathSamplingDataset('ck21b_bench',N=32,H=32,SPP=32,batch_size=32768,device='cuda',bounce=2)
    fixture.reload(211700)
    small=len(fixture.xi); count=128**4
    print('fixture',small,'full footprint',count,flush=True)
    host={k:getattr(fixture,k).repeat((count//small,1)) for k in FIELDS}
    host['mask']=fixture.mask.repeat(count//small)
    small_packed=torch.cat([*(getattr(fixture,k) for k in FIELDS),fixture.mask.float()[:,None]],dim=1)
    packed=small_packed.repeat((count//small,1))
    indices=torch.randperm(count,generator=torch.Generator().manual_seed(211701))
    indices_gpu=indices.cuda()
    B.dr.sync_thread(); B.dr.flush_malloc_cache(); torch.cuda.empty_cache(); gc.collect()
    result['tuple_count']=count; result['packed_bytes']=packed.numel()*packed.element_size(); result['legacy_host_bytes']=sum(x.numel()*x.element_size() for x in host.values())

    def baseline(index):
        first=(index%(count//32768))*32768
        ind=indices_gpu[first:first+32768].cpu()
        return {key:value[ind].cuda() for key,value in host.items()}
    # Equal ordered fields and paired gradients/Adam updates, not test-set tuning.
    left,ol=make_model(anchor); right,orr=make_model(anchor)
    iterator=CUDAPrefetch(packed,indices,32768,max_batches=3)
    field_equal=True; comparisons=[]
    for i,fast in enumerate(iterator):
        native=baseline(i)
        field_equal &= all(torch.equal(native[k],fast[k]) and native[k].dtype==fast[k].dtype and fast[k].is_contiguous() for k in native)
        la,ga=step(left,ol,native,True); lb,gb=step(right,orr,fast,True)
        pa=torch.cat([p.detach().flatten() for p in left.parameters()]); pb=torch.cat([p.detach().flatten() for p in right.parameters()])
        comparisons.append({'loss_abs':abs(la-lb),'gradient_max_abs':float((ga-gb).abs().max()),'gradient_rel_l2':float((ga-gb).norm()/ga.norm().clamp_min(1e-12)),'parameter_max_abs':float((pa-pb).abs().max())})
    result['field_bit_equal']=field_equal; result['paired_optimizer']=comparisons
    result['correctness_pass']=field_equal and all(x['loss_abs']<=1e-6 and x['gradient_rel_l2']<=1e-4 and x['parameter_max_abs']<=2e-6 for x in comparisons)
    del left,right,ol,orr,iterator; gc.collect(); torch.cuda.empty_cache(); save(result)
    if not result['correctness_pass']: raise RuntimeError('Pipeline changes numerical training contract')
    print('CORRECTNESS',json.dumps(comparisons),flush=True)

    times={}
    for label in ['baseline','prefetch','baseline','prefetch']:
        model,optimizer=make_model(anchor)
        fast=CUDAPrefetch(packed,indices,32768,max_batches=72) if label=='prefetch' else None
        for i in range(8): step(model,optimizer,next(fast) if fast is not None else baseline(i))
        torch.cuda.synchronize(); start=time.perf_counter()
        for i in range(8,72): step(model,optimizer,next(fast) if fast is not None else baseline(i))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        times.setdefault(label,[]).append(elapsed/64)
        if fast is not None: fast.close()
        del model,optimizer,fast; gc.collect(); torch.cuda.empty_cache()
        print('TIMING',label,elapsed/64,flush=True)
    result['seconds_per_complete_step']=times
    result['speedup']=float(np.median(times['baseline'])/np.median(times['prefetch']))
    result['status']='admitted' if result['correctness_pass'] and result['speedup']>=1.2 else 'not_admitted'
    save(result)
    # CPU/CUDA attribution includes gather, transfers, optimizer and scalar checks.
    for label in ['baseline','prefetch']:
        model,optimizer=make_model(anchor)
        fast=CUDAPrefetch(packed,indices,32768,max_batches=12) if label=='prefetch' else None
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False) as prof:
            for i in range(12): step(model,optimizer,next(fast) if fast is not None else baseline(i))
            torch.cuda.synchronize()
        (BASE/'results'/f'profile_{label}.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=24))
        prof.export_chrome_trace(str(BASE/'results'/f'profile_{label}.json'))
        if fast is not None: fast.close()
        del model,optimizer,fast; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
