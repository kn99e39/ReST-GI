"""Read-only CPU diagnostics of existing WL59 actions. No model/oracle imports or fitting."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import torch

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(p):return torch.load(p,map_location='cpu',mmap=True,weights_only=False)

def angular(y):
    bm=torch.arange(64);bm=(bm%8)//2+4*((bm//8)//2)
    z=torch.zeros(*y.shape[:-2],16,3,dtype=y.dtype)
    return z.index_add_(-2,bm,y)

def agg(y,parent):
    z=angular(y)
    return torch.zeros(len(y),int(parent.max())+1,16,3,dtype=y.dtype).index_add_(1,parent,z).double()

def project(y,basis):
    # y (..., angular bins), basis (angular bins, rank); SVD handles numerical rank.
    u,s,_=torch.linalg.svd(basis.double(),full_matrices=False)
    rank=int((s>max(basis.shape)*torch.finfo(torch.float64).eps*s[0]).sum())
    q=u[:,:rank]
    return y@q@q.T,rank

def relative(y,t):return float(torch.linalg.vector_norm(y-t)/torch.linalg.vector_norm(t).clamp(min=1e-30))
def cosine(a,b):return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp(min=1e-30))
def rank_floor(y,rank=8):
    sv=torch.linalg.svdvals(y.reshape(-1,y.shape[-1]).double())
    return float(sv[rank:].square().sum().sqrt()),sv.tolist()

def ballistic(g,x):
    B=g['B'];ys=[]
    for c in range(3):
        yy=torch.sparse.mm(B,x[:,:,:,c].reshape(len(x),-1).T).T.reshape(len(x),g['P'],64)
        ys.append(yy)
    return torch.stack(ys,-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source-commit',required=True)
    args=ap.parse_args();root=args.root;torch.set_num_threads(4);torch.set_grad_enabled(False);t0=time.perf_counter()
    report=dict(source_commit=args.source_commit,script_sha256=sha(__file__),device='CPU',torch=torch.__version__,python=platform.python_version(),primary='pt0',seeds={},inputs={},notes=['effective residual = total reference minus existing approximate B; no true ballistic/scattered oracle separation','span bounds relax positivity, spatial and input constraints; small bound does not prove representability','no training, reference generation, runtime cache mutation or GPU execution'])
    def record(p):
        p=Path(p);report['inputs'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,resolved=str(p.resolve()))
    xp=root/'data/lights.pt';record(xp);lights=load(xp);x=lights['test']
    geoms={};balls={}
    for pose in ('rest','pt0'):
        gp=root/'data/geometry'/f'{pose}.pt';record(gp);g=load(gp);geoms[pose]=g
        balls[pose]=agg(ballistic(g,x),g['parent'])
    parent=geoms['rest']['parent'];seed_residuals={};inputs={}
    for seed in (5901,5902):
        cp=root/'checkpoints'/f'P_{seed}.pt';record(cp);ck=load(cp)['model']
        apath=root/'out'/f'actions_{seed}.pt';record(apath);obj=load(apath)
        pred={p:agg(obj['predictions'][p]['F'],parent) for p in ('rest','pt0')}
        ref={p:agg(obj['reference'][p],parent) for p in ('rest','pt0')}
        noise={p:agg(obj['noise'][p],parent) for p in ('rest','pt0')}
        delta=ref['pt0']-ref['rest'];yd=pred['pt0']-pred['rest'];bd=balls['pt0']-balls['rest']
        dn=torch.hypot(noise['pt0'],noise['rest']);snr=delta.flatten(1,2).norm(dim=1)/dn.flatten(1,2).norm(dim=1).clamp(min=1e-30)
        O=angular(torch.softmax(ck['out_logits'],dim=1)).double() # (8,16,3)
        A=torch.softmax(ck['in_logits'],dim=0).double()
        out=dict(dynamic=[],directional=[],absolute={},secondary={},input_projection=[])
        for pose in ('rest','pt0'):
            er=pred[pose]-ref[pose]
            out['absolute'][pose]=dict(mean_relative_L2=[float((er[:,:,:,c].flatten(1).norm(dim=1)/ref[pose][:,:,:,c].flatten(1).norm(dim=1)).mean()) for c in range(3)])
        seed_residuals[seed]={}
        for c in range(3):
            basis=O[:,:,c].T
            ids=(snr[:16,c]>=3).nonzero().flatten()
            yy=yd[ids,:,:,c];tt=delta[ids,:,:,c];bb=bd[ids,:,:,c];eff=tt-bb
            projected,rank=project(eff,basis)
            se=dn[ids,:,:,c]/(2**.5) # descriptive mean-reference noise per-pose quadrature combination
            rr=yy-tt
            seed_residuals[seed][f'delta_{c}']=rr
            learned=yy-bb
            fitted,_=project(learned,basis)
            assert relative(learned,fitted)<2e-5,'Stored predictor must lie in derived scattered output span'
            out['dynamic'].append(dict(channel=c,fields=ids.tolist(),error=relative(yy,tt),ballistic_delta_over_target=float(bb.norm()/tt.norm()),effective_scatter_delta_over_target=float(eff.norm()/tt.norm()),frozen_output_span_floor=float((eff-projected).norm()/tt.norm()),output_rank=rank,model_span_residual=relative(learned,fitted),mean_reference_noise_norm_ratio=float(se.norm()/tt.norm())))
            ypair=pred['pt0'][48::2,:,:,c]-pred['pt0'][49::2,:,:,c]
            tpair=ref['pt0'][48::2,:,:,c]-ref['pt0'][49::2,:,:,c]
            bpair=balls['pt0'][48::2,:,:,c]-balls['pt0'][49::2,:,:,c]
            for k in range(8):
                ef=tpair[k]-bpair[k];pr,_=project(ef,basis);rf,sv=rank_floor(ef)
                out['directional'].append(dict(pair=k,channel=c,error=relative(ypair[k],tpair[k]),frozen_output_span_floor=float((ef-pr).norm()/tpair[k].norm()),free_rank8_floor=rf/float(tpair[k].norm()),singular_values=sv,ballistic_contrast_over_target=float(bpair[k].norm()/tpair[k].norm())))
                seed_residuals[seed][f'pair_{k}_{c}']=ypair[k]-tpair[k]
            # Do declared pair differences vanish under the actual frozen incoming projection?
            remaining=x.double()*(1-geoms['pt0']['beta'].double())[None,:,:,None]
            q=torch.einsum('biac,rac->birc',remaining,A)
            for k in range(8):
                raw=(remaining[48+2*k,:,:,c]-remaining[49+2*k,:,:,c]);enc=q[48+2*k,:,:,c]-q[49+2*k,:,:,c]
                out['input_projection'].append(dict(pair=k,channel=c,raw_norm=float(raw.norm()),encoded_norm=float(enc.norm()),exact_collision=bool(torch.count_nonzero(enc)==0)))
        for pose in ('cf0','cf1','cf2','cf3'):
            pp=agg(obj['predictions'][pose]['F'],parent);tt=agg(obj['reference'][pose],parent)
            out['secondary'][pose]=dict(mean_relative_L2=[float(((pp-tt)[:,:,:,c].flatten(1).norm(dim=1)/tt[:,:,:,c].flatten(1).norm(dim=1)).mean()) for c in range(3)])
        report['seeds'][str(seed)]=out
        del obj
    report['cross_seed_residual_cosine']={k:cosine(seed_residuals[5901][k],seed_residuals[5902][k]) for k in seed_residuals[5901]}
    # Metadata construction, not a new inference campaign: report exact input populations and geometry features.
    for split,xx in lights.items():
        nz=(xx.sum((1,3))>0).sum(1)
        inputs[split]=dict(fields=len(xx),one_hot_angular_fields=int((nz==1).sum()),field_sha256=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in xx])
    report['light_populations']=inputs
    # Original supervision precision gap on SAME rest/test fields only, never substituted for train-label noise.
    old=root.parent/'rest_wl57'
    op=old/'data/targets/rest_r0.pt'
    if op.exists():
        record(op);historical=load(op)
        # Historical test lights differ from A2. Do not compare them numerically to fresh test actions.
        report['old_target_keys']=list(historical)
        report['training_noise_control_available']=False
    for name,path in [('old_light_metadata',old/'out/light_manifest.json'),('test_light_metadata',root.parent/'rest_wl57_recovery/out/new_light_manifest.json')]:
        if path.exists():record(path);report[name]=json.loads(path.read_text())
    report['seconds']=time.perf_counter()-t0
    import resource
    report['maxrss_platform_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump(report,sys.stdout,indent=2,allow_nan=False);print()

if __name__=='__main__':main()
