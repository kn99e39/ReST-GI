"""Frozen-checkpoint probe responses, replay and factorisation gates. Reads NO physical probe data."""
import sys,json,time,os
import numpy as np
from common66 import CFG,ROOT,ORIG,HOME,GEOMS,NBIN,sha,write_new,save_new,environment,verify_sources,ports

def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-300))

def direct_scatter(g,co,i,d):
    """D_i A^T G_i from frozen coefficients, independent of Transport.apply: (16,P,64,3)."""
    import torch
    W,U,V,A,O=co['W'].double(),co['U'].double(),co['V'].double(),co['A'].double(),co['O'].double()
    e=(g['src']==i).nonzero().flatten()
    T=torch.zeros(g['P'],W.shape[1],3,dtype=torch.float64,device=W.device).index_add_(0,g['dst'][e],W[e])
    T=T+torch.einsum('jkrc,krc->jrc',U,V[i])
    return torch.einsum('a,rac,jrc,roc->ajoc',d.double(),A,T,O)

def main():
    import torch
    C59,RG=environment();from runtime import Transport,full,apply
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    if (ROOT/'model_complete.json').exists():raise FileExistsError('model outputs are immutable')
    hashes=verify_sources();pr=ports();t0=time.perf_counter();records=dict(replay={},factorisation={},span={},checkpoints={})
    lights=torch.load(ORIG/'data/lights.pt',map_location='cpu')['test'];allck={**CFG['checkpoints']['primary'],**CFG['checkpoints']['secondary_descriptive']}
    for name in GEOMS:
        g=C59.loaded_geometry(name);out={}
        for s in pr['ports']:
            i=s['port'];x=torch.zeros(NBIN,g['P'],NBIN,3,device='cuda');x[torch.arange(NBIN),i,torch.arange(NBIN)]=1.
            ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(NBIN,-1).T).T.reshape(NBIN,g['P'],64) for c in range(3)],-1)
            out[s['slot']]=ball.cpu()
        save_new(ROOT/'model'/f'ball_{name}.pt',dict(ball=out,geometry=name,signature=g['signature']))
        for ck,rel_path in allck.items():
            m=Transport().cuda();m.load_state_dict(torch.load(HOME/rel_path,map_location='cpu')['model']);m.eval()
            c=full(m,g);co=c['co'];res={}
            if name=='pt0' and ck.startswith('WL59'):
                seed=int(ck[-4:]);stored=torch.load(ORIG/'out'/f'actions_{seed}.pt',map_location='cpu')['predictions']['pt0']['F'][[8,9,62,63]]
                yy=torch.cat([apply(m,c,lights[[f]].cuda()).cpu() for f in (8,9,62,63)])
                records['replay'][ck]=rel(yy,stored);assert records['replay'][ck]<=CFG['gates']['replay_relative_L2'],'frozen replay'
            A=co['A'].double().cpu()
            for s in pr['ports']:
                i=s['port'];d=torch.tensor(s['geometries'][name]['d'],dtype=torch.float64,device='cuda')
                x=torch.zeros(NBIN,g['P'],NBIN,3,device='cuda');x[torch.arange(NBIN),i,torch.arange(NBIN)]=1.
                total=torch.cat([apply(m,c,x[k:k+4]) for k in range(0,NBIN,4)])
                ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,cc].reshape(NBIN,-1).T).T.reshape(NBIN,g['P'],64) for cc in range(3)],-1)
                scatter=total-ball;direct=direct_scatter(g,co,i,d)
                fz=rel(scatter.double(),direct);key=f'{ck}/{name}/s{s["slot"]}';records['factorisation'][key]=fz
                assert fz<=CFG['gates']['factorisation_relative_L2'],('factorisation',key,fz)
                Z=(scatter.double()/d[:,None,None,None]).cpu()  # actual apply output, not the direct algebra
                sp=[]
                for cc in range(3):
                    At=A[:,:,cc].T;Pz=At@torch.linalg.pinv(At);r=Z[...,cc].reshape(NBIN,-1);sp.append(float(((r-Pz@r).norm()/r.norm()).item()))
                    O=co['O'][:,:,cc].double().cpu();B=torch.einsum('ra,ro->rao',At.T,O).reshape(8,-1).T;Pb=B@torch.linalg.pinv(B)
                    blk=Z[...,cc].permute(1,0,2).reshape(g['P'],-1);sp.append(float(((blk-blk@Pb.T).norm()/blk.norm()).item()))
                records['span'][key]=sp;assert max(sp)<=CFG['gates']['factorisation_relative_L2'],('span',key,sp)
                res[s['slot']]=scatter.cpu()
            save_new(ROOT/'model'/f'{ck}_{name}.pt',dict(scatter=res,A=co['A'].cpu(),O=co['O'].cpu(),checkpoint=ck,geometry=name,signature=g['signature']))
            records['checkpoints'][ck]=dict(path=str(HOME/rel_path),sha256=CFG['checkpoints']['sha256'][ck],parameters=sum(p.numel() for p in m.parameters()))
            del m,c,co;torch.cuda.empty_cache()
        del g;torch.cuda.empty_cache()
    write_new(ROOT/'model_complete.json',dict(records,verified_hashes=hashes,seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        physical_probe_data_read=False,files={p.name:sha(p) for p in sorted((ROOT/'model').glob('*.pt'))}))
    print('MODEL_COMPLETE',json.dumps(records['replay']),max(records['factorisation'].values()),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
