"""Prospectively fixed new source population; evaluation only, no optimizer access."""
import os,sys,json,time
from pathlib import Path
ROOT=Path.home()/'rest_direction64';ORIG=Path.home()/'rest_persist_v1';HERE=Path(__file__).resolve().parent
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG);os.environ['WL57_WALKER']='fast';os.environ['WL57_ACTION_CHUNK']='32000000'
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'));sys.path.insert(0,str(HERE.parent/'fnto_components63'))
import torch,numpy as np
from common import RG
from r57_data import light_fields,aggregate,dump,save,sha
import r57_action_oracle as AO,r57_fastwalk as FW
from tag_reference import tagged_action

def main():
    assert not (ROOT/'independent_complete.json').exists();torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;g=torch.load(ORIG/'data/geometry/rest.pt');g['coarse']=RG.mesh('rest')[0]
    x,meta=light_fields(g,64,6408103,'test');x=x[48:];old=torch.load(ORIG/'data/lights.pt')
    import hashlib
    digest=lambda z:hashlib.sha256(z.numpy().tobytes()).hexdigest()
    oldhash={digest(f) for fields in old.values() for f in fields};hashes=[digest(f) for f in x];assert not oldhash.intersection(hashes);assert len(set(hashes))==16
    for p in range(8):assert torch.allclose(x[2*p].sum(1),x[2*p+1].sum(1),atol=1e-8,rtol=1e-6)
    save(x,ROOT/'independent_lights.pt');dump({'seed':6408103,'retained_generator_indices':list(range(48,64)),'sha256':hashes,'previous_splits_disjoint':True,'pre_model_outcome':True},ROOT/'independent_population.json')
    g=torch.load(ORIG/'data/geometry/pt0.pt');fresh=RG.geometry('pt0',ballistic=False)
    for k in ('xyz','rest_xyz','area','node','pair','prior','parent'):assert torch.equal(fresh[k].cpu(),g[k]),k
    mesh,V=RG.mesh('pt0');fm,VC,parent=RG.KO.refined(mesh,V,4);dom=RG.KO.DomainX(fm,VC)
    first=[];second=[];receipts=[];t0=time.perf_counter()
    for f in range(16):
        ff=[];ss=[]
        for rep in range(2):
            for half in range(2):
                seed=640000000+f*100+rep*10+half;p=ROOT/'independent_halves'/f'f{f}_r{rep}_h{half}.pt';p.parent.mkdir(exist_ok=True)
                if p.exists():r=torch.load(p)
                else:
                    r=tagged_action(AO,FW,dom,x[f],32000000,seed);save(r,p)
                assert r['physical_passed'];assert all(v<=1e-12 for k,v in r['gates_A_B'].items() if type(v)!=bool);assert all(v for v in r['gates_A_B'].values() if type(v)==bool)
                ff.append(torch.stack([aggregate(r['component_first'][k],g['parent'])[0] for k in ('T','B','Scat')]))
                ss.append(torch.stack([aggregate(r['component_second'][k],g['parent'])[0] for k in ('T','B','Scat')]))
                receipts.append(dict(field=f,replica=rep,half=half,seed=seed,N=r['N'],seconds=r['seconds'],sha256=sha(p),physical=True,gates=r['gates_A_B'],stats=r['stats'],rng=r['rng_state_sha256'],peak_GPU_bytes=r['peak_GPU_bytes']))
                print('FRESH',f,rep,half,r['seconds'],flush=True)
        first.append(torch.stack(ff));second.append(torch.stack(ss))
    first=torch.stack(first);second=torch.stack(second);mean=first.mean(1);variance=((32000000*second-first.square())/31999999).sum(1)/16
    replicas=torch.stack([first[:,:2].mean(1),first[:,2:].mean(1)])
    contrast=lambda y:y[::2]-y[1::2]
    norm=lambda y:y.flatten(-3,-2).norm(dim=-2)
    d=contrast(mean);noise=(contrast(replicas[0])-contrast(replicas[1]))/(2**.5);snr=norm(d)/norm(noise).clamp(min=1e-30)
    np.savez_compressed(ROOT/'independent_reference.npz',first=first.numpy(),second=second.numpy(),reference=mean.numpy(),variance=variance.numpy(),replicas=replicas.numpy(),snr=snr.numpy())
    dump(dict(seconds=time.perf_counter()-t0,photons=16*4*32000000,receipts=receipts,SNR_pairs_components_RGB=snr.tolist(),total_qualified_cells=int((snr[:,0]>=3).sum()),total_all24_qualified=bool((snr[:,0]>=3).all()),component_all72_qualified=bool((snr>=3).all()),evaluation_only=True),ROOT/'independent_complete.json')
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
