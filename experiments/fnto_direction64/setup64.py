import os,sys,json,hashlib,platform,subprocess,time
from pathlib import Path
ROOT=Path.home()/'rest_direction64';ORIG=Path.home()/'rest_persist_v1';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'))
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
import torch,numpy as np
from common import RG
from r57_data import sha,dump,aggregate
from runtime import Transport
from train import loss_metric
from objective64 import contrast,total
CFG=json.loads((HERE/'protocol/direction_v1.json').read_text())
def setup():
    assert not (ROOT/'calibration.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources={}
    for rel,expected in CFG['source_expected_sha256'].items():
        actual=sha(ORIG/'repo'/rel);assert actual==expected,rel;sources[rel]=actual
    identities={}
    for rel,expected in CFG['input_expected_sha256'].items():
        actual=sha(ORIG/rel);assert actual==expected,rel;identities[rel]=actual
    for folder in ('geometry','training_targets','targets'):
        for p in sorted((ORIG/'data'/folder).glob('*.pt')):identities[str(p.relative_to(ORIG))]=sha(p)
    # Calibration reads only train values; validation/test labels never enter calibration.
    ta=torch.load(ORIG/'data/training_targets/rest_r0.pt')['train']
    noise_path=Path.home()/'rest_wl57/data/targets/rest_r1.pt'
    tb=torch.load(noise_path)['train'];identities[str(noise_path)]=sha(noise_path)
    olda=torch.load(Path.home()/'rest_wl57/data/targets/rest_r0.pt')['train']
    assert torch.equal(ta,olda),'Calibration r0 must be actual training labels'
    g=RG.device(torch.load(ORIG/'data/geometry/rest.pt'));eta=[];precision={}
    for coarse in (True,False):
        a=aggregate(ta[80:].cuda(),g['parent'],coarse).double();b=aggregate(tb[80:].cuda(),g['parent'],coarse).double()
        mid=(a+b)/2;signal=mid[::2]-mid[1::2];noise=((a[::2]-a[1::2])-(b[::2]-b[1::2]))/(2**.5)
        mean=.5*(mid[::2]+mid[1::2]);norm=lambda z:z.flatten(1,2).square().sum(1)
        ratios=norm(noise)/norm(mean).clamp(min=1e-30);eta.append(max(1e-4,float(ratios.median())))
        precision[str(coarse)]={'noise_to_contrast':(norm(noise)/norm(signal).clamp(min=1e-30)).sqrt().cpu().tolist(),'noise_to_pairmean_squared':ratios.cpu().tolist()}
    torch.manual_seed(5901);np.random.seed(5901);m=Transport().cuda();lights=torch.load(ORIG/'data/lights.pt')['train'];delta_vals=[];dir_vals=[];pair_tests=[]
    for i in range(8):
        x=lights[80+2*i:82+2*i];assert torch.allclose(x[0].sum(1),x[1].sum(1),rtol=1e-6,atol=1e-8);assert not torch.equal(x[0],x[1]);pair_tests.append(True)
    with torch.no_grad():
        for name in ('tr00','tr04','tr08','tr11'):
            gg=RG.device(torch.load(ORIG/'data/geometry'/f'{name}.pt'));tt=torch.load(ORIG/'data/training_targets'/f'{name}_r0.pt')['train']
            for p in range(8):
                ix=slice(80+2*p,82+2*p);x=lights[ix].cuda();t,tr=tt[ix].cuda(),ta[ix].cuda();y,yr=m(gg,x),m(g,x)
                dy,dt=aggregate(y-yr,gg['parent']),aggregate(t-tr,gg['parent']);base=aggregate(tr,g['parent'])
                den=dt.flatten(1,2).square().sum(1)+.0001*base.flatten(1,2).square().sum(1)+1e-20
                delta_vals.append(float(((dy-dt).flatten(1,2).square().sum(1)/den).mean()))
                dir_vals.append(float(.5*(contrast(y,t,gg['parent'],eta)+contrast(yr,tr,g['parent'],eta))))
    weight=.25*float(np.median(delta_vals))/float(np.median(dir_vals));assert np.isfinite(weight) and weight>0
    # Focused mathematical and actual-model controls.
    x=lights[80:82].cuda();t=ta[80:82].cuda();y=m(g,x)
    c=contrast(y,t,g['parent'],eta);sw=contrast(y.flip(0),t.flip(0),g['parent'],eta)
    assert torch.allclose(c,sw)
    gpu_zero=float(contrast(t,t,g['parent'],eta));assert gpu_zero<=1e-10
    assert float(contrast(t.cpu().double(),t.cpu().double(),g['parent'].cpu(),eta))==0
    base=loss_metric(y,t,g['parent']);assert total(base,c,0) is base
    m.zero_grad();base.backward();gr=[p.grad.clone() if p.grad is not None else None for p in m.parameters()]
    m.zero_grad();y2=m(g,x);b2=loss_metric(y2,t,g['parent']);total(b2,contrast(y2,t,g['parent'],eta),0).backward()
    maxdiff=max(float((a-p.grad).abs().max()) for a,p in zip(gr,m.parameters()) if a is not None)
    assert maxdiff<=1e-7,maxdiff
    from r57_eval import physics
    with torch.no_grad():ph=physics(m,g,x,m.coefficients(g))
    assert ph['passed'];assert sum(p.numel() for p in m.parameters())==157024
    tests={'all8_training_pairs_equal_spatial_distinct_angular':all(pair_tests),'swap_invariant':True,'CPU_double_exact_target_zero':True,'GPU_identical_target_roundoff':gpu_zero,'lambda_zero_identical_tensor':True,'baseline_gradient_max_abs_difference':maxdiff,'source_independent_runtime_unchanged':True,'physical':ph,'parameter_count':157024}
    # Original train/eval inputs symlinked, original files never written.
    for arm in ('A','B'):
        ar=ROOT/'arms'/arm;ar.mkdir(parents=True);(ar/'data').symlink_to(ORIG/'data',target_is_directory=True);(ar/'out').mkdir();(ar/'checkpoints').mkdir()
        (ar/'out/gate1.json').write_text((ORIG/'out/gate1.json').read_text());(ar/'repo_commit.txt').write_text((ROOT/'source_commit.txt').read_text())
    dump(dict(eta=eta,**{'lambda':weight},rule=CFG['calibration'],train_only=True),ROOT/'fixed_objective.json')
    dump(dict(precision=precision,eta=eta,**{'lambda':weight},initial_delta=delta_vals,initial_directional=dir_vals,tests=tests,identities=identities,sources=sources,environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':subprocess.check_output(['nvidia-smi'],text=True)},source_commit=(ROOT/'source_commit.txt').read_text().strip()),ROOT/'calibration.json')
    print('CALIBRATION_TESTS_PASS',eta,weight,tests,flush=True)
if __name__=='__main__':setup();sys.stdout.flush();os._exit(0)
