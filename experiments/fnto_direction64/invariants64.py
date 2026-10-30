"""Post-training checks of unchanged rigid/source-independent cache contracts."""
import os,sys,json
from pathlib import Path
ROOT=Path.home()/'rest_direction64';ORIG=Path.home()/'rest_persist_v1'
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'))
import torch
from common import loaded_geometry,rigid_geometry
from runtime import Transport,full,selective,apply
from r57_eval import physics
from r57_data import dump

def main():
    assert not (ROOT/'trained_invariants.json').exists();torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    rest=loaded_geometry('rest');rigid=rigid_geometry(ballistic=False);x=torch.load(ORIG/'data/lights.pt')['test'][:2].cuda();out={}
    rel=lambda a,b:float((a-b).abs().sum()/b.abs().sum().clamp(min=1e-30))
    for arm in ('A','B'):
        for seed in (5901,5902):
            m=Transport().cuda().eval();m.load_state_dict(torch.load(ROOT/'arms'/arm/'checkpoints'/f'P_{seed}.pt')['model'])
            c=full(m,rest);base=apply(m,c,x);co=id(c['co']);c,a=selective(m,c,rest);noop=rel(apply(m,c,x),base);ident=id(c['co'])==co
            c,a=selective(m,c,rigid);er=rel(apply(m,c,x),base);ph=physics(m,rest,x,m.coefficients(rest))
            retained=a['raw_retained_fraction']==1.;source=rel(base[:1],base[1:])
            passed=ph['passed'] and noop<=1e-5 and ident and er<=1e-6 and retained and source>0
            out[f'{arm}_{seed}']=dict(passed=passed,physics=ph,noop=noop,noop_cache_identity=ident,rigid=er,rigid_all_raw_retained=retained,independent_incident_output_difference=source,parameters=sum(p.numel() for p in m.parameters()))
            assert passed,out[f'{arm}_{seed}'];del c,m;torch.cuda.empty_cache()
    dump(out,ROOT/'trained_invariants.json');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
