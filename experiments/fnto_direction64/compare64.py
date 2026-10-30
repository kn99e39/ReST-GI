"""Frozen A/B comparison: original primary + fresh directional + tagged diagnostics."""
import os,sys,json
from pathlib import Path
ROOT=Path.home()/'rest_direction64';ORIG=Path.home()/'rest_persist_v1';HERE=Path(__file__).resolve().parent
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'))
import torch,numpy as np
from common import RG
from runtime import Transport
from r57_data import aggregate,dump,sha

def directional(y,t):
    norm=lambda z:z.flatten(1,2).norm(dim=1)
    d=y[::2]-y[1::2];q=t[::2]-t[1::2]
    yc=d-d.mean((1,2),keepdim=True);tc=q-q.mean((1,2),keepdim=True)
    return dict(error=(norm(d-q)/norm(q).clamp(min=1e-30)).tolist(),amplitude=(norm(d)/norm(q).clamp(min=1e-30)).tolist(),correlation=((yc*tc).sum((1,2))/(norm(yc)*norm(tc)).clamp(min=1e-30)).tolist())
def ballistic(g,x):
    return torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(len(x),-1).T).T.reshape(len(x),g['P'],64) for c in range(3)],-1)
def main():
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    fresh=np.load(ROOT/'independent_reference.npz');ref=torch.from_numpy(fresh['reference']);x=torch.load(ROOT/'independent_lights.pt').cuda()
    pair7=np.load(Path.home()/'rest_components63/coarse_components.npz');r7=torch.from_numpy(pair7['reference']);oldx=torch.load(ORIG/'data/lights.pt')['test'][62:64].cuda()
    g=RG.device(torch.load(ORIG/'data/geometry/pt0.pt'));b=aggregate(ballistic(g,x),g['parent']).cpu().double();b7=aggregate(ballistic(g,oldx),g['parent']).cpu().double()
    results={};arrays={};historical=json.loads((ORIG/'out/heldout.json').read_text())
    for arm in ('A','B'):
        root=ROOT/'arms'/arm;held=json.loads((root/'out/heldout.json').read_text())
        for seed in (5901,5902):
            ck=torch.load(root/'checkpoints'/f'P_{seed}.pt');assert ck['seed']==seed and ck['steps']==6000
            m=Transport().cuda();m.load_state_dict(ck['model']);m.eval();co=m.coefficients(g)
            y=torch.cat([m.apply(g,x[i:i+2],co).cpu() for i in range(0,16,2)]);ym=aggregate(y.cuda(),g['parent']).cpu().double()
            y7=aggregate(m.apply(g,oldx,co),g['parent']).cpu().double();pred=torch.stack([ym,b,ym-b],1);pred7=torch.stack([y7,b7,y7-b7],1)
            row={'checkpoint_sha256':sha(root/'checkpoints'/f'P_{seed}.pt'),'fresh':{},'pair7_historically_inspected':{},'original':held[f'P_{seed}']}
            for c,name in enumerate(('T','B','Scat')):
                row['fresh'][name]=directional(pred[:,c],ref[:,c]);row['pair7_historically_inspected'][name]=directional(pred7[:,c],r7[:,c])
            # Explicit baseline drift versus preserved WL59 scores; no selection or retuning.
            h=np.array(historical[f'P_{seed}']['poses']['pt0']['directional_pairs']['F']['errors']);a=np.array(row['original']['poses']['pt0']['directional_pairs']['F']['errors'])
            row['historical_direction_error_change']=(a-h).tolist();results[f'{arm}_{seed}']=row
            arrays[f'{arm}_{seed}_fresh']=pred.numpy();arrays[f'{arm}_{seed}_pair7']=pred7.numpy();del m,co,y;torch.cuda.empty_cache()
    decisions={};precision=json.loads((ROOT/'independent_complete.json').read_text())
    for seed in (5901,5902):
        a,b=results[f'A_{seed}'],results[f'B_{seed}'];pa=a['original']['poses']['pt0']['directional_pairs']['F'];pb=b['original']['poses']['pt0']['directional_pairs']['F'];mask=np.array(pa['detectable'])&np.array(pb['detectable']);ea=np.array(pa['errors']);eb=np.array(pb['errors'])
        fa=np.array(a['fresh']['T']['error']);fb=np.array(b['fresh']['T']['error'])
        decisions[str(seed)]={'original_direction_mean_A':float(ea[mask].mean()),'original_direction_mean_B':float(eb[mask].mean()),'original_mean_ratio_B_A':float(eb[mask].mean()/ea[mask].mean()),'original_failures_A':int(((ea>.25)&mask).sum()),'original_failures_B':int(((eb>.25)&mask).sum()),'old_pairs0_6_mean_ratio_B_A':float(eb[:7].mean()/ea[:7].mean()),'fresh_mean_ratio_B_A':float(fb.mean()/fa.mean()),'fresh_failures_A':int((fa>.25).sum()),'fresh_failures_B':int((fb>.25).sum()),'complete_A':a['original']['useful_and_physical'],'complete_B':b['original']['useful_and_physical']}
    direction_ok=all(v['original_mean_ratio_B_A']<=.9 and v['original_failures_B']<=v['original_failures_A'] for v in decisions.values())
    beyond=precision['total_all24_qualified'] and all(v['old_pairs0_6_mean_ratio_B_A']<=.9 and v['fresh_mean_ratio_B_A']<=.9 for v in decisions.values())
    complete=all(v['complete_B'] for v in decisions.values())
    dump(dict(models=results,decisions=decisions,directional_improvement=direction_ok,beyond_pair7=beyond,complete_correction=complete,coupled_GI_eligible=complete,fresh_precision_qualified=precision['total_all24_qualified'],note='Single objective/2seeds; fresh sources on known pt0 only; original Pair7 informed intervention; no external baseline superiority'),ROOT/'comparison.json')
    np.savez_compressed(ROOT/'comparison_components.npz',**arrays)
    print(json.dumps(decisions,indent=2),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
