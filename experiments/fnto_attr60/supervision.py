"""Existing-label precision and frozen-input coverage; CPU only, no new reference."""
import argparse,json,time
from pathlib import Path
import torch
from analyze_stored import agg,load,sha,ballistic

def norms(z):return z.flatten(1,2).norm(dim=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();root=a.root;torch.set_num_threads(4);torch.set_grad_enabled(False);t0=time.perf_counter()
    g=load(root/'data/geometry/rest.pt');lights=load(root/'data/lights.pt');old=root.parent/'rest_wl57';paths=[root/'data/geometry/rest.pt',root/'data/lights.pt']
    out=dict(source_commit=a.source_commit,script_sha256=sha(__file__),device='CPU',inputs={},same_population_rest_noise={},coverage={})
    p0=old/'data/targets/rest_r0.pt';p1=old/'data/targets/rest_r1.pt';paths += [p0,p1];ys=[load(p) for p in (p0,p1)]
    for split in ('train','validation'):
        r=[agg(y[split],g['parent']) for y in ys];mid=.5*(r[0]+r[1]);n=(r[0]-r[1])/(2**.5)
        for_pairs=lambda z:z[-16::2]-z[-15::2]
        signal=for_pairs(mid);noise=for_pairs(n)
        out['same_population_rest_noise'][split]=dict(single_replica_action_noise_mean=(norms(n)/norms(mid)).mean(0).tolist(),single_replica_directional_noise=(norms(noise)/norms(signal)).tolist(),method='Two original sparse-operator target replicas, identical rest geometry and split fields. (r0-r1)/sqrt(2) estimates a single-label noise norm; midpoint denominator. Not primary A2 test noise, not deformation-label noise, not a bias estimate.')
    train=lights['train'].double();trflat=train.reshape(len(train),-1);trnorm=trflat.norm(dim=1)
    for split,x in lights.items():
        xx=x.double();sp=xx.sum(2);ess=sp.sum(1).square()/sp.square().sum(1)
        bins=xx.sum((1,3)).argmax(1)
        ids=list(range(len(x)-16,len(x)))
        pop=[]
        for i in ids:
            flat=xx[i].flatten();cs=trflat@flat/(trnorm*flat.norm()).clamp(min=1e-30)
            pop.append(dict(field=i,incoming_bin=int(bins[i]),effective_material_cells_RGB=ess[i].tolist(),nearest_train_cosine=float(cs.max()),nearest_train_index=int(cs.argmax())))
        out['coverage'][split]=dict(directional_fields=pop,focused_effective_cells_RGB=ess[:16].tolist())
    # Positivity floor is for individual total actions, never an oracle collided decomposition.
    out['ballistic_positivity']={}
    for pose in ('rest','pt0'):
        gp=root/'data/geometry'/f'{pose}.pt';paths.append(gp);gg=load(gp);ba=agg(ballistic(gg,lights['test']),g['parent'])
        refs=[]
        for rep in (0,1):
            p=root/'data/targets'/f'{pose}_r{rep}.pt';paths.append(p);refs.append(agg(load(p)['test'],g['parent']))
        tt=.5*(refs[0]+refs[1]);floor=norms((ba-tt).clamp(min=0))/norms(tt)
        out['ballistic_positivity'][pose]=dict(per_field_channel=floor.tolist(),max_RGB=floor.max(0).values.tolist(),field62_63=floor[62:64].tolist(),method='Optimistic fixed-B nonnegative-scatter absolute-action lower bound ||positive(Bx-reference)||/||reference||. Includes reference error and approximate B; no isolated physical B oracle.')
    out['missing_control']='No independent tr02_r1 target exists. Full train-geometry delta uncertainty cannot be inferred from rest-only replica differences.'
    for p in dict.fromkeys(paths):out['inputs'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,resolved=str(p.resolve()))
    out['seconds']=time.perf_counter()-t0;print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':main()
