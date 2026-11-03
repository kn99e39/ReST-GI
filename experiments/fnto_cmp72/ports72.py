"""te2 geometry identity gate and confirmation-port selection (geometry only; no response, model output or error is read).
Excludes exact-symmetry images of every previously referenced port (fairness for Candidate A). Writes ROOT/ports.json (immutable)."""
import sys,json
import numpy as np
import torch
from common72 import CFG,ROOT,INP68,R71,write_new,sha,tensor_sha,environment
import sym72 as SY

KEYS=('xyz','rest_xyz','area','node','pair','prior','parent','src','dst','graph')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    RG=environment();torch.set_grad_enabled(False)
    if (ROOT/'ports.json').exists():raise FileExistsError('immutable')
    G={'rest':load(INP68/'wl59/geometry_rest.pt'),'te2':load(ROOT/'inputs/geometry_te2.pt')}
    fresh=RG.geometry('te2',ballistic=True)
    for k in KEYS:assert torch.equal(fresh[k].cpu(),G['te2'][k]),('te2 identity',k)
    bf,ba=fresh['beta'].cpu().double(),G['te2']['beta'].double()
    ident=dict(features_bitwise_equal=True,beta_rel_L2=float((bf-ba).norm()/ba.norm()),archived_sha256=sha(ROOT/'inputs/geometry_te2.pt'),pose=list(RG.POSES['te2']))
    x0=G['rest']['rest_xyz'].numpy().astype(np.float64);face=np.rint(G['rest']['node'][:,15].numpy()*5).astype(int)
    for g in G.values():assert np.array_equal(np.rint(g['node'][:,15].numpy()*5).astype(int),face) and np.allclose(g['rest_xyz'].numpy(),x0)
    ref=json.loads((R71/'ports.json').read_text())['slots'];refp=sorted({int(v['port']) for v in ref.values()})
    perms=[SY.permutation(x0,op) for op in SY.ops(True)];images=sorted({int(p[i]) for p in perms for i in refp})
    chosen=[];rec={}
    for s in CFG['confirmation']['slots']:
        m=np.where(face==s['face'])[0];t=np.array(s['target']);dist=np.round(np.linalg.norm(x0[m]-t,axis=1),9);order=m[np.lexsort((m,dist))]
        for i in order:
            if i in images or i in chosen:continue
            if s['face'] in (4,5) and min(.05-abs(x0[i,0]),.05-abs(x0[i,1]))<.01-1e-12:continue
            if min(float((1-g['beta'][i]).min()) for g in G.values())<.5:continue
            blk=[j for j in images+chosen if face[j]==face[i]]
            if blk and min(np.linalg.norm(x0[i]-x0[j]) for j in blk)<.01-1e-12:continue
            chosen.append(int(i));break
        else:raise RuntimeError(f"STOP: no admissible port for slot {s['slot']}")
        i=chosen[-1];rec[str(s['slot'])]=dict(port=i,name=s['name'],face=int(face[i]),rest_xyz=x0[i].tolist(),d_min={n:float((1-g['beta'][i]).min()) for n,g in G.items()},
            dz_mm={n:float((g['xyz'][i,2]-g['rest_xyz'][i,2])*1e3) for n,g in G.items()},min_dist_to_reference_images_mm=float(min(np.linalg.norm(x0[i]-x0[j]) for j in images if face[j]==face[i])*1e3) if any(face[j]==face[i] for j in images) else None,
            beta_row_sha256={n:tensor_sha(g['beta'][i]) for n,g in G.items()})
    write_new(ROOT/'ports.json',dict(slots=rec,confirmation_ports=chosen,referenced_ports=refp,symmetry_images=len(images),te2_identity=ident,information_used='geometry only'))
    print('PORTS',chosen,json.dumps(ident),flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
