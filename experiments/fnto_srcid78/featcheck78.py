"""WL78 pre-freeze feature check (geometry only; no reference, no model): W+UV-Q inputs at rest / pt0, probe statistics, input identity of
interior ports, and the input map of the Stage 0 correspondence pairs (source port and its 16-NN graph neighbourhood) under the exact
operation, expressed in the bin frames.
    python featcheck78.py <out.json>"""
import sys,json
import numpy as np,torch
from common78 import environment
RG=environment();torch.set_grad_enabled(False)
import fit71,sym72 as SY,q78 as Q
from scipy.spatial import cKDTree
out={}
for name in ('rest','pt0'):
    g=RG.device(fit71.guarded(f'geometry_{name}'));q=Q.QG(RG,g);node=q['node'][:,:16];t,b,n=Q.frames(RG,g)
    out[name]=dict(info=q['_q78_info'],node_min=node.min(0).values.tolist(),node_max=node.max(0).values.tolist(),
        distinct_inputs=int(torch.unique(node.round(decimals=6),dim=0).shape[0]),P=g['P'])
    if name=='rest':
        G=g;N=node;T,B,Nn=t,b,n
x=G['rest_xyz'].double().cpu().numpy();tree=cKDTree(x);pairs={}
for (ps,pf) in ((3439,224),(408,260),(527,1568),(2085,1716),(2085,976)):
    cand=[];nf=Nn[pf].double().cpu().numpy()
    for op in SY.ops(True):
        M=SY.matrix(op);d=x[pf]-M@x[ps]
        if abs(d@nf)<1e-9 and np.allclose(M@Nn[ps].double().cpu().numpy(),nf,atol=1e-6):cand.append((float(np.abs(d).sum()),op,M,d))
    _,op,M,d=min(cand,key=lambda c:c[0]);dd,ii=tree.query(x@M.T+d);mp=np.where(dd<1e-6,ii,-1);best=(op,M,d,mp)
    op,M,d,mp=best;Mt=torch.as_tensor(M,dtype=torch.float32,device='cuda')
    # frame map Q (3x3) taking the source port's frame components to the far port's frame components
    def fmat(i):return torch.stack([T[i],B[i],Nn[i]])
    Qm=fmat(pf)@Mt@fmat(ps).T
    def mapped(i,j):
        Qi=fmat(j)@Mt@fmat(i).T;v=N[i].clone()
        for sl in ((0,3),(6,9),(10,13)):v[sl[0]:sl[1]]=Qi@v[sl[0]:sl[1]]
        return v
    nb=G['graph'][ps].tolist();rows=[]
    for i in [ps]+nb:
        j=int(mp[i]);rows.append(float((N[j]-mapped(i,j)).abs().max()) if j>=0 else None)
    pairs[f'{ps}->{pf}']=dict(op=list(map(str,op)),delta_mm=(d*1e3).round(3).tolist(),frame_map=Qm.round(decimals=4).tolist(),
        source_input_maxdiff=rows[0],neighbourhood_input_maxdiff=max(r for r in rows if r is not None),
        source_raw=N[ps].tolist(),far_raw=N[pf].tolist())
out['pairs']=pairs;json.dump(out,open(sys.argv[1],'w'),indent=1);print(json.dumps(out)[:4000])
