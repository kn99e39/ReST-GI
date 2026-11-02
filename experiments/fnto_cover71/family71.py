"""Batched restriction of the unchanged WL59 network to a set of source ports (same algebra as family68.network_columns,
one batched pass instead of a Python loop per port; infrastructure only, certified against family68 in test71)."""
import torch
import torch.nn.functional as F
from family68 import M,TR,C

def edges(g,ports):
    """Edge ids of each source (n,128): src is np.repeat(arange(P),128) in r57_geometry, verified here."""
    key=tuple(int(i) for i in ports);cache=g.setdefault('_edges71',{})
    if key not in cache:
        e=torch.stack([(g['src']==i).nonzero().flatten() for i in key])
        assert bool((g['src'][e]==torch.as_tensor(key,device=e.device)[:,None]).all()) and e.shape[1]==128
        assert all(len(set(r.tolist()))==128 for r in g['dst'][e]),'distinct receivers per source'
        cache[key]=e
    return cache[key]

def network_columns(model,g,ports,state=None):
    """T (n,P,M,C), A (M,16,C), O (M,64,C) for the given source ports."""
    st=model.state(g) if state is None else state;z=st['z'];P=g['P'];nh=model.node_head(z)
    U=torch.softmax(nh[:,:TR*M*C].view(P,TR,M,C),dim=0)
    V=F.softplus(nh[:,TR*M*C:2*TR*M*C].view(P,TR,M,C));mass=torch.sigmoid(nh[:,2*TR*M*C:].view(P,M,C))
    e=edges(g,ports);src=g['src'][e];dst=g['dst'][e];n=e.shape[0]
    h=st['s'][src]+st['d'][dst]+model.pair_layer(g['pair'][e])
    Wr=F.softplus(model.edge(h)).view(n,-1,M,C)*g['prior'][e][:,:,None,:]
    pi=torch.as_tensor([int(i) for i in ports],device=z.device)
    scale=mass[pi]/(Wr.sum(1)+V[pi].sum(1)+1e-12)
    T=torch.zeros(n,P,M,C,device=z.device,dtype=Wr.dtype).scatter_add(1,dst[:,:,None,None].expand(-1,-1,M,C),Wr*scale[:,None])
    T=T+torch.einsum('jkrc,nkrc->njrc',U,V[pi]*scale[:,None])
    return T,torch.softmax(model.in_logits,dim=0),torch.softmax(model.out_logits,dim=1)
