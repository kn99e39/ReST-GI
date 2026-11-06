"""WL75 rank-generic W+UV intrinsic K (Stage 0 contract). The WL59 'P' runtime network (runtime.Transport) with the mixture rank M
as a constructor argument; every other layer, the node rank TR = 16, the geometry features and the passivity normalisation are unchanged.

  K_scatter[i,j,a,o,c] = d[i,a] sum_{r<M} A[r,a,c] T[i,j,r,c] O[r,o,c],  d = 1 - beta (geometry-only ballistic budget)
  T[i,j,r,c] = (W[e(i,j),r,c] + sum_k U[j,k,r,c] V[i,k,r,c]) * mass[i,r,c] / (sum_e W[e,r,c] + sum_k V[i,k,r,c])
  A = softmax(in_logits (M,16,C), dim r), O = softmax(out_logits (M,64,C), dim o)   (global, frozen during spatial fitting)
  => T >= 0, sum_j T[i,j,r,c] = mass in (0,1), K >= 0, sum_{j,o} K <= d <= 1: positive, linear, zero-preserving and passive for any M.

Rank-dependent parameters (the only shape changes): edge[-1] Linear(128, M*C); node_head[-1] Linear(144, 2*TR*M*C + M*C) (U, V, mass);
in_logits (M,16,C); out_logits (M,64,C). Layout: edge row r*C+c; node_head rows t*M*C+r*C+c (U), TR*M*C + same (V), 2*TR*M*C + r*C+c (mass).
"""
import os,json
import torch
from torch import nn
import torch.nn.functional as F
from common75 import CFG,ROOT,R74,sha,tensor_sha,environment

TR,C,NB=16,3,16
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

environment()   # WL59 runtime path (idempotent)
from runtime import Transport as _P59  # noqa: E402

class TransportM(_P59):
    def __init__(self,M):
        super().__init__();self.M=M
        h=self.edge[-1].in_features;nh=self.node_head[-1].in_features
        self.edge[-1]=nn.Linear(h,M*C);self.node_head[-1]=nn.Linear(nh,2*TR*M*C+M*C)
        self.in_logits=nn.Parameter(torch.zeros(M,16,C));self.out_logits=nn.Parameter(torch.zeros(M,64,C))

    def coefficients(self,g,state=None):
        """r57_model.Transport.coefficients with M = self.M (identical arithmetic)."""
        st=self.state(g) if state is None else state;src,dst=g['src'],g['dst'];M=self.M;parts=[]
        chunk=int(os.environ.get('WL57_EDGE_CHUNK','65536'))
        for i in range(0,len(src),chunk):
            sl=slice(i,min(i+chunk,len(src)))
            h=st['s'][src[sl]]+st['d'][dst[sl]]+self.pair_layer(g['pair'][sl])
            parts.append(F.softplus(self.edge(h)).view(-1,M,C)*g['prior'][sl][:,None,:])
        W=torch.cat(parts,0);nh=self.node_head(st['z']);p=g['P']
        U=torch.softmax(nh[:,:TR*M*C].view(p,TR,M,C),dim=0)
        V=F.softplus(nh[:,TR*M*C:2*TR*M*C].view(p,TR,M,C))
        mass=torch.sigmoid(nh[:,2*TR*M*C:].view(p,M,C))
        norm=torch.zeros(p,M,C,device=nh.device,dtype=nh.dtype).index_add_(0,src,W)+V.sum(1)
        scale=mass/(norm+1e-12)
        return dict(W=W*scale[src],U=U,V=V*scale[:,None],A=torch.softmax(self.in_logits,dim=0),O=torch.softmax(self.out_logits,dim=1))

    def apply(self,g,x,coeff):
        """r57_model.Transport.apply with M = self.M: x (batch,P,16,RGB) incident flux -> (batch,P,64,RGB) outgoing flux."""
        M=self.M;q=torch.einsum('biac,rac->birc',x*(1-g['beta'])[None,:,:,None],coeff['A'])
        y=torch.zeros(x.shape[0],g['P'],M,C,device=x.device,dtype=x.dtype)
        chunk=int(os.environ.get('WL57_EDGE_CHUNK','65536'))
        for i in range(0,len(g['src']),chunk):
            sl=slice(i,min(i+chunk,len(g['src'])))
            y=y.index_add(1,g['dst'][sl],q[:,g['src'][sl]]*coeff['W'][sl][None])
        latent=torch.einsum('birc,ikrc->bkrc',q,coeff['V'])
        y=y+torch.einsum('jkrc,bkrc->bjrc',coeff['U'],latent)
        scatter=torch.einsum('bjrc,roc->bjoc',y,coeff['O'])
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(x.shape[0],-1).T).T.reshape(x.shape[0],g['P'],64) for c in range(3)],-1)
        return scatter+ball

    def column_sums(self,g,coeff):
        s=torch.zeros(g['P'],self.M,C,device=g['src'].device).index_add_(0,g['src'],coeff['W'])+coeff['V'].sum(1)
        return g['beta'][:,:,None]+(1-g['beta'])[:,:,None]*torch.einsum('irc,rac->iac',s,coeff['A'])

def columns(model,g,ports,state=None):
    """family71.network_columns with M = model.M (identical arithmetic): T (n,P,M,C), A (M,16,C), O (M,64,C)."""
    import family71 as FW
    M=model.M;st=model.state(g) if state is None else state;z=st['z'];P=g['P'];nh=model.node_head(z)
    U=torch.softmax(nh[:,:TR*M*C].view(P,TR,M,C),dim=0)
    V=F.softplus(nh[:,TR*M*C:2*TR*M*C].view(P,TR,M,C));mass=torch.sigmoid(nh[:,2*TR*M*C:].view(P,M,C))
    e=FW.edges(g,ports);src=g['src'][e];dst=g['dst'][e];n=e.shape[0]
    h=st['s'][src]+st['d'][dst]+model.pair_layer(g['pair'][e])
    Wr=F.softplus(model.edge(h)).view(n,-1,M,C)*g['prior'][e][:,:,None,:]
    pi=torch.as_tensor([int(i) for i in ports],device=z.device)
    scale=mass[pi]/(Wr.sum(1)+V[pi].sum(1)+1e-12)
    T=torch.zeros(n,P,M,C,device=z.device,dtype=Wr.dtype).scatter_add(1,dst[:,:,None,None].expand(-1,-1,M,C),Wr*scale[:,None])
    T=T+torch.einsum('jkrc,nkrc->njrc',U,V[pi]*scale[:,None])
    return T,torch.softmax(model.in_logits,dim=0),torch.softmax(model.out_logits,dim=1)

# ------------------------------------------------------------------ rank embedding (duplication map r -> r mod 8)
def rank_rows(M,M0=8):
    """Row indices into the rank-M0 rank-dependent layers that give the rank-M layers under r -> r mod M0."""
    r=torch.arange(M)%M0;c=torch.arange(C);t=torch.arange(TR)
    e=(r[:,None]*C+c[None]).flatten()
    U=(t[:,None,None]*M0*C+r[None,:,None]*C+c[None,None]).flatten();V=U+TR*M0*C;ms=2*TR*M0*C+e
    return e,torch.cat([U,V,ms])

def embed_rank(st8,M,logits=True):
    """State dict of a rank-M network whose mixture r is a copy of rank-8 mixture r mod 8. With logits=True the in/out logits are
    repeated too (each copy's A is halved by the softmax over 2x as many equal logits): the rank-M operator equals the rank-8 one."""
    e,nh=rank_rows(M);st={k:v.clone() for k,v in st8.items()}
    st['edge.3.weight']=st8['edge.3.weight'][e].clone();st['edge.3.bias']=st8['edge.3.bias'][e].clone()
    st['node_head.2.weight']=st8['node_head.2.weight'][nh].clone();st['node_head.2.bias']=st8['node_head.2.bias'][nh].clone()
    if logits:k=M//8;st['in_logits']=st8['in_logits'].repeat(k,1,1);st['out_logits']=st8['out_logits'].repeat(k,1,1)
    return st

# ------------------------------------------------------------------ frozen G2 basis (selected by the fitting objective only)
def basis():
    """Among the WL74 G2 restarts the one with the lowest final FITTING objective J (S16-R fitting replicas 0-3), read from the basis
    files only (never WL74 Stage 1/2 scores). Provenance asserted against the protocol (id, sha256, rank, J_final)."""
    bc=CFG['basis'];J={b:float(load(R74/'bases'/f'{b}.pt',mmap=True)['J_final']) for b in bc['candidates']};bid=min(J,key=J.get)
    f=R74/'bases'/f'{bid}.pt';h=sha(f);assert bid==bc['selected'] and h==bc['sha256'],('basis provenance',bid,h)
    b=load(f);assert b['rank']==16 and b['arm']=='G2' and abs(b['J_final']-bc['J_final'])<=1e-12
    a,o=b['in_logits'],b['out_logits'];assert a.shape==(16,16,3) and o.shape==(16,64,3) and a.dtype==torch.float64
    meta=dict(id=bid,sha256=h,J_final_by_restart=J,in_logits_sha256=tensor_sha(a),out_logits_sha256=tensor_sha(o),dtype_stored=str(a.dtype),
              dtype_model='float32 (rounded once at load)',fit_seconds=float(b['seconds']))
    return a.float(),o.float(),meta

def init_model(init,M=16):
    """Predeclared initialisation: trunk (encoder, messages, src/dst/pair layers, edge and node_head hidden layers) = P_590x bitwise;
    rank-dependent final layers = P_590x rows of mixture r mod 8 (the same duplication that defines G2_r0's restart-0 start from the
    WL68 basis); in/out logits = the frozen G2 basis. No new random channel."""
    import fit71
    st8=fit71.guarded(init)['model'];st=embed_rank(st8,M,logits=False);a,o,meta=basis();st['in_logits']=a;st['out_logits']=o
    m=TransportM(M).cuda();m.load_state_dict(st);return m,meta

def load_run(path):
    """A WL75 run checkpoint -> rank-16 network in eval mode."""
    x=load(path);m=TransportM(x['model']['in_logits'].shape[0]).cuda();m.load_state_dict(x['model']);m.eval();return m

def parameter_counts(m):
    allp=sum(p.numel() for p in m.parameters());ao=m.in_logits.numel()+m.out_logits.numel()
    rk=sum(p.numel() for n,p in m.named_parameters() if n.startswith(('edge.3','node_head.2')))
    return dict(learned_parameters_excl_fixed_AO=allp-ao,fixed_AO_logits=ao,rank_dependent_final_layers=rk,total=allp)
