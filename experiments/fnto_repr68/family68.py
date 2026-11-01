"""Exact admissible response family of the WL59 P intrinsic K for localized one-hot probes (no IO).

For source port i, incoming bin a, channel c (WL59 runtime algebra, r57_model.Transport.coefficients/apply):
  scatter[a,j,o,c] = d[i,a] * sum_r A[r,a,c] T[j,r,c] O[r,o,c],   d = 1 - beta_model (geometry-only quadrature)
  T[j,r,c] = sum_{e: src=i, dst=j} W[e,r,c] + sum_k U[j,k,r,c] V[i,k,r,c]
  A = softmax(in_logits, dim r): every column a is a probability vector over 8 mixtures, global (all ports/geometries)
  O = softmax(out_logits, dim o): every mixture is a probability distribution over 64 outgoing bins, global
  W = softplus(edge MLP) * prior * mass/norm on the 128 nearest rest-coordinate neighbours of i
  U = softmax over ALL destination ports (rank 16 per mixture/channel, per geometry), V = softplus * mass/norm
  => T >= 0 and sum_j T[j,r,c] = mass[i,r,c] = sigmoid(.) in (0,1)   (exact passivity normalisation).
The coefficient-level class FCoef keeps A, O (global simplex), T >= 0 and sum_j T <= 1 and frees only the map from
geometry features to T. With 4 probe sources per geometry and rank 16 >= 4 every such T is approached by W+UV, so FCoef
is the closure of the network family's response set on the probe population minus geometry conditioning.
"""
import torch
import torch.nn.functional as F

M,TR,C,NB=8,16,3,16

def bin_map(device='cpu'):
    import numpy as np
    f=2;b=np.arange(64);return torch.as_tensor((b%8)//f+4*((b//8)//f),device=device)

def network_columns(model,g,ports):
    """Exact restriction of Transport.coefficients to the given source ports. Returns T (n,P,M,C), A (M,16,C), O (M,64,C)."""
    st=model.state(g);z=st['z'];P=g['P'];nh=model.node_head(z)
    U=torch.softmax(nh[:,:TR*M*C].view(P,TR,M,C),dim=0)
    V=F.softplus(nh[:,TR*M*C:2*TR*M*C].view(P,TR,M,C));mass=torch.sigmoid(nh[:,2*TR*M*C:].view(P,M,C))
    out=[];cache=g.setdefault('_edges68',{})
    for i in ports:
        if i not in cache:cache[i]=(g['src']==i).nonzero().flatten()
        e=cache[i]
        h=st['s'][g['src'][e]]+st['d'][g['dst'][e]]+model.pair_layer(g['pair'][e])
        Wr=F.softplus(model.edge(h)).view(-1,M,C)*g['prior'][e][:,None,:]
        scale=mass[i]/(Wr.sum(0)+V[i].sum(0)+1e-12)
        T=torch.zeros(P,M,C,device=Wr.device,dtype=Wr.dtype).index_add(0,g['dst'][e],Wr*scale[None])
        out.append(T+torch.einsum('jkrc,krc->jrc',U,V[i]*scale[None]))
    return torch.stack(out),torch.softmax(model.in_logits,dim=0),torch.softmax(model.out_logits,dim=1)

def predict(T,d,A,O,space,parent=None,bm=None):
    """T (n,P,M,C), d (n,16) -> scatter (n,16,Q,nout,C) in 'native' (P,64) / 'fine16' (P,16) / 'coarse' (240,16)."""
    if space!='native':O=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,bm,O)
    if space=='coarse':T=torch.zeros(T.shape[0],240,M,C,device=T.device,dtype=T.dtype).index_add(1,parent,T)
    return torch.einsum('na,rac,njrc,roc->najoc',d,A,T,O)

def aggregate(y,space,parent,bm):
    """y (...,P,64,C) flux -> space (sums; r57 aggregate semantics)."""
    if space=='native':return y
    z=torch.zeros(*y.shape[:-2],NB,C,device=y.device,dtype=y.dtype).index_add(y.dim()-2,bm,y)
    if space=='fine16':return z
    return torch.zeros(*z.shape[:-3],240,NB,C,device=y.device,dtype=y.dtype).index_add(z.dim()-3,parent,z)

def Hmat(device,dtype=torch.float32):return torch.eye(NB,device=device,dtype=dtype)-torch.full((NB,NB),1./NB,device=device,dtype=dtype)

class Target:
    """Diagnostic fitting target for one geometry: n probe sources, scatter S (n,16,P,64,C) and model budget d (n,16)."""
    def __init__(self,S,d,parent,bm):
        self.d=d;self.parent=parent;self.bm=bm;Hm=Hmat(S.device,S.dtype)
        self.S={'native':S,'fine16':aggregate(S,'fine16',parent,bm),'coarse':aggregate(S,'coarse',parent,bm)}
        self.HZ=torch.einsum('ab,nbqoc->naqoc',Hm,self.S['coarse']/d[:,:,None,None,None])
        self.den={s:v.square().sum((1,2,3)) for s,v in self.S.items()};self.den['con']=self.HZ.square().sum((1,2,3))
        # Channel-major copies (C, n*a*j, o) for so_contract (execution amendment 2; same values, layout only).
        self.Sc={s:self.S[s].permute(4,0,1,2,3).reshape(S.shape[-1],-1,self.S[s].shape[-2]).contiguous() for s in ('native','fine16')}

def so_contract(S,O,Sc=None):
    """X[n,a,j,r,c] = sum_o S[n,a,j,o,c] O[r,o,c]. Given Sc (S laid out channel-major) the O-gradient is a wide GEMM
    instead of a ~245k-long cuBLAS reduction (amendment 2: same contraction, float32 round-off only)."""
    if Sc is None:return torch.einsum('najoc,roc->najrc',S,O)
    n,a,j=S.shape[:3]
    return torch.bmm(O.permute(2,0,1),Sc.transpose(1,2)).view(S.shape[-1],O.shape[0],n,a,j).permute(2,3,4,1,0)

def sq_error(T,d,A,O,S,den,Sc=None):
    """|S_hat - S|^2 per (source, channel) without materialising S_hat (exact algebraic identity):
    |S_hat|^2 = sum_a d^2 sum_rs A_ra A_sa (T'T)_rs (OO')_rs ; <S_hat,S> = sum_a d sum_r A_ra sum_j T_jr (S O')_ajr."""
    GT=torch.einsum('njrc,njsc->nrsc',T,T);GO=torch.einsum('roc,soc->rsc',O,O)
    hh=torch.einsum('na,rac,sac,nrsc,rsc->nc',d*d,A,A,GT,GO)
    hs=torch.einsum('na,rac,njrc,najrc->nc',d,A,T,so_contract(S,O,Sc))
    return hh-2*hs+den

def objective(T,A,O,tg,parts=False):
    """Predeclared diagnostic objective: mean over (probe source, channel) of
    [mean over spaces of |S_hat-S|^2/|S|^2 (native, fine16, coarse)] + |H(Z_hat-Z)|^2/|HZ|^2 (coarse contrast)."""
    out={}
    O16=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,tg.bm,O)
    out['native']=sq_error(T,tg.d,A,O,tg.S['native'],tg.den['native'],tg.Sc['native'])/tg.den['native']
    out['fine16']=sq_error(T,tg.d,A,O16,tg.S['fine16'],tg.den['fine16'],tg.Sc['fine16'])/tg.den['fine16']
    for s in ('coarse',):
        out[s]=(predict(T,tg.d,A,O,s,tg.parent,tg.bm)-tg.S[s]).square().sum((1,2,3))/tg.den[s]
    zc=predict(T,tg.d,A,O,'coarse',tg.parent,tg.bm)/tg.d[:,:,None,None,None]
    out['con']=(torch.einsum('ab,nbqoc->naqoc',Hmat(T.device,T.dtype),zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']
    J=((out['native']+out['fine16']+out['coarse'])/3+out['con']).mean()
    return (J,{k:v.detach() for k,v in out.items()}) if parts else J

def objective_direct(T,A,O,tg):
    """Reference implementation (materialised prediction); used by the tests to certify objective()."""
    v=sum((predict(T,tg.d,A,O,s,tg.parent,tg.bm)-tg.S[s]).square().sum((1,2,3))/tg.den[s] for s in ('native','fine16','coarse'))/3
    zc=predict(T,tg.d,A,O,'coarse',tg.parent,tg.bm)/tg.d[:,:,None,None,None]
    return (v+(torch.einsum('ab,nbqoc->naqoc',Hmat(T.device,T.dtype),zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']).mean()

class FCoef(torch.nn.Module):
    """Coefficient-level admissible class: global simplex A/O, per-source T >= 0 with sum_j T = sigmoid(mass) < 1."""
    def __init__(self,A_logit,O_logit,mass_logit,sp_logit):
        super().__init__()
        self.in_logits=torch.nn.Parameter(A_logit.clone());self.out_logits=torch.nn.Parameter(O_logit.clone())
        self.mass=torch.nn.ParameterList([torch.nn.Parameter(m.clone()) for m in mass_logit])
        self.sp=torch.nn.ParameterList([torch.nn.Parameter(s.clone()) for s in sp_logit])
    def columns(self,k):
        return torch.sigmoid(self.mass[k])[:,None]*torch.softmax(self.sp[k],dim=1),torch.softmax(self.in_logits,0),torch.softmax(self.out_logits,1)
    @staticmethod
    def logits_from(T):
        """Inverse map of an admissible T (n,P,M,C) with 0 < sum_j T < 1 (mass logit, spatial logit)."""
        m=T.sum(1).clamp(1e-12,1-1e-7);return torch.log(m)-torch.log1p(-m),torch.log(T.clamp(min=1e-30))

def project_capped(v,cap=1.):
    """Euclidean projection of each column v[..., :] (last dim) onto {x >= 0, sum x <= cap}."""
    w=v.clamp(min=0);over=w.sum(-1)>cap
    if over.any():
        u=v[over];s,_=torch.sort(u,dim=-1,descending=True);cs=s.cumsum(-1)-cap
        k=torch.arange(1,u.shape[-1]+1,device=u.device,dtype=u.dtype);cond=s-cs/k>0
        rho=cond.to(u.dtype).cumsum(-1).argmax(-1,keepdim=True)
        theta=cs.gather(-1,rho)/(rho+1).to(u.dtype);w=w.clone();w[over]=(u-theta).clamp(min=0)
    return w

def fista(f,x0,proj,iters,L0=1.,eta=2.):
    """Accelerated projected gradient with backtracking and adaptive restart for a convex smooth f. Returns x, history."""
    with torch.enable_grad():return _fista(f,x0,proj,iters,L0,eta)

def _fista(f,x0,proj,iters,L0,eta):
    x=proj(x0);y=x.clone();t=1.;L=L0;hist=[]
    fx=float(f(x))
    for it in range(iters):
        y=y.detach().requires_grad_(True);fy=f(y);g,=torch.autograd.grad(fy,y);fy=float(fy)
        while True:
            xn=proj(y.detach()-g/L);dx=xn-y.detach()
            fn=float(f(xn))
            if fn<=fy+float((g*dx).sum())+.5*L*float(dx.square().sum())+1e-14*abs(fy):break
            L*=eta
        tn=.5*(1+(1+4*t*t)**.5)
        if fn>fx:y=x.clone();t=1.;continue  # adaptive restart (monotone)
        y=xn+((t-1)/tn)*(xn-x);x,fx,t=xn,fn,tn;L/=1.2
        if it%50==0 or it==iters-1:hist.append(dict(iter=it,f=fx,L=L))
    # Projected-gradient residual as a convergence/optimality certificate proxy.
    xg=x.detach().requires_grad_(True);g,=torch.autograd.grad(f(xg),xg);pg=float((x-proj(x-g/L)).norm()*L)
    return x.detach(),dict(history=hist,final=fx,projected_gradient=pg,L=L)
