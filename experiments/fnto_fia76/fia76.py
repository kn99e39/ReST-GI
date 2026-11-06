"""WL76 candidate FIA: first-interaction anchored intrinsic K (scattered part), isolated experiment.

  K_scat(i,a -> j,o,c) = albedo_c   * SS(i,a -> j,o)                                              exact single scatter (geometry-only quadrature)
                       + albedo_c^2 * sum_k W(i,a -> k) * R_theta(k -> j,o,c)                       learned, source-independent
  W(i,a -> k) = E_{entry, first collision x, isotropic w} [ c * (1 - exp(-sigma t2)) * 1{second collision y nearest to anchor k} ]
  SS(i,a -> j,o) = E [ c * exp(-sigma t2) * 1{the ray from x along w exits through (j, o)} ]
  sum_k W + sum_{j,o} SS = E[c] = 1 - beta(i,a)   (exact identity per quadrature sample: every sample's weight c is split)
Both W and SS are fixed-seed quadratures of geometry + sigma_t only (like the existing ballistic B; deployable, no reference, no learned part);
per cell they are rescaled so that sum W + sum SS = 1 - beta_B exactly (the budget of the existing ballistic quadrature).
Anchors: 4 depths (0.25, 0.75, 1.5, 3.0 mm) below every port along its current inward normal (current geometry; 4P anchors).
R_theta(k -> j,o,c) = rho_c(k) [ (1 - tau_c(k)) sum_{j in N(k)} softmax_j(log G_c(r_kj) + f_D) sum_m alpha_m(k,j,c) L_m(o,c)
                                +      tau_c(k)  sum_J softmax_J(log G_c(r_kJ) + f_T) (area_j / area_J) sum_m alphaT_m(k,J,c) L_m(o,c) ]
  the response of an isotropic emitter at the anchor (source-independent by the Markov property + g = 0); rho, tau = sigmoid, N(k) = all ports within the 512-th
  nearest distance (ties included), J = 240 coarse cells (tail), G_c = exp(-r/L_c) / (r^2 + (1 mm)^2) with learned L_c, L_m = 3 ring lobes per channel
  (a distribution over the 4 polar rings of the 8x8 concentric bin grid, uniform inside a ring = equal projected solid angle).
  Inputs are rotation/reflection-invariant: distances, receiver-normal cosines, normal products, sorted free paths from the anchor/receiver
  to the boundary (8 lateral + 2 normal directions) -> exactly equivariant under the slab symmetries (bins permute within rings).
Passive by construction: sum_{j,o} K_scat <= albedo E[c e1] + albedo^2 E[c (1 - e1)] max rho <= 1 - beta.
"""
import os,math,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common76 import CFG

C=3;NB=64;DEPTHS=(.00025,.00075,.0015,.003);NN=512;LOBES=3
def rings():
    b=torch.arange(64);u=b%8;v=b//8;return (torch.maximum((u-3.5).abs(),(v-3.5).abs())-.5).long()   # 0..3, counts 4/12/20/28
RING=rings();RCOUNT=torch.bincount(RING,minlength=4).float()

# ------------------------------------------------------------------ geometry-only operators
def lateral_dirs(n,k=8):
    """k unit directions in the tangent plane of each normal n (arbitrary in-plane origin; only sorted distances are used)."""
    a=torch.where(n[:,2:3].abs()<.9,torch.tensor([[0.,0.,1.]],device=n.device),torch.tensor([[1.,0.,0.]],device=n.device))
    t=F.normalize(torch.cross(n,a,dim=1),dim=1);b=torch.cross(n,t,dim=1);ang=torch.arange(k,device=n.device)*(2*math.pi/k)
    return torch.cos(ang)[None,:,None]*t[:,None]+torch.sin(ang)[None,:,None]*b[:,None]

def free_paths(dom,x,n,cap=.03):
    """Distances from x to the boundary along -n (inward), +n and 8 lateral directions -> (len(x), 10) [inward, outward, sorted lateral]."""
    dirs=torch.cat([-n[:,None],n[:,None],lateral_dirs(n)],1);m=dirs.shape[1]
    t,_=dom.isect(x[:,None].expand(-1,m,-1).reshape(-1,3).contiguous(),dirs.reshape(-1,3).contiguous());t=t.nan_to_num(posinf=cap).clamp(max=cap).view(-1,m)
    return torch.cat([t[:,:2],t[:,2:].sort(1).values],1)

class Geo:
    """Per-geometry deployable state that is not learned: anchors, their features, receiver features, neighbour lists, coarse tail data."""
    def __init__(self,g,dom):
        t0=time.perf_counter();self.g=g;self.dom=dom;P=g['P'];dev='cuda'
        x=g['xyz'].float();n=F.normalize(g['node'][:,9:12].float(),dim=1);self.P=P;self.x=x;self.n=n
        d=torch.tensor(DEPTHS,device=dev);self.anchor=(x[:,None]-d[None,:,None]*n[:,None]).reshape(-1,3).contiguous()   # anchor id = port*4 + layer
        an=n[:,None].expand(-1,4,-1).reshape(-1,3).contiguous();self.layer=torch.arange(4,device=dev).repeat(P)
        fp=free_paths(dom,self.anchor,an);self.afeat=torch.cat([fp/.01,torch.exp(-fp/.001),d.repeat(P)[:,None]/.001],1)            # (4P, 21)
        rf=free_paths(dom,x-.00025*n,n);self.rfeat=torch.cat([rf/.01,torch.exp(-rf/.001),g['node'][:,12:15].float(),n[:,2:3].abs()],1)   # (P, 24): area ratio, thickness, normal spread, |n_z| (the face id is not symmetry-invariant)
        from scipy.spatial import cKDTree
        ax=self.anchor.double().cpu().numpy();self.atree=cKDTree(ax);dd,ii=cKDTree(x.double().cpu().numpy()).query(ax,k=NN+NN//4)
        # every port within the NN-th neighbour distance, ties included (a hard NN cut breaks ties by index and is not symmetry-equivariant)
        thr=dd[:,NN-1:NN]*(1+1e-6);assert (dd[:,-1:]>thr).all(),'tie band exceeds the padded neighbour list'
        self.nbr=torch.as_tensor(ii,device=dev);self.nmask=torch.as_tensor(dd<=thr,device=dev)
        par=g['parent'];area=g['area'].float();self.parent=par;self.area=area
        aJ=torch.zeros(240,device=dev).index_add_(0,par,area);self.cx=torch.zeros(240,3,device=dev).index_add_(0,par,x*area[:,None])/aJ[:,None]
        self.cn=F.normalize(torch.zeros(240,3,device=dev).index_add_(0,par,n*area[:,None]),dim=1);self.afrac=area/aJ[par]
        self.cfeat=torch.zeros(240,self.rfeat.shape[1],device=dev).index_add_(0,par,self.rfeat*area[:,None])/aJ[:,None]
        self.seconds=time.perf_counter()-t0

def quadrature(G,cells,q,sd,chunk=2_000_000):
    """W (anchor) and SS (exit) sparse rows for the given input cells (port*16 + bin), q samples each, fixed seed. Returns dict of
    COO lists: W: (cell, anchor, weight), SS: (cell, port*64 + bin, weight); weights achromatic and already divided by q."""
    import r57_geometry as RG
    import phys76 as PH
    dom=G.dom;gen=torch.Generator(device='cuda').manual_seed(sd);Wr=[];Wc=[];Wv=[];Sr=[];Sc=[];Sv=[];per=max(1,chunk//q);sig=dom.sigma_t;GG=RG.G
    for c0 in range(0,len(cells),per):
        cl=cells[c0:c0+per].repeat_interleave(q);n=len(cl)
        x0,d,nrm,_,_=RG.TR.cell_source(dom,cl,1,16,gen);x0=x0-dom.eps*nrm
        t,_=dom.isect(x0,d);ok=torch.isfinite(t);c=torch.zeros_like(t);c[ok]=-torch.expm1(-sig*t[ok])
        u=torch.rand(n,device='cuda',generator=gen);s=torch.zeros_like(t);s[ok]=-torch.log1p(-u[ok]*c[ok])/sig;s=torch.minimum(s,t.nan_to_num(posinf=0.)*(1-1e-7))
        x1=x0+s[:,None]*d;w1=PH.iso(n,gen);t2,tri=dom.isect(x1,w1);f=torch.isfinite(t2)&ok
        e=torch.zeros_like(t);e[f]=torch.exp(-sig*t2[f]);u2=torch.rand(n,device='cuda',generator=gen)
        s2=torch.zeros_like(t);s2[f]=-torch.log1p(-u2[f]*(1-e[f]))/sig;s2=torch.minimum(s2,t2.nan_to_num(posinf=0.)*(1-1e-7))
        y=x1+s2[:,None]*w1;wc=c*(1-e)/q;ws=c*e/q
        i=f.nonzero().squeeze(1);k=torch.as_tensor(G.atree.query(y[i].double().cpu().numpy(),workers=-1)[1],device='cuda')
        Wr.append(cl[i]);Wc.append(k);Wv.append(wc[i].double())
        ex=f.clone();ex[f]=(w1[f]*dom.n[tri[f]]).sum(1)>0;j=ex.nonzero().squeeze(1);tj=tri[j]
        ob=GG.local_to_bin(GG.to_local(w1[j],dom.n[tj],dom.t[tj],dom.b[tj]),8);Sr.append(cl[j]);Sc.append(dom.tri_port[tj]*64+ob);Sv.append(ws[j].double())
    cat=lambda L:torch.cat(L) if L else torch.empty(0,device='cuda')
    def coalesce(r,c,v,ncol):   # amendment 2: sum duplicate (cell, column) entries immediately (same operator, float64 sums; bounds memory)
        if not len(r):return r,c,v
        key=r.long()*ncol+c.long();u,inv=torch.unique(key,return_inverse=True)
        return u//ncol,u%ncol,torch.zeros(len(u),device=v.device,dtype=torch.float64).index_add_(0,inv,v)
    return dict(W=coalesce(cat(Wr),cat(Wc),cat(Wv),4*G.P),SS=coalesce(cat(Sr),cat(Sc),cat(Sv),G.P*64),q=q,seed=sd)

# ------------------------------------------------------------------ learned response R_theta
def mlp(i,h,o):
    m=nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o));nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias);return m

class FIA(nn.Module):
    def __init__(self,hidden=64):
        super().__init__();A,R=21,24;Pf=4
        self.rho=mlp(A,hidden,C);self.tau=mlp(A,hidden,C)
        self.fD=mlp(A+R+Pf,hidden,C);self.fT=mlp(A+R+Pf,hidden,C);self.fA=mlp(A+R+Pf,hidden,LOBES*C);self.fAT=mlp(A+R+Pf,hidden,LOBES*C)
        self.logL=nn.Parameter(torch.log(torch.tensor([.0181,.0058,.0018])))
        self.lobe=nn.Parameter(torch.zeros(LOBES,4,C))   # ring logits; 0 = uniform over bins (Lambertian in projected solid angle)
        with torch.no_grad():self.tau[-1].bias.fill_(-2.)
    def lobes(self):
        p=torch.softmax(self.lobe,1)/RCOUNT.to(self.lobe.device)[None,:,None];return p[:,RING.to(self.lobe.device)]   # (LOBES, 64, C), rows sum to 1
    def pair(self,G,k,xj,nj,fj):
        """Pair features for anchors k (n,) and receivers given by positions xj (n,m,3), normals nj (n,m,3), features fj (n,m,R)."""
        r=xj-G.anchor[k][:,None];dist=r.norm(dim=-1).clamp(min=1e-5);cos=(r*nj).sum(-1)/dist;nn_=(nj*G.n[k//4][:,None]).sum(-1)
        pf=torch.stack([dist/.01,cos,nn_,torch.exp(-dist/.002)],-1);af=G.afeat[k][:,None].expand(-1,xj.shape[1],-1)
        return torch.cat([af,fj,pf],-1),dist
    def prior(self,dist):
        L=self.logL.exp();return -dist[...,None]/L-torch.log(dist[...,None].square()+1e-6)   # (.., C)
    def response(self,G,k):
        """R for anchors k: local part (n,NN,LOBES,C) on G.nbr[k] and tail part (n,240,LOBES,C) on coarse cells, already weighted."""
        rho=torch.sigmoid(self.rho(G.afeat[k]));tau=torch.sigmoid(self.tau(G.afeat[k]))
        nb=G.nbr[k];x,d=self.pair(G,k,G.x[nb],G.n[nb],G.rfeat[nb])
        D=torch.softmax((self.prior(d)+self.fD(x)).masked_fill(~G.nmask[k][...,None],float('-inf')),1);a=torch.softmax(self.fA(x).view(*x.shape[:2],LOBES,C),2)
        loc=(rho*(1-tau))[:,None,None]*D[:,:,None]*a
        n=len(k);xt,dt=self.pair(G,k,G.cx[None].expand(n,-1,-1),G.cn[None].expand(n,-1,-1),G.cfeat[None].expand(n,-1,-1))
        T=torch.softmax(self.prior(dt)+self.fT(xt),1);at=torch.softmax(self.fAT(xt).view(n,240,LOBES,C),2)
        tail=(rho*tau)[:,None,None]*T[:,:,None]*at
        return loc,tail

def predict(model,G,Wq,SSq,cells,albedo):
    """Dense native prediction (len(cells), P, 64, C) of K_scat for the given input cells (port*16 + bin) from their quadrature rows."""
    dev='cuda';P=G.P;n=len(cells);lut=torch.full((G.P*16,),-1,dtype=torch.long,device=dev);lut[cells]=torch.arange(n,device=dev)
    out=torch.zeros(n,P*64,C,device=dev)
    r,c_,v=SSq;ri=lut[r];sel=ri>=0
    if sel.any():out=out.index_put((ri[sel],c_[sel]),v[sel].float()[:,None]*albedo[None],accumulate=True)
    r,k,v=Wq;ri=lut[r];sel=ri>=0
    if sel.any():
        rs=ri[sel];uk,inv=torch.unique(k[sel],return_inverse=True)
        key=rs*len(uk)+inv;uq,pi=torch.unique(key,return_inverse=True);pv=torch.zeros(len(uq),device=dev).index_add(0,pi,v[sel].float())   # coalesced (cell, anchor) pairs
        pc=uq//len(uk);pa=uq%len(uk);Wd=torch.zeros(n,len(uk),device=dev).index_put((pc,pa),pv,accumulate=True)                       # dense (n, anchors) for the tail
        loc,tail=model.response(G,uk);L=model.lobes()                                                  # loc (A,NN,LOBES,C), tail (A,240,LOBES,C), L (LOBES,64,C)
        idx=(pc[:,None]*P+G.nbr[uk][pa]).reshape(-1)                                                   # sparse scatter of the local part: only nonzero (cell, anchor) pairs
        acc=torch.zeros(n*P,LOBES,C,device=dev).index_add(0,idx,(pv[:,None,None,None]*loc[pa]).reshape(-1,LOBES,C)).view(n,P,LOBES,C)
        tw=torch.einsum('na,aJmc->nJmc',Wd,tail);acc=acc+tw[:,G.parent]*G.afrac[None,:,None,None]
        out=out+(torch.einsum('njmc,moc->njoc',acc,L)*(albedo**2)[None,None,None]).reshape(n,P*64,C)
    return out.view(n,P,64,C)

# ------------------------------------------------------------------ deterministic per-port quadrature and the full runtime operator
def quad_seed(gi,port):return 762_000_000+gi*10_000_000+port*16
def quad_ports(G,ports,gi,q):
    """W / SS rows for every bin of the given ports; each port uses its own fixed seed (identical rows wherever a port is requested)."""
    parts=[quadrature(G,torch.arange(16,device='cuda')+16*int(p),q,quad_seed(gi,int(p))) for p in ports]
    out={k:list(torch.cat([x[k][i] for x in parts]) for i in range(3)) for k in ('W','SS')}
    # budget consistency with the geometry's ballistic B: rescale each cell's W and SS so that sum W + sum SS = 1 - beta_B(cell) exactly
    # (both are quadratures of the same E[c]; without this, independent quadrature noise could make beta + scattered mass exceed 1)
    tot=torch.zeros(G.P*16,device='cuda',dtype=torch.float64).index_add_(0,out['W'][0],out['W'][2]).index_add_(0,out['SS'][0],out['SS'][2])
    sc=torch.where(tot>0,(1-G.g['beta'].reshape(-1).double())/tot.clamp(min=1e-300),torch.zeros_like(tot))
    for k in ('W','SS'):out[k][2]=out[k][2]*sc[out[k][0]]
    return {k:tuple(v) for k,v in out.items()}

class Runtime:
    """Full deployable operator for one geometry: K x = B x + albedo SS x + albedo^2 R^T (W^T x). State: sparse W (cells x anchors),
    sparse SS (cells x P*64), learned R on every anchor (local NN neighbours + coarse tail), ring lobes. Built once per geometry."""
    def __init__(self,model,G,gi,q,chunk=1024):
        torch.cuda.synchronize();t0=time.perf_counter();P=G.P;self.G=G;dev='cuda'
        Q=quad_ports(G,range(P),gi,q);torch.cuda.synchronize();t1=time.perf_counter()
        r,k,v=Q['W'];self.W=torch.sparse_coo_tensor(torch.stack([r,k]),v.float(),(P*16,4*P)).coalesce()
        r,c,v=Q['SS'];self.SS=torch.sparse_coo_tensor(torch.stack([r,c]),v.float(),(P*16,P*64)).coalesce()
        with torch.no_grad():
            locs=[];tails=[]
            for a0 in range(0,4*P,chunk):
                ka=torch.arange(a0,min(a0+chunk,4*P),device=dev);lo,ta=model.response(G,ka);locs.append(lo);tails.append(ta)
            self.loc=torch.cat(locs);self.tail=torch.cat(tails);self.L=model.lobes();self.rho=torch.sigmoid(model.rho(G.afeat))
        torch.cuda.synchronize();t2=time.perf_counter()
        self.seconds=dict(quadrature=t1-t0,response=t2-t1,total=t2-t0)
        self.bytes=dict(W=self.W._nnz()*20,SS=self.SS._nnz()*20,local=self.loc.numel()*4+G.nbr.numel()*8,tail=self.tail.numel()*4,anchors=G.afeat.numel()*4+G.anchor.numel()*4)
    def apply(self,g,x,albedo):
        """x (b, P, 16, C) incident flux -> (b, P, 64, C) outgoing flux (ballistic B + scattered)."""
        b=x.shape[0];P=self.G.P;xc=x.reshape(b,P*16,C)
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(b,-1).T).T.reshape(b,P,64) for c in range(C)],-1)
        ss=torch.stack([torch.sparse.mm(self.SS.t(),xc[:,:,c].T).T for c in range(C)],-1).view(b,P,64,C)*albedo
        e=torch.stack([torch.sparse.mm(self.W.t(),xc[:,:,c].T).T for c in range(C)],-1)            # (b, anchors, C)
        acc=torch.zeros(b,P,LOBES,C,device=x.device)
        for a0 in range(0,e.shape[1],2048):
            sl=slice(a0,a0+2048);acc=acc.index_add(1,self.G.nbr[sl].reshape(-1),(e[:,sl,None,None,:]*self.loc[sl][None]).reshape(b,-1,LOBES,C))
        acc=acc+torch.einsum('bac,aJmc->bJmc',e,self.tail)[:,self.G.parent]*self.G.afrac[None,:,None,None]
        return ball+ss+torch.einsum('bjmc,moc->bjoc',acc,self.L)*(albedo**2)
    def column_sums(self,g,albedo):
        ones=torch.ones(self.G.P*64,1,device='cuda');s1=torch.sparse.mm(self.SS,ones)[:,0];wr=torch.sparse.mm(self.W,self.rho)   # (cells,), (cells, C)
        return g['beta'].reshape(-1)[:,None]+albedo[None]*s1[:,None]+albedo[None]**2*wr

def physics(rt,g,x,albedo):
    """r57_eval.physics checks for the FIA runtime: zero input, positivity, linearity, column sums <= 1, rebuild identity is checked separately."""
    y=rt.apply(g,x,albedo);z=rt.apply(g,x*0,albedo);s=rt.apply(g,x[:1]+x[1:2],albedo)
    lin=float((s-y[:2].sum(0,keepdim=True)).abs().sum()/s.abs().sum().clamp(min=1e-30));col=float(rt.column_sums(g,albedo).max())
    return dict(zero=float(z.abs().max()),minimum=float(y.min()),linearity=lin,max_colsum=col,
                passed=bool(float(z.abs().max())<=1e-8 and float(y.min())>=-1e-8 and lin<=1e-5 and col<=1+1e-6))

def objective(Sh,tg,idx):
    """WL68 objective J (family68.objective definition) for predicted sources idx (rows of the target tg); returns the per-(source, channel)
    terms summed (caller divides by the total count). tg: dict with S (n,16,P,64,C) native target, f16, co, HZ, den (native, fine16, coarse, con), d."""
    import family68 as FM
    S=tg['S'][idx];nat=(Sh-S).square().sum((1,2,3))/tg['den']['native'][idx]
    f16=(FM.aggregate(Sh,'fine16',tg['parent'],tg['bm'])-tg['f16'][idx]).square().sum((1,2,3))/tg['den']['fine16'][idx]
    co_=FM.aggregate(Sh,'coarse',tg['parent'],tg['bm']);co=(co_-tg['co'][idx]).square().sum((1,2,3))/tg['den']['coarse'][idx]
    con=(torch.einsum('ab,nbqoc->naqoc',FM.Hmat(Sh.device,Sh.dtype),co_/tg['d'][idx][:,:,None,None,None])-tg['HZ'][idx]).square().sum((1,2,3))/tg['den']['con'][idx]
    return ((nat+f16+co)/3+con).sum(),dict(native=nat.detach(),fine16=f16.detach(),coarse=co.detach(),con=con.detach())

def target(S,d,parent,bm):
    import family68 as FM
    t=FM.Target(S,d,parent,bm);return dict(S=S,f16=t.S['fine16'],co=t.S['coarse'],HZ=t.HZ,den=t.den,d=d,parent=parent,bm=bm)
