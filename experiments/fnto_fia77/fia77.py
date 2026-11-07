"""WL77 candidate FIA-AZ: WL76 FIA with ONE change - the post-collision outgoing angular representation of the local response.

Everything geometric is imported unchanged from fnto_fia76.fia76: Geo (anchors, features, neighbours, coarse tail), the exact single-scatter SS
and second-collision W quadratures (same seeds, q, budget rescaling), the WL68 objective J, the physics checks.

WL76 local outgoing distribution at receiver j for anchor k (channel c):  O(o) = sum_{m<3} alpha_m(k,j) L_m(o),
  L_m = ring lobes: a distribution over the 4 polar rings of the 8x8 concentric bin grid, uniform inside a ring. Removed degrees of freedom:
  every azimuthal variation inside a ring (64 - 4 = 60 of the 64 per-receiver directions; the post hoc WL76 floor 0.45 of the reference
  multiple-scatter native signal lies there).
WL77 local outgoing distribution:  O(o) = sum_{m<3} alpha_m L_m(o) + alpha_az Lambda(o),  alpha = softmax over 4 components (per pair, channel)
  Lambda(o) = nu(o) [1 + s a1 cos(phi_o - phi_r) + s a2 cos 2(phi_o - phi_r)] / Z,   |a1|, |a2| <= 1/2  (a = tanh(u)/2)  =>  Lambda >= 0
  nu: a learned ring profile (distribution over rings, uniform inside), phi_o: azimuth of bin o's centre direction in the receiver's local
  frame (t_j, b_j), phi_r: azimuth of the tangential part of r = x_j - y_k (anchor -> receiver) in the same frame (geometry, not learned),
  s = |r_t| / (|r_t| + 1 mm) (azimuth undefined when r has no tangential part), Z = sum_o of the bracket times nu (exact per pair).
  cos(phi_o - phi_r) is invariant under in-plane rotations and reflections of the local frame, so the response is exactly D4-equivariant
  wherever the slab symmetry permutes the bins. Nonnegative, normalised (sum_o O = 1): positivity / linearity / passivity unchanged.
  Basis form used by the runtime (exact): Lambda = (1/Z) [nu + s a1 cos phi_r nu cos phi_o + s a1 sin phi_r nu sin phi_o
                                                           + s a2 cos 2phi_r nu cos 2phi_o + s a2 sin 2phi_r nu sin 2phi_o]
  -> 8 fixed per-bin basis functions (3 ring lobes + 5 azimuthal) with per-pair coefficients.
The coarse tail keeps WL76's ring lobes (receivers are 240 coarse cells without a single tangent frame); its share tau is reported.
"""
import os,sys,math,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from common77 import CFG
p=str(Path(__file__).resolve().parent.parent/'fnto_fia76')
if p not in sys.path:sys.path.append(p)
import fia76 as F76   # noqa: E402  (unchanged geometry, quadratures, objective)
from fia76 import Geo,quadrature,quad_ports,quad_seed,objective,target,physics,mlp,RING,RCOUNT,C,LOBES,NN   # noqa: E402,F401
NBASIS=LOBES+5

def bin_azimuth():
    """phi_o (64,) and sin(theta_o) (64,) of the bin centre directions (p53_geom concentric mapping; bin = iu + 8 iv)."""
    import r57_geometry as RG
    b=torch.arange(64);u=((b%8).double()+.5)/8;v=((b//8).double()+.5)/8;x,y=RG.G.square_to_disk(u,v)
    return torch.atan2(y,x).float(),torch.sqrt(x*x+y*y).float()

def port_frames(G):
    """Receiver local frame (t, b) of each port: the frame of its first triangle (bins are defined in the triangle frames of the port)."""
    tri=G.dom.port_tris[:,0];return G.dom.t[tri],G.dom.b[tri]

class FIA77(F76.FIA):
    def __init__(self,hidden=64):
        super().__init__(hidden);A,R=21,24;Pf=4
        self.fA=mlp(A+R+Pf,hidden,(LOBES+1+2)*C)          # mixture logits over 3 ring lobes + the azimuthal lobe, and (u1, u2)
        self.nu=nn.Parameter(torch.zeros(4,C))              # ring profile of the azimuthal lobe (0 = uniform over bins)
        with torch.no_grad():self.fA[-1].bias.view(LOBES+3,C)[LOBES].fill_(-2.)   # predeclared init: azimuthal lobe weight ~ 0.04, a = 0
    def basis(self):
        """(NBASIS, 64, C): the 3 ring lobes and nu x {1, cos phi, sin phi, cos 2phi, sin 2phi}."""
        phi,_=self._phi();dev=self.lobe.device
        nu=(torch.softmax(self.nu,0)/RCOUNT.to(dev)[:,None])[RING.to(dev)]               # (64, C)
        h=torch.stack([torch.ones_like(phi),torch.cos(phi),torch.sin(phi),torch.cos(2*phi),torch.sin(2*phi)])   # (5, 64)
        return torch.cat([self.lobes(),h[:,:,None]*nu[None]],0)
    def _phi(self):
        if not hasattr(self,'_phic'):self._phic=[t.to(self.lobe.device) for t in bin_azimuth()]
        return self._phic
    def response(self,G,k):
        """loc (n, NN', NBASIS, C) per-pair basis coefficients on G.nbr[k]; tail (n, 240, LOBES, C) unchanged (ring lobes)."""
        rho=torch.sigmoid(self.rho(G.afeat[k]));tau=torch.sigmoid(self.tau(G.afeat[k]))
        nb=G.nbr[k];x,d=self.pair(G,k,G.x[nb],G.n[nb],G.rfeat[nb])
        D=torch.softmax((self.prior(d)+self.fD(x)).masked_fill(~G.nmask[k][...,None],float('-inf')),1)
        o=self.fA(x).view(*x.shape[:2],LOBES+3,C);alpha=torch.softmax(o[:,:,:LOBES+1],2);a=.5*torch.tanh(o[:,:,LOBES+1:])   # (n,NN,4,C), (n,NN,2,C)
        t,b=G.frames if hasattr(G,'frames') else port_frames(G);r=G.x[nb]-G.anchor[k][:,None];rn=(r*G.n[nb]).sum(-1,keepdim=True);rt=r-rn*G.n[nb]
        cr=(rt*t[nb]).sum(-1);sr=(rt*b[nb]).sum(-1);nt=torch.sqrt(cr*cr+sr*sr).clamp(min=1e-9);s=(nt/(nt+.001))[...,None]          # (n,NN), (n,NN,1)
        c1,s1=(cr/nt)[...,None],(sr/nt)[...,None];c2,s2=c1*c1-s1*s1,2*s1*c1
        phi,_=self._phi();nu=(torch.softmax(self.nu,0)/RCOUNT.to(phi.device)[:,None])[RING.to(phi.device)]
        Cn=torch.stack([(nu*torch.cos(n*phi)[:,None]).sum(0) for n in (1,2)]);Sn=torch.stack([(nu*torch.sin(n*phi)[:,None]).sum(0) for n in (1,2)])   # (2, C)
        a1=s*a[:,:,0];a2=s*a[:,:,1]
        Z=1+a1*(Cn[0]*c1+Sn[0]*s1)+a2*(Cn[1]*c2+Sn[1]*s2)
        az=alpha[:,:,LOBES]/Z;coef=torch.stack([az,az*a1*c1,az*a1*s1,az*a2*c2,az*a2*s2],2)                                   # (n,NN,5,C)
        loc=(rho*(1-tau))[:,None,None]*D[:,:,None]*torch.cat([alpha[:,:,:LOBES],coef],2)
        n=len(k);xt,dt=self.pair(G,k,G.cx[None].expand(n,-1,-1),G.cn[None].expand(n,-1,-1),G.cfeat[None].expand(n,-1,-1))
        T=torch.softmax(self.prior(dt)+self.fT(xt),1);at=torch.softmax(self.fAT(xt).view(n,240,LOBES,C),2)
        return loc,(rho*tau)[:,None,None]*T[:,:,None]*at

def attach_frames(G):
    G.frames=port_frames(G);return G

def predict(model,G,Wq,SSq,cells,albedo):
    """fia76.predict with the 8-function outgoing basis (identical spatial / quadrature arithmetic)."""
    dev='cuda';P=G.P;n=len(cells);lut=torch.full((G.P*16,),-1,dtype=torch.long,device=dev);lut[cells]=torch.arange(n,device=dev)
    out=torch.zeros(n,P*64,C,device=dev)
    r,c_,v=SSq;ri=lut[r];sel=ri>=0
    if sel.any():out=out.index_put((ri[sel],c_[sel]),v[sel].float()[:,None]*albedo[None],accumulate=True)
    r,k,v=Wq;ri=lut[r];sel=ri>=0
    if sel.any():
        rs=ri[sel];uk,inv=torch.unique(k[sel],return_inverse=True)
        key=rs*len(uk)+inv;uq,pi=torch.unique(key,return_inverse=True);pv=torch.zeros(len(uq),device=dev).index_add(0,pi,v[sel].float())
        pc=uq//len(uk);pa=uq%len(uk);Wd=torch.zeros(n,len(uk),device=dev).index_put((pc,pa),pv,accumulate=True)
        loc,tail=model.response(G,uk);B=model.basis()
        idx=(pc[:,None]*P+G.nbr[uk][pa]).reshape(-1)
        acc=torch.zeros(n*P,NBASIS,C,device=dev).index_add(0,idx,(pv[:,None,None,None]*loc[pa]).reshape(-1,NBASIS,C)).view(n,P,NBASIS,C)
        tw=torch.einsum('na,aJmc->nJmc',Wd,tail);acc[:,:,:LOBES]=acc[:,:,:LOBES]+tw[:,G.parent]*G.afrac[None,:,None,None]
        out=out+(torch.einsum('njmc,moc->njoc',acc,B)*(albedo**2)[None,None,None]).reshape(n,P*64,C)
    return out.view(n,P,64,C)

class Runtime(F76.Runtime):
    """fia76.Runtime (same quadrature, CSR SS^T and W^T) with an 8-function local basis: one shared int32 CSR index structure over
    (port, basis) x anchor and one value array per channel."""
    def __init__(self,model,G,gi,q,chunk=512,keep_reference=False):
        torch.cuda.synchronize();t0=time.perf_counter();P=G.P;self.G=G;dev='cuda';attach_frames(G)
        Q=quad_ports(G,range(P),gi,q);torch.cuda.synchronize();t1=time.perf_counter()
        r,k,v=Q['W'];self.W=torch.sparse_coo_tensor(torch.stack([r,k]),v.float(),(P*16,4*P)).coalesce()
        r,c,v=Q['SS'];self.SS=torch.sparse_coo_tensor(torch.stack([r,c]),v.float(),(P*16,P*64)).coalesce();del Q
        A=4*P;NNp=G.nbr.shape[1]
        with torch.no_grad():
            locs=[];tails=[]
            for a0 in range(0,A,chunk):
                ka=torch.arange(a0,min(a0+chunk,A),device=dev);lo,ta=model.response(G,ka);locs.append(lo);tails.append(ta)
            loc=torch.cat(locs);del locs;self.tail=torch.cat(tails);self.B=model.basis();self.rho=torch.sigmoid(model.rho(G.afeat))
        keep=G.nmask.reshape(-1);rows=(G.nbr.reshape(-1)[keep][:,None]*NBASIS+torch.arange(NBASIS,device=dev)[None]).reshape(-1)          # (nnz,)
        cols=torch.arange(A,device=dev).repeat_interleave(NNp)[keep][:,None].expand(-1,NBASIS).reshape(-1)
        perm=torch.argsort(rows*A+cols);rows_s=rows[perm];self.crow=torch.zeros(P*NBASIS+1,dtype=torch.int64,device=dev);self.crow[1:]=torch.cumsum(torch.bincount(rows_s,minlength=P*NBASIS),0)
        self.crow=self.crow.int();self.col=cols[perm].int();del rows,cols,rows_s
        self.vals=[loc[...,c].reshape(A*NNp,NBASIS)[keep].reshape(-1)[perm].contiguous() for c in range(C)];del perm
        if keep_reference:self.loc=loc
        del loc;torch.cuda.empty_cache()
        self.Rt=[torch.sparse_csr_tensor(self.crow,self.col,self.vals[c],(P*NBASIS,A)) for c in range(C)]
        self.Tt=[self.tail[...,c].reshape(A,240*LOBES).T.contiguous() for c in range(C)]
        self.SSt=self.SS.t().coalesce().to_sparse_csr();self.Wt=self.W.t().coalesce().to_sparse_csr()
        self.ss_rowsum=torch.zeros(P*16,device=dev).index_add_(0,self.SS.indices()[0],self.SS.values())
        if not keep_reference:del self.SS,self.tail;torch.cuda.empty_cache()
        torch.cuda.synchronize();t2=time.perf_counter();self.seconds=dict(quadrature=t1-t0,response_and_assembly=t2-t1,total=t2-t0)
        nb=lambda t:t.values().numel()*4+t.col_indices().numel()*t.col_indices().element_size()+t.crow_indices().numel()*t.crow_indices().element_size()
        self.bytes=dict(W=nb(self.Wt),SS=nb(self.SSt),local=self.crow.numel()*4+self.col.numel()*4+sum(v.numel()*4 for v in self.vals),tail=sum(t.numel()*4 for t in self.Tt),
                        anchors=G.afeat.numel()*4+G.anchor.numel()*4)
    def apply(self,g,x,albedo):
        b=x.shape[0];P=self.G.P;xc=x.reshape(b,P*16,C);sc=[];acc=[]
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(b,-1).T).T.reshape(b,P,64) for c in range(C)],-1)
        for c in range(C):
            X=xc[:,:,c].T.contiguous();sc.append(self.SSt@X);e=self.Wt@X
            loc=(self.Rt[c]@e).view(P,NBASIS,b);tl=(self.Tt[c]@e).view(240,LOBES,b)[self.G.parent]*self.G.afrac[:,None,None]
            loc[:,:LOBES]=loc[:,:LOBES]+tl;acc.append(loc)
        ss=torch.stack(sc,-1).view(P,64,b,C).permute(2,0,1,3)*albedo
        return ball+ss+torch.einsum('pmbc,moc->bpoc',torch.stack(acc,-1),self.B)*(albedo**2)
    def apply_reference(self,g,x,albedo):
        """Dense per-slot scatter of the same coefficients (equivalence test)."""
        b=x.shape[0];P=self.G.P;xc=x.reshape(b,P*16,C)
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(b,-1).T).T.reshape(b,P,64) for c in range(C)],-1)
        ss=torch.stack([torch.sparse.mm(self.SS.t(),xc[:,:,c].T).T for c in range(C)],-1).view(b,P,64,C)*albedo
        e=torch.stack([torch.sparse.mm(self.W.t(),xc[:,:,c].T).T for c in range(C)],-1);acc=torch.zeros(b,P,NBASIS,C,device=x.device)
        for a0 in range(0,e.shape[1],1024):
            sl=slice(a0,a0+1024);acc=acc.index_add(1,self.G.nbr[sl].reshape(-1),(e[:,sl,None,None,:]*self.loc[sl][None]).reshape(b,-1,NBASIS,C))
        acc[:,:,:LOBES]=acc[:,:,:LOBES]+torch.einsum('bac,aJmc->bJmc',e,self.tail)[:,self.G.parent]*self.G.afrac[None,:,None,None]
        return ball+ss+torch.einsum('bjmc,moc->bjoc',acc,self.B)*(albedo**2)

FIA=FIA77   # adapters (fit77 / eval77) use FI.FIA, FI.predict, FI.Runtime
