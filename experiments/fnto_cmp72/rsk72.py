"""Candidate B — relational shared kernel (RSK). Persistent rule = three MLPs (phi_loc, phi_far, phi_m); every other quantity is
rebuilt from the current geometry (frames, position-free descriptors, 240 coarse cells, 128-neighbour graph, pair relations).
T[i->j] = mass_i * (w_loc(i->j) + w_far(i->J(j)) a_j/a_J) / (sum w_loc + sum w_far); T >= 0, sum_j T = mass_i in (0,1).
Exposes the WL59 Transport API used by the evaluators: state, coefficients, apply, column_sums, forward, in_logits/out_logits."""
import torch
from torch import nn
import torch.nn.functional as F

M,C,NC=8,3,240

def mlp(i,h,o,bias=0.):
    m=nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))
    with torch.no_grad():m[-1].bias.fill_(bias)
    return m

def frames(n):
    """Deterministic orthonormal frame per port from its normal (position-free)."""
    z=torch.tensor([0.,0.,1.],device=n.device).expand_as(n);x=torch.tensor([1.,0.,0.],device=n.device).expand_as(n)
    side=n[:,2].abs()<.9
    t=torch.where(side[:,None],torch.cross(z,n,dim=1),x-(x*n).sum(1,keepdim=True)*n);t=t/t.norm(dim=1,keepdim=True).clamp(min=1e-12)
    return t,torch.cross(n,t,dim=1)

def prepare(g):
    """Geometry-dependent RSK state (cached in g['_rsk72']); no learned quantity."""
    if '_rsk72' in g:return g['_rsk72']
    nd=g['node'];x=g['xyz'];n=nd[:,9:12]/nd[:,9:12].norm(dim=1,keepdim=True);t,b=frames(n);P=g['P']
    nonco=(n@n.T).abs()<.5;dist=torch.cdist(x,x)
    edge=torch.where(nonco,dist,torch.full_like(dist,1e9)).min(1).values.clamp(max=.05)/.01
    desc=torch.cat([n,nd[:,13:14],edge[:,None],nd[:,12:13],nd[:,14:15],nd[:,6:9]],1)  # (P,10): normal, thickness, edge distance, area ratio, normal spread, displacement
    par=g['parent'];a=g['area'];aJ=torch.zeros(NC,device=x.device).index_add_(0,par,a);share=a/aJ[par]
    cx=torch.zeros(NC,3,device=x.device).index_add_(0,par,x*a[:,None])/aJ[:,None]
    cd=torch.zeros(NC,10,device=x.device).index_add_(0,par,desc*a[:,None])/aJ[:,None]
    cn=cd[:,:3]/cd[:,:3].norm(dim=1,keepdim=True).clamp(min=1e-12);cd=torch.cat([cn,cd[:,3:]],1)
    kern=torch.exp(-torch.cdist(x,cx)/.02)*aJ[None];ctx=(kern@cd)/kern.sum(1,keepdim=True)
    st=dict(x=x,n=n,t=t,b=b,desc=desc,parent=par,share=share,cx=cx,cd=cd,aJ=aJ/aJ.mean(),ctx=ctx)
    g['_rsk72']=st;return st

def rel(st,i,xd,nd_):
    """Relation from sources i (n,) to targets with positions xd (n,K,3), normals nd_ (n,K,3), in the source frame."""
    D=(xd-st['x'][i][:,None])/.01;f=lambda v:torch.stack([(v*st[k][i][:,None]).sum(-1) for k in ('t','b','n')],-1)
    return torch.cat([f(D),D.norm(dim=-1,keepdim=True),f(nd_)],-1)

class RSK(nn.Module):
    def __init__(self,in_logits,out_logits,hidden=96):
        super().__init__()
        self.phi_loc=mlp(27,hidden,M*C);self.phi_far=mlp(28,hidden,M*C,-7.);self.phi_m=nn.Sequential(nn.Linear(20,64),nn.SiLU(),nn.Linear(64,M*C))
        with torch.no_grad():self.phi_m[-1].bias.fill_(1.1)
        self.in_logits=nn.Parameter(in_logits.clone(),requires_grad=False);self.out_logits=nn.Parameter(out_logits.clone(),requires_grad=False)
    # ---------------------------------------------------------------- core
    def parts(self,g,ports):
        """(local weights (n,128,M,C) on edges, far weights (n,240,M,C), mass (n,M,C), edge ids (n,128))."""
        import family71 as FW
        st=prepare(g);pi=torch.as_tensor(ports,device=st['x'].device);e=FW.edges(g,ports);dst=g['dst'][e];n=len(ports)
        rl=rel(st,pi,st['x'][dst],st['n'][dst]);di=st['desc'][pi][:,None].expand(-1,dst.shape[1],-1)
        wl=F.softplus(self.phi_loc(torch.cat([rl,di,st['desc'][dst]],-1))).view(n,-1,M,C)*g['prior'][e][:,:,None,:]
        rf=rel(st,pi,st['cx'][None].expand(n,-1,-1),st['cd'][None,:,:3].expand(n,-1,-1))
        wf=F.softplus(self.phi_far(torch.cat([rf,st['desc'][pi][:,None].expand(-1,NC,-1),st['cd'][None].expand(n,-1,-1),st['aJ'][None,:,None].expand(n,-1,1)],-1))).view(n,NC,M,C)
        mass=torch.sigmoid(self.phi_m(torch.cat([st['desc'][pi],st['ctx'][pi]],-1))).view(n,M,C)
        return wl,wf,mass,e
    def columns(self,g,ports):
        """T (n,P,M,C) for the given source ports, A (M,16,C), O (M,64,C)."""
        st=prepare(g);wl,wf,mass,e=self.parts(g,ports);n=len(ports);P=g['P']
        scale=mass/(wl.sum(1)+wf.sum(1)+1e-12)
        T=torch.zeros(n,P,M,C,device=wl.device,dtype=wl.dtype).scatter_add(1,g['dst'][e][:,:,None,None].expand(-1,-1,M,C),wl*scale[:,None])
        T=T+(wf*scale[:,None])[:,st['parent']]*st['share'][None,:,None,None]
        return T,torch.softmax(self.in_logits,0),torch.softmax(self.out_logits,1)
    # ---------------------------------------------------------------- WL59 runtime API
    def state(self,g):return prepare(g)
    def coefficients(self,g,state=None,chunk=256):
        P=g['P'];WL=[];WF=[]
        for c0 in range(0,P,chunk):
            ps=list(range(c0,min(P,c0+chunk)));wl,wf,mass,_=self.parts(g,ps);s=mass/(wl.sum(1)+wf.sum(1)+1e-12);WL.append(wl*s[:,None]);WF.append(wf*s[:,None])
        return dict(WL=torch.cat(WL),WF=torch.cat(WF),A=torch.softmax(self.in_logits,0),O=torch.softmax(self.out_logits,1))
    def apply(self,g,x,co):
        """x (batch,P,16,RGB) incident flux -> (batch,P,64,RGB) outgoing flux (same algebra as runtime.Transport.apply)."""
        st=prepare(g);q=torch.einsum('biac,rac->birc',x*(1-g['beta'])[None,:,:,None],co['A']);P=g['P']
        src=g['src'].view(P,-1);dst=g['dst'].view(P,-1)
        y=torch.zeros(x.shape[0],P,M,C,device=x.device,dtype=x.dtype).index_add(1,dst.reshape(-1),(q[:,:,None]*co['WL'][None]).reshape(x.shape[0],-1,M,C))
        Y=torch.einsum('birc,iJrc->bJrc',q,co['WF']);y=y+Y[:,st['parent']]*st['share'][None,:,None,None]
        scatter=torch.einsum('bjrc,roc->bjoc',y,co['O'])
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(x.shape[0],-1).T).T.reshape(x.shape[0],P,64) for c in range(3)],-1)
        return scatter+ball
    def column_sums(self,g,co):
        s=co['WL'].sum(1)+co['WF'].sum(1)
        return g['beta'][:,:,None]+(1-g['beta'])[:,:,None]*torch.einsum('irc,rac->iac',s,co['A'])
    def forward(self,g,x,state=None):return self.apply(g,x,self.coefficients(g))

def param_count(m):return sum(p.numel() for n,p in m.named_parameters() if n not in ('in_logits','out_logits'))
