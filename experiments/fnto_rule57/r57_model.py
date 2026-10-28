"""ONE matched passive spatial/directional kernel family; D direct geometry vs R learned geometry-state encoder.
K = analytic unscattered B + learned scattered response on the remaining incident budget.
All caches depend on geometry/parameters only, never illumination or target K.
"""
import os
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import r57_geometry as RG

M=RG.MIX;T=RG.RANK;C=3

class Transport(nn.Module):
    def __init__(self,arm):
        super().__init__();assert arm in ('D','R');self.arm=arm
        nd,h,nh=(140,96,128) if arm=='D' else (32,128,144)
        if arm=='R':
            self.encoder=nn.Linear(140,32)
            self.messages=nn.ModuleList([nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,32)) for _ in range(2)])
            self.global_enc=nn.Linear(32,32)
        self.src_layer=nn.Linear(nd,h);self.dst_layer=nn.Linear(nd,h,bias=False);self.pair_layer=nn.Linear(6,h,bias=False)
        self.edge=nn.Sequential(nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,M*C))
        self.node_head=nn.Sequential(nn.Linear(nd,nh),nn.SiLU(),nn.Linear(nh,2*T*M*C+M*C))
        self.in_logits=nn.Parameter(torch.randn(M,16,C)*.05)
        self.out_logits=nn.Parameter(torch.randn(M,64,C)*.05)
        nn.init.zeros_(self.edge[-1].weight);nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.node_head[-1].weight);nn.init.zeros_(self.node_head[-1].bias)
        with torch.no_grad():
            self.node_head[-1].bias[T*M*C:2*T*M*C]=-4
            # Initialization only; all values learned. RGB approximate reflectance, not reference calibration.
            self.node_head[-1].bias[2*T*M*C:]=torch.tensor([2.,1.,0.]).repeat(M)

    def state(self,g):
        z=g['node']
        if self.arm=='R':
            z=F.silu(self.encoder(z))
            for layer in self.messages:z=z+.25*layer(torch.cat([z,z[g['graph']].mean(1)],1))
            z=z+F.silu(self.global_enc(z.mean(0,keepdim=True)))
        # D gets the exact same legitimate partial affine caching as R.
        return dict(z=z,s=self.src_layer(z),d=self.dst_layer(z))

    def coefficients(self,g,state=None):
        st=self.state(g) if state is None else state;src,dst=g['src'],g['dst'];parts=[]
        chunk=int(os.environ.get('WL57_EDGE_CHUNK','65536'))
        def edge_part(s,d,p,prior):
            h=s+d+self.pair_layer(p)
            return F.softplus(self.edge(h)).view(-1,M,C)*prior[:,None,:]
        for i in range(0,len(src),chunk):
            sl=slice(i,min(i+chunk,len(src)))
            args=(st['s'][src[sl]],st['d'][dst[sl]],g['pair'][sl],g['prior'][sl])
            if torch.is_grad_enabled() and os.environ.get('WL57_CHECKPOINT_EDGES','0')=='1':parts.append(checkpoint(edge_part,*args,use_reentrant=False))
            else:parts.append(edge_part(*args))
        W=torch.cat(parts,0)
        nh=self.node_head(st['z']);p=g['P']
        U=torch.softmax(nh[:,:T*M*C].view(p,T,M,C),dim=0)
        V=F.softplus(nh[:,T*M*C:2*T*M*C].view(p,T,M,C))
        mass=torch.sigmoid(nh[:,2*T*M*C:].view(p,M,C))
        norm=torch.zeros(p,M,C,device=nh.device,dtype=nh.dtype).index_add_(0,src,W)+V.sum(1)
        scale=mass/(norm+1e-12)
        return dict(W=W*scale[src],U=U,V=V*scale[:,None],
            A=torch.softmax(self.in_logits,dim=0),O=torch.softmax(self.out_logits,dim=1))

    def apply(self,g,x,coeff):
        """x (batch, P,16,RGB) incident FLUX -> (batch,P,64,RGB) outgoing FLUX."""
        # beta is achromatic zero-collision probability; shared physical budget split.
        q=torch.einsum('biac,rac->birc',x*(1-g['beta'])[None,:,:,None],coeff['A'])
        y=torch.zeros(x.shape[0],g['P'],M,C,device=x.device,dtype=x.dtype)
        # Chunking bounds temporary activation memory; no transport edges are pruned.
        chunk=int(os.environ.get('WL57_EDGE_CHUNK','65536'))
        for i in range(0,len(g['src']),chunk):
            sl=slice(i,min(i+chunk,len(g['src'])))
            y=y.index_add(1,g['dst'][sl],q[:,g['src'][sl]]*coeff['W'][sl][None])
        latent=torch.einsum('birc,ikrc->bkrc',q,coeff['V'])
        y=y+torch.einsum('jkrc,bkrc->bjrc',coeff['U'],latent)
        scatter=torch.einsum('bjrc,roc->bjoc',y,coeff['O'])
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(x.shape[0],-1).T).T.reshape(x.shape[0],g['P'],64) for c in range(3)],-1)
        return scatter+ball

    def forward(self,g,x,state=None):return self.apply(g,x,self.coefficients(g,state))

    def column_sums(self,g,coeff):
        s=torch.zeros(g['P'],M,C,device=g['src'].device).index_add_(0,g['src'],coeff['W'])+coeff['V'].sum(1)
        # U sums to one and O sums to one; A sums over r to one per incoming bin.
        return g['beta'][:,:,None]+(1-g['beta'])[:,:,None]*torch.einsum('irc,rac->iac',s,coeff['A'])

def parameter_count(arm):return sum(p.numel() for p in Transport(arm).parameters())
