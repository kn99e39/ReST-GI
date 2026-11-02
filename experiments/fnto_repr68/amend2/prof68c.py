"""Variants of the S x_o O contraction (forward + O-gradient) and their accuracy vs float64. Non-evidence."""
import sys,time,torch
sys.path.insert(0,'.')
from common68 import environment
import fit68 as F,family68 as FM
RG=environment();torch.backends.cuda.matmul.allow_tf32=False
pb=F.Problem(RG);m=F.model_from('P_5901')
tg=pb.tg['rest'];S=tg.S['native'];n,a,P,no,C=S.shape
O0=torch.softmax(m.out_logits.detach(),1)            # (8,64,3)
G=torch.randn(n,a,P,8,C,device='cuda')               # upstream gradient for X
def tm(fn,k=20):
    fn();torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(k):fn()
    torch.cuda.synchronize();return round((time.perf_counter()-t)/k*1e3,3)
def run_einsum(O):return torch.einsum('najoc,roc->najrc',S,O)
Sc=S.permute(4,0,1,2,3).reshape(C,-1,no).contiguous()  # (C, naj, o)
def run_bmm(O):return torch.bmm(Sc,O.permute(2,1,0)).view(C,n,a,P,8).permute(1,2,3,4,0)
def run_bmmT(O):return torch.bmm(O.permute(2,0,1),Sc.transpose(1,2)).view(C,8,n,a,P).permute(2,3,4,1,0)
ref_S=S.double();ref=torch.einsum('najoc,roc->najrc',ref_S,O0.double());refg=torch.einsum('najoc,najrc->roc',ref_S,G.double())
for name,f in [('einsum',run_einsum),('bmm',run_bmm),('bmmT',run_bmmT)]:
    O=O0.clone().requires_grad_(True)
    def fb():
        O.grad=None;X=f(O);X.backward(G)
    fwd=lambda:f(O0)
    X=f(O0);fb()
    ex=float((X.double()-ref).norm()/ref.norm());eg=float((O.grad.double()-refg).norm()/refg.norm())
    print(name,'fwd ms',tm(fwd),'fwd+bwd ms',tm(fb),'rel err fwd',ex,'grad',eg,flush=True)
