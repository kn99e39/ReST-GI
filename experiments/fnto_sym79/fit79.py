"""WL79 Stage 3 training: W+UV-Q (q78.TransportQ: rank 16, frozen G2 A/O, Q78 layers, inputs and initialisation rule incl. the reinit seeds)
with ONE change, the symmetry-consistent presentation of the existing rest S16-R supervision:

  at optimizer step t, rest source k (S16-R order, k = 0..15) is presented under the exact rest operation OPS[(k + t) mod 16]
  (OPS = sym72.ops(True), identity first): image port perm[p_k] of the rest geometry, label = the exact image of its fitting-replica mean
  (sym79 maps: receiver ports, incoming and outgoing bins), model budget = 1 - beta of the rest geometry at the image port.
  Every step contains all 16 operations once (Latin schedule); every source sees every operation once per 16 steps.

Unchanged: objective J (ato74.objective_m, mean of the rest and pt0 terms) + 1.0 L_sym (WL72 term, identical code and labels); the pt0 term
(identity presentation: dented-pose maps are not exact, Stage 0); 16 rest + 16 pt0 physical samples per update; Adam, cosine 3e-3 -> 3e-5,
20,000 steps, clip 1.0, checkpoints 2,000 / 6,000 / 10,000, final iterate (no selection); no new label, loss term or hyperparameter.
Records (initial / final J) use the identity presentation, i.e. the same objective as Q78's records.
Implementation: the transformed targets are rebuilt in place each step from the original fitting targets by exact gathers (no MC data);
the transformed native target is produced directly in the channel-major layout (family68.Target.Sc) and the aggregates are recomputed.
    python fit79.py run <run_id> | probe <run_id> <steps>"""
import sys,os,json,time,math
import torch
from common79 import CFG,ROOT,save_new,write_new,environment,dev
RG=environment()
import fit75
import q78 as Q
import ato74 as AT
import sym72 as SY
import sym79 as S9
import family68 as FM
TRAIN=('rest','pt0')

class SymRest:
    """In-place symmetry presentation of a family68.Target (rest geometry, n sources). set(t) installs the step-t Latin presentation and
    returns the image ports; identity() restores the original target (bitwise the original tensors)."""
    def __init__(self,RG,tg,ports,g):
        self.tg=tg;self.ports=list(ports);self.g=g;n=len(ports);P=g['P'];C=3;self.n,self.P,self.C=n,P,C
        assert 'native' not in tg.S,'native target must be released (layout copy Sc is the source)'
        self.orig=dict(Sc_native=tg.Sc['native'],Sc_fine16=tg.Sc['fine16'],fine16=tg.S['fine16'],coarse=tg.S['coarse'],d=tg.d,HZ=tg.HZ,den=dict(tg.den))
        self.ops=SY.ops(True);x0=g['rest_xyz'].double().cpu().numpy();self.maps=[S9.Map(RG,'rest',op,x0) for op in self.ops]
        for m in self.maps:m.check()
        dv=tg.Sc['native'].device
        self.inv_in=[[m.inv_in(p).to(dv) for p in self.ports] for m in self.maps];self.inv_out=[m._inv_out.to(dv) for m in self.maps]
        self.inv16=[m._inv16.to(dv) for m in self.maps];self.img=[[int(m.perm[p]) for p in self.ports] for m in self.maps]
        self.dimg=[(1-g['beta'][torch.as_tensor(im,device=g['beta'].device)]).float() for im in self.img]
        self.buf=torch.empty_like(tg.Sc['native']);self.fbuf=torch.empty_like(tg.S['fine16']);self.H=FM.Hmat(dv,self.buf.dtype)
        self.Sc0=self.orig['Sc_native'].view(C,n,16,P*64);self.F0=self.orig['fine16'].reshape(n,16,P*16,C)
    def ops_at(self,t):return [(k+t)%len(self.ops) for k in range(self.n)]
    def identity(self):
        o=self.orig;tg=self.tg;tg.Sc['native']=o['Sc_native'];tg.Sc['fine16']=o['Sc_fine16'];tg.S['fine16']=o['fine16'];tg.S['coarse']=o['coarse']
        tg.d=o['d'];tg.HZ=o['HZ'];tg.den=dict(o['den']);return self.ports
    def set(self,t):
        n,P,C=self.n,self.P,self.C;tg=self.tg;ok=self.ops_at(t);B=self.buf.view(C,n,16,P*64);Fb=self.fbuf.view(n,16,P*16,C)
        with torch.no_grad():
            for k,o in enumerate(ok):
                B[:,k]=self.Sc0[:,k].index_select(1,self.inv_in[o][k]).index_select(2,self.inv_out[o])
                Fb[k]=self.F0[k].index_select(0,self.inv_in[o][k]).index_select(1,self.inv16[o])
            d=torch.stack([self.dimg[o][k] for k,o in enumerate(ok)])
            coarse=torch.zeros(n,16,240,16,C,device=Fb.device,dtype=Fb.dtype).index_add(2,tg.parent,self.fbuf)
            HZ=torch.einsum('ab,nbqoc->naqoc',self.H,coarse/d[:,:,None,None,None])
            tg.Sc['native']=self.buf;tg.S['fine16']=self.fbuf;tg.Sc['fine16']=self.fbuf.permute(4,0,1,2,3).reshape(C,-1,16).contiguous()
            tg.S['coarse']=coarse;tg.d=d;tg.HZ=HZ
            tg.den=dict(native=self.buf.view(C,n,-1).square().sum(-1).T.contiguous(),fine16=self.fbuf.square().sum((1,2,3)),coarse=coarse.square().sum((1,2,3)),con=HZ.square().sum((1,2,3)))
        return [self.img[o][k] for k,o in enumerate(ok)]

def build(rid,RG):
    import fit68,fit72,fit73
    spec=CFG['runs'][rid];pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
    m,meta=Q.init_model(RG,spec['init'],spec['reinit_seed'],spec['rank']);params=fit68.group_params(m,'net')
    assert not m.in_logits.requires_grad and not m.out_logits.requires_grad
    cols=lambda g,ports:Q.columns(m,g,ports);lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F);del pb.F
    for k in list(gs):Q.QG(RG,gs[k])     # W+UV-Q inputs once per geometry (geometry-only)
    rest=SymRest(RG,pb.tg['rest'],pb.ports,gs['rest']);state=dict(n=0)
    def loss():
        if torch.is_grad_enabled():ports=rest.set(state['n']);state['n']+=1     # one presentation per optimizer step
        else:ports=rest.identity()                                             # records: the original (Q78-comparable) objective
        Jr=AT.objective_m(*cols(gs['rest'],ports),rest.tg);Jp=AT.objective_m(*cols(gs['pt0'],pb.ports),pb.tg['pt0']);J=(Jr+Jp)/2;terms=[]
        for key,src,lab in sym:
            for c0 in range(0,len(src),64):
                ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
        L=torch.cat(terms).mean();return J+lam*L,J,L
    loss.rest=rest;loss.state=state
    return spec,m,meta,params,loss,pb,sym

def run(rid,probe=None):
    fit75.build=build;fit75.ROOT=ROOT;fit75.dev=dev;fit75.CFG=dict(fit75.CFG,optimizer=CFG['optimizer'],runs=CFG['runs'])
    os.environ.setdefault('R75_PROBE_TAG','');return fit75.run(rid,probe)

if __name__=='__main__':
    if sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='probe':run(sys.argv[2],int(sys.argv[3]))
    sys.stdout.flush();os._exit(0)
