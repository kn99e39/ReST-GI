"""WL78 Stage 3: W+UV-Q rank 16 with the frozen G2 basis, exactly the WL75 A75 recipe otherwise (fit75.build / run): S16-R fitting targets
(replicas 0-3), objective J + 1.0 L_sym, 'net' group (A/O frozen), Adam, cosine 3e-3 -> 3e-5, 20,000 steps, clip 1.0, checkpoints
2,000 / 6,000 / 10,000, final iterate (no selection). The only differences: the geometry inputs (q78.QG) and the re-initialised input layers.
    python fit78.py run <run_id> | probe <run_id> <steps>"""
import sys,os,json,time,math
import torch
from common78 import CFG,ROOT,save_new,write_new,environment,dev
RG=environment()
import fit75
import q78 as Q
import ato74 as AT
import sym72 as SY
TRAIN=('rest','pt0')

def build(rid,RG):
    import fit68,fit72,fit73
    spec=CFG['runs'][rid];pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
    m,meta=Q.init_model(RG,spec['init'],spec['reinit_seed'],spec['rank']);params=fit68.group_params(m,'net')
    assert not m.in_logits.requires_grad and not m.out_logits.requires_grad
    cols=lambda g,ports:Q.columns(m,g,ports);lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F);del pb.F
    for k in list(gs):Q.QG(RG,gs[k])     # every training / symmetry-image geometry gets its WL78 inputs once (geometry-only)
    def loss():
        Js=[AT.objective_m(*cols(gs[g],pb.ports),pb.tg[g]) for g in TRAIN];J=sum(Js)/len(Js);terms=[]
        for key,src,lab in sym:
            for c0 in range(0,len(src),64):
                ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
        L=torch.cat(terms).mean();return J+lam*L,J,L
    return spec,m,meta,params,loss,pb,sym

# fit75.run with this module's build (identical loop, checkpoints, records); paths under the WL78 root
def run(rid,probe=None):
    fit75.build=build;fit75.ROOT=ROOT;fit75.dev=dev;fit75.CFG=dict(fit75.CFG,optimizer=CFG['optimizer'],runs=CFG['runs'])
    os.environ.setdefault('R75_PROBE_TAG','');return fit75.run(rid,probe)

if __name__=='__main__':
    if sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='probe':run(sys.argv[2],int(sys.argv[3]))
    sys.stdout.flush();os._exit(0)
