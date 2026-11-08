"""WL80 Stage 3: controlled 2 x 2 learned experiment on the unchanged W+UV-Q operator (q78.TransportQ: rank 16, WL78 inputs, layers, capacity).
  Factor A (frozen angular basis): A0 = the historical G2 basis (rank75.basis) | A1 = the selected orbit-balanced basis G1 (basis80.select,
           G2-aligned mixture order), stored float64, rounded once to float32 at load (as rank75.basis)
  Factor B (training presentation): B0 = the historical presentation (Q78) | B1 = exact rest-symmetry presentation (fit79.SymRest, the
           frozen WL79 Latin schedule: rest source k at step t under OPS[(k + t) mod 16]; pt0 identity)
  Arms: A0B0 = the frozen Q78 control (never retrained); A0B1, A1B0, A1B1 trained here with the two matched spatial seeds.
Initialisation (predeclared): the Q78 rule (q78.init_model: P_590x trunk, rank-dependent rows of mixture r mod 8, encoder / pair_layer
re-initialised with seed 7801 / 7802), then in_logits / out_logits := the arm's frozen basis. G1's mixtures were re-indexed onto G2 by the optimal
joint-pattern assignment, so mixture r keeps the latent identity used by the rank-row rule; the trunk is bitwise the Q78 initial trunk.
Unchanged: objective J (mean of rest and pt0 terms of ato74.objective_m) + 1.0 L_sym (WL72), Adam, cosine 3e-3 -> 3e-5, 20,000 steps, clip 1.0,
checkpoints 2,000 / 6,000 / 10,000, final iterate primary (no selection), 16 rest + 16 pt0 physical samples per update.
    python fit80.py run <run_id> | probe <run_id> <steps>"""
import sys,os,json
import torch
from common80 import CFG,ROOT,save_new,write_new,environment,dev,load,sha,tensor_sha
RG=environment()
import fit75
import fit79
import q78 as Q
import rank75 as RK
import ato74 as AT
import sym72 as SY

def basis(arm):
    """(in_logits, out_logits) float32 and provenance of the arm's frozen basis."""
    if arm.startswith('A0'):a,o,meta=RK.basis();return a,o,dict(meta,basis='G2')
    f=ROOT/'basis/G1.pt';b=load(f);return b['in_logits'].float(),b['out_logits'].float(),dict(basis='G1',sha256=sha(f),in_logits_sha256=tensor_sha(b['in_logits']),
        out_logits_sha256=tensor_sha(b['out_logits']),dtype_stored='float64',dtype_model='float32 (rounded once at load)')

def init_model(spec):
    m,meta=Q.init_model(RG,spec['init'],spec['reinit_seed'],spec['rank']);a,o,bm=basis(spec['arm'])
    with torch.no_grad():m.in_logits.copy_(a.cuda());m.out_logits.copy_(o.cuda())
    return m,dict(meta,arm=spec['arm'],angular=bm)

def build(rid,RG):
    import fit68,fit72,fit73
    spec=CFG['runs'][rid];pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
    m,meta=init_model(spec);params=fit68.group_params(m,'net')
    assert not m.in_logits.requires_grad and not m.out_logits.requires_grad
    cols=lambda g,ports:Q.columns(m,g,ports);lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F);del pb.F
    for k in list(gs):Q.QG(RG,gs[k])
    symp=spec['arm'].endswith('B1');rest=fit79.SymRest(RG,pb.tg['rest'],pb.ports,gs['rest']) if symp else None;state=dict(n=0)
    def loss():
        if symp and torch.is_grad_enabled():ports=rest.set(state['n']);state['n']+=1
        else:ports=rest.identity() if symp else pb.ports
        Jr=AT.objective_m(*cols(gs['rest'],ports),pb.tg['rest']);Jp=AT.objective_m(*cols(gs['pt0'],pb.ports),pb.tg['pt0']);J=(Jr+Jp)/2;terms=[]
        for key,src,lab in sym:
            for c0 in range(0,len(src),64):
                ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
        L=torch.cat(terms).mean();return J+lam*L,J,L
    return spec,m,meta,params,loss,pb,sym

def run(rid,probe=None):
    fit75.build=build;fit75.ROOT=ROOT;fit75.dev=dev;fit75.CFG=dict(fit75.CFG,optimizer=CFG['optimizer'],runs=CFG['runs'])
    os.environ.setdefault('R75_PROBE_TAG','');return fit75.run(rid,probe)

if __name__=='__main__':
    if sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='probe':run(sys.argv[2],int(sys.argv[3]))
    sys.stdout.flush();os._exit(0)
