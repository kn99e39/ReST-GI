"""WL80 Stage 0 contract reconciliation (geometry and archived WL79 records only; no model output, no reference response value is read).
 C1 WL79 Stage 0 / 1 records: present, unchanged (sha256), contract VALID, Stage 1 ANGULAR BASIS LIMIT recorded.
 C2 group composition of the exact rest maps for all 16 x 16 pairs (ports, incoming and outgoing bins) and inverses.
 C3 feature-construction non-equivariance of the W+UV-Q inputs (classes recorded, never treated as physical):
    probe ties: for each operation, every port whose node inputs are not exactly equivariant is classified as TIE (the tie set of its probe
    contains the image of the probe chosen at the source, i.e. an index-ordered tie break) or OTHER;
    128-NN neighbourhood: every unmapped edge is classified as TIE (its distance equals the 128th-neighbour distance to 1e-7 m: float32 rest coordinates) or OTHER.
 C4 symmetry classes: A exact whole-domain D4h (the only class that authorises a label permutation), B approximate local-context
    equivalence (historical pairs 101<-50, 105<-52, 107<-53: op + in-face translation), C arbitrary translation (not a symmetry).
    python stage0_80.py <out.json>"""
import sys,os,json
import numpy as np
import torch
from common80 import ROOT,R79,environment,load,sha
RG=environment();torch.set_grad_enabled(False)
import fit71,sym72 as SY,sym79 as S9,q78 as Q
out={}
# ---- C1
rec={p:sha(R79/p) for p in ('stage0/geometry.json','stage0/refs.json','stage1/oracle.json','stage1/gate.json','stage1/closure.json')}
g1=json.loads((R79/'stage1/gate.json').read_text());r0=json.loads((R79/'stage0/refs.json').read_text())['summary']
out['C1']=dict(sha256=rec,stage1=g1['classification'],stage0_contract=g1['stage0'],references=r0,no_wl79_stage5_references=not (R79/'refs').exists(),
               no_wl79_ports_frozen=not (R79/'ports79.json').exists())
# ---- C2
g=RG.device(fit71.guarded('geometry_rest'));x0=g['rest_xyz'].double().cpu().numpy();P=g['P'];OPS=SY.ops(True);M={op:S9.Map(RG,'rest',op,x0) for op in OPS}
def comp(a,b):
    m=SY.matrix(a)@SY.matrix(b);return [c for c in OPS if np.allclose(SY.matrix(c),m)][0]
bad=0;n=0
for a in OPS:
    for b in OPS:
        c=comp(a,b);ma,mb,mc=M[a],M[b],M[c];n+=1
        ok=torch.equal(ma.perm[mb.perm],mc.perm) and torch.equal(torch.gather(ma.pin[mb.perm],1,mb.pin),mc.pin) and torch.equal(torch.gather(ma.pout[mb.perm],1,mb.pout),mc.pout)
        bad+=not ok
out['C2']=dict(pairs=n,failures=bad,all_exact=all(m.exact for m in M.values()))
# ---- C3
q=Q.QG(RG,g);node=q['node'][:,:16].double().cpu();xx=g['xyz'].double();nn=torch.nn.functional.normalize(g['node'][:,9:12].double(),dim=1)
D=torch.cdist(xx,xx);perp=(nn@nn.T).abs()<Q.PERP;D1=torch.where(perp,D,torch.full_like(D,float('inf')));r1=D1.min(1).values
ties1=[(D1[i]-r1[i]).abs().le(1e-7).nonzero().flatten().tolist() for i in range(P)]
Dc=D.cpu();src=g['src'].cpu();dst=g['dst'].cpu();K=128;dd=Dc[src,dst].view(P,K);kth=dd.max(1).values
Eset=set((src*P+dst).tolist());c3={}
for op,mp in M.items():
    if op==('id',False):continue
    exp=node.clone()
    for s0 in (0,6,10):exp[:,s0:s0+3]=torch.einsum('pij,pj->pi',mp.Q,node[:,s0:s0+3])
    diff=(node[mp.perm]-exp).abs().max(1).values;badp=torch.nonzero(diff>1e-5).flatten().tolist()
    cls={'TIE':0,'OTHER':0}
    for p in badp:   # probe 1 of p maps to one of the tie candidates of probe 1 at the image port -> index tie break
        j=int(D1[p].argmin());cand=ties1[int(mp.perm[p])];cls['TIE' if (int(mp.perm[j]) in cand and len(cand)>1) or len(ties1[p])>1 else 'OTHER']+=1
    img=(mp.perm[src]*P+mp.perm[dst]).tolist();un=[e for e,k in enumerate(img) if k not in Eset]
    et={'TIE':0,'OTHER':0}
    for e in un:et['TIE' if abs(float(dd.view(-1)[e])-float(kth[e//K]))<=1e-7 or abs(float(Dc[int(mp.perm[src[e]]),int(mp.perm[dst[e]])])-float(kth[int(mp.perm[src[e]])]))<=1e-12 else 'OTHER']+=1
    c3[f"{op[0]}{'+z' if op[1] else ''}"]=dict(node_non_equivariant_ports=len(badp),node_classes=cls,max_node_diff=float(diff.max()),unmapped_edges=len(un),edge_classes=et)
out['C3']=c3
out['C4']=dict(A='exact whole-domain D4h (16 rest operations): label permutation authorised (sym79 maps)',B='approximate local-context equivalence (op + in-face translation; e.g. 101<-50): diagnostic comparison only, never a label',
               C='translation of a source inside the finite slab: not a symmetry',dented='pt0 / pt0v / te0 maps are not exact (WL79 Stage 0): identity presentation only')
out['valid']=bool(out['C2']['failures']==0 and out['C2']['all_exact'] and g1['stage0']['valid'])   # C3 is documentation (feature construction), never a label
json.dump(out,open(sys.argv[1],'w'),indent=1);print('STAGE0_80',json.dumps(dict(C2=out['C2'],C3={k:(v['node_classes'],v['edge_classes']) for k,v in c3.items()},valid=out['valid'])),flush=True)
sys.stdout.flush();os._exit(0)
