"""WL79 Stage 5 prospective confirmation population (geometry only; ports73 Slab / ports76 machinery). Frozen before any candidate score.
Excluded: every previously referenced or scored source position and ALL its exact rest images: WL66-72 referenced (slots 0-15, 20-27, 40-47),
S16-R (training) and its 16-operation orbit (= the images exposed by the WL79 symmetry presentation), the WL73 confirmation ports, the WL74
Stage 4 ports (scored in WL75), the WL76 / WL77A Part 0 ports (scored in WL78).
Static rest slots: contexts TBE / TBC / SM / SR / SV / INT x strata near (|distance to the training orbit - 5 mm| minimal) / far (maximal
distance to the training orbit) x coarse-cell sub-position (hi = most extreme available of corner > edge > center, lo = least extreme), 10 mm
spacing, d_min >= 0.5 (ports76 rules; 'near' now measured to the exposed orbit, which contains the actual sources). No new deformation pose:
Stage 6 uses the existing qualified deformation references.
    python ports79.py propose <out.json> | freeze"""
import sys,json
import numpy as np
import torch
import os
from common79 import CFG,ROOT,R73,R74,R77A,INP68,write_new,sha,environment,HERE
sys.path.insert(0,str(HERE.parent/'fnto_fia76'))
import ports73 as P73
import ports76 as P76

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
CTX=('TBE','TBC','SM','SR','SV','INT');STRATA=('near','far');SUB=('hi','lo')
def specs():
    out=[];k=0
    for c in CTX:
        for st in STRATA:
            for sb in SUB:out.append(dict(slot=k,context=c,stratum=st,subcell=sb));k+=1
    return out

def design():
    G={'rest':load(INP68/'wl59/geometry_rest.pt')};sl=P73.Slab(G['rest']);x=sl.x;p71,p72=P73.referenced()
    w73=json.loads((R73/'ports.json').read_text());w74=json.loads((R74/'ports4.json').read_text());w76=json.loads((R77A/'ports76.json').read_text())
    S16R={**{int(k):v for k,v in w73['kept'].items()},**{int(k):v['port'] for k,v in w73['new'].items()}};actual=list(S16R.values());orbit=sl.images(actual)
    prev=sorted(set(p71.values())|set(p72.values())|set(actual)|{v['port'] for v in w73['confirmation'].values()}|{v['port'] for v in w74['ports'].values()}
                |{v['port'] for v in w76['ports'].values() if v['port'] is not None})
    excl=set(sl.images(prev))|set(orbit);conf={};lab=np.array([P76.subcell_label(G['rest'],i) if sl.ctx[i] else '' for i in range(sl.P)])
    for spec in specs():
        m=np.where(sl.ctx==spec['context'])[0]
        if spec['context']=='INT':m=m[sl.e[m]>=12.5]
        m=np.array([i for i in m if int(i) not in excl and P73.dmin(G['rest'],i)>=.5])
        chosen=[v for v in conf.values() if v is not None]
        if len(m) and chosen:m=m[sl.dist(m,chosen)>=10.]
        if len(m):rk=np.array([P76.RANK[lab[i]] for i in m]);m=m[rk==(rk.max() if spec['subcell']=='hi' else rk.min())]
        if not len(m):conf[spec['slot']]=None;continue
        key=np.abs(np.round(sl.dist(m,orbit),6)-5.) if spec['stratum']=='near' else -np.round(sl.dist(m,orbit),6)
        i=int(m[np.lexsort((m,key))[0]]);conf[spec['slot']]=i;excl|=set(sl.images([i]))
    rows={}
    for s,i in conf.items():
        sp=[q for q in specs() if q['slot']==s][0]
        if i is None:rows[str(s)]=dict(port=None,spec=sp);continue
        rows[str(s)]=dict(port=i,spec=sp,face=int(sl.face[i]),context=str(sl.ctx[i]),subcell=str(lab[i]),rest_xyz_mm=[round(float(v),4) for v in x[i]],edge_mm=float(sl.e[i]),
            d_to_S16R_actual_mm=round(float(sl.dist([i],actual)[0]),4),d_to_training_orbit_mm=round(float(sl.dist([i],orbit)[0]),4),
            d_to_previous_scored_mm=round(float(sl.dist([i],prev)[0]),4),orbit_size=len({int(p[i]) for p in sl.perms}))
    return rows,prev,orbit

def main(mode,out=None):
    environment();torch.set_grad_enabled(False);rows,prev,orbit=design();got={k:v['port'] for k,v in rows.items()}
    if mode=='propose':open(out,'w').write(json.dumps(dict(ports=rows,excluded_previous=len(prev),training_orbit=len(orbit)),indent=1,default=float));print('PROPOSE',json.dumps(got),flush=True);return
    if (ROOT/'ports79.json').exists():raise FileExistsError('immutable')
    fz=CFG['stage5'];assert got=={k:v for k,v in fz['ports'].items()},('frozen lists differ',got)
    write_new(ROOT/'ports79.json',dict(ports=rows,excluded_previous=len(prev),training_orbit=orbit,information_used='geometry only'))
    print('PORTS79',json.dumps(got),flush=True)

if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();import os;os._exit(0)
