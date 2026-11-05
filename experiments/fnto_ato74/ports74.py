"""WL74 Stage 4 (conditional) confirmation population, geometry only (ports73 machinery unchanged): exclude every previously scored
source position and its exact images (WL66-72 referenced, S16-R, all WL73 confirmation ports); boundary contexts TBE/TBC/SM/SR and INT,
'near' (|dA - 5 mm| to an actual S16-R source) and 'far' (max distance to any S16-R image), 10 mm spacing; a new deformation pose by the
WL73 pose rule with te0 added to the probed poses, and its footprint ports.
    python ports74.py propose <out.json> | freeze"""
import sys,json
import numpy as np
import torch
from common74 import CFG,ROOT,R73,INP68,write_new,sha,tensor_sha,environment
import ports73 as P73
import sym72 as SY

ST=CFG['stage4']
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def pose_rule(RG):
    r=ST['pose'];pt0=CFG['geometry_pt0']
    used=[np.array(pt0[2:])*1e3 if u=='pt0' else np.array(RG.POSES[u][2:])*1e3 for u in r['probed_dented_poses']];sc={}
    for n in r['candidates']:
        c=np.array(RG.POSES[n][2:])*1e3;sc[n]=min(float(np.linalg.norm(c-np.array(SY.D4[k],dtype=float)@u)) for u in used for k in SY.D4)
    return sorted(sc,key=lambda n:(-round(sc[n],6),-RG.POSES[n][1],n))[0],{n:round(v,4) for n,v in sc.items()}

def design(G,RG):
    sl=P73.Slab(G['rest']);x=sl.x;p71,p72=P73.referenced();w73=json.loads((R73/'ports.json').read_text())
    S16R={**{int(k):v for k,v in w73['kept'].items()},**{int(k):v['port'] for k,v in w73['new'].items()}}
    prev=sorted(set(p71.values())|set(p72.values())|set(S16R.values())|{v['port'] for v in w73['confirmation'].values()})
    trainR=sl.images(list(S16R.values()));actual=list(S16R.values());excl=set(sl.images(prev));conf={}
    for spec in ST['static']:
        m=np.where(sl.ctx==spec['context'])[0]
        if 'e_min_mm' in spec:m=m[sl.e[m]>=spec['e_min_mm']]
        if 'e_mm' in spec:m=m[sl.e[m]==spec['e_mm']]
        m=np.array([i for i in m if int(i) not in excl and P73.dmin(G['rest'],i)>=.5])
        if len(m) and conf:m=m[sl.dist(m,list(conf.values()))>=ST['spacing_mm']]
        if not len(m):raise RuntimeError(f'STOP: no admissible Stage 4 port for slot {spec["slot"]}')
        key=np.abs(np.round(sl.dist(m,actual),6)-ST['near_mm']) if spec['stratum']=='near' else -np.round(sl.dist(m,trainR),6)
        i=int(m[np.lexsort((m,key))[0]]);conf[spec['slot']]=i;excl|=set(sl.images([i]))
    pose=ST['pose']['selected'];pz=RG.POSES[pose];cz=np.array(pz[2:])*1e3;Rz=pz[1]*1e3
    for spec in ST['footprint']:
        t=np.array([cz[0]+spec['offset_mm'][0],cz[1]+spec['offset_mm'][1],spec['z_mm']]);m=np.where(sl.face==spec['face'])[0]
        m=m[np.hypot(*(x[m,:2]-cz).T)<=ST['footprint_radius_fraction']*Rz];order=m[np.lexsort((m,np.round(np.linalg.norm(x[m]-t,axis=1),6)))]
        for i in order:
            if int(i) in excl or min(P73.dmin(G[g],i) for g in ('rest',pose))<.5:continue
            conf[spec['slot']]=int(i);excl|=set(sl.images([int(i)]));break
        else:raise RuntimeError(f'STOP: no admissible footprint port for slot {spec["slot"]}')
    rows={}
    for s,i in conf.items():
        sp=[q for q in ST['static']+ST['footprint'] if q['slot']==s][0]
        rows[str(s)]=dict(port=i,spec=sp,face=int(sl.face[i]),context=str(sl.ctx[i]),rest_xyz_mm=[round(float(v),4) for v in x[i]],edge_mm=float(sl.e[i]),
            d_to_S16R_actual_mm=round(float(sl.dist([i],actual)[0]),4),d_to_S16R_images_mm=round(float(sl.dist([i],trainR)[0]),4),
            d_to_previous_scored_mm=round(float(sl.dist([i],prev)[0]),4),d_min={k:P73.dmin(G[k],i) for k in G},
            dz_mm=float((G[pose]['xyz'][i,2]-G[pose]['rest_xyz'][i,2])*1e3) if pose in G else None,beta_row_sha256={k:tensor_sha(G[k]['beta'][i]) for k in G})
    return rows,prev

def main(mode,out=None):
    RG=environment();torch.set_grad_enabled(False);pose=ST['pose']['selected']
    best,scores=pose_rule(RG);assert best==pose,('pose rule',best,pose)
    G={'rest':load(INP68/'wl59/geometry_rest.pt'),pose:load(ROOT/'inputs'/f'geometry_{pose}.pt')}
    rows,prev=design(G,RG);got={k:v['port'] for k,v in rows.items()}
    if mode=='propose':
        open(out,'w').write(json.dumps(dict(ports=rows,pose_scores=scores,excluded_previous=len(prev)),indent=1,default=float));print('PROPOSE',json.dumps(got),flush=True);return
    if (ROOT/'ports4.json').exists():raise FileExistsError('immutable')
    assert got==CFG['frozen']['stage4_ports'],('frozen lists differ',got)
    fresh=RG.geometry(pose,ballistic=True);A=G[pose]
    for k in P73.KEYS:
        if k=='node':assert float((fresh[k].cpu().double()-A[k].double()).abs().max())<=1e-12,('identity',k)
        else:assert torch.equal(fresh[k].cpu(),A[k]),('identity',k)
    ident=dict(node_max_abs_delta=float((fresh['node'].cpu().double()-A['node'].double()).abs().max()),beta_rel_L2=float((fresh['beta'].cpu().double()-A['beta'].double()).norm()/A['beta'].double().norm()),
        archived_sha256=sha(ROOT/'inputs'/f'geometry_{pose}.pt'),pose=list(RG.POSES[pose]))
    write_new(ROOT/'ports4.json',dict(ports=rows,pose=dict(selected=pose,scores=scores),identity=ident,excluded_previous=len(prev),information_used='geometry only'))
    print('PORTS4',json.dumps(got),json.dumps(ident),flush=True)

if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();import os;os._exit(0)
