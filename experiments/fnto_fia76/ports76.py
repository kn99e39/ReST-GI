"""WL76 Stage 4 confirmation population, geometry only (ports73 Slab machinery). Excludes every previously scored source position and all
of its exact slab images: WL66-72 referenced (slots 0-15, 20-27, 40-47), S16-R, the 29 WL73 confirmation ports, the 16 WL74 Stage 4 ports
(scored in WL75). Static slots: contexts TBE / TBC / SM / SR / SV / INT x strata near (|d_actual - 5 mm| minimal) / far (maximal distance
to the S16-R images) x coarse-cell sub-position pair (hi = most extreme available of corner > edge > center, lo = least extreme; paired controls), 10 mm spacing, d_min >= 0.5. Deformation: a new pose by
the WL73 pose rule with cf1 added to the probed poses, 4 footprint ports.
    python ports76.py propose <out.json> | freeze"""
import sys,json
import numpy as np
import torch
from common76 import CFG,ROOT,R73,R74,INP68,write_new,sha,tensor_sha,environment
import ports73 as P73
import sym72 as SY

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
CTX=('TBE','TBC','SM','SR','SV','INT');STRATA=('near','far');SUB=('hi','lo')
RANK={'center':0,'edge':1,'corner':2}
def specs():
    out=[];k=0
    for c in CTX:
        for st in STRATA:
            for sb in SUB:out.append(dict(slot=k,context=c,stratum=st,subcell=sb));k+=1
    return out
FOOT=[dict(slot=24,face=5,offset_mm=[0,0],z_mm=5.),dict(slot=25,face=4,offset_mm=[0,0],z_mm=-5.),dict(slot=26,face=5,offset_mm=[7.5,0],z_mm=5.),dict(slot=27,face=4,offset_mm=[-7.5,0],z_mm=-5.)]

def subcell_label(g,i):
    x=g['rest_xyz'].double().numpy()*1e3;par=g['parent'].numpy();mem=np.where(par==par[i])[0];X=x[mem];ext=0
    for ax in range(3):
        u=np.unique(np.round(X[:,ax],4))
        if len(u)<2:continue
        k=int(np.searchsorted(u,np.round(x[i,ax],4)));ext+=k in (0,len(u)-1)
    return {0:'center',1:'edge'}.get(ext,'corner')

def pose_rule(RG):
    r=dict(candidates=['te1','te3','va0','va1','cf0'],probed=['pt0','cf2','cf3','te2','te0','cf1']);pt0=np.array(__import__('common68').PT0[2:])*1e3;sc={}
    used=[pt0 if u=='pt0' else np.array(RG.POSES[u][2:])*1e3 for u in r['probed']]
    for n in r['candidates']:
        c=np.array(RG.POSES[n][2:])*1e3;sc[n]=min(float(np.linalg.norm(c-np.array(SY.D4[k],dtype=float)@u)) for u in used for k in SY.D4)
    return sorted(sc,key=lambda n:(-round(sc[n],6),-RG.POSES[n][1],n))[0],{n:round(v,4) for n,v in sc.items()}

def design(RG):
    G={'rest':load(INP68/'wl59/geometry_rest.pt')};pose,scores=pose_rule(RG);gp=RG.geometry(pose,ballistic=True);G[pose]={k:(v.cpu() if isinstance(v,torch.Tensor) else v) for k,v in gp.items()}
    sl=P73.Slab(G['rest']);x=sl.x;p71,p72=P73.referenced();w73=json.loads((R73/'ports.json').read_text());w74=json.loads((R74/'ports4.json').read_text())
    S16R={**{int(k):v for k,v in w73['kept'].items()},**{int(k):v['port'] for k,v in w73['new'].items()}};actual=list(S16R.values());trainR=sl.images(actual)
    prev=sorted(set(p71.values())|set(p72.values())|set(actual)|{v['port'] for v in w73['confirmation'].values()}|{v['port'] for v in w74['ports'].values()})
    excl=set(sl.images(prev));conf={};lab=np.array([subcell_label(G['rest'],i) if sl.ctx[i] else '' for i in range(sl.P)])
    for spec in specs():
        m=np.where(sl.ctx==spec['context'])[0]
        if spec['context']=='INT':m=m[sl.e[m]>=12.5]
        m=np.array([i for i in m if int(i) not in excl and P73.dmin(G['rest'],i)>=.5])
        chosen=[v for v in conf.values() if v is not None]
        if len(m) and chosen:m=m[sl.dist(m,chosen)>=10.]
        if len(m):   # sub-cell control: 'hi' = the most extreme available coarse-cell position (corner > edge > center), 'lo' = the least extreme
            rk=np.array([RANK[lab[i]] for i in m]);m=m[rk==(rk.max() if spec['subcell']=='hi' else rk.min())]
        if not len(m):conf[spec['slot']]=None;continue
        key=np.abs(np.round(sl.dist(m,actual),6)-5.) if spec['stratum']=='near' else -np.round(sl.dist(m,trainR),6)
        i=int(m[np.lexsort((m,key))[0]]);conf[spec['slot']]=i;excl|=set(sl.images([i]))
    pz=RG.POSES[pose];cz=np.array(pz[2:])*1e3;Rz=pz[1]*1e3
    for spec in FOOT:
        t=np.array([cz[0]+spec['offset_mm'][0],cz[1]+spec['offset_mm'][1],spec['z_mm']]);m=np.where(sl.face==spec['face'])[0]
        m=m[np.hypot(*(x[m,:2]-cz).T)<=.6*Rz];order=m[np.lexsort((m,np.round(np.linalg.norm(x[m]-t,axis=1),6)))]
        conf[spec['slot']]=None
        for i in order:
            if int(i) in excl or min(P73.dmin(G[g],i) for g in ('rest',pose))<.5:continue
            conf[spec['slot']]=int(i);excl|=set(sl.images([int(i)]));break
    rows={}
    for s,i in conf.items():
        sp=[q for q in specs()+FOOT if q['slot']==s][0]
        if i is None:rows[str(s)]=dict(port=None,spec=sp);continue
        rows[str(s)]=dict(port=i,spec=sp,face=int(sl.face[i]),context=str(sl.ctx[i]),subcell=str(lab[i]),rest_xyz_mm=[round(float(v),4) for v in x[i]],edge_mm=float(sl.e[i]),
            d_to_S16R_actual_mm=round(float(sl.dist([i],actual)[0]),4),d_to_S16R_images_mm=round(float(sl.dist([i],trainR)[0]),4),d_to_previous_scored_mm=round(float(sl.dist([i],prev)[0]),4))
    return rows,prev,pose,scores

def main(mode,out=None):
    RG=environment();torch.set_grad_enabled(False);rows,prev,pose,scores=design(RG);got={k:v['port'] for k,v in rows.items()}
    if mode=='propose':open(out,'w').write(json.dumps(dict(ports=rows,pose=pose,pose_scores=scores,excluded_previous=len(prev)),indent=1,default=float));print('PROPOSE',pose,json.dumps(got),flush=True);return
    if (ROOT/'ports76.json').exists():raise FileExistsError('immutable')
    fz=CFG['stage4'];assert pose==fz['pose'] and got=={k:v for k,v in fz['ports'].items()},('frozen lists differ',pose,got)
    fresh=RG.geometry(pose,ballistic=True);(ROOT/'inputs').mkdir(parents=True,exist_ok=True);import r57_geometry as R
    __import__('common68').save_new(ROOT/'inputs'/f'geometry_{pose}.pt',R.serial(fresh))
    write_new(ROOT/'ports76.json',dict(ports=rows,pose=dict(selected=pose,scores=scores),geometry_sha256=sha(ROOT/'inputs'/f'geometry_{pose}.pt'),excluded_previous=len(prev),information_used='geometry only'))
    print('PORTS76',json.dumps(got),flush=True)

if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();import os;os._exit(0)
