"""WL73 Stage 0 (geometry only; no response, model output or error is read):
  * S16 coverage audit (face, boundary context, edge distance, thickness, location, pt0 displacement, distances to other supervised
    sources and their exact-symmetry images, image coverage),
  * the S16-R replacement rule (remove exact-symmetry duplicates among flat-interior S16 sources; add one source in each of four
    unsupported boundary contexts),
  * the stratified static confirmation population (context x distance to the S16-R training images) and the te0 deformation
    footprint ports, plus the deformation-pose rule and the te0 geometry identity gate.
    python ports73.py propose <out.json>   (design: prints the result of the rules; writes nothing in ROOT)
    python ports73.py freeze               (chain: recompute, assert identical to the frozen protocol lists, write ROOT/ports.json)"""
import sys,json
import numpy as np
import torch
from common73 import CFG,ROOT,INP68,R71,R72,write_new,sha,tensor_sha,environment
import sym72 as SY

KEYS=('xyz','rest_xyz','area','node','pair','prior','parent','src','dst','graph')
ST=CFG['stage0']
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def geoms(te0=True):
    G={g:load(INP68/'wl59'/f'geometry_{g}.pt') for g in ('rest','pt0')}
    if te0:G['te0']=load(ROOT/'inputs/geometry_te0.pt')
    return G

class Slab:
    """Rest material coordinates (mm), faces, boundary contexts, exact-symmetry permutations (rest: D4 x z-flip, 16 ops incl. identity)."""
    def __init__(self,g):
        self.x0=g['rest_xyz'].numpy().astype(np.float64);self.x=self.x0*1e3;self.face=np.rint(g['node'][:,15].numpy()*5).astype(int);self.P=len(self.x)
        ax,ay,az=np.abs(self.x).T;tb=np.isin(self.face,(4,5));sx=np.isin(self.face,(0,1))
        self.e=np.round(np.where(tb,np.minimum(50-ax,50-ay),np.where(sx,50-ay,50-ax)),3)   # top/bottom: nearest in-plane edge; side: along-edge distance to the vertical edge
        self.E=np.round(np.where(tb,np.maximum(50-ax,50-ay),np.nan),3)                       # top/bottom: farther in-plane edge
        self.row=np.round(np.where(tb,np.nan,az),3)                                           # side: |z| (1.25 mid row, 3.75 rim row)
        ctx=np.empty(self.P,dtype=object)
        ctx[tb&(self.e>=10)]='INT';ctx[tb&(self.e<10)&(self.E>=10)]='TBE';ctx[tb&(self.E<10)]='TBC'
        ctx[~tb&(self.e>=10)&(self.row==1.25)]='SM';ctx[~tb&(self.e>=10)&(self.row==3.75)]='SR';ctx[~tb&(self.e<10)]='SV'
        assert all(c is not None for c in ctx);self.ctx=ctx
        self.perms=[SY.permutation(self.x0,op) for op in SY.ops(True)]
        from scipy.spatial import cKDTree
        self.tree=cKDTree
    def images(self,ports):return sorted({int(p[i]) for p in self.perms for i in ports})
    def orbit_key(self,i):return min(int(p[i]) for p in self.perms)
    def dist(self,ports,to):
        """min 3-D rest distance (mm) from each of `ports` to the set `to`."""
        if not len(to):return np.full(len(ports),np.inf)
        return self.tree(self.x[list(to)]).query(self.x[list(ports)])[0]

def dmin(g,i):return float((1-g['beta'][i]).min())

def pose_rule(RG):
    """Unscored deformation pose: among in-family WL57 poses never used for a port-probe reference and not WL59 training poses,
    the one whose dent centre is farthest from every exact D4 image of the dent centres of the probed dented poses; tie: larger radius, name."""
    r=ST['deformation_pose'];used=[np.array(RG.POSES[u][2:])*1e3 if u!='pt0' else np.array(CFG['geometry']['pt0'][2:])*1e3 for u in r['probed_dented_poses']]
    sc={}
    for n in r['candidates']:
        c=np.array(RG.POSES[n][2:])*1e3
        sc[n]=min(float(np.linalg.norm(c-np.array(SY.D4[k],dtype=float)@u)) for u in used for k in SY.D4)
    best=sorted(sc,key=lambda n:(-round(sc[n],6),-RG.POSES[n][1],n))[0]
    return best,{n:round(v,4) for n,v in sc.items()}

def referenced():
    p71={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    p72={int(k):v['port'] for k,v in json.loads((R72/'ports.json').read_text())['slots'].items()}
    return p71,p72

def design(G,RG):
    sl=Slab(G['rest']);x=sl.x;p71,p72=referenced();S16={s:p71[s] for s in range(16)}
    R0=sorted(set(p71.values())|set(p72.values()))
    # ---- removal: exact-symmetry duplicates among flat-interior S16 sources (keep the lowest slot of each orbit)
    pt0=CFG['geometry']['pt0'];c0=np.array(pt0[2:])*1e3;R=pt0[1]*1e3;rm=ST['removal']
    flat=[s for s in range(16) if sl.ctx[S16[s]]=='INT' and float(np.hypot(*(x[S16[s],:2]-c0)))>=R+rm['outside_pt0_footprint_mm']]
    orb={};[orb.setdefault(sl.orbit_key(S16[s]),[]).append(s) for s in flat]
    removed=sorted(s for v in orb.values() if len(v)>1 for s in sorted(v)[1:])
    if len(removed)!=rm['count']:raise RuntimeError(f'STOP: removal rule gives {removed}, not {rm["count"]} sources (infeasible)')
    kept={s:S16[s] for s in range(16) if s not in removed}
    # ---- replacements: one source per unsupported boundary context
    new={};excl=set(sl.images(R0))
    for spec in ST['replacements']:
        t=np.array(spec['target_mm']);m=np.where((sl.face==spec['face'])&(sl.ctx==spec['context']))[0]
        if 'e_mm' in spec:m=m[sl.e[m]==spec['e_mm']]
        order=m[np.lexsort((m,np.round(np.linalg.norm(x[m]-t,axis=1),6)))]
        for i in order:
            if int(i) in excl or min(dmin(G[g],i) for g in ('rest','pt0'))<ST['min_collided_budget']:continue
            new[spec['slot']]=int(i);excl|=set(sl.images([int(i)]));break
        else:raise RuntimeError(f'STOP: no admissible replacement for slot {spec["slot"]}')
    S16R={**kept,**new};trainR=sl.images(list(S16R.values()));train16=sl.images(list(S16.values()))
    # ---- static confirmation (rest): context x distance to the S16-R training images
    conf={};excl=set(sl.images(R0+list(new.values())));actual=list(S16R.values())
    for spec in ST['confirmation_static']:
        m=np.where(sl.ctx==spec['context'])[0]
        if 'face' in spec:m=m[sl.face[m]==spec['face']]
        if 'e_min_mm' in spec:m=m[sl.e[m]>=spec['e_min_mm']]
        if 'e_mm' in spec:m=m[sl.e[m]==spec['e_mm']]
        if 'row_mm' in spec:m=m[sl.row[m]==spec['row_mm']]
        if spec['stratum']=='near_removed':rp=S16[spec['removed_slot']];m=m[sl.face[m]==sl.face[rp]]
        m=np.array([i for i in m if int(i) not in excl and dmin(G['rest'],i)>=ST['min_collided_budget']])
        if len(m) and conf:m=m[sl.dist(m,list(conf.values()))>=ST['static_spacing_mm']]
        if not len(m):raise RuntimeError(f'STOP: no admissible confirmation port for slot {spec["slot"]}')
        # near: 5 mm from an actual S16-R training port (hence also from its images); far: farthest from every image of an S16-R port
        if spec['stratum']=='near_removed':key=np.abs(np.round(sl.dist(m,[rp]),6)-ST['near_mm'])   # 5 mm from the actual removed S16 source
        else:key=np.abs(np.round(sl.dist(m,actual),6)-ST['near_mm']) if spec['stratum']=='near' else -np.round(sl.dist(m,trainR),6)
        i=int(m[np.lexsort((m,key))[0]]);conf[spec['slot']]=i;excl|=set(sl.images([i]))
    # ---- te0 footprint (deformation) ports
    pose=ST['deformation_pose']['selected'];pz=RG.POSES[pose];cz=np.array(pz[2:])*1e3;Rz=pz[1]*1e3;fp=ST['footprint']
    for spec in fp['ports']:
        t=np.array(spec['target_mm']);m=np.where(sl.face==spec['face'])[0];m=m[np.hypot(*(x[m,:2]-cz).T)<=fp['radius_fraction']*Rz]
        order=m[np.lexsort((m,np.round(np.linalg.norm(x[m]-t,axis=1),6)))]
        for i in order:
            if int(i) in excl or min(dmin(G[g],i) for g in ('rest',pose))<ST['min_collided_budget']:continue
            conf[spec['slot']]=int(i);excl|=set(sl.images([int(i)]));break
        else:raise RuntimeError(f'STOP: no admissible footprint port for slot {spec["slot"]}')
    return sl,dict(S16=S16,removed=removed,kept=kept,new=new,S16R=S16R,conf=conf,R0=R0,train16=train16,trainR=trainR,flat=flat)

def describe_port(sl,G,i,d,pose):
    x=sl.x[i];g=G.get(pose)
    out=dict(port=int(i),face=int(sl.face[i]),context=str(sl.ctx[i]),rest_xyz_mm=[round(float(v),4) for v in x],edge_mm=float(sl.e[i]),
        side_row_mm=None if np.isnan(sl.row[i]) else float(sl.row[i]),orbit_size=len({int(p[i]) for p in sl.perms}),
        d_to_S16R_images_mm=round(float(sl.dist([i],d['trainR'])[0]),4),d_to_S16_images_mm=round(float(sl.dist([i],d['train16'])[0]),4),
        d_to_S16R_actual_mm=round(float(sl.dist([i],list(d['S16R'].values()))[0]),4),d_to_S16_actual_mm=round(float(sl.dist([i],list(d['S16'].values()))[0]),4),
        d_to_removed_actual_mm=round(float(sl.dist([i],[d['S16'][s] for s in d['removed']])[0]),4),
        d_min={k:dmin(G[k],i) for k in G},beta_row_sha256={k:tensor_sha(G[k]['beta'][i]) for k in G})
    if g is not None:out['dz_mm_'+pose]=float((g['xyz'][i,2]-g['rest_xyz'][i,2])*1e3);out['thickness_mm_'+pose]=float(g['node'][i,13]*10)
    return out

def audit(sl,G,d):
    """Stage 0 S16 / S16-R coverage audit (geometry only)."""
    pt0=np.array(CFG['geometry']['pt0'][2:])*1e3;rows={}
    for s,i in d['S16'].items():
        others=[j for t,j in d['S16'].items() if t!=s]
        rows[s]=dict(port=i,face=int(sl.face[i]),context=str(sl.ctx[i]),rest_xyz_mm=[round(float(v),4) for v in sl.x[i]],edge_mm=float(sl.e[i]),
            thickness_mm_rest=float(G['rest']['node'][i,13]*10),thickness_mm_pt0=float(G['pt0']['node'][i,13]*10),
            dz_mm_pt0=float((G['pt0']['xyz'][i,2]-G['pt0']['rest_xyz'][i,2])*1e3),d_pt0_centre_xy_mm=round(float(np.hypot(*(sl.x[i,:2]-pt0))),4),
            nearest_other_S16_mm=round(float(sl.dist([i],others)[0]),4),nearest_other_S16_image_mm=round(float(sl.dist([i],sl.images(others))[0]),4),
            orbit_key=sl.orbit_key(i),orbit_size=len({int(p[i]) for p in sl.perms}),flat_interior=s in d['flat'],removed=s in d['removed'])
    cov=lambda ports:{c:int(sum(sl.ctx[j]==c for j in ports)) for c in ('INT','TBE','TBC','SM','SR','SV')}
    return dict(per_slot=rows,contexts_all_ports=cov(range(sl.P)),S16_sources=cov(d['S16'].values()),S16R_sources=cov(d['S16R'].values()),
        S16_image_union=dict(size=len(d['train16']),contexts=cov(d['train16']),min_edge_mm=float(sl.e[d['train16']].min())),
        S16R_image_union=dict(size=len(d['trainR']),contexts=cov(d['trainR'])))

def te0_identity(RG,G):
    """Archived WL57 te0 geometry vs a fresh build here: features bitwise except node |delta| <= 1e-12 (WL72 amendment 1 rule)."""
    pose=ST['deformation_pose']['selected'];fresh=RG.geometry(pose,ballistic=True);A=G[pose]
    for k in KEYS:
        if k=='node':assert fresh[k].shape==A[k].shape and float((fresh[k].cpu().double()-A[k].double()).abs().max())<=1e-12,('identity',k)
        else:assert torch.equal(fresh[k].cpu(),A[k]),('identity',k)
    dn=fresh['node'].cpu()!=A['node'];bf,ba=fresh['beta'].cpu().double(),A['beta'].double()
    return dict(features_bitwise_equal=not bool(dn.any()),node_non_bitwise_elements=dn.nonzero().tolist(),
        node_max_abs_delta=float((fresh['node'].cpu().double()-A['node'].double()).abs().max()),beta_rel_L2=float((bf-ba).norm()/ba.norm()),
        archived_sha256=sha(ROOT/'inputs/geometry_te0.pt'),pose=list(RG.POSES[pose]))

def result(RG,G):
    sl,d=design(G,RG);pose=ST['deformation_pose']['selected']
    best,scores=pose_rule(RG);assert best==pose,('pose rule',best,pose)
    spec={s['slot']:s for s in ST['confirmation_static']+ST['footprint']['ports']}
    return dict(removed={str(s):d['S16'][s] for s in d['removed']},kept={str(s):p for s,p in d['kept'].items()},
        new={str(s):describe_port(sl,G,p,d,'pt0')|dict(context_target=[r for r in ST['replacements'] if r['slot']==s][0]['context']) for s,p in d['new'].items()},
        S16R={str(s):p for s,p in sorted(d['S16R'].items())},
        confirmation={str(s):describe_port(sl,G,p,d,pose)|dict(spec=spec[s]) for s,p in d['conf'].items()},
        pose=dict(selected=pose,scores=scores),audit=audit(sl,G,d),excluded_referenced_ports=d['R0'],information_used='geometry only')

def main(mode,out=None):
    RG=environment();torch.set_grad_enabled(False);G=geoms()
    if mode=='propose':
        r=result(RG,G);open(out,'w').write(json.dumps(r,indent=1,default=float));print('PROPOSE',json.dumps(dict(removed=r['removed'],new={k:v['port'] for k,v in r['new'].items()},
            conf={k:v['port'] for k,v in r['confirmation'].items()})),flush=True);return
    if (ROOT/'ports.json').exists():raise FileExistsError('immutable')
    r=result(RG,G);fz=CFG['frozen']
    got=dict(removed=sorted(int(s) for s in r['removed']),new={k:v['port'] for k,v in r['new'].items()},confirmation={k:v['port'] for k,v in r['confirmation'].items()})
    assert got['removed']==fz['removed_slots'] and got['new']==fz['new_ports'] and got['confirmation']==fz['confirmation_ports'],('frozen lists differ',got)
    r['te0_identity']=te0_identity(RG,G);write_new(ROOT/'ports.json',r);print('PORTS',json.dumps(got),json.dumps(r['te0_identity']),flush=True)

if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();import os;os._exit(0)
