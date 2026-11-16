"""CK2 bx38 geometry and frozen ER as a native Mitsuba/8DNA scene.

Only the panel and checker box A are one prebaked asset. Everything else,
including the point emitter, is outside its AABB. All scene coordinates are
uniformly normalized together; the asset instance transform stays identity
because the released 8DNA query code assumes orthonormal instance matrices.
"""
import importlib, json, math, os, sys
from pathlib import Path
import numpy as np
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

BASE=Path(os.environ.get('CK21B_BASE',str(Path.home()/'rest_8dna_ck21b')))
SOURCE=Path(os.environ.get('CK21B_SOURCE',str(Path.home()/'rest_x93d_ck2'/'code')))
EXP=SOURCE/'experiments'
sys.path.insert(0,str(EXP/'ecro86'))
sys.path.insert(0,str(EXP/'fnto_compiled83'))
import scene86 as S
import closure83_dr as CD
SC=json.loads((EXP/'ecro_local93d'/'addendum_checker'/'scene_ck2.json').read_text())
ORIGINAL_STATES=np.load(EXP/'ecro86'/'data'/'mats86.npz')['states']

def lambert_state(albedo):
    st=np.zeros(69,np.float32)
    st[[0,8,16]]=albedo; st[24:38]=0.3; st[38:45]=0.5; st[45]=1
    st[53:69]=1
    return st

STATES=np.concatenate([ORIGINAL_STATES,np.stack([lambert_state(.3),lambert_state(.8)])])
RECTS=S.stationary_rects()+S.box_faces(SC['geom']['A'],4)+S.box_faces(SC['geom']['B'],5)
ASSET_RECTS=[5,6,7,8,9,10,11]
VERTICES=np.array([r[0]+a*r[3]*r[1]+b*r[4]*r[2] for j in ASSET_RECTS for r in [RECTS[j]] for a in [-1,1] for b in [-1,1]])
LO,HI=VERTICES.min(0),VERTICES.max(0)
CENTER=(LO+HI)/2
SCALE=float(np.linalg.norm((HI-LO)/2)*1.01)
def norm_point(x): return (np.asarray(x)-CENTER)/SCALE

class FrozenER(mi.BSDF):
    def __init__(self,props):
        super().__init__(props)
        self.row=int(props.get('row',0)); self.checker=bool(props.get('checker',False))
        self.lambert=self.checker or self.row>=3
        self.uniform=None if self.lambert else CD.DrClosure83('ER',STATES[self.row])
        self.m_flags=mi.BSDFFlags.GlossyReflection|mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide
        if self.checker: self.m_flags |= mi.BSDFFlags.SpatiallyVarying
        self.m_components=[self.m_flags]
    def closure(self,si):
        if self.uniform is not None: return self.uniform
        mid=mi.UInt32(self.row)
        if self.checker:
            iu=mi.UInt32(dr.clamp(dr.floor(si.uv.x*3),0,2))
            iv=mi.UInt32(dr.clamp(dr.floor(si.uv.y*3),0,2))
            mid=dr.select(dr.eq((iu+iv)%2,0),mi.UInt32(6),mi.UInt32(7))
        return CD.DrClosure83('ER',STATES,mid)
    def albedo(self,si):
        if not self.checker: return mi.Float(float(STATES[self.row,0]))
        iu=mi.UInt32(dr.clamp(dr.floor(si.uv.x*3),0,2)); iv=mi.UInt32(dr.clamp(dr.floor(si.uv.y*3),0,2))
        return dr.select(dr.eq((iu+iv)%2,0),mi.Float(float(STATES[6,0])),mi.Float(float(STATES[7,0])))
    def sample(self,ctx,si,u1,u2,active=True):
        if self.lambert:
            wo=mi.warp.square_to_cosine_hemisphere(u2)
            valid=mi.Bool(active)&(si.wi.z>CD.EPS)&(wo.z>CD.EPS)&(dr.norm(si.wi+wo)>1e-6)
            pdf=dr.select(valid,wo.z/math.pi,0)
            w=mi.Color3f(self.albedo(si))
        else: wo,pdf,w,valid=self.closure(si).sample(si.wi,u1,u2,active)
        bs=dr.zeros(mi.BSDFSample3f); bs.wo=wo; bs.pdf=pdf; bs.eta=1
        bs.sampled_type=mi.UInt32(+mi.BSDFFlags.GlossyReflection); bs.sampled_component=mi.UInt32(0)
        return bs,mi.Color3f(dr.select(valid,w,0))
    def eval(self,ctx,si,wo,active=True):
        if self.lambert:
            valid=mi.Bool(active)&(si.wi.z>CD.EPS)&(wo.z>CD.EPS)&(dr.norm(si.wi+wo)>1e-6)
            return mi.Color3f(dr.select(valid,self.albedo(si)*wo.z/math.pi,0))
        return mi.Color3f(self.closure(si).eval(si.wi,wo,active))
    def pdf(self,ctx,si,wo,active=True):
        if self.lambert:
            valid=mi.Bool(active)&(si.wi.z>CD.EPS)&(wo.z>CD.EPS)&(dr.norm(si.wi+wo)>1e-6)
            return dr.select(valid,wo.z/math.pi,0)
        return self.closure(si).pdf(si.wi,wo,active)
    def eval_pdf(self,ctx,si,wo,active=True):
        if self.lambert: return self.eval(ctx,si,wo,active),self.pdf(ctx,si,wo,active)
        f,p=self.closure(si).eval_pdf(si.wi,wo,active); return mi.Color3f(f),p
    def to_string(self): return 'CK21B frozen WL86 ER'

mi.register_bsdf('ck21b_er',lambda p:FrozenER(p))
_BSDFS={}

def rect(j):
    c,a,b,h1,h2,nu,nv,mid=RECTS[j]
    mat=S.rect_matrix(norm_point(c),a,b,h1/SCALE,h2/SCALE)
    key=(int(mid),6<=j<12)
    if key not in _BSDFS: _BSDFS[key]=mi.load_dict({'type':'ck21b_er','row':key[0],'checker':key[1]})
    return mi.load_dict({'type':'rectangle','to_world':mi.ScalarTransform4f(mat),'bsdf':_BSDFS[key]})

def make_scene(res=512,training=False,max_depth=7):
    shapes={j:rect(j) for j in range(18)}
    group={'type':'shapegroup',**{f'r{j}':shapes[j] for j in ASSET_RECTS}}
    cam=SC['camera']
    d={'type':'scene','camera':{'type':'perspective','fov':cam['fov_deg'],'to_world':mi.ScalarTransform4f.look_at(origin=norm_point(cam['origin']).tolist(),target=norm_point(cam['target']).tolist(),up=[0,0,1]),'film':{'type':'hdrfilm','width':res,'height':res,'rfilter':{'type':'box'}}},'group0':group,'instance0':{'type':'instance','shapegroup':{'type':'ref','id':'group0'}},'integrator':{'type':'prb','max_depth':-1 if training else max_depth,'rr_depth':5 if training else 64}}
    if training:
        d['background']={'type':'constant','radiance':1.0}
    else:
        d.update({f'r{j}':shapes[j] for j in range(18) if j not in ASSET_RECTS})
        d['sphere']={'type':'sphere','center':norm_point(S.FX['sphere']['center']).tolist(),'radius':S.FX['sphere']['radius']/SCALE,'bsdf':{'type':'ck21b_er','row':2}}
        for i,light in enumerate(SC['lights']):
            d[f'light{i}']={'type':'point','position':norm_point(light['pos']).tolist(),'intensity':{'type':'rgb','value':(np.asarray(light['rgb'])/SCALE**2).tolist()}}
    return d,shapes

def training_scene(res,*args,**kwargs): return make_scene(res,training=True)[0]

def cpu_contract():
    def disjoint(a,b): return bool(np.any(a.max(0)<b[0]-1e-6)|np.any(a.min(0)>b[1]+1e-6))
    outside={}
    for j in range(18):
        if j in ASSET_RECTS: continue
        r=RECTS[j]; corners=np.array([r[0]+a*r[3]*r[1]+b*r[4]*r[2] for a in [-1,1] for b in [-1,1]])
        outside[f'r{j}']=disjoint(corners,(LO,HI))
    sc=np.array(S.FX['sphere']['center']); sr=S.FX['sphere']['radius']
    outside['sphere']=bool(np.any(sc+sr<LO)|np.any(sc-sr>HI))
    for i,l in enumerate(SC['lights']): outside[f'light{i}']=bool(np.any(np.array(l['pos'])<LO)|np.any(np.array(l['pos'])>HI))
    return dict(asset_aabb=[LO.tolist(),HI.tolist()],normalization_center=CENTER.tolist(),normalization_scale=SCALE,normalized_max_radius=float(np.linalg.norm(norm_point(VERTICES),axis=1).max()),external_objects_aabb_disjoint=outside,all_external_disjoint=all(outside.values()),scene=SC)
