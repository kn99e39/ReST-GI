"""WL78 W+UV-Q: the WL75 rank-16 W+UV operator (rank75.TransportM, identical layers, shapes, A/O and passivity normalisation) with
geometry-relative, source-identity-free conditioning. ONLY the 16 local node inputs read by the encoder (node[:, :16]) and the 6 pair inputs
read by pair_layer are replaced; every other tensor of the geometry (edges, graph, prior, B, beta, parent) is untouched.

Node features (16; all in the port's own bin frame (t, b, n) = the frame of its first triangle, i.e. the frame that defines its 16 incoming
and 64 outgoing bins):
  0-2  rest -> current displacement (d.t, d.b, d.n) / 6 mm                       (A75: the same vector in material axes)
  3    area ratio, 4 local thickness / 1 cm, 5 normal spread                      (A75 columns 12-14, bitwise)
  6-8  u1 w1: unit offset to boundary probe 1 in the port frame, times w1
  9    w1 = max(0, 1 - r1 / 20 mm)^2
  10-12 u2 w2: the same for boundary probe 2;  13 w2
  14   max(0, 1 - r1 / 5 mm)^2,  15 max(0, 1 - r2 / 5 mm)^2
  boundary probe 1 = nearest port (current positions) whose area-averaged normal is perpendicular to the port's (|n_j.n_i| < 0.5);
  boundary probe 2 = nearest such port that is also perpendicular to probe 1's normal (a second, non-parallel boundary plane).
  Removed: absolute rest / current position (6), normal in material axes (3), ordinal face id (1).
Pair features (6): rest offset x0_j - x0_i in the SOURCE port frame / 1 cm (A75: material axes) + A75 columns 3-5 bitwise (|dr|, current
distance / 1 cm, n_i.n_j).
Consequences: no feature depends on the coordinate origin or on material-axis orientation; ports whose boundaries are all >= 20 mm away and
whose local deformation is equal have identical inputs; a frame-consistent isometry of the local context (exact slab operation, in-face
translation) maps the inputs by the same rotation/reflection that maps the port's bins."""
import os,json
import torch
import torch.nn.functional as F
import rank75 as RK

RC,RS,PERP=0.02,0.005,0.5
_POSES=None
def _register(RG):
    global _POSES
    if _POSES is None:
        import fit72,sym72 as SY
        RG.POSES.setdefault('pt0',tuple(fit72.PT0));assert tuple(RG.POSES['pt0'])==tuple(fit72.PT0),'pt0 pose'
        for op in SY.ops(False):
            pose=SY.pose_image(fit72.PT0,op);RG.POSES.setdefault(fit72.variant_name(pose),tuple(pose))
        _POSES=True

def frames(RG,g):
    """Bin frame (t, b, n) of every port of geometry g (first-triangle frame, as in the bin definition)."""
    import probes68 as P68
    _register(RG);dom=P68.domain(RG,g['name']);assert dom.P==g['P'];tri=dom.port_tris[:,0]
    t,b,n=dom.t[tri].float(),dom.b[tri].float(),dom.n[tri].float()
    nn=F.normalize(g['node'][:,9:12].float(),dim=1);assert float((n-nn).norm(dim=1).max())<0.2,'frame vs geometry normal'
    return t,b,n

def features(RG,g):
    """(node16 (P,16), pair6 (E,6), info) from the geometry alone."""
    t,b,n=frames(RG,g);x=g['xyz'].float();x0=g['rest_xyz'].float();P=g['P'];nn=F.normalize(g['node'][:,9:12].float(),dim=1)
    fr=lambda v,i=None:torch.stack([(v*t).sum(-1),(v*b).sum(-1),(v*n).sum(-1)],-1) if i is None else torch.stack([(v*t[i]).sum(-1),(v*b[i]).sum(-1),(v*n[i]).sum(-1)],-1)
    D=torch.cdist(x.double(),x.double());perp=(nn@nn.T).abs()<PERP;inf=torch.tensor(float('inf'),dtype=torch.float64,device=x.device)
    D1=torch.where(perp,D,inf);r1,j1=D1.min(1)
    perp2=perp&((nn[j1]@nn.T).abs()<PERP);D2=torch.where(perp2,D,inf);r2,j2=D2.min(1)
    def probe(j,r):
        ok=torch.isfinite(r);rr=torch.where(ok,r,torch.ones_like(r)).float();o=(x[j]-x)/rr[:,None]
        w=(1-rr/RC).clamp(min=0)**2*ok;ws=(1-rr/RS).clamp(min=0)**2*ok;return fr(o)*w[:,None],w,ws
    u1,w1,ws1=probe(j1,r1);u2,w2,ws2=probe(j2,r2)
    node=torch.cat([fr(x-x0)/.006,g['node'][:,12:15].float(),u1,w1[:,None],u2,w2[:,None],ws1[:,None],ws2[:,None]],1);assert node.shape==(P,16)
    src,dst=g['src'],g['dst'];pair=torch.cat([fr(x0[dst]-x0[src],src)/.01,g['pair'][:,3:6].float()],1)
    info=dict(no_probe1=int((~torch.isfinite(r1)).sum()),no_probe2=int((~torch.isfinite(r2)).sum()),
              ties1=int(((D1-r1[:,None]).abs()<1e-9).sum(1).gt(1).sum()),ties2=int(((D2-r2[:,None]).abs()<1e-9).sum(1).gt(1).sum()))
    return node.contiguous(),pair.contiguous(),info

_CACHE={}
def QG(RG,g):
    """Geometry view with the WL78 inputs (node[:, :16] and pair replaced); cached per geometry object; idempotent."""
    if g.get('_q78'):return g
    key=id(g)
    if key in _CACHE and _CACHE[key][0] is g:return _CACHE[key][1]
    node16,pair,info=features(RG,g);node=g['node'].clone();node[:,:16]=node16
    q=dict(g,node=node,pair=pair,_q78=True,_q78_info=info);_CACHE[key]=(g,q);return q

class TransportQ(RK.TransportM):
    """rank75.TransportM with every geometry routed through QG (state, coefficients, column sums, apply see the same tensors)."""
    def __init__(self,M,RG):super().__init__(M);self._RG=RG
    def state(self,g):return super().state(QG(self._RG,g))
    def coefficients(self,g,state=None):return super().coefficients(QG(self._RG,g),state)
    def apply(self,g,x,coeff):return super().apply(QG(self._RG,g),x,coeff)
    def column_sums(self,g,coeff):return super().column_sums(QG(self._RG,g),coeff)

def columns(m,g,ports):return RK.columns(m,QG(m._RG,g),ports)

def init_model(RG,init,seed,M=16):
    """A75 initialisation (rank75.init_model: P_590x trunk, rank rows r mod 8, frozen G2 A/O) except the two input layers that read the
    replaced features (encoder Linear(16,32), pair_layer Linear(6,h)), which are re-initialised with the PyTorch default under `seed`."""
    a,meta=RK.init_model(init,M);m=TransportQ(M,RG).cuda();m.load_state_dict(a.state_dict());del a
    g=torch.Generator(device='cpu').manual_seed(seed);torch.manual_seed(seed)
    enc=torch.nn.Linear(16,32);pl=torch.nn.Linear(m.pair_layer.in_features,m.pair_layer.out_features,bias=False)
    with torch.no_grad():
        m.encoder.weight.copy_(enc.weight);m.encoder.bias.copy_(enc.bias);m.pair_layer.weight.copy_(pl.weight)
    meta=dict(meta,reinitialised=['encoder.weight','encoder.bias','pair_layer.weight'],reinit_seed=seed);return m,meta

def load_run(RG,path):
    x=torch.load(path,map_location='cpu',weights_only=False);m=TransportQ(x['model']['in_logits'].shape[0],RG).cuda();m.load_state_dict(x['model']);m.eval();return m
