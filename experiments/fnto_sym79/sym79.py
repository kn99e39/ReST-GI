"""WL79 exact physical symmetry contract of the slab fixture (Stage 0).

Group. The homogeneous slab (100 x 100 x 10 mm, origin at its centre) is invariant under D4h = D4 (in-plane rotations / reflections about
x = 0, y = 0 and the diagonals) x {identity, z-flip}: 16 operations (sym72.ops(True)). A dented pose (dent centre c) is mapped to the pose with
centre op(c) (sym72.pose_image; z-flip excluded because the dent is on the top face only). Self-symmetries of a dented pose are the operations
with op(c) = c (pt0, c = (5, -5) mm: {id, asw}; te0, c = 0: all of D4).

Index convention (all maps computed from geometry only):
  perm[p]   = the port at op(x0_p), rest material coordinates (sym72.permutation: exact bijection or error)
  Q[p]      = F_dst[perm p] . M . F_src[p]^T (3 x 3): components in the bin frame (t, b, n) of port p in the source geometry -> components in
              the bin frame of the image port in the image geometry. F = rows (t, b, n) of the port's first triangle (the bin definition).
  pin[p,a]  = the N = 4 (incoming) bin of the image port that contains Q[p] applied to the centre direction of bin a of port p
  pout[j,o] = the N = 8 (outgoing) bin of the image receiver, likewise
  p16[j,f]  = the N = 4 map at receiver j (fine16 outgoing bins and incoming bins share the 4 x 4 layout)
Transformed response of source perm[p] in the image geometry:
  S'[pin[p,a], perm[j], pout[j,o], c] = S[a, j, o, c]        i.e. K(gG) = P_out(g) K(G) P_in(g)^-1
with P_out acting jointly on (receiver port, outgoing bin) and P_in on the source's incoming bins. The model budget transforms as
d'[pin[p,a]] = d[a]. A map is EXACT when the mapped bin centres coincide with the target bin centres (residual <= 1e-9) and every per-port map is
a bijection; approximate maps are never used to build labels (exact() asserts)."""
import numpy as np
import torch
import sym72 as SY

NIN,NOUT=4,8;TOL=1e-9

def centres(G,N):
    k=torch.arange(N*N,dtype=torch.float64);u=((k%N)+.5)/N;v=((k//N)+.5)/N;x,y=G.square_to_disk(u,v)
    return torch.stack([x,y,torch.sqrt((1-x*x-y*y).clamp(min=0))],1)

_F={}
def frames(RG,name):
    """(P, 3, 3) float64 rows (t, b, n) of every port's first-triangle frame (the bin frame) in geometry `name`."""
    if name not in _F:
        import probes68 as P68,q78
        q78._register(RG);dom=P68.domain(RG,name);tri=dom.port_tris[:,0]
        _F[name]=torch.stack([dom.t[tri],dom.b[tri],dom.n[tri]],1).double().cpu()
    return _F[name]

def pose_of(RG,name):
    import q78
    q78._register(RG);return tuple(RG.POSES[name])

def image_name(RG,name,op):
    """Name of the geometry op(G) among the registered poses (rest -> rest; pt0 -> pt0 or a pt0v variant; te0 -> te0)."""
    pose=pose_of(RG,name)
    if pose[0]==0:return name
    img=SY.pose_image(pose,op)
    for k,v in RG.POSES.items():
        if tuple(v)==tuple(img) and (k==name or k.startswith(name.split('v')[0])):return k
    raise KeyError(('no registered image pose',name,op,img))

class Map:
    """Exact symmetry map (perm, Q, pin, pout, p16) from geometry src to geometry dst under op; plus exactness diagnostics."""
    def __init__(self,RG,src,op,rest_xyz,dst=None,delta=None,post=None):
        """delta (3,) metres: op followed by a translation (NOT a symmetry; diagnostic correspondences only, partial port map, perm = -1
        where op(x) + delta is not a port); post (3, 3): an extra map applied in the image frame (negative controls only)."""
        self.src=src;self.op=tuple(op);self.dst=dst or image_name(RG,src,op);self.M=torch.as_tensor(SY.matrix(op),dtype=torch.float64)
        x=np.asarray(rest_xyz,dtype=np.float64);self.partial=delta is not None or post is not None
        if delta is None:self.perm=torch.as_tensor(SY.permutation(x,op),dtype=torch.long)
        else:
            from scipy.spatial import cKDTree
            dd,ii=cKDTree(x).query(x@SY.matrix(op).T+np.asarray(delta));self.perm=torch.as_tensor(np.where(dd<1e-6,ii,-1),dtype=torch.long)
        P=len(self.perm);pc=self.perm.clamp(min=0)
        Fs=frames(RG,src);Fd=frames(RG,self.dst);self.Q=Fd[pc]@self.M@Fs.transpose(1,2)
        if post is not None:self.Q=torch.as_tensor(post,dtype=torch.float64)@self.Q
        self.maps={};self.residual={};self.bijective={}
        mk=self.perm>=0      # partial (diagnostic) maps: exactness is assessed on the mapped ports only
        for N in (NIN,NOUT):
            L=centres(RG.G,N);w=torch.einsum('pij,kj->pki',self.Q,L);b=RG.G.local_to_bin(w,N).long()
            self.maps[N]=b;self.residual[N]=float((w-L[b]).norm(dim=-1)[mk].max())
            self.bijective[N]=bool((torch.sort(b,1).values==torch.arange(N*N)[None]).all(1)[mk].all())
        self.pin=self.maps[NIN];self.pout=self.maps[NOUT];self.p16=self.maps[NIN]
        # frame-map structure: normal preserved, tangent block a signed permutation (a D4 element of the bin square)
        self.normal_err=float((self.Q[:,2]-torch.tensor([0,0,1.],dtype=torch.float64)).abs().max().item())
        tb=self.Q[:,:2,:2];self.tangent_err=float((tb-tb.round()).abs().max())
        self.exact=all(self.residual[N]<=TOL and self.bijective[N] for N in (NIN,NOUT)) and not self.partial
        if self.partial:return
        self.inv=torch.empty(P,dtype=torch.long);self.inv[self.perm]=torch.arange(P)
        tgt=(self.perm[:,None]*64+self.pout).reshape(-1);self._inv_out=torch.empty_like(tgt);self._inv_out[tgt]=torch.arange(len(tgt))
        t16=(self.perm[:,None]*16+self.p16).reshape(-1);self._inv16=torch.empty_like(t16);self._inv16[t16]=torch.arange(len(t16))

    def tangent_type(self,p):
        """The D4 element of port p's tangent map as a 2 x 2 integer matrix (rows of Q[p][:2,:2])."""
        return tuple(int(v) for v in self.Q[p,:2,:2].round().reshape(-1).tolist())

    def check(self):assert self.exact,('symmetry map not exact',self.src,self.op,self.dst,self.residual,self.bijective)

    def inv_in(self,p):
        i=torch.empty(16,dtype=torch.long);i[self.pin[p]]=torch.arange(16);return i

    def response(self,S,p):
        """S (..., 16, P, 64, C) response of source p in src -> response of source perm[p] in dst (exact gather)."""
        self.check();sh=S.shape;dev=S.device
        X=S.index_select(-4,self.inv_in(p).to(dev)).reshape(*sh[:-3],sh[-3]*sh[-2],sh[-1])
        return X.index_select(-2,self._inv_out.to(dev)).reshape(sh)

    def response_partial(self,S,p):
        """Diagnostic correspondences (translation / negative controls): mapped receivers only; returns (S', mask of mapped image receivers).
        Requires bijective per-port bin maps (asserted); never used to build a training label."""
        assert all(self.bijective.values()) and self.residual[NIN]<=TOL and self.residual[NOUT]<=TOL,'bins do not permute'
        J=torch.nonzero(self.perm>=0).flatten();Jp=self.perm[J];out=torch.zeros_like(S);mask=torch.zeros(S.shape[-3],dtype=torch.bool);mask[Jp]=True
        X=S.index_select(-4,self.inv_in(p).to(S.device))
        for a in range(16):out[...,a,Jp[:,None],self.pout[J],:]=X[...,a,J[:,None],torch.arange(64)[None],:]
        return out,mask

    def fine16(self,X,p):
        """X (..., 16, P, 16, C) fine16 response (or zero-collision B16) of source p -> image source (same maps, N = 4 receiver bins)."""
        self.check();sh=X.shape;dev=X.device
        Y=X.index_select(-4,self.inv_in(p).to(dev)).reshape(*sh[:-3],sh[-3]*sh[-2],sh[-1])
        return Y.index_select(-2,self._inv16.to(dev)).reshape(sh)

    def budget(self,d,p):
        """d (..., 16) incoming budget of port p -> of the image port (d'[pin[a]] = d[a])."""
        return d.index_select(-1,self.inv_in(p).to(d.device))

    def receiver(self,v):
        """Per-receiver tensor v (P, ...) of the source geometry -> image geometry ordering (v'[perm j] = v[j])."""
        return v.index_select(0,self.inv.to(v.device))

def commutation(mp,parent,bm):
    """Aggregation consistency: fine16 parent of every mapped outgoing bin equals the N = 4 map of its parent bin (per receiver), and the
    coarse cell partition (port -> 240 cells) is mapped onto itself (a cell permutation exists)."""
    bm=bm.cpu();par=parent.cpu();f16=(bm[mp.pout]==torch.gather(mp.p16,1,bm[None].expand(len(mp.perm),-1))).all().item()
    img=par[mp.perm];cmap={}
    ok=True
    for p in range(len(par)):
        c=int(par[p]);v=int(img[p])
        if cmap.setdefault(c,v)!=v:ok=False;break
    ok=ok and len(set(cmap.values()))==len(cmap)
    return dict(fine16_commutes=bool(f16),coarse_cells_permute=bool(ok),coarse_perm=[cmap[c] for c in sorted(cmap)] if ok else None)

def mesh_check(RG,src,dst,M):
    """Refined (level-4) current meshes: max distance of op(src vertices) to dst vertices, and the fraction of op(src triangles) that are
    dst triangles (vertex-index triples matched by position). 1.0 / 0 = the physical surfaces are identical."""
    from scipy.spatial import cKDTree
    out={}
    def fm(name):
        m,V=RG.mesh(name);f,VC,_=RG.KO.refined(m,V,4);return np.asarray(VC,np.float64),np.asarray(f['F'])
    Vs,Fs=fm(src);Vd,Fd=fm(dst);Y=Vs@np.asarray(M).T;d,i=cKDTree(Vd).query(Y)
    out['vertex_max_dist_m']=float(d.max())
    ts=np.sort(i[Fs],1);td=np.sort(Fd,1);setd={tuple(r) for r in td.tolist()}
    out['triangle_match_fraction']=float(np.mean([tuple(r) in setd for r in ts.tolist()]))
    return out
