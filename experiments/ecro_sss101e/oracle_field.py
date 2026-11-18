"""Independent exposed-domain E2–E5 field, using only the original suffix tracer."""
import math

import numpy as np


class FirstSurface:
    """Inject the geometry-only native attachment once. Delegate all later rays."""
    def __init__(self, scene, hit):
        self.full_scene, self.hit = scene, hit

    def ray_intersect(self, *args, **kwargs):
        if self.hit is not None:
            hit, self.hit = self.hit, None
            return hit
        return self.full_scene.ray_intersect(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.full_scene, name)


def clip_polygon(poly, axis, value, sign):
    cut=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        da,db=sign*(a[axis]-value),sign*(b[axis]-value)
        if (da>=0)!=(db>=0):
            cut.append(a+(b-a)*(da/(da-db)))
        if db>=0:
            cut.append(b)
    return np.array(cut,dtype=np.float64).reshape(-1,2)


def exposed_floor_areas(geom):
    """Independent projected convex-polygon intersection with every floor cell."""
    data=geom.g["B"]
    bx,by,_=geom.BX[1].reshape(3,3)
    c,h=np.array(data["c"]),np.array(data["h"])
    corners=np.array([(c+u*h[0]*bx+v*h[1]*by)[:2] for u,v in ((-1,-1),(1,-1),(1,1),(-1,1))])
    exposed=[]
    for i in np.flatnonzero(geom.PSID==0):
        p=geom.PO[i]
        q=p+geom.PE1[i]+geom.PE2[i]
        intersection=corners
        for axis,value,sign in [(0,p[0],1),(0,q[0],-1),(1,p[1],1),(1,q[1],-1)]:
            intersection=clip_polygon(intersection,axis,value,sign)
        x,y=intersection.T
        area=abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))*0.5
        exposed.append(1.-area/geom.PAREA[i])
    return np.array(exposed)


class ExposedOracle:
    def __init__(self, R, rt):
        self.R,self.rt=R,rt
        self.dr,self.mi=R.dr,R.mi
        self.geom=rt.inc.G
        self.scene=rt.scene
        assert len(self.geom.PSID)==1270 and R.NB==64
        self.ids=R.U(self.geom.PSID.astype(np.uint32))
        self.ns=rt.N*64
        by_name={shape.id():shape for shape in self.scene.shapes()}
        self.shape_scenes=[]
        for name in R.PT.SHAPE_NAMES:
            self.shape_scenes.append(self.mi.load_dict({"type":"scene",name:by_name[name]}))
        box=self.geom.g["B"]
        assert abs(box["c"][2]-box["h"][2])<1e-12
        assert self.geom.g["A"]["c"][2]-self.geom.g["A"]["h"][2]>0
        assert self.geom.SC[2]==self.geom.SR
        self.box_center=np.array(box["c"],dtype=np.float64)
        self.box_half=np.array(box["h"],dtype=np.float64)
        self.floor_fraction=exposed_floor_areas(self.geom)
        assert np.all(self.floor_fraction>=-1e-10)
        assert abs(np.sum((1-self.floor_fraction)*self.geom.PAREA[self.geom.PSID==0])-
                   4*self.box_half[0]*self.box_half[1])<1e-12

    def probes(self, spp, seed, launch, diagnostic=None):
        dr,mi,R,rt=self.dr,self.mi,self.R,self.rt
        n=self.ns*spp
        sd=dr.opaque(mi.UInt64,seed*1000003+launch)
        rng=mi.PCG32(size=n,initstate=sd,initseq=mi.UInt64(0))
        slots=dr.arange(R.U,n)%self.ns
        pid,bn=slots//64,slots%64
        sid=dr.gather(R.U,self.ids,pid)
        u0,u1,u2,u3=(rng.next_float32() for _ in range(4))
        p,normal,(tx,ty,tz)=rt.patch_point(pid,u0,u1)
        local=rt.dir_in_bin(bn,u2,u3)
        z=1.25*u2-1.
        phase=(R.F(bn%16)+u3)*(2*math.pi/16)-math.pi
        rr=dr.sqrt(dr.maximum(1-z*z,0.))
        local=dr.select((sid==6)&(bn<16),mi.Vector3f(rr*dr.cos(phase),rr*dr.sin(phase),z),local)
        w=tx*local.x+ty*local.y+tz*local.z
        bx,by,_=self.geom.BX[1].reshape(3,3)
        rp=p-mi.Point3f(*self.box_center.tolist())
        u=dr.dot(rp,mi.Vector3f(*bx.tolist()))
        v=dr.dot(rp,mi.Vector3f(*by.tolist()))
        covered=(sid==0)&(dr.abs(u)<self.box_half[0])&(dr.abs(v)<self.box_half[1])
        underside=(sid==8)&(normal.z<-.99)
        tangent=(sid==6)&(p.z<=0)
        buried=covered|underside|tangent
        back=~buried&(dr.dot(w,normal)<=1e-6)
        candidate=~buried&~back

        attach=mi.Ray3f(p+normal*0.002,-normal)
        attach.maxt=0.003
        hits=dr.zeros(mi.SurfaceInteraction3f,n)
        hits.t=R.F(float("inf"))
        for i,scene in enumerate(self.shape_scenes):
            select=candidate&(sid==i)
            attached=scene.ray_intersect(attach,select)
            dr.eval(attached)
            hits=dr.select(select,attached,hits)
            dr.eval(hits)
        shape_match=rt.shape_id(hits)==sid
        point_delta=dr.norm(hits.p-p)
        normal_match=dr.dot(hits.n,normal)>=0.999999
        attached=hits.is_valid()&shape_match&(point_delta<=4e-6)&normal_match
        actual_pid,actual_bin=rt.patch_bin(hits,sid,w)
        address_match=(actual_pid==pid)&(actual_bin==bn)
        native_front=hits.to_local(w).z>1e-6
        near_tie=(dr.minimum(dr.minimum(dr.minimum(u0,1-u0),dr.minimum(u1,1-u1)),
                    dr.minimum(dr.minimum(u2,1-u2),dr.minimum(u3,1-u3)))<=8e-6)
        near_tie |= dr.abs(dr.dot(w,normal)-1e-6)<=8e-6
        # A finite-precision planar sample that rounds onto its closed outline
        # can be missed by triangle traversal; this is a numerical edge tie.
        area_edge=dr.minimum(dr.minimum(u0,1-u0),dr.minimum(u1,1-u1))
        rounded_miss=~hits.is_valid()&(sid!=6)&(area_edge<=4*np.finfo(np.float32).eps)
        tie=candidate&((attached&(~address_match|~native_front)&near_tie)|rounded_miss)
        invalid=candidate&~tie&(~attached|~address_match|~native_front)
        valid=candidate&attached&address_match&native_front
        state={"valid":valid,"buried":buried,"back":back,"tie":tie,"invalid":invalid,
               "negative_ring_zero":valid&(sid==6)&(local.z<0),"point_error":dr.select(valid,point_delta,0.)}
        if diagnostic is not None:
            bad_indices=dr.compress(invalid)
            dr.eval(bad_indices)
            if dr.width(bad_indices)>0:
                values={"slot":slots,"expected_pid":pid,"expected_sid":sid,"expected_bin":bn,
                        "actual_pid":actual_pid,"actual_bin":actual_bin,"actual_sid":rt.shape_id(hits),
                        "point_error_m":point_delta,"normal_dot":dr.dot(hits.n,normal),
                        "native_valid":hits.is_valid(),"native_front":native_front,
                        "analytic_front_cos":dr.dot(w,normal),"is_near_tie":near_tie,
                        "u0":u0,"u1":u1,"u2":u2,"u3":u3,
                        "requested_x":p.x,"requested_y":p.y,"requested_z":p.z,
                        "attached_x":hits.p.x,"attached_y":hits.p.y,"attached_z":hits.p.z}
                diagnostic.append({"launch":launch,**{k:np.asarray(dr.gather(type(v),v,bad_indices)).tolist()
                                                     for k,v in values.items()}})
        return slots,mi.Ray3f(p+w*0.002,-w),hits,rng,sd,state

    def trace(self,ray,hits,rng,sd,valid,mode):
        original=self.rt.scene
        try:
            self.rt.scene=FirstSurface(original,hits)
            args={"probe_ok":valid}
            if mode=="probe":
                args["probe_y"]=hits.p
            return self.rt.trace(ray,self.dr.width(ray),rng,sd,"ER",5,self.rt.lt,mode,**args)
        finally:
            self.rt.scene=original

    def accumulate(self,spp,seed,batch,callback=None):
        dr,R=self.dr,self.R
        accum={}
        max_distance=0.
        for launch,done in enumerate(range(0,spp,batch)):
            local_spp=min(batch,spp-done)
            slots,ray,hit,rng,sd,info=self.probes(local_spp,seed,launch)
            output,count=self.trace(ray,hit,rng,sd,info["valid"],"probe")
            assert not bool(dr.any(count!=info["valid"])[0])
            max_distance=max(max_distance,float(dr.max(info.pop("point_error"))[0]))
            for key,mask in info.items():
                output[key]=dr.select(mask,R.F(1),R.F(0))
            for i in range(3):
                a,b=output[f"f2{i}"],output[f"fh{i}"]
                square=dr.square(a+b)
                output[f"square{i}"]=square
            sums=self.rt.accum(output,slots,self.ns)
            for key,val in sums.items():
                accum[key]=accum.get(key,0.)+val
            if callback is not None:
                callback(launch+1,done+local_spp)
        count=accum["valid"]
        counts={k:accum[k].reshape(self.rt.N,64).astype(np.uint32) for k in
                ("valid","buried","back","tie","invalid","negative_ring_zero")}
        assert np.all(sum(counts[k] for k in ("valid","buried","back","tie","invalid"))==spp)
        assert int(counts["invalid"].sum())==0, {k:int(v.sum()) for k,v in counts.items()}
        assert counts["tie"].sum()<=5e-5*(spp*self.ns-counts["buried"].sum()-counts["back"].sum())
        def channels(label):
            return np.stack([accum[f"{label}{i}"] for i in range(3)],axis=1)
        f2,fh=channels("f2"),channels("fh")
        normalize=np.maximum(count,1)[:,None]
        mean=(f2+fh)/normalize
        variance=np.maximum(0.,channels("square")-(f2+fh)**2/normalize)/np.maximum(count-1,1)[:,None]
        assert np.isfinite(mean).all() and np.min(mean)>=0 and np.isfinite(variance).all()
        holder={"proposed":np.full((self.rt.N,64),spp,np.uint32),
                "B":mean.reshape(1270,64,3).astype(np.float32),
                "f2":(f2/normalize).reshape(1270,64,3).astype(np.float32),
                "fh":(fh/normalize).reshape(1270,64,3).astype(np.float32),
                "sample_variance":variance.reshape(1270,64,3).astype(np.float32),
                "max_attachment_error_m":np.array(max_distance),**counts}
        return holder

    def floor_check(self,packet):
        floor=self.geom.PSID==0
        total=packet["proposed"][floor].sum(1,dtype=np.float64)
        measured=total-packet["buried"][floor].sum(1,dtype=np.float64)
        expected=total*self.floor_fraction
        scale=np.sqrt(np.maximum(total*self.floor_fraction*(1-self.floor_fraction),0.))
        error=np.abs(measured-expected)
        assert np.all(error<=6*scale+1)
        return {
            "partially_covered_patches":int(np.count_nonzero((self.floor_fraction<1-1e-8)&(self.floor_fraction>1e-8))),
            "max_standardized_count_error":float(np.max(error/np.maximum(scale,1.))),
            "analytic_total_area_removed_m2":float(np.sum((1-self.floor_fraction)*self.geom.PAREA[floor])),
        }


def suffix_controls(oracle, seed):
    R,dr,mi=oracle.R,oracle.dr,oracle.mi
    _,raw,hit,_,sd,info=oracle.probes(1,seed,0)
    idx=dr.arange(R.U,2048)*39
    ray=dr.gather(mi.Ray3f,raw,idx)
    pinned=dr.gather(mi.SurfaceInteraction3f,hit,idx)
    ok=dr.gather(mi.Bool,info["valid"],idx)
    native=oracle.scene.ray_intersect(ray,ok)
    same_surface=(native.is_valid()&(oracle.rt.shape_id(native)==oracle.rt.shape_id(pinned))&
                  (dr.norm(native.p-pinned.p)<4e-6)&(native.to_local(-ray.d).z>1e-6))
    ok &= same_surface
    n=int(dr.count(ok)[0])
    assert n>800
    def rng():
        return mi.PCG32(size=2048,initstate=mi.UInt64(seed*1000003+13),initseq=mi.UInt64(0))
    probe,_=oracle.trace(ray,pinned,rng(),sd,ok,"probe")
    pinned_cam,_=oracle.trace(ray,pinned,rng(),sd,ok,"cam")
    free_cam,_=oracle.rt.trace(ray,2048,rng(),sd,"ER",5,oracle.rt.lt,"cam",probe_ok=ok)
    max_err=0.
    err,energy=0.,0.
    for base,other in (("f2","e2"),("fh","h")):
        for c in range(3):
            a,b,d=probe[f"{base}{c}"],pinned_cam[f"{other}{c}"],free_cam[f"{other}{c}"]
            max_err=max(max_err,float(dr.max(dr.abs(a-b))[0]))
            err+=float(dr.sum(dr.abs(a-d))[0])
            energy+=float(dr.sum(dr.abs(d))[0])
    assert max_err<=1e-6 and err/energy<=1e-3, (max_err,err/energy)
    return {"paired_native_rays":n,"event_identity_max_abs":max_err,
            "unpinned_suffix_relative_l1":err/energy}
