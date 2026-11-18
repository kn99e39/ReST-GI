"""Read-only instrumentation of event-2 camera queries and original physical suffixes."""
from collections import namedtuple
from contextlib import contextmanager
import inspect
import math
import textwrap
import types

import numpy as np

Query = namedtuple("Query", "point ray hit weight pid bn pix state inc")
LUMA = (0.2126, 0.7152, 0.0722)


def make_observed(trace):
    """Add one observation at the active HIGH gather, preserving source statements."""
    source = textwrap.dedent(inspect.getsource(trace))
    before = "        if mode=='ecro' and k==2:\n"
    assert source.count(before) == 1
    line = "            self.observe_high(si,ray,active,thr,pid,bn,rng)\n"
    scope = {}
    exec(compile(source.replace(before, before + line), "wl101f_observed_pt100", "exec"),
         trace.__globals__, scope)
    return scope["trace"]


class QueryReadout:
    def __init__(self, R, rt, camera):
        self.R, self.rt, self.cam = R, rt, camera
        self.dr, self.mi = R.dr, R.mi
        self.scene = rt.scene
        self.res = camera["res"]
        self.n = self.res ** 2
        self.slots = rt.N * 64
        self.original_trace = rt.trace
        self.observed = types.MethodType(make_observed(R.pt100.Ctx.trace), rt)
        self.zero = R.PT.fl(np.zeros((rt.N,64,3), dtype=np.float32))
        self.ones = R.PT.fl(np.ones((rt.N,64,3), dtype=np.float32))
        geom = rt.inc.G
        self.geom = geom
        packed = np.zeros((rt.N,11), dtype=np.float32)
        packed[:,:3] = geom.PO
        for sl, edge in ((slice(3,6),geom.PE1),(slice(6,9),geom.PE2)):
            packed[:,sl] = edge / np.maximum(1e-20, np.square(edge).sum(1)[:,None])
        packed[:,9:] = geom.PSPH
        self.coord = R.F(packed.ravel())

    @contextmanager
    def observe(self, pixel):
        dr, mi, rt = self.dr, self.mi, self.rt
        recorded = []
        def capture(hit, ray, active, throughput, pid, bn, rng):
            idx = dr.compress(active & (dr.max(throughput) > 0))
            grab = lambda x: dr.gather(type(x), x, idx)
            recorded.append((idx, Query(grab(hit.p),grab(ray),grab(hit),grab(throughput),
                                        grab(pid),grab(bn),grab(pixel),grab(mi.UInt64(rng.state)),
                                        grab(mi.UInt64(rng.inc)))))
        rt.observe_high = capture
        rt.trace = self.observed
        try:
            yield recorded
        finally:
            rt.trace = self.original_trace
            del rt.observe_high

    def rays(self, spp, seed, launch, pixels=None, res=None):
        R, dr, mi = self.R,self.dr,self.mi
        res = self.res if res is None else res
        n = res ** 2 if pixels is None else dr.width(pixels)
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        rng = R.pcg_046(n*spp,sd,mi.UInt64(0))
        if pixels is None:
            ray, pix = R.PT.gen_rays(self.cam,res,spp,rng)
        else:
            o,fwd,right,up,s = R.PT.cam_basis(self.cam)
            pix = dr.gather(R.U,pixels,dr.arange(R.U,n*spp)%n)
            jx,jy = rng.next_float32(),rng.next_float32()
            x = (R.F(pix%res)+jx)*(2./res)-1.
            y = 1.-(R.F(pix//res)+jy)*(2./res)
            direction = dr.normalize(mi.Vector3f(*[
                float(fwd[k])+s*x*float(right[k])+s*y*float(up[k]) for k in range(3)]))
            ray = mi.Ray3f(mi.Point3f(*o.tolist()),direction)
        return ray,pix,rng,sd

    def prefix(self, ray, pix, rng, sd, table=None):
        dr,rt = self.dr,self.rt
        with self.observe(pix) as capture:
            out,_ = rt.trace(ray,dr.width(pix),rng,sd,"ER",2,rt.lt,"ecro",
                             B=self.zero if table is None else table)
            [(idx,q)] = capture
            dr.eval(out, *q)
        return out,idx,q

    def suffix(self, q, sd, proxy, independent_seed=None):
        R, dr, mi, rt = self.R,self.dr,self.mi,self.rt
        n=dr.width(q.pix)
        if independent_seed is None:
            rng=mi.PCG32(size=n)
            rng.state,rng.inc=q.state,q.inc
        else:
            sd=dr.opaque(mi.UInt64,independent_seed)
            rng=R.pcg_046(n,sd,mi.UInt64(0))
        try:
            rt.scene=proxy(rt.scene,q.hit)
            out,count=rt.trace(q.ray,n,rng,sd,"ER",5,rt.lt,"probe",probe_y=q.point,
                               probe_ok=mi.Bool(True))
            dr.eval(out,count)
            assert int(dr.count(count)[0]) == n
        finally:
            rt.scene=self.scene
        return out

    def gather(self, table, indices, q):
        dr,R=self.dr,self.R
        return [q.weight[c]*dr.gather(R.F,table,indices*3+c) for c in range(3)]

    def coordinates(self, point, w, pid, bn):
        R,dr,mi=self.R,self.dr,self.mi
        g=lambda j:dr.gather(R.F,self.coord,pid*11+j)
        displacement=point-mi.Point3f(g(0),g(1),g(2))
        x=dr.dot(displacement,mi.Vector3f(g(3),g(4),g(5)))
        y=dr.dot(displacement,mi.Vector3f(g(6),g(7),g(8)))
        q=(point-mi.Point3f(*self.geom.SC.tolist()))/self.geom.SR
        sphere=pid>=self.geom.SBASE
        a=dr.select(sphere,(dr.atan2(q.y,q.x)+math.pi)*(R.S.SPH_NPHI/(2*math.pi))-g(9),x)
        b=dr.select(sphere,(q.z+1)*(R.S.SPH_NZ/2)-g(10),y)
        tx,ty,tz=self.rt.patch_frame(pid)
        z=dr.dot(w,tz)
        c=dr.select(sphere & (bn<16),(z+1)/1.25,z*4-R.F(bn//16))
        d=(dr.atan2(dr.dot(w,ty),dr.dot(w,tx))+math.pi)*(16/(2*math.pi))-R.F(bn%16)
        return a,b,c,d

    def addresses(self, q, tolerance):
        dr,R=self.dr,self.R
        coords=self.coordinates(q.point,-q.ray.d,q.pid,q.bn)
        finite=dr.isfinite(coords[0])
        error=R.F(0)
        for x in coords:
            finite &= dr.isfinite(x)
            error=dr.maximum(error,dr.maximum(-x,x-1))
        dr.eval(finite,error)
        maximum=float(dr.max(error)[0])
        assert dr.all(finite)[0] and maximum <= tolerance, maximum
        a,b,c,d=coords
        s=R.U(a>=.5)+2*R.U(b>=.5)
        t=R.U(c>=.5)+2*R.U(d>=.5)
        slot=q.pid*64+q.bn
        return slot,s,t,maximum

    def smoke(self, seed, proxy, gates):
        R,dr,rt=self.R,self.dr,self.rt
        ray,pix,rng,sd=self.rays(1,seed,0,res=64)
        prefix,idx,q=self.prefix(ray,pix,rng,sd,self.ones)
        full_ray,_,full_rng,_=self.rays(1,seed,0,res=64)
        full,_=rt.trace(full_ray,dr.width(pix),full_rng,sd,"ER",6,rt.lt,"cam")
        suffix=self.suffix(q,sd,proxy)
        errors={}
        for label in ("e1","e2","h"):
            predicted,reference=[],[]
            for c in range(3):
                if label=="h":
                    p=dr.zeros(R.F,dr.width(pix))
                    dr.scatter(p,q.weight[c]*(suffix[f"f2{c}"]+suffix[f"fh{c}"]),idx)
                else:
                    p=prefix[f"{label}{c}"]
                predicted.append(np.asarray(p))
                reference.append(np.asarray(full[f"{label}{c}"]))
            pred,ref=np.asarray(predicted,dtype=np.float64),np.asarray(reference,dtype=np.float64)
            errors[label]={"relative_l1":float(np.abs(pred-ref).sum()/np.abs(ref).sum()),
                           "max_absolute":float(np.abs(pred-ref).max())}
            assert errors[label]["relative_l1"]<=gates["smoke_event_relative_l1"], errors
            assert errors[label]["max_absolute"]<=gates["smoke_event_max_absolute"], errors
        for c in range(3):
            direct=np.asarray(dr.gather(R.F,prefix[f"hl{c}"],idx))
            np.testing.assert_array_equal(direct,np.asarray(q.weight[c]))
        _,_,_,maximum=self.addresses(q,gates["address_coordinate_tolerance"])
        a,b,c,d=self.coordinates(q.point,-q.ray.d,q.pid,q.bn)
        p,_,(tx,ty,tz)=rt.patch_point(q.pid,a,b)
        ring=(q.pid>=self.geom.SBASE)&(q.bn<16)
        z=dr.select(ring,c*1.25-1,(R.F(q.bn//16)+c)/4)
        phi=(R.F(q.bn%16)+d)*(2*math.pi/16)-math.pi
        radius=dr.sqrt(dr.maximum(1-z*z,0))
        w=tx*radius*dr.cos(phi)+ty*radius*dr.sin(phi)+tz*z
        perr=float(dr.max(dr.norm(p-q.point))[0])
        werr=float(dr.max(dr.norm(w+q.ray.d))[0])
        assert perr<=gates["address_reconstruction_metres"] and werr<=gates["address_reconstruction_direction"],(perr,werr)
        again_ray,again_pix,again_rng,again_sd=self.rays(1,seed,0,res=64)
        _,_,again=self.prefix(again_ray,again_pix,again_rng,again_sd,self.ones)
        for lhs,rhs in zip((q.pid,q.bn,q.pix,q.state,q.inc),(again.pid,again.bn,again.pix,again.state,again.inc)):
            np.testing.assert_array_equal(np.asarray(lhs),np.asarray(rhs))
        return {"active_queries":dr.width(q.pix),"event_errors":errors,"coordinate_excursion":maximum,
                "point_reconstruction_m":perr,"direction_reconstruction":werr,"deterministic_replay":True}
