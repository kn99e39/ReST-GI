"""Physical material adapters around preserved pt86 geometry/renderer methods.

The learned ECRO is NOT implemented here. SSS uses an index-matched homogeneous
sphere, complete entry->first-exit random walks, with ballistic light shadows.
One such excursion counts as one optical-domain interaction, not one collision.
"""
import math
import numpy as np
import common100 as C
import pt86 as PT

mi,dr,F,U=PT.mi,PT.dr,PT.F,PT.U


class Ctx(PT.Ctx):
    def __init__(self,geom,case):
        self.case=case
        super().__init__(geom)
        self.mats.states=C.materials(case)
        self.mats.cls=U(np.array([2,0,1,2,2,2,2,2],np.uint32))
        self.sigma=12.;self.albedo=mi.Color3f(.92,.80,.65)

    def set_tables(self,g):
        C.checker(g)
        super().set_tables(g)
        self.PMATU=U(g.PMAT.astype(np.uint32))

    def sphere_exit(self,p,w):
        q=p-mi.Point3f(*self.SC)
        b=dr.dot(q,w)
        return dr.maximum(-b+dr.sqrt(dr.maximum(b*b+self.SR*self.SR-dr.squared_norm(q),0.)),0.)

    def sss_walk(self,si,ray,rng,lights,active):
        """Adjoint volume walk; isotropic NEE at every collision, first exit.

        No implicit air paths are learned/owned by this material routine.
        Max possible residual RGB weight after 128 collisions < 2.4e-5.
        """
        p=si.p+ray.d*1e-6;w=mi.Vector3f(ray.d)
        beta=mi.Color3f(1.);source=mi.Color3f(0.)
        ep=mi.Point3f(si.p);ew=mi.Vector3f(ray.d);eb=mi.Color3f(0.)
        act=mi.Bool(active);depth=U(0)
        loop=mi.Loop(name='WL100_sphere_first_exit',state=lambda:(rng,p,w,beta,source,ep,ew,eb,act,depth))
        while loop(act & (depth<128)):
            end=self.sphere_exit(p,w)
            distance=-dr.log(dr.maximum(1.-rng.next_float32(),1e-12))/self.sigma
            coll=act & (distance<end)
            leave=act & ~coll
            ep=mi.Point3f(dr.select(leave,p+w*end,ep));ew=mi.Vector3f(dr.select(leave,w,ew));eb=mi.Color3f(dr.select(leave,beta,eb))
            p=mi.Point3f(p+w*dr.minimum(distance,end))
            beta=mi.Color3f(beta*self.albedo)
            for l in range(lights.n):
                dl=lights.pos[l]-p;d2=dr.squared_norm(dl);dist=dr.sqrt(d2)
                wl=dl/dr.maximum(dist,1e-12)
                de=self.sphere_exit(p,wl)
                on=coll & (dr.dot(-wl,lights.axis[l])>lights.cc[l]) & (dist>de+1e-5)
                shadow=mi.Ray3f(p+wl*(de+1e-4),wl)
                shadow.maxt=dr.maximum(dist-de-2e-4,0.)
                occ=self.scene.ray_test(shadow,on)
                source=mi.Color3f(source+dr.select(on & ~occ,beta*(1/(4*math.pi))*dr.exp(-self.sigma*de)*lights.rgb[l]/dr.maximum(d2,1e-12),0.))
            u=rng.next_float32();v=rng.next_float32()
            z=1-2*u;r=dr.sqrt(dr.maximum(1-z*z,0.));a=2*math.pi*v
            w=mi.Vector3f(r*dr.cos(a),r*dr.sin(a),z)
            act=coll;depth+=1
        return source,ep,ew,eb

    def direct(self,si,wi,cl,lights,active,not_sss):
        """Direct source + ballistic source through sphere, kept disjoint.

        Ballistic crossing is an additional domain interaction and therefore
        returned separately for event k+1, including a no-double-count rule.
        """
        direct=mi.Color3f(0.);through=mi.Color3f(0.)
        for l in range(lights.n):
            dl=lights.pos[l]-si.p;d2=dr.squared_norm(dl);dist=dr.sqrt(d2)
            wl=dl/dr.maximum(dist,1e-12);wo=si.to_local(wl)
            on=active & not_sss & (wo.z>1e-6) & (dr.dot(-wl,lights.axis[l])>lights.cc[l])
            if self.case=='sss':
                sh=self.scene.ray_intersect(si.spawn_ray_to(lights.pos[l]),on)
                clear=on & ~sh.is_valid()
                cross=on & sh.is_valid() & dr.eq(self.shape_id(sh),6)
                # Continue from first sphere hit to its far-side exit.
                de=self.sphere_exit(sh.p+wl*1e-6,wl)
                sr=mi.Ray3f(sh.p+wl*(de+1e-4),wl)
                sr.maxt=dr.maximum(dist-sh.t-de-2e-4,0.)
                clear2=cross & ~self.scene.ray_test(sr,cross)
                f=cl.eval(wi,dr.select(on,wo,mi.Vector3f(0,0,1)),on)
                direct+=dr.select(clear,f*lights.rgb[l]/dr.maximum(d2,1e-12),0.)
                through+=dr.select(clear2,f*lights.rgb[l]*dr.exp(-self.sigma*de)/dr.maximum(d2,1e-12),0.)
            else:
                clear=on & ~self.scene.ray_test(si.spawn_ray_to(lights.pos[l]),on)
                f=cl.eval(wi,dr.select(clear,wo,mi.Vector3f(0,0,1)),clear)
                direct+=dr.select(clear,f*lights.rgb[l]/dr.maximum(d2,1e-12),0.)
        return direct,through

    def trace(self,ray,N,rng,sd,ref,kmax,lights,mode,B=None,probe_y=None,probe_ok=None,plain=False):
        assert ref=='ER' and not plain
        thr=mi.Color3f(1.);active=mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
        out={};okp=None
        def add(name,color):
            for c,ch in enumerate((color.x,color.y,color.z)):
                key=f'{name}{c}';out[key]=out.get(key,F(0.))+ch
        def contribution(k,color):
            if mode=='probe':
                if k>=2:add('f2' if k==2 else 'fh',color)
            else:add('e1' if k==1 else ('e2' if k==2 else 'h'),color)
        for k in range(1,kmax+1):
            si=self.scene.ray_intersect(ray,active);active &= si.is_valid()
            sid=self.shape_id(si);pid,bn=self.patch_bin(si,sid,-ray.d)
            mid=dr.gather(U,self.PMATU,pid,active)
            wi=si.to_local(-ray.d)
            if mode=='probe' and k==1:
                active &= dr.norm(si.p-probe_y)<1e-3;okp=mi.Bool(active)
            active &= wi.z>1e-6;wi=dr.select(active,wi,mi.Vector3f(0,0,1))
            if mode=='ecro' and k==2:
                i=(pid*PT.NB+bn)*3
                hl=mi.Color3f(*[dr.gather(F,B,i+c,active) for c in range(3)])
                add('hl',dr.select(active,thr*hl,0.))
            cl=PT.CD.DrClosure83('ER',self.mats.states,mid)
            is_sss=active & dr.eq(mid,2) if self.case=='sss' else mi.Bool(False)
            if self.case=='sss':
                qs,ep,ew,eb=self.sss_walk(si,ray,rng,lights,is_sss)
            else:qs=mi.Color3f(0.)
            d,ballistic=self.direct(si,wi,cl,lights,active,~is_sss)
            if not(mode=='probe' and k==1):contribution(k,thr*(d+qs))
            # No NEE at an unmodelled source vertex beyond the fixed depth.
            if k<kmax:contribution(k+1,thr*ballistic)
            if k==kmax:break
            wo,pdf,weight,valid=cl.sample(wi,rng.next_float32(),mi.Point2f(rng.next_float32(),rng.next_float32()),active & ~is_sss)
            newray=si.spawn_ray(si.to_world(wo))
            if self.case=='sss':
                newray.o=dr.select(is_sss,ep+ew*1e-4,newray.o)
                newray.d=dr.select(is_sss,ew,newray.d)
                weight=dr.select(is_sss,eb,weight)
                valid=dr.select(is_sss,dr.max(eb)>0.,valid)
            thr*=dr.select(valid,weight,0.);active &= valid;ray=newray
        return out,okp

    def field_components(self,lights,spp,seed,b=64):
        ns=self.N*PT.NB;acc={};launch=0
        for done in range(0,spp,b):
            bb=min(b,spp-done);N=ns*bb
            sd=dr.opaque(mi.UInt64,seed*1000003+launch);launch+=1
            rng=mi.PCG32(size=N,initstate=sd,initseq=mi.UInt64(0))
            slot=dr.arange(U,N)%ns;pid=slot//PT.NB;bn=slot%PT.NB
            p,nn,(t,bv,n)=self.patch_point(pid,rng.next_float32(),rng.next_float32())
            local=self.dir_in_bin(bn,rng.next_float32(),rng.next_float32())
            w=t*local.x+bv*local.y+n*local.z;ok=dr.dot(w,nn)>1e-3
            ray=mi.Ray3f(p+w*2e-3,-w)
            a,okp=self.trace(ray,N,rng,sd,'ER',5,lights,'probe',probe_y=p,probe_ok=ok)
            a['count']=dr.select(okp,F(1.),F(0.))
            reduced=self.accum(a,slot,ns)
            for key,val in reduced.items():acc[key]=acc.get(key,0.)+val
        cnt=acc['count'].reshape(self.N,PT.NB)
        def pack(name):return (np.stack([acc.get(name+str(c),np.zeros(ns)) for c in range(3)],-1).reshape(self.N,PT.NB,3)/np.maximum(cnt,1)[...,None]).astype(np.float32)
        return pack('f2'),pack('fh')
