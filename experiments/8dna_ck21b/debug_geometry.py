import numpy as np
import bridge as B
mi,dr=B.mi,B.dr
dr.set_flag(dr.JitFlag.LoopRecord,False); dr.set_flag(dr.JitFlag.VCallRecord,False)
d,sh=B.make_scene(512); scene=mi.load_dict(d)
print('norm',B.CENTER,B.SCALE,flush=True)
print('camera',scene.sensors()[0],flush=True)
print('shapes',[(s.id(),str(s.bbox())) for s in scene.shapes()],flush=True)
for uv in [[.5,.5],[.2,.5],[.8,.5]]:
    ray,_=scene.sensors()[0].sample_ray(0,0,mi.Point2f(*uv),mi.Point2f(.5))
    hit=scene.ray_intersect(ray)
    print('SAMPLE',uv,'ray',ray,'hit',hit,'worldp',np.asarray(hit.p)*B.SCALE+B.CENTER,flush=True)
ix=np.tile(np.arange(512),512); iy=np.repeat(np.arange(512),512)
ray,_=scene.sensors()[0].sample_ray(0,0,mi.Point2f((ix+.5)/512,(iy+.5)/512),mi.Point2f(.5))
hit=scene.ray_intersect(ray)
print('sizes',dr.width(ray.d),dr.width(hit.p),dr.width(hit.uv),'valid',np.sum(np.array(hit.is_valid())),'front',np.sum(np.array(hit.wi.z)>0),flush=True)
for j in [5,6,7,8,9,10,11]: print('shape',j,'eq',np.sum(np.array(dr.eq(hit.shape,mi.ShapePtr(sh[j])))),flush=True)
