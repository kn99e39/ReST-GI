"""Pre-freeze physical and model-contract checks, including independent volpath."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import common100 as C
from pt100 import Ctx,PT,mi,dr


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rgb=lambda x:{'type':'rgb','value':x}
    cam={'origin':[.9,-2.4,.6],'target':[.9,.4,.4],'fov_deg':28.,'res':64}
    ls=[{'pos':[.9,-.6,1.2],'rgb':[30.,30.,30.]}]
    cfg=C.config('sss','train',99998);ctx=Ctx(cfg['geom'],'sss')
    sphere={'type':'sphere','center':ctx.SC,'radius':ctx.SR,'bsdf':{'type':'null'},
            'interior':{'type':'homogeneous','sigma_t':12.,'albedo':rgb([.92,.80,.65])}}
    scene=mi.load_dict({'type':'scene','sph':sphere})
    ctx.scene=scene
    # This validation scene contains exactly one shape, the sphere. Invalid
    # intersections are still excluded by si.is_valid() in the renderer.
    ctx.shape_id=lambda si:PT.U(6)
    a,ta=ctx.render(PT.Lights(ls),'ER',cam,64,2048,100070,'cam',kmax=1,b=8)
    first=PT.pack(a,64,2048)['e1']
    native={'type':'scene','integrator':{'type':'volpath','max_depth':256,'rr_depth':256},
        'sensor':{'type':'perspective','fov':28.,'to_world':mi.ScalarTransform4f.look_at(origin=cam['origin'],target=cam['target'],up=[0,0,1]),
                  'sampler':{'type':'independent'},'film':{'type':'hdrfilm','width':64,'height':64,'rfilter':{'type':'box'}}},
        'sph':sphere,'light':{'type':'point','position':ls[0]['pos'],'intensity':rgb(ls[0]['rgb'])}}
    nscene=mi.load_dict(native);natives=[]
    for seed in [100071,100072]:
        x=mi.render(nscene,spp=2048,seed=seed);dr.eval(x);dr.sync_thread();natives.append(np.asarray(x).copy())
    reference=.5*(natives[0]+natives[1]);roi=(first.mean(-1)>0)&(reference.mean(-1)>0)
    ratio=first.sum(axis=(0,1))/reference.sum(axis=(0,1))
    noise=np.sqrt(np.mean(((natives[0]-natives[1])/2)[roi]**2))/np.sqrt(np.mean(reference[roi]**2))
    # All optical-domain NEE is independently reproduced by native volpath.
    tests={'sss_native_mean_energy':bool(np.max(np.abs(ratio-1))<.05),
           'sss_finite_nonnegative':bool(np.isfinite(first).all() and np.min(first)>=0),
           'sss_reference_noise':bool(noise<.05)}
    np.savez_compressed(root/'sss_native_check.npz',ours=first,native=reference,half0=natives[0],half1=natives[1],roi=roi)
    # Actual renderer light-connection through a diameter of the sphere,
    # compared to an independent Lambert * Beer calculation.
    bc=Ctx(cfg['geom'],'sss')
    ray=mi.Ray3f(mi.Point3f(.9,-.3,1.),mi.Vector3f(0,0,-1))
    hit=bc.scene.ray_intersect(ray)
    wi=hit.to_local(-ray.d);cl=PT.CD.DrClosure83('ER',C.materials('sss'),PT.U(0))
    ld=PT.Lights([{'pos':[.9,1.1,.8],'rgb':[30.,30.,30.]}])
    direct,ball=bc.direct(hit,wi,cl,ld,mi.Bool(True),mi.Bool(True))
    dr.eval(direct,ball)
    measured=float(np.asarray(ball).ravel()[0]);expected=.55*(.8/np.sqrt(2.6))/np.pi*30/2.6*np.exp(-12*.8)
    ball_error=abs(measured/expected-1)
    tests['ballistic_beer_renderer']=bool(ball_error<.001 and np.max(np.asarray(direct))==0)
    import torch
    import model86 as M
    nparams=sum(p.numel() for p in M.ECRO().parameters())
    tests['model_parameter_count']=nparams==735664
    near=C.config('near','train',99999);nc=Ctx(near['geom'],'near')
    gb=nc.gbuffer(dict(near['camera'],res=128),128)
    tests['near_panel_visible']=int(np.sum(gb['sid']==5))>1000
    tests['renderer_material_feature_table']=bool(np.array_equal(nc.mats.states,C.materials('near')))
    hit,bn,_=nc.state_rays();gs=M.GeoState(nc.G,hit,bn,torch.device('cuda'))
    model=M.ECRO().cuda();model.mat_states=torch.as_tensor(C.materials('near'),device='cuda')
    with torch.no_grad():
        f=model.factors(gs);s=torch.as_tensor(nc.source(PT.Lights(near['lights'][0])),device='cuda').unsqueeze(0)
        a=model.apply(f,gs,s);zero=model.apply(f,gs,torch.zeros_like(s));linear=model.apply(f,gs,1.5*s)
        linear_error=float((linear-1.5*a).abs().sum()/a.abs().sum())
    tests['zero_superposition']=bool(float(zero.abs().max())==0 and linear_error<1e-5)
    result={'tests':tests,'pass':all(tests.values()),'native_energy_ratio_rgb':ratio.tolist(),
            'native_noise_relative_rms':float(noise),'custom_timing':ta,
            'native_scope':'Mitsuba homogeneous volpath; isolated sphere, same point light/camera; sigma_t=12/m, known null-interface epsilon small at this density',
            'params':nparams,'ballistic_renderer_relative_error':ball_error,'linearity_relative_error':linear_error,
            'mitsuba':mi.__version__,'drjit':dr.__version__,'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0),'boundary_offset_qualification':'low-density sphere only; not dense-media validity'}
    C.jd(result,root/'preflight.json');print(json.dumps(result),flush=True)
    assert result['pass'],'STOP_INVALID_PREFLIGHT'


if __name__=='__main__':
    main(sys.argv[1]);sys.stdout.flush();os._exit(0)
