import glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import bridge as B
mi,dr=B.mi,B.dr
dr.set_flag(dr.JitFlag.LoopRecord,False); dr.set_flag(dr.JitFlag.VCallRecord,False)
BASE=B.BASE; OLD=Path.home()/'rest_x93d_ck2'

def save(data,name='preflight.json'):
    (BASE/'results').mkdir(parents=True,exist_ok=True)
    (BASE/'results'/name).write_text(json.dumps(data,indent=2)+'\n')

def reference():
    z=[np.load(p) for p in sorted((OLD/'results'/'renders').glob('REF_ER_CK_bx38_CK2_T1_r*.npz'))]
    return np.mean([x['e1'].astype(np.float64)+x['e2']+x['h'] for x in z],axis=0)

def render(scene,integrator,spp,seed,chunk=4):
    import torch
    total=None; start=time.perf_counter()
    for offset in range(0,spp,chunk):
        n=min(chunk,spp-offset); image=mi.render(scene,integrator=integrator,spp=n,seed=seed+offset//chunk)
        dr.eval(image); torch.cuda.synchronize(); a=np.array(image,dtype=np.float64)
        if not np.isfinite(a).all(): raise ValueError('Nonfinite render')
        total=a*n if total is None else total+a*n
        if (offset+n)%128==0: print('render',offset+n,'/',spp,'seconds',round(time.perf_counter()-start,2),flush=True)
    return total/spp,time.perf_counter()-start

def main():
    import torch
    result={'status':'running','cpu_contract':B.cpu_contract(),'hashes':{}}
    for p in [B.EXP/'ecro86'/'data'/'mats86.npz',B.EXP/'fnto_compiled83'/'closure83_dr.py',B.EXP/'ecro_local93d'/'addendum_checker'/'scene_ck2.json']:
        result['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(result)
    if not result['cpu_contract']['all_external_disjoint']: raise RuntimeError('Invalid native asset hull')
    print('CPU contract PASS',flush=True)
    # Same frozen closure implementation, independent plugin calls and channel checks.
    n=4096; rs=np.random.default_rng(21110)
    def dirs():
        x=rs.normal(size=(n,3)); x[:,2]=np.abs(x[:,2])+0.01; x/=np.linalg.norm(x,axis=1,keepdims=True); return mi.Vector3f(x.astype(np.float32))
    si=dr.zeros(mi.SurfaceInteraction3f,n); si.wi=dirs(); wo=dirs()
    si.uv=mi.Point2f(rs.random((n,2),dtype=np.float32)); u1=mi.Float(rs.random(n,dtype=np.float32)); u2=mi.Point2f(rs.random((n,2),dtype=np.float32))
    maxdiff=0
    for row in range(8):
        plugin=mi.load_dict({'type':'ck21b_er','row':row})
        direct=B.CD.DrClosure83('ER',B.STATES,mi.UInt32(row))
        f,p=plugin.eval_pdf(mi.BSDFContext(),si,wo)
        rf,rp=direct.eval_pdf(si.wi,wo)
        bs,w=plugin.sample(mi.BSDFContext(),si,u1,u2)
        rw,rpdf,rweight,rvalid=direct.sample(si.wi,u1,u2)
        for a,b in [(f,rf),(p,rp),(bs.wo,rw),(bs.pdf,rpdf),(w,dr.select(rvalid,rweight,0))]:
            maxdiff=max(maxdiff,float(np.max(np.abs(np.array(a)-np.array(b)))))
    result['closure_max_abs_diff']=maxdiff
    plugin=mi.load_dict({'type':'ck21b_er','row':4,'checker':True})
    mid=dr.select(dr.eq((mi.UInt32(dr.floor(si.uv.x*3))+mi.UInt32(dr.floor(si.uv.y*3)))%2,0),mi.UInt32(6),mi.UInt32(7))
    direct=B.CD.DrClosure83('ER',B.STATES,mid)
    f,p=plugin.eval_pdf(mi.BSDFContext(),si,wo); rf,rp=direct.eval_pdf(si.wi,wo)
    bs,w=plugin.sample(mi.BSDFContext(),si,u1,u2); rw,rpdf,rweight,rvalid=direct.sample(si.wi,u1,u2)
    for a,b in [(f,rf),(p,rp),(bs.wo,rw),(bs.pdf,rpdf),(w,dr.select(rvalid,rweight,0))]: maxdiff=max(maxdiff,float(np.max(np.abs(np.array(a)-np.array(b)))))
    result['closure_max_abs_diff']=maxdiff
    if maxdiff>1e-6: raise RuntimeError('Frozen ER plugin mismatch')
    print('Closure PASS',maxdiff,flush=True)
    dictionary,shapes=B.make_scene(512,max_depth=7); scene=mi.load_dict(dictionary)
    side=512; ix=np.tile(np.arange(side),side); iy=np.repeat(np.arange(side),side)
    ray,_=scene.sensors()[0].sample_ray(0,0,mi.Point2f((ix+.5)/side,(iy+.5)/side),mi.Point2f(.5))
    hit=scene.ray_intersect(ray); front=hit.is_valid()&(hit.wi.z>1e-6)
    panel=front&dr.eq(hit.shape,mi.ShapePtr(shapes[5]))
    box=mi.Bool(False); pid=mi.UInt32(0)
    g=B.S.Geom(B.SC['geom'])
    for j in [5,6,7,8,9,10,11]:
        match=front&dr.eq(hit.shape,mi.ShapePtr(shapes[j]))
        if j!=5: box |= match
        nu,nv=g.RG[j]
        iu=mi.UInt32(dr.clamp(dr.floor(hit.uv.x*int(nu)),0,int(nu)-1)); iv=mi.UInt32(dr.clamp(dr.floor(hit.uv.y*int(nv)),0,int(nv)-1))
        pid=dr.select(match,mi.UInt32(int(g.RBASE[j]))+iu+int(nu)*iv,pid)
    old=np.load(OLD/'results'/'pre'/'masks_ck2.npz')
    conv=lambda x:np.broadcast_to(np.array(x),(side*side,)).reshape(side,side)
    result['mask_agreement']={k:float(np.mean(conv(v)==old[k])) for k,v in [('frame',front),('panel',panel),('target_direct',box)]}
    target=conv(box); npid=conv(pid).astype(int)
    result['box_patch_agreement']=float(np.mean(npid[target]==old['pid'][target]))
    pmat=old['PMAT_bx38']; iu=mi.UInt32(dr.clamp(dr.floor(hit.uv.x*3),0,2)); iv=mi.UInt32(dr.clamp(dr.floor(hit.uv.y*3),0,2))
    mids=conv(dr.select(dr.eq((iu+iv)%2,0),mi.UInt32(6),mi.UInt32(7)))
    result['checker_material_agreement']=float(np.mean(mids[target]==pmat[old['pid'][target]]))
    print('Geometry',json.dumps({k:result[k] for k in ['mask_agreement','box_patch_agreement','checker_material_agreement']}),flush=True)
    save(result)
    if min(result['mask_agreement'].values())<.999 or result['checker_material_agreement']<.999: raise RuntimeError('CK2 bridge geometry/UV/camera mismatch')
    image,seconds=render(scene,scene.integrator(),512,21120)
    np.save(BASE/'results'/'native_er_d6_512spp.npy',image)
    ref=reference(); roi=old['ROI_T']
    rel=lambda m:float(np.abs(image[m]-ref[m]).sum()/max(ref[m].sum(),1e-30))
    result['radiometry']={'spp':512,'seconds':seconds,'whole_rel_l1':rel(np.ones((512,512),bool)),'reflection_rel_l1':rel(roi),'whole_signed_bias':float((image-ref).sum()/ref.sum()),'native_mean':float(image.mean()),'stored_mean':float(ref.mean())}
    result['status']='pass' if abs(result['radiometry']['whole_signed_bias'])<.02 and result['radiometry']['whole_rel_l1']<.1 else 'failed_radiometry'
    save(result); print(json.dumps(result,indent=2),flush=True)
    if result['status']!='pass': raise RuntimeError('Native radiometry bridge not qualified')
    six,_=render(scene,scene.integrator(),128,21190)
    deep=mi.load_dict({'type':'prb','max_depth':17,'rr_depth':64})
    sixteen,sec16=render(scene,deep,128,21190)
    np.savez_compressed(BASE/'results'/'depth_diagnostic.npz',six=six,sixteen=sixteen)
    result['depth_diagnostic']={'paired_spp':128,'native_d16_seconds':sec16,'whole_signed_tail_over_d6':float((sixteen-six).sum()/max(six.sum(),1e-30)),'reflection_signed_tail_over_d6':float((sixteen[roi]-six[roi]).sum()/max(six[roi].sum(),1e-30)),'note':'Paired finite-spp diagnostic, not a converged infinite-depth reference. Native neural asset internal depth remains unbounded.'}
    save(result); print('DEPTH',json.dumps(result['depth_diagnostic']),flush=True)

if __name__=='__main__': main()
