"""WL84-style, one shared radiometric/error scale; never substitutes another method for 8DNA."""
import gc, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import bridge as B
from preflight import render,reference

def make(checkpoint,stage,spp=256):
    import torch
    import mitsuba as mi
    import drjit as dr
    import models.integrator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dr.set_flag(dr.JitFlag.LoopRecord,False); dr.set_flag(dr.JitFlag.VCallRecord,False)
    base=B.BASE; out=base/'results'; checkpoint=Path(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu')
    global_step=int(saved.get('global_step',0)); saved_epoch=int(saved.get('epoch',-1))
    required=30*8192 if stage=='final_epoch30' else 8192
    if global_step<required: raise RuntimeError(f'Checkpoint is not qualified for {stage}: {global_step} < {required}')
    del saved
    d,_=B.make_scene(512,max_depth=7); scene=mi.load_dict(d)
    integrator=mi.load_dict({'type':'neuralpath','max_depth':10,'rr_depth':5,'device':0,'model_type':'8dna','asset_instance0':str(checkpoint)})
    reps=[]; timings=[]
    # render(chunk=1) increments its seed once per spp, so reserve disjoint ranges.
    seeds=[21201,21201+spp]
    for seed in seeds:
        image,sec=render(scene,integrator,spp,seed,chunk=1); reps.append(image); timings.append(sec)
    neural=np.mean(reps,axis=0); np.save(out/f'8dna_{stage}.npy',neural)
    mi.Bitmap(neural.astype(np.float32)).write(str(out/f'8dna_{stage}.exr'))
    del integrator; gc.collect(); torch.cuda.empty_cache()
    old=Path.home()/'rest_x93d_ck2'; ref=reference()
    variants=[]
    for p in sorted((old/'results'/'renders').glob('CK_bx38_own_s8601_C0_CK2_T1_r*.npz')):
        with np.load(p) as z: variants.append(z['hl'].astype(np.float64))
    refs=[np.load(p) for p in sorted((old/'results'/'renders').glob('REF_ER_CK_bx38_CK2_T1_r*.npz'))]
    low=np.mean([z['e1'].astype(np.float64)+z['e2'] for z in refs],axis=0)
    rest=low+np.mean(variants,axis=0)
    masks=np.load(old/'results'/'pre'/'masks_ck2.npz'); roi=masks['ROI_T']
    ys,xs=np.nonzero(roi); pad=16; crop=(slice(max(0,ys.min()-pad),min(512,ys.max()+pad+1)),slice(max(0,xs.min()-pad),min(512,xs.max()+pad+1)))
    errors=[np.abs(rest-ref).mean(-1),np.abs(neural-ref).mean(-1)]
    maximum=max(float(e.max()) for e in errors)
    tm=lambda x:np.clip(np.maximum(x,0)/(1+np.maximum(x,0)),0,1)**(1/2.2)
    fig,axs=plt.subplots(2,5,figsize=(18,8.2),squeeze=False)
    metrics={}
    for label,im in [('rest_gi',rest),('8dna',neural)]:
        metrics[label]={region:float(np.abs(im[m]-ref[m]).sum()/max(ref[m].sum(),1e-30)) for region,m in [('frame',np.ones_like(roi)),('reflection',roi)]}
    model_label='8DNA (1 / 30 epochs, PREVIEW)' if stage=='epoch01_preview' else '8DNA (30 / 30 epochs)'
    titles=['CK2 REF-ER\n4 x 8192 spp; 6 events','ReST-GI ECRO C0, s8601\nREF LOW + learned H; 2 x 4096 spp',f'{model_label}\nnew CK2 asset; 2 x {spp} spp','|ReST-GI - REF|','|8DNA - REF|']
    for row in range(2):
        cut=(slice(None),slice(None)) if row==0 else crop
        for col,im in enumerate([ref,rest,neural]): axs[row,col].imshow(tm(im[cut])); axs[row,col].set_title(titles[col] if row==0 else ['Fixed reflection crop',f'Reflection rel L1 {metrics["rest_gi"]["reflection"]:.4f}',f'Reflection rel L1 {metrics["8dna"]["reflection"]:.4f}'][col],fontsize=10)
        for col,e in zip([3,4],errors):
            mapped=axs[row,col].imshow(np.sqrt(e[cut]),cmap='viridis',vmin=0,vmax=math.sqrt(maximum))
            axs[row,col].set_title(titles[col] if row==0 else 'Same absolute HDR error scale',fontsize=10)
        for ax in axs[row]: ax.axis('off')
    fig.subplots_adjust(left=.015,right=.91,top=.87,bottom=.14,wspace=.08,hspace=.2)
    cax=fig.add_axes([.93,.20,.012,.56]); fig.colorbar(mapped,cax=cax,label='sqrt(mean RGB |HDR difference|), shared scale')
    fig.suptitle('CK2 checkerboard interreflection: references and candidates (WL84-style layout)',fontsize=15)
    fig.text(.02,.065,'Frozen ER materials, original CK2 camera/light, bx38 checker (0.3/0.8). Display: x/(1+x), gamma 2.2; exposure 1 for all images.\n8DNA owns panel + checker box A; room/other objects are external. Native intrinsic paths are not the same as the stored six-event target.\nStatic qualitative comparison; different training lifecycles/budgets. '+('PRELIMINARY: only 1 of 30 native training epochs completed.' if stage=='epoch01_preview' else 'Fixed final native 30-epoch fit.'),fontsize=9)
    path=out/f'ck2_references_and_candidates_{stage}.png'; fig.savefig(path,dpi=140); plt.close(fig)
    payload={'stage':stage,'checkpoint_epoch':saved_epoch,'checkpoint_global_step':global_step,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'neural_spp_per_replicate':spp,'neural_seeds':seeds,'neural_render_seconds':timings,'metrics_descriptive_only':metrics,'error_scale_max_mean_rgb_abs':maximum,'figure':str(path),'figure_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reference_role':'Stored frozen-ER six-event reference. Native intrinsic full transport is a different path-depth regime, so no strict fidelity/speed winner is declared.'}
    (out/f'figure_{stage}.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('FIGURE',json.dumps(payload),flush=True)
    return payload

if __name__=='__main__': make(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 256)
