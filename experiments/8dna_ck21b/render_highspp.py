"""Fixed completed-checkpoint, user-requested 4096-spp CK2 rendering."""
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
UP = Path.home()/'NeuralShaderDemo/external/8dna26'
sys.path.insert(0,str(UP)); os.chdir(UP)
import bridge as B
import numpy as np
import torch
import models.integrator
from PIL import Image
from preflight import reference
mi, dr = B.mi, B.dr
dr.set_flag(dr.JitFlag.LoopRecord,False); dr.set_flag(dr.JitFlag.VCallRecord,False)
EPOCHS, TARGET = 18, 4096
STEM = f'8dna_ck2_epoch{EPOCHS:02d}_{TARGET}spp'
OUT = B.BASE/'results'/STEM
OUT.mkdir(exist_ok=True)
def write_json(name,value):
    temporary=OUT/(name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(OUT/name)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tm(x):
    x=np.maximum(x,0)
    return np.clip(x/(1+x),0,1)**(1/2.2)
def png(name,x): Image.fromarray(np.rint(tm(x)*255).astype(np.uint8)).save(OUT/name)

source=B.BASE/f'training_packed/CK2_bx38/epoch={EPOCHS-1}-step={EPOCHS*8192}.ckpt'
anchor=OUT/'checkpoint.ckpt'
if not anchor.exists(): shutil.copy2(source,anchor)
saved=torch.load(anchor,map_location='cpu')
assert saved['epoch']==EPOCHS-1 and saved['global_step']==EPOCHS*8192
del saved
plan={'epochs_completed':EPOCHS,'checkpoint_global_step':EPOCHS*8192,
    'checkpoint_source':str(source),'checkpoint':str(anchor),'checkpoint_sha256':sha(anchor),
    'resolution':[512,512],'target_spp':TARGET,'max_depth':10,'rr_depth':5,
    'display':'exposure1, Reinhard x/(1+x), gamma2.2; no denoising',
    'scope':'Intermediate fixed 18/30-epoch fit; native full transport differs from stored six-event reference. No convergence or dynamic superiority claim.',
    'scheduling':'Read-only render concurrent with continuing fit. All speed observations include contention.'}
write_json('request.json',plan)
if (OUT/'complete.json').exists(): raise RuntimeError('Preserve completed output')
scene=mi.load_dict(B.make_scene(512,max_depth=7)[0])
integrator=mi.load_dict({'type':'neuralpath','max_depth':10,'rr_depth':5,
    'device':0,'model_type':'8dna','asset_instance0':str(anchor)})

# Calibrate batching only; no sample-based quality selection. Probe samples
# never enter the requested image. All successful probes are recorded.
probes=[]
for size in [1,4,8]:
    torch.cuda.reset_peak_memory_stats()
    try:
        for repeat in range(2):
            start=time.perf_counter()
            image=mi.render(scene,integrator=integrator,spp=size,seed=24000+size*10+repeat)
            dr.eval(image); torch.cuda.synchronize()
            values=np.array(image,dtype=np.float64)
            if not np.isfinite(values).all(): raise ValueError('Nonfinite batch probe')
            seconds=time.perf_counter()-start
            print('PROBE',size,repeat,seconds,'peak',torch.cuda.max_memory_allocated(),flush=True)
        probes.append({'spp':size,'steady_seconds':seconds,'seconds_per_spp':seconds/size,
                       'torch_peak_bytes':torch.cuda.max_memory_allocated()})
        del image,values
    except torch.cuda.OutOfMemoryError:
        probes.append({'spp':size,'status':'oom_not_admitted'})
        gc.collect(); torch.cuda.empty_cache(); dr.flush_malloc_cache()
        break
valid=[p for p in probes if 'seconds_per_spp' in p]
chunk=min(valid,key=lambda p:p['seconds_per_spp'])['spp']
plan.update({'batch_spp':chunk,'batch_probes':probes,'seed_base':23101})
write_json('request.json',plan)
total=np.zeros((512,512,3),np.float64); done=0
accumulator=OUT/'accumulator.npz'
if accumulator.exists():
    with np.load(accumulator) as z:
        if str(z['checkpoint_sha256'])!=plan['checkpoint_sha256'] or int(z['chunk'])!=chunk:
            raise RuntimeError('Accumulator identity mismatch')
        total=z['total']; done=int(z['done'])
started=time.perf_counter(); initial=done
while done<TARGET:
    count=min(chunk,TARGET-done)
    image=mi.render(scene,integrator=integrator,spp=count,seed=23101+done//chunk)
    dr.eval(image); torch.cuda.synchronize()
    values=np.array(image,dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite render')
    total+=values*count; done+=count
    if done%128==0 or done==TARGET:
        elapsed=time.perf_counter()-started
        progress={**plan,'status':'rendering','spp_completed':done,'render_seconds_this_session':elapsed,
                  'seconds_per_spp':elapsed/max(done-initial,1),
                  'estimated_remaining_seconds':elapsed/max(done-initial,1)*(TARGET-done)}
        with open(OUT/'accumulator.tmp.npz','wb') as stream:
            np.savez(stream,total=total,done=done,chunk=chunk,checkpoint_sha256=plan['checkpoint_sha256'])
        (OUT/'accumulator.tmp.npz').replace(accumulator)
        write_json('progress.json',progress)
        print('PROGRESS',done,'/',TARGET,'seconds',elapsed,'remaining',progress['estimated_remaining_seconds'],flush=True)

neural=(total/TARGET).astype(np.float32)
np.save(OUT/(STEM+'.npy'),neural)
mi.Bitmap(neural).write(str(OUT/(STEM+'.exr')))
png(STEM+'.png',neural)
del integrator; gc.collect(); torch.cuda.empty_cache()
old=Path.home()/'rest_x93d_ck2'; ref=reference()
variants=[]
for path in sorted((old/'results/renders').glob('CK_bx38_own_s8601_C0_CK2_T1_r*.npz')):
    with np.load(path) as z: variants.append(z['hl'].astype(np.float64))
lows=[]
for path in sorted((old/'results/renders').glob('REF_ER_CK_bx38_CK2_T1_r*.npz')):
    with np.load(path) as z: lows.append(z['e1'].astype(np.float64)+z['e2'])
rest=np.mean(lows,axis=0)+np.mean(variants,axis=0)
with np.load(old/'results/pre/masks_ck2.npz') as z: roi=z['ROI_T']
ys,xs=np.nonzero(roi); pad=16
crop=(slice(max(0,ys.min()-pad),min(512,ys.max()+pad+1)),slice(max(0,xs.min()-pad),min(512,xs.max()+pad+1)))
metrics={name:{region:float(np.abs(im[m]-ref[m]).sum()/max(ref[m].sum(),1e-30))
    for region,m in [('frame',np.ones_like(roi)),('reflection',roi)]} for name,im in [('rest_gi',rest),('8dna',neural)]}
errors=[np.abs(rest-ref).mean(-1),np.abs(neural-ref).mean(-1)]
maximum=max(float(e.max()) for e in errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,5,figsize=(18,8.2),squeeze=False)
titles=['CK2 REF-ER\n4 x 8192 spp; 6 events','ReST-GI ECRO C0, s8601\nREF LOW + learned H; 2 x 4096 spp',
        f'8DNA ({EPOCHS}/30 epochs, INTERMEDIATE)\n1 x {TARGET} spp','|ReST-GI - REF|','|8DNA - REF|']
for row in range(2):
    cut=(slice(None),slice(None)) if row==0 else crop
    for col,im in enumerate([ref,rest,neural]):
        axs[row,col].imshow(tm(im[cut]))
        axs[row,col].set_title(titles[col] if row==0 else ['Fixed reflection crop',
            f'Reflection rel L1 {metrics["rest_gi"]["reflection"]:.4f}',
            f'Reflection rel L1 {metrics["8dna"]["reflection"]:.4f}'][col],fontsize=10)
    for col,error in zip([3,4],errors):
        mapped=axs[row,col].imshow(np.sqrt(error[cut]),cmap='viridis',vmin=0,vmax=math.sqrt(maximum))
        axs[row,col].set_title(titles[col] if row==0 else 'Same absolute HDR error scale',fontsize=10)
    for ax in axs[row]: ax.axis('off')
fig.subplots_adjust(left=.015,right=.91,top=.87,bottom=.14,wspace=.08,hspace=.2)
cax=fig.add_axes([.93,.20,.012,.56]); fig.colorbar(mapped,cax=cax,label='sqrt(mean RGB |HDR difference|), shared scale')
fig.suptitle('CK2 checkerboard interreflection: references and candidates',fontsize=15)
fig.text(.02,.065,f'Fixed {EPOCHS}/30-epoch checkpoint; {TARGET} spp total, no denoising. Display: x/(1+x), gamma 2.2, exposure 1.\nOriginal frozen ER materials/camera/light. Native asset full transport differs from the six-event stored target.\nStatic qualitative comparison; different training lifecycles/budgets. ReST-GI retains its stored 2 x 4096 spp and REF LOW + predicted H composition.',fontsize=9)
comparison='ck2_references_and_candidates_epoch18_4096spp.png'
fig.savefig(OUT/comparison,dpi=140); plt.close(fig)
files={name:sha(OUT/name) for name in [STEM+'.png',STEM+'.npy',STEM+'.exr',comparison]}
write_json('complete.json',{**plan,'status':'complete','spp_completed':done,
    'render_seconds_this_session':time.perf_counter()-started,'files':files,
    'metrics_descriptive_only':metrics})
print('COMPLETE',json.dumps(files),flush=True)
