"""Postprocess completed neural rendering using one stored 4096-spp C0 replicate."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path.home()/'rest_8dna_ck21b'
stem='8dna_ck2_epoch18_4096spp'
out=base/'results'/stem
marker=out/'matched4096_complete.json'
if marker.exists():
    print('Matched comparison already complete'); raise SystemExit(0)
record=json.loads((out/'complete.json').read_text())
assert record['status']=='complete' and record['spp_completed']==4096
assert record['epochs_completed']==18 and record['checkpoint_global_step']==147456
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
neural_path=out/(stem+'.npy')
assert sha(neural_path)==record['files'][neural_path.name]
neural=np.load(neural_path)
old=Path.home()/'rest_x93d_ck2/results'
rest_path=old/'renders/CK_bx38_own_s8601_C0_CK2_T1_r0.npz'
rest_meta=json.loads(rest_path.with_suffix('.json').read_text())
assert rest_meta['spp']==4096
ref_paths=sorted((old/'renders').glob('REF_ER_CK_bx38_CK2_T1_r*.npz'))
assert len(ref_paths)==4
lows=[]; refs=[]
for path in ref_paths:
    with np.load(path) as z:
        low=z['e1'].astype(np.float64)+z['e2']
        lows.append(low); refs.append(low+z['h'])
ref=np.mean(refs,axis=0)
with np.load(rest_path) as z: rest=np.mean(lows,axis=0)+z['hl'].astype(np.float64)
with np.load(old/'pre/masks_ck2.npz') as z: roi=z['ROI_T']
ys,xs=np.nonzero(roi); pad=16
crop=(slice(max(0,ys.min()-pad),min(512,ys.max()+pad+1)),slice(max(0,xs.min()-pad),min(512,xs.max()+pad+1)))
def tm(x):
    x=np.maximum(x,0)
    return np.clip(x/(1+x),0,1)**(1/2.2)
metrics={name:{region:float(np.abs(im[m]-ref[m]).sum()/max(ref[m].sum(),1e-30))
    for region,m in [('frame',np.ones_like(roi)),('reflection',roi)]} for name,im in [('rest_gi',rest),('8dna',neural)]}
errors=[np.abs(rest-ref).mean(-1),np.abs(neural-ref).mean(-1)]
maximum=max(float(e.max()) for e in errors)
fig,axs=plt.subplots(2,5,figsize=(18,8.2),squeeze=False)
titles=['CK2 REF-ER\n4 x 8192 spp; 6 events',
    'ReST-GI ECRO C0, s8601\n4096 spp, r0; REF LOW + learned H',
    '8DNA (18/30 epochs, INTERMEDIATE)\n4096 spp total; no denoising',
    '|ReST-GI - REF|','|8DNA - REF|']
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
fig.suptitle('CK2 checkerboard interreflection: references and candidates (4096 spp)',fontsize=15)
fig.text(.02,.065,'8DNA: 4096 spp total. ReST-GI learned H: 4096 spp, plus stored high-spp REF LOW. Display: x/(1+x), gamma 2.2, exposure 1.\nReST-GI: frozen C0 s8601 r0 (seed 86600). 8DNA: fixed 18/30-epoch learned asset. No denoising.\nOriginal frozen ER materials/camera/light. Native asset full transport differs from the six-event target; descriptive static comparison.',fontsize=9)
filename='ck2_references_and_candidates_epoch18_matched4096spp.png'
fig.savefig(out/filename,dpi=140); plt.close(fig)
files={name:record['files'][name] for name in [stem+'.png',stem+'.npy',stem+'.exr']}
files[filename]=sha(out/filename)
matched={**record,'files':files,'metrics_descriptive_only':metrics,
    'rest_gi_spp':4096,'rest_gi_replicate':'r0','rest_gi_render_metadata':rest_meta,
    'rest_gi_composition':'Stored high-spp reference LOW + single 4096-spp learned H; inherited baseline contract.',
    'source_hashes':{str(p):sha(p) for p in [rest_path,rest_path.with_suffix('.json'),*ref_paths]},
    'original_render_manifest_sha256':sha(out/'complete.json')}
temporary=out/'matched4096_complete.tmp'
temporary.write_text(json.dumps(matched,indent=2)+'\n'); temporary.replace(marker)
print('MATCHED_COMPARISON_COMPLETE',filename,files[filename],flush=True)
