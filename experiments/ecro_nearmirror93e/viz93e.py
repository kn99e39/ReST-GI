"""WL93E linear-HDR measurements and shared-exposure reviewer sheets.

python viz93e.py ROOT OUTPUT [--preview]
No render imports, image denoising, per-image exposure, or trained parameters.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image

NAMES = ['S1','G075','G040','G010','MIRROR']
LABELS = ['Original S1 / frozen ER','GGX alpha = 0.075','GGX alpha = 0.04','GGX alpha = 0.01','Ideal mirror / e2 qualified']
COLORS = ['#469b9b','#ad6339','#d8a138','#2776c1','#252525']


def jd(data,path):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))


def tm(x,key):
    y = np.maximum(x,0)*key
    return np.clip(y/(1+y),0,1)**(1/2.2)


def lum(x):
    # Match pt86 / historical checker measurements (arithmetic RGB mean).
    return x.mean(-1).astype(np.float64)


def read(root,mat,var,res,seeds):
    zs = [np.load(root/'results/renders'/f'{mat}_{var}_{res}_{s}.npz') for s in seeds]
    reps = [{k:z[k].astype(np.float64) for k in z.files} for z in zs]
    return {k:np.mean([r[k] for r in reps],axis=0) for k in reps[0]}, reps


def bbox(mask,pad=12):
    y,x=np.nonzero(mask)
    return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),
            min(mask.shape[1],int(x.max())+pad+1),min(mask.shape[0],int(y.max())+pad+1)]


def sl(box):
    x0,y0,x1,y1=box
    return np.s_[y0:y1,x0:x1]


def fit(x,y,mask):
    xx,yy=x[mask],y[mask]
    slope,intercept=np.linalg.lstsq(np.column_stack([xx,np.ones(len(xx))]),yy,rcond=None)[0]
    return {'slope':float(slope),'intercept':float(intercept),
            'correlation':float(np.corrcoef(xx,yy)[0,1]),'pixels':int(mask.sum())}


def image_save(path,x,key):
    Image.fromarray(np.round(tm(x,key)*255).astype(np.uint8)).save(path)


def main(root,out,preview=False):
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    res,seeds=(512,[931490]) if preview else (1024,[931500,931501])
    data,reps={},{}
    for name in NAMES:
        data[name],reps[name]=read(root,name,'bx38',res,seeds)
    m=np.load(root/f'results/maps_{res}.npz')
    roi=m['roi']; panel=m['panel']; direct=m['direct']
    box=bbox(roi,8); crop=sl(box)
    key=.5/float(np.median(lum(data['S1']['full'])[lum(data['S1']['full'])>0]))
    if preview:
        fig,ax=plt.subplots(2,5,figsize=(20,10))
        for j,n in enumerate(NAMES):
            ax[0,j].imshow(tm(data[n]['full'],key)); ax[0,j].set_title(LABELS[j])
            ax[1,j].imshow(tm(data[n]['full'][crop],key))
        for a in ax.flat:a.axis('off')
        fig.suptitle('WL93E physical-only preview / same CK2 scene, lighting and exposure / no model selection',fontsize=14)
        fig.tight_layout(); fig.savefig(out/'preview.png',dpi=140); plt.close(fig)
        return
    uniform={}; ureps={}
    for name in NAMES:
        uniform[name],ureps[name]=read(root,name,'none',res,seeds)
    # Exposure is fixed to the ORIGINAL S1 UNTEXTURED reference for all sheets.
    key=.5/float(np.median(lum(uniform['S1']['full'])[lum(uniform['S1']['full'])>0]))
    fig,ax=plt.subplots(2,5,figsize=(25,11),gridspec_kw={'height_ratios':[1,1]})
    for j,n in enumerate(NAMES):
        ax[0,j].imshow(tm(data[n]['full'],key)); ax[0,j].set_title(LABELS[j],fontsize=15)
        ax[0,j].contour(roi,levels=[.5],colors=['#00ffff'],linewidths=.6)
        ax[1,j].imshow(tm(data[n]['full'][crop],key),interpolation='nearest')
    for a in ax.flat:a.axis('off')
    ax[0,0].text(.02,.03,'cyan = reflected checker ROI',transform=ax[0,0].transAxes,color='cyan',fontsize=10)
    ax[1,0].text(.02,.03,'same native-pixel crop',transform=ax[1,0].transAxes,color='white',fontsize=11)
    fig.suptitle('WL93E | 1024 x 1024 / 4096 spp | Full-radiance DIAGNOSTICS: higher-order pixel RMS noise is not converged\nSame geometry/light/camera/exposure. Mirror full/h also omit light-side delta caustics. Use the qualified e2 comparison for edge conclusions.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(out/'figNM0_full_and_roi.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(3,5,figsize=(24,16))
    for j,n in enumerate(NAMES):
        ax[0,j].imshow(tm(data[n]['reflection'][crop],key));ax[0,j].set_title(LABELS[j],fontsize=13)
        ax[1,j].imshow(tm(data[n]['e2'][crop],key))
        ax[2,j].imshow(tm((data[n]['h']*panel[...,None])[crop],key))
    for a in ax.flat:a.axis('off')
    for i,label in enumerate(['Panel reflection = events 2-6','Exactly 2 events','Higher order = events 3-6']):
        ax[i,0].text(0,-.07,label,transform=ax[i,0].transAxes,fontsize=13)
    fig.suptitle('Path contributions / shared exposure / no per-component rescaling\ne2 is the qualified edge reference. Reflection/h are noise-unconverged diagnostics; mirror h also omits light-side delta caustics.',fontsize=14)
    fig.tight_layout(rect=(0,.02,1,.95));fig.savefig(out/'figNM1_path_contributions.png',dpi=160);plt.close(fig)
    # Main qualified visual: LOW gives scene context; e2 isolates the physical
    # checker reflection without the unqualified higher-order noise.
    fig,ax=plt.subplots(2,5,figsize=(25,11))
    for j,n in enumerate(NAMES):
        ax[0,j].imshow(tm(data[n]['e1']+data[n]['e2'],key));ax[0,j].set_title(LABELS[j],fontsize=14)
        ax[0,j].contour(roi,levels=[.5],colors=['cyan'],linewidths=.6)
        ax[1,j].imshow(tm(data[n]['e2'][crop],key),interpolation='nearest')
    for a in ax.flat:a.axis('off')
    ax[0,0].text(.02,.03,'LOW: events 1-2',transform=ax[0,0].transAxes,color='cyan',fontsize=11)
    ax[1,0].text(.02,.03,'Panel e2 reflection / native-pixel crop',transform=ax[1,0].transAxes,color='white',fontsize=10)
    fig.suptitle('QUALIFIED CHECKER-REFLECTION REFERENCE | 1024 x 1024 / 4096 spp / same scene, lighting and exposure\nTop: LOW (events 1-2), not full radiance. Bottom: exactly-two-event panel reflection. Original S1 is a multilobe historical control.',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'figNM5_qualified_low_and_e2.png',dpi=180);plt.close(fig)
    # Geometric edge alignment from independent OBB mapping; common mask.
    faces,counts=np.unique(m['face'][roi],return_counts=True)
    face_id=int(faces[np.argmax(counts)])
    face=roi & (m['face']==face_id)
    bright=m['parity'].astype(bool)
    # Opposite parity outside this face is suppressed in the distance transform.
    dbright=ndi.distance_transform_edt(~(face & bright))
    ddark=ndi.distance_transform_edt(~(face & ~bright))
    distance=np.where(bright,ddark-.5,-dbright+.5)
    interior=face & (ndi.distance_transform_edt(face)>30)
    # Reject near checker corners where two edge normals overlap. A local
    # geometry test, independent of reference radiance/model error.
    gy,gx=np.gradient(distance)
    straight=interior & ((np.abs(gx)<.12)|(np.abs(gy)<.12))
    bins=np.arange(-60,61,dtype=float)
    def profile(x):
        return np.array([float(np.mean(x[straight & (distance>=b-.5) & (distance<b+.5)]))
                         if np.any(straight & (distance>=b-.5) & (distance<b+.5)) else np.nan for b in bins])
    ratios={n:lum(data[n]['e2'])/np.maximum(lum(uniform[n]['e2']),1e-12) for n in NAMES}
    mirror_ratio=ratios['MIRROR']
    profiles={n:profile(ratios[n]) for n in NAMES}
    metric={}; profile_noise={}
    mask=roi & (m['edge']>.04)
    for n in NAMES:
        r=ratios[n]
        dark=float(np.median(r[mask & ~bright]));light=float(np.median(r[mask & bright]))
        contrast=(light-dark)/(light+dark)
        p=profiles[n]
        # Relative to ideal levels .6 / 1.6: failure to reach thresholds is
        # reported as unresolved, rather than renormalising away lost contrast.
        valid=np.isfinite(p)
        monotone=np.maximum.accumulate(np.clip(p[valid],.6,1.6))
        xx=bins[valid]
        def crossing(value):
            if np.min(monotone)>value or np.max(monotone)<value:return None
            return float(np.interp(value,monotone,xx))
        x10,x90=crossing(.7),crossing(1.5)
        noise={}
        for k in ['full','e2','h','reflection']:
            mean=lum(data[n][k]); delta=(lum(reps[n][0][k])-lum(reps[n][1][k]))/2
            noise[k]={'relative_rms_sem_of_mean':float(np.sqrt(np.mean(delta[roi]**2))/max(np.sqrt(np.mean(mean[roi]**2)),1e-20)),
                      'absolute_rms_sem_of_mean':float(np.sqrt(np.mean(delta[roi]**2)))}
        r0=lum(reps[n][0]['e2'])/np.maximum(lum(ureps[n][0]['e2']),1e-12)
        r1=lum(reps[n][1]['e2'])/np.maximum(lum(ureps[n][1]['e2']),1e-12)
        pn=np.abs(profile(r0)-profile(r1))/2
        profile_noise[n]=pn
        metric[n]={'dark_ratio_median':dark,'bright_ratio_median':light,'michelson_contrast':float(contrast),
                   'texture_ratio_response_vs_mirror':fit(mirror_ratio-1,r-1,mask),
                   'edge_10_90_width_pixels':None if x10 is None or x90 is None else x90-x10,
                   'edge_x10':x10,'edge_x90':x90,
                   'edge_width_scope':'Ideal .6/1.6 ratio levels; pooled signed-distance profile on largest reflected face; null if thresholds not attained in +/-60 px',
                   'profile_sem_max':float(np.nanmax(pn)), 'noise':noise}
    for n in NAMES:
        metric[n]['contrast_retention_vs_ideal']=metric[n]['michelson_contrast']/metric['MIRROR']['michelson_contrast']
    expected=m['expected_e2']
    # Remove checker boundaries, silhouette, self-shadow transitions and dark
    # pixels. Analytic point-light e2 is tested on fully visible front face.
    analytical=mask & (ndi.distance_transform_edt(roi)>8) & (expected>1e-4)
    measured=lum(data['MIRROR']['e2'])
    ar=np.sqrt(np.mean((measured[analytical]-expected[analytical])**2))/np.sqrt(np.mean(expected[analytical]**2))
    proof={'analytic_delta_e2_relative_rms':float(ar),'pixels':int(analytical.sum()),
           'description':'independent NumPy ideal mirror to OBB, Lambert albedo and point-light inverse-square/cosine; centre-ray expectation away from cell/silhouette edges, compared to subpixel-integrated render'}
    fig=plt.figure(figsize=(25,13))
    gs=fig.add_gridspec(2,5,height_ratios=[1.25,1])
    for j,n in enumerate(NAMES):
        a=fig.add_subplot(gs[0,j]);a.imshow(np.where(roi,ratios[n],np.nan)[crop],vmin=.6,vmax=1.6,cmap='coolwarm',interpolation='nearest')
        a.set_title(LABELS[j]+f'\ncontrast retention {metric[n]["contrast_retention_vs_ideal"]:.3f}',fontsize=13);a.axis('off')
    a=fig.add_subplot(gs[1,:3])
    for j,n in enumerate(NAMES):
        pp=profiles[n];pn=profile_noise[n]
        a.plot(bins,pp,color=COLORS[j],label=LABELS[j]);a.fill_between(bins,pp-pn,pp+pn,color=COLORS[j],alpha=.15)
    a.axhline(.7,color='gray',linestyle=':');a.axhline(1.5,color='gray',linestyle=':')
    a.set(xlabel='Signed distance from ideal checker boundary (native pixels; dark to bright)',ylabel='Textured / uniform radiance (exactly 2 events)',ylim=(.55,1.65));a.legend(fontsize=10);a.grid(alpha=.2)
    a=fig.add_subplot(gs[1,3:]);a.axis('off')
    lines=['Linear HDR metrics; no tone mapping in numbers','10-90% width uses ideal ratio levels 0.7 / 1.5','Shading = paired-replica profile difference / 2','', 'Material          retention   width px   corr']
    for n in NAMES:
        mm=metric[n];w=mm['edge_10_90_width_pixels'];ws='unreached' if w is None else f'{w:.2f}'
        lines.append(f'{n:8s}          {mm["contrast_retention_vs_ideal"]:.3f}     {ws:>9s}   {mm["texture_ratio_response_vs_mirror"]["correlation"]:.3f}')
    a.text(0,1,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=11)
    fig.suptitle('Checker edge / ideal-mirror comparison | same pixel correspondence, ratio scale and paired uniform control\nS1 is a historical multilobe control; only the grey GGX / mirror group is a roughness-only comparison.',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'figNM2_edges_and_contrast.png',dpi=170);plt.close(fig)
    # Explicitly separate the direct box from its mirror image.
    overview=tm(data['MIRROR']['full'],key)
    fig,a=plt.subplots(figsize=(11,10));a.imshow(overview)
    a.contour(roi,levels=[.5],colors=['cyan'],linewidths=1)
    a.contour(direct,levels=[.5],colors=['yellow'],linewidths=1)
    a.set_title('Ideal mirror correspondence\nCYAN: reflected checker / YELLOW: directly viewed box',fontsize=15);a.axis('off')
    fig.tight_layout();fig.savefig(out/'figNM3_correspondence.png',dpi=150);plt.close(fig)
    # S1 only, and expressly a material-distribution diagnostic. Do not run the
    # frozen model with the new GGX/delta states.
    model_metrics={}; model_present=all((root/'results/renders'/f'ECRO{s}_bx38_{res}_{seeds[0]}.npz').exists() for s in [8601,8602])
    if model_present:
        fig,ax=plt.subplots(3,3,figsize=(16,16))
        hs=[data['S1']['h']];fulls=[data['S1']['full']];labels=['REF-ER / original S1']
        for sd in [8601,8602]:
            mh,mr=read(root,f'ECRO{sd}','bx38',res,seeds)
            mu,_=read(root,f'ECRO{sd}','none',res,seeds)
            hs.append(mh['hl']);fulls.append(data['S1']['e1']+data['S1']['e2']+mh['hl']);labels.append(f'Frozen ECRO C0 / seed {sd}')
            model_metrics[str(sd)]={'higher_order_relative_rms':float(np.sqrt(np.mean((lum(mh['hl'])[roi]-lum(data['S1']['h'])[roi])**2))/np.sqrt(np.mean(lum(data['S1']['h'])[roi]**2))),
                'higher_order_texture_response':fit(lum(data['S1']['h']-uniform['S1']['h']),lum(mh['hl']-mu['hl']),roi)}
        for j in range(3):
            ax[0,j].imshow(tm(fulls[j],key));ax[0,j].set_title(labels[j])
            ax[1,j].imshow(tm(fulls[j][crop],key));ax[2,j].imshow(tm(hs[j][crop],key))
        for a in ax.flat:a.axis('off')
        fig.suptitle('S1 MATCHED-MATERIAL comparison only | common exposure and physical LOW + learned higher-order term\nChecker itself is post-hoc material-distribution OOD. This is NOT a near-mirror fidelity qualification.',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'figNM4_frozen_ecro_S1_only.png',dpi=180);plt.close(fig)
    for n in NAMES:
        image_save(out/f'{n}_full_1024.png',data[n]['full'],key)
        image_save(out/f'{n}_reflection_roi.png',data[n]['reflection'][crop],key)
        image_save(out/f'{n}_higher_order_roi.png',data[n]['h'][crop],key)
        image_save(out/f'{n}_qualified_e2_roi.png',data[n]['e2'][crop],key)
    gates={'near_mirror_noise':metric['G010']['noise']['e2']['relative_rms_sem_of_mean']<=.02,
           'near_mirror_profile_noise':metric['G010']['profile_sem_max']<=.03,
           'analytic_delta_e2':bool(ar<=.01),
           'roi_pixels':int(roi.sum())>=10000}
    result={'name':'wl93e_nearmirror_reference_v1','architecture_verdict':'NOT APPLICABLE',
            'resolution':res,'spp_per_condition':4096,'replica_seeds':seeds,'metrics':metric,
            'qualified_scope':'Exactly-two-event panel reflection e2 and its texture edge/contrast; not complete full radiance or higher-order convergence',
            'higher_order_convergence':'NOT QUALIFIED: ROI h relative RMS SEM is 0.127 / 0.199 / 0.294 / 0.487 for S1/G075/G040/G010; rare bright paths dominate. Full/h images and model-h metrics are descriptive diagnostics only.',
            'roi':{'bbox_xyxy_exclusive':box,'pixels':int(roi.sum()),'direct_pixels':int(direct.sum()),
                   'coordinate_convention':'origin top-left, x right, y down; same primary-centre-ray pixels for every material',
                   'edge_face':face_id,'edge_pixels':int(straight.sum()),'metric_pixels':int(mask.sum())},
            'analytic_mirror_control':proof,'qualification':gates,'qualified':all(gates.values()),
            'exposure_key':key,'tone_map':'(key*x/(1+key*x))**(1/2.2); identical all RGB/component/model images',
            'noise_definition':'spatial RMS of independent replica difference / 2, relative to spatial RMS of mean, for ROI; estimated SEM of 2-replica mean, not pixelwise confidence',
            'path_scope':'Reflection on primary panel pixels = e2+h; exact two-event e2; higher order h=events 3..6. Finite depth, no denoise/clamp. Ideal mirror e2 is qualified; its full/reflection/h omit light-side point-light delta caustics and are restricted diagnostics only. GGX finite lobes do not have this delta support exclusion.',
            'frozen_ecro_material_scope':'Only original six frozen states; pure GGX and mirror not supported. S1 comparison retains panel state but checker 0.3/0.8 is material-distribution OOD; no architecture verdict.',
            'model_S1_OOD_diagnostic':model_metrics}
    jd(result,out/'metrics_nearmirror93e.json')
    np.savez_compressed(out/'edge_profiles.npz',distance_pixels=bins,**profiles,**{n+'_sem':v for n,v in profile_noise.items()})
    # Editable standalone reviewer page: original 1024 images plus full sheets.
    html=['<!doctype html><meta charset="utf-8"><title>WL93E Near-Mirror Reference</title>',
          '<style>body{font:17px system-ui;background:#131820;color:#eee;margin:30px}img{max-width:100%}a{color:#80d6ff}.row{display:flex;gap:12px}.row figure{margin:0;flex:1}h2{margin-top:40px}</style>',
          '<h1>WL93E — Near-mirror checker reflection</h1>',
          '<p>Same CK2 geometry, camera, lighting and checker. 1024×1024, 4096 spp per condition. Shared exposure; no denoise. Architecture verdict: NOT APPLICABLE.</p>',
          '<p>The exactly-two-event panel reflection (e2) is qualified for edge/contrast conclusions. Full radiance and higher-order images remain noise-unconverged diagnostics; near-mirror h has 48.7% pixel RMS SEM from rare bright paths.</p>',
          '<p>S1 is the original multilobe frozen ER material. The neutral GGX group changes alpha only; the ideal mirror is a true delta event. Frozen ECRO does not support the new material states.</p>',
          '<p>Ideal mirror scope: its exactly-two-event panel reflection is the qualified edge control. Its full and higher-order images omit light-side point-light delta caustics and are restricted diagnostics, not complete ground truth.</p>',
          '<div class="row">']
    for n,label in zip(NAMES,LABELS):html.append(f'<figure><a href="{n}_full_1024.png"><img src="{n}_full_1024.png"></a><figcaption>{label}</figcaption></figure>')
    html.append('</div>')
    for filename,label in [('figNM5_qualified_low_and_e2.png','Qualified checker reflection: LOW context and e2'),('figNM0_full_and_roi.png','Full frames and reflection crops / noise-unconverged diagnostics'),('figNM1_path_contributions.png','Separated reflection and higher-order contributions'),('figNM2_edges_and_contrast.png','Ideal mirror edge comparison'),('figNM3_correspondence.png','Direct versus reflected texture'),('figNM4_frozen_ecro_S1_only.png','Frozen ECRO: original S1 only / checker OOD')]:
        if (out/filename).exists():html.append(f'<h2>{label}</h2><a href="{filename}"><img src="{filename}"></a>')
    html.append('<p><a href="metrics_nearmirror93e.json">Machine-readable measurements</a></p>')
    (out/'index.html').write_text('\n'.join(html),encoding='utf8')
    jd({'artifacts':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest_nearmirror93e.json'],
        'provenance':'render93e.py and viz93e.py; raw inputs ../results/renders; full provenance recorded in execution_manifest.json',
        'architecture_verdict':'NOT APPLICABLE'},out/'manifest_nearmirror93e.json')
    print(json.dumps({'qualification':gates,'metrics':{n:{k:v for k,v in metric[n].items() if k in ['contrast_retention_vs_ideal','edge_10_90_width_pixels','profile_sem_max']} for n in NAMES}}),flush=True)


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],'--preview' in sys.argv)
