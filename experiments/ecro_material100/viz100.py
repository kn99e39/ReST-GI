"""Held-out ECRO material figures: physical reference, prediction and error maps.

Only reads final checkpoints/renderer arrays. Does not tune or select a model.
python viz100.py ROOT evaluation/100
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def jd(x,path):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean3(x):return np.asarray(x,np.float64).mean(-1)
def tm(x,key):
    y=np.maximum(x,0)*key
    return np.clip(y/(1+y),0,1)**(1/2.2)
def cropbox(mask):
    y,x=np.nonzero(mask);h,w=mask.shape
    return [max(0,int(x.min())-8),max(0,int(y.min())-8),min(w,int(x.max())+9),min(h,int(y.max())+9)]
def crop(box):return np.s_[box[1]:box[3],box[0]:box[2]]


def metrics(p,r,mask):
    pp,rr=p[mask],r[mask];diff=pp-rr
    mse=float(np.mean(diff**2));power=float(np.mean(rr**2));peak=float(np.max(rr))
    return {'relative_rmse':float(np.sqrt(mse/max(power,1e-30))),
            'relative_l1':float(np.sum(np.abs(diff))/max(np.sum(np.abs(rr)),1e-30)),
            'psnr_hdr_peak_db':float(10*np.log10(max(peak*peak,1e-30)/max(mse,1e-30))),
            'psnr_definition':'10 log10(max(reference ROI)^2 / RGB MSE), linear HDR, no clipping',
            'pixels':int(mask.sum())}


def main(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    target_noise=json.loads((root/'dataset_noise.json').read_text()) if (root/'dataset_noise.json').exists() else {}
    summaries={};html=['<!doctype html><meta charset="utf-8"><title>WL100 ECRO Near Mirror and SSS</title>',
        '<style>body{font:17px system-ui;background:#171c24;color:#eee;margin:30px}img{max-width:100%}a{color:#8ddbff}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #617080}</style>',
        '<h1>WL100 — pre-RB-DTO-E ECRO training</h1>',
        '<p>Two independent fixed ECRO OWN fits, 24,000 steps each. Same 735,664 trainable parameters. Fresh physical datasets: near-mirror GGX and real volume SSS. Held-out geometry/light evaluations, not training images.</p>',
        '<p>ECRO predicts the higher-order boundary table. Physical material response and the first two optical-domain interactions remain explicit. SSS internal collisions are grouped into one entry-to-first-exit interaction. This is not a claim that the neural model learns the entire local SSS response.</p>']
    for case,title in [('near','Near-mirror GGX alpha=0.01'),('sss','Subsurface scattering / homogeneous RTE sphere')]:
        training=json.loads((root/case/'training.json').read_text())
        assert training['steps']==24000 and training['params']==735664
        rows=[]
        for index in [0,1,2]:
            base=root/case/'renders';zs=[np.load(base/f'REF_{index}_{h}.npz') for h in [0,1]]
            avg={k:.5*(zs[0][k].astype(np.float64)+zs[1][k].astype(np.float64)) for k in ['e1','e2','h']}
            prediction=np.load(base/f'PRED_{index}.npz')
            ref=avg['e1']+avg['e2']+avg['h'];pred=prediction['e1']+prediction['e2']+prediction['hl']
            maps=np.load(base/f'map_{index}.npz');mask=maps['sid']==(5 if case=='near' else 6)
            fullmask=maps['ok'];box=cropbox(mask)
            fullhalves=[sum(z[k].astype(np.float64) for k in ['e1','e2','h']) for z in zs]
            noise=.5*(fullhalves[0]-fullhalves[1]);nh=.5*(zs[0]['h'].astype(np.float64)-zs[1]['h'].astype(np.float64))
            row={'test_index':index,'mask':mask,'fullmask':fullmask,'box':box,'ref':ref,'pred':pred,
                 'h':avg['h'],'hl':prediction['hl'].astype(np.float64),'noise':noise,'nh':nh,
                 'error':np.mean(np.abs(pred-ref),-1),'herror':np.mean(np.abs(prediction['hl']-avg['h']),-1)}
            row['metrics']={'full':metrics(pred,ref,fullmask),'roi':metrics(pred,ref,mask),
                'higher_order_roi':metrics(row['hl'],row['h'],mask),
                'reference_noise_roi':float(np.sqrt(np.mean(noise[mask]**2))/max(np.sqrt(np.mean(ref[mask]**2)),1e-30)),
                'reference_noise_full':float(np.sqrt(np.mean(noise[fullmask]**2))/max(np.sqrt(np.mean(ref[fullmask]**2)),1e-30)),
                'reference_noise_h_roi':float(np.sqrt(np.mean(nh[mask]**2))/max(np.sqrt(np.mean(avg['h'][mask]**2)),1e-30)),
                'physical_h_fraction_roi':float(avg['h'][mask].sum()/max(ref[mask].sum(),1e-30)),
                'predicted_h_fraction_roi':float(prediction['hl'][mask].sum()/max(pred[mask].sum(),1e-30)),
                'roi_box_xyxy':box,
                'reference_used_at_inference':False}
            # Post-hoc descriptive diagnostic after observing heavy-tailed
            # reference noise. Independent halves cancel reference variance
            # in expectation; prediction readout variance remains. No gate or
            # exact-fidelity claim is based on this estimate.
            for tag,pp,aa,bb in [('full',pred,fullhalves[0],fullhalves[1]),
                                 ('h',row['hl'],zs[0]['h'],zs[1]['h'])]:
                numerator=float(np.mean(((pp-aa)*(pp-bb))[mask]))
                denominator=float(np.mean((aa.astype(np.float64)*bb)[mask]))
                row['metrics']['posthoc_reference_noise_corrected_'+tag+'_roi']={
                    'relative_rmse':float(np.sqrt(numerator/denominator)) if numerator>=0 and denominator>0 else None,
                    'cross_mse':numerator,'reference_cross_power':denominator,
                    'scope':'Post-hoc descriptive cross-half estimate; reference variance cancels in expectation. Prediction readout noise remains; no confidence interval or architecture verdict.'}
            rows.append(row)
        # A SINGLE exposure and error scale per dataset, shared across every
        # view, reference/prediction and full/higher-order row. All clipping
        # applies to visualisation only, never the metrics above.
        positives=np.concatenate([mean3(r['ref'])[r['fullmask']] for r in rows])
        key=.5/float(np.median(positives[positives>0]))
        escale=float(np.quantile(np.concatenate([r['error'][r['fullmask']] for r in rows]+[r['herror'][r['fullmask']] for r in rows]),.995))
        escale=max(escale,1e-8)
        for r in rows:
            index=r['test_index'];sl=crop(r['box'])
            fig,ax=plt.subplots(2,6,figsize=(24,10))
            values=[r['ref'],r['pred'],r['error'],r['h'],r['hl'],r['herror']]
            labels=['Physical reference / full','ECRO physical LOW + learned hl','Absolute error / full','Reference higher order h','ECRO predicted higher order hl','Absolute error / higher order']
            for j,(v,label) in enumerate(zip(values,labels)):
                for i,sub in enumerate([np.s_[:,:],sl]):
                    if j in [2,5]:im=ax[i,j].imshow(v[sub],vmin=0,vmax=escale,cmap='magma',interpolation='nearest')
                    else:ax[i,j].imshow(tm(v[sub],key),interpolation='nearest')
                    ax[i,j].axis('off')
                ax[0,j].set_title(label,fontsize=11)
            ax[0,0].contour(r['mask'],levels=[.5],colors=['cyan'],linewidths=.7)
            mm=r['metrics'];fig.suptitle(f'WL100 | {title} | held-out test {index} | 512 x 512 | ECRO OWN, 24,000 steps\nROI rel RMSE {mm["roi"]["relative_rmse"]:.3f}; reference-half noise {mm["reference_noise_roi"]:.3f}; h fraction {mm["physical_h_fraction_roi"]:.3f}. Same exposure / error scale for all tests in this dataset.',fontsize=14)
            # Leave the final colourbar margin fixed; tight_layout does not
            # understand axes spanning multiple rows and would overlap it.
            fig.subplots_adjust(left=.01,right=.92,bottom=.02,top=.88,wspace=.05,hspace=.04)
            cax=fig.add_axes([.94,.15,.012,.65]);cb=fig.colorbar(im,cax=cax)
            cb.set_label('mean |RGB error|, linear HDR; shared 99.5% display cap')
            name=f'{case}_test{index}_comparison.png';fig.savefig(out/name,dpi=170);plt.close(fig)
            for kind,v in [('reference',r['ref']),('prediction',r['pred'])]:
                Image.fromarray(np.uint8(np.round(tm(v,key)*255))).save(out/f'{case}_test{index}_{kind}_512.png')
            norm=np.clip(r['error']/escale,0,1);rgb=plt.get_cmap('magma')(norm)[...,:3]
            Image.fromarray(np.uint8(np.round(rgb*255))).save(out/f'{case}_test{index}_error_512.png')
        fig,ax=plt.subplots(3,3,figsize=(15,15))
        for i,r in enumerate(rows):
            ax[i,0].imshow(tm(r['ref'],key));ax[i,1].imshow(tm(r['pred'],key));im=ax[i,2].imshow(r['error'],vmin=0,vmax=escale,cmap='magma')
            ax[i,0].set_ylabel('Held-out '+str(r['test_index']))
            ax[i,1].set_xlabel('ROI rel RMSE '+format(r['metrics']['roi']['relative_rmse'],'.3f'))
        for j,label in enumerate(['Physical reference','Trained ECRO / LOW + hl','Absolute error / linear HDR']):ax[0,j].set_title(label)
        for a in ax.flat:a.set_xticks([]);a.set_yticks([])
        fig.suptitle('WL100 — '+title+'\nIndependent fixed model; same exposure and error-map scale across all held-out cases',fontsize=16)
        fig.subplots_adjust(top=.91,left=.02,right=.92,bottom=.03,hspace=.07,wspace=.03)
        cax=fig.add_axes([.94,.15,.012,.65]);fig.colorbar(im,cax=cax,label='mean |RGB error|')
        fig.savefig(out/f'{case}_overview.png',dpi=170);plt.close(fig)
        fig,ax=plt.subplots(3,2,figsize=(10,15))
        for i,r in enumerate(rows):
            ax[i,0].imshow(r['error'],vmin=0,vmax=escale,cmap='magma')
            im=ax[i,1].imshow(np.mean(np.abs(r['noise']),-1),vmin=0,vmax=escale,cmap='magma')
            ax[i,0].set_ylabel('Held-out '+str(r['test_index']))
        ax[0,0].set_title('Prediction vs reference / abs error');ax[0,1].set_title('Reference half-difference / 2')
        for a in ax.flat:a.set_xticks([]);a.set_yticks([])
        fig.suptitle(title+' / noise control\nSame absolute HDR scale. Half-difference is a noise estimate, not a per-pixel confidence interval.',fontsize=13)
        fig.subplots_adjust(top=.91,left=.02,right=.90,bottom=.03,hspace=.03,wspace=.03)
        cax=fig.add_axes([.93,.15,.016,.65]);fig.colorbar(im,cax=cax,label='mean absolute RGB / HDR')
        fig.savefig(out/f'{case}_noise_control.png',dpi=150);plt.close(fig)
        summaries[case]={'training':training,'target_noise':target_noise.get(case),
                         'exposure_key':key,'absolute_error_display_max':escale,
                         'metrics':[dict(test_index=r['test_index'],**r['metrics']) for r in rows],
                         'reference_spp':8192,'prediction_spp':4096,'tone_map':'(key*x/(1+key*x))**(1/2.2)',
                         'error_definition':'mean absolute RGB difference in linear HDR; shared cap across this dataset; metrics never clipped',
                         'reference_noise_definition':'independent half difference / 2 estimates SEM of averaged 8192-spp reference; prediction readout has separate MC noise',
                         'scope':'separate fixed-material ECRO dataset, unchanged architecture, one seed; explicit physical material/LOW, learned global higher order only'}
        html.extend([f'<h2>{title}</h2>',f'<p>{training["train_items"]} train geometries, {training["val_items"]} validation, {training["test_items"]} test; 24,000 steps; training {training["train_s"]/60:.1f} minutes (actual possibly contended wall).</p>',
                     f'<a href="{case}_overview.png"><img src="{case}_overview.png"></a>'])
        if case in target_noise:
            n=target_noise[case]['test'];html.append(f'<p>Held-out boundary-field target noise: relative L1 {n["independent_half_noise_rel_l1"]:.3f}, relative RMS {n["independent_half_noise_relative_rms_of_mean"]:.3f}. Near-mirror point-light targets have rare-path heavy tails; those field errors do not isolate neural representation failure. Image reference-half noise is reported separately in each figure.</p>')
        html.append(f'<h3>Reference noise control</h3><a href="{case}_noise_control.png"><img src="{case}_noise_control.png"></a>')
        if case=='sss' and (root/'sss/opaque_control.npz').exists():
            direct=.5*(np.load(root/'sss/renders/REF_0_0.npz')['e1'].astype(float)+np.load(root/'sss/renders/REF_0_1.npz')['e1'])
            opaque=np.load(root/'sss/opaque_control.npz')['e1']
            sphere=rows[0]['mask'];sl=crop(rows[0]['box'])
            shadow=sphere & (mean3(opaque)<1e-8)
            fig,ax=plt.subplots(1,3,figsize=(15,6))
            for a,v,label in zip(ax,[opaque,direct,np.maximum(direct-opaque,0)],
                    ['Opaque Lambert / direct source','Actual SSS / direct source','Positive SSS minus opaque']):
                a.imshow(tm(v[sl],key));a.set_title(label);a.axis('off')
            fig.suptitle('Post-hoc PHYSICAL control, same geometry / light / camera / exposure\nThe SSS source response reaches the camera-facing shadowed hemisphere. Not a model or training comparison. RGB parameters matched; emergent reflectance is not.',fontsize=11)
            fig.tight_layout(rect=(0,0,1,.86));fig.savefig(out/'sss_physical_nonlocal_control.png',dpi=160);plt.close(fig)
            summaries[case]['posthoc_physical_control']={'camera_shadow_pixels':int(shadow.sum()),
                'sss_mean_direct_radiance_on_opaque_shadow':float(np.mean(direct[shadow])),
                'opaque_mean_direct_radiance_on_same_pixels':float(np.mean(opaque[shadow])),
                'scope':'physical-only, same RGB parameter not matched emergent reflectance; no learned evidence'}
            html.append('<h3>Physical SSS versus opaque control</h3><p>Post-hoc material-physics visual control, not training/model evidence.</p><img src="sss_physical_nonlocal_control.png">')
        for r in rows:
            i=r['test_index'];html.append(f'<h3>Held-out test {i}: full / higher-order and ROI</h3><a href="{case}_test{i}_comparison.png"><img src="{case}_test{i}_comparison.png"></a>')
    # Both curves are measured; no restart / best-test checkpoint selection.
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for a,case in zip(ax,['near','sss']):
        tr=summaries[case]['training'];log=tr['log'];xs=[r['step'] for r in log]
        for j,label in enumerate(['whole','single','global']):a.semilogy(xs,[r['losses'][j] for r in log],label=label)
        a.set(title=case+' / OWN component objectives',xlabel='step',ylabel='relative squared loss');a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    html.append('<h2>Fixed-run learning curves</h2><img src="training_curves.png"><p><a href="metrics100.json">Full numerical results</a> · <a href="manifest100.json">Provenance and hashes</a></p>')
    (out/'index.html').write_text('\n'.join(html),encoding='utf8')
    jd({'batch':'WL100','cases':summaries,'architecture_adoption':'NOT APPLICABLE',
        'preflight':json.loads((root/'preflight/preflight.json').read_text()),
        'limits':['one seed per material dataset','bounded new scene/light family, no broad generalisation or dynamic/economic claim',
                  'local SSS physics and LOW explicitly traced; learned ECRO owns higher-order table',
                  'six optical-domain interactions; native SSS isolated-sphere agreement only within 5% bound',
                  'sharp point-light reference outliers/noise must be considered with error maps']},out/'metrics100.json')
    freeze=json.loads((root/'freeze_manifest.json').read_text());repo=Path(__file__).resolve().parent.parent
    for f,h in freeze['source_hashes'].items():assert sha(repo/f)==h,'FROZEN_SOURCE_CHANGED '+f
    jd({'batch':'WL100','base_commit':freeze['base_commit'],'freeze':freeze,
        'chain_status':(root/'chain.status').read_text(),
        'physical_input_material_asset':{'path':str((repo/'ecro86/data/mats86.npz').resolve()),
            'sha256':sha(repo/'ecro86/data/mats86.npz'),
            'local_snapshot':'results/ecro_material100/input_mats86.npz'},
        'model_checkpoints':{case:{'path':str((root/case/'model.pt').resolve()),'sha256':sha(root/case/'model.pt')} for case in ['near','sss']},
        'generated_data':{case:{sp:json.loads((root/case/f'gen_{sp}.json').read_text()) for sp in ['train','val','test']} for case in ['near','sss']},
        'final_scripts':{p.name:sha(p) for p in Path(__file__).resolve().parent.iterdir() if p.is_file()},
        'images':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest100.json']},out/'manifest100.json')
    print('FIGURES_COMPLETE',str(out),json.dumps({case:{'train_s':s['training']['train_s'],'roi_rmse':[r['roi']['relative_rmse'] for r in s['metrics']]} for case,s in summaries.items()}),flush=True)


if __name__=='__main__':main(*sys.argv[1:3])
