"""Complete-population precision qualification and isolated partial label bank; no fitting."""
import argparse,json,time,sys,resource
from pathlib import Path
import torch,numpy as np
from common65 import CFG,sha,write_new,save_new
from moments65 import aggregate,variance,norm,ratio,loss_scale

def rows3(values,**meta):
    return [dict(**meta,channel='RGB'[c],**{k:float(v[c]) for k,v in values.items()}) for c in range(3)]
def summarized(first,second):
    mu=first.mean(1);v=variance(first,second,CFG['N_per_half']).sum(1)/16
    assert float(v.min())>=-1e-20
    return dict(mean=mu,var=v.clamp(min=0),rep=torch.stack([first[:,:2].mean(1),first[:,2:].mean(1)]))
def change(ref,kind):
    if kind=='pair':
        d=ref['mean'][::2]-ref['mean'][1::2];v=ref['var'][::2]+ref['var'][1::2];rep=ref['rep'][:,::2]-ref['rep'][:,1::2]
    else:raise ValueError(kind)
    noise=(rep[0]-rep[1])/(2**.5);se=v.sum((1,2)).sqrt();sig=norm(d)
    return d,dict(SNR=sig/norm(noise).clamp(min=1e-300),mean_SE_ratio=se/sig.clamp(min=1e-300),signal_norm=sig,SE_norm=se)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);a=ap.parse_args();r=a.root;orig=a.original
    assert not (r/'phaseA_status.json').exists();complete=json.loads((r/'reference_complete.json').read_text());assert complete['packets']==832 and complete['photons']==26624000000 and complete['physical']
    receipts={(v['pose'],v['field'],v['replica'],v['half']):v for v in complete['records']};torch.set_num_threads(8);torch.set_grad_enabled(False);t0=time.perf_counter();absolute=[];pairs=[];deltas=[];discrep=[];geometry=[];labels=[];gates=CFG['gates'];rest=None
    for pose in CFG['poses']:
        g=torch.load(orig/'data/geometry'/f'{pose}.pt');parent=g['parent'];allfirst={True:[],False:[]};allsecond={True:[],False:[]};means=[];variances=[]
        for field in CFG['fields']:
            fieldfirst={True:[],False:[]};fieldsecond={True:[],False:[]};mean=None;var=None
            for rep in range(2):
                for half in range(2):
                    p=r/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt';receipt=receipts[pose,field,rep,half];assert sha(p)==receipt['sha256'];packet=torch.load(p,map_location='cpu');assert all(packet['observer_checks'].values())
                    f,s=packet['first64'],packet['second'];mean=f/4 if mean is None else mean+f/4;v=variance(f,s,CFG['N_per_half'])/16;var=v if var is None else var+v
                    for coarse in (True,False):fieldfirst[coarse].append(aggregate(f,parent,coarse)[0]);fieldsecond[coarse].append(aggregate(s,parent,coarse)[0])
                    del packet,f,s,v
            means.append(mean.float()[0]);variances.append(var.float()[0])
            for coarse in (True,False):allfirst[coarse].append(torch.stack(fieldfirst[coarse]));allsecond[coarse].append(torch.stack(fieldsecond[coarse]))
        bank=torch.stack(means);vv=torch.stack(variances);assert bank.shape==(16,g['P'],64,3) and torch.isfinite(bank).all() and (bank>=0).all()
        lp=r/'labels'/f'{pose}.pt';vp=r/'label_variance'/f'{pose}.pt';save_new(lp,dict(fields=CFG['fields'],response=bank,split='train',geometry=pose,source_sha256=sha(orig/'data/training_targets'/f'{pose}_r0.pt'),original_field_sha256=json.loads((r/'preflight.json').read_text())['all96_train_field_sha256']));save_new(vp,dict(fields=CFG['fields'],variance_of_final_mean=vv,diagonal_only=True))
        labels.append(dict(pose=pose,path=str(lp),sha256=sha(lp),variance_path=str(vp),variance_sha256=sha(vp),bytes=lp.stat().st_size+vp.stat().st_size));del means,variances,bank,vv,mean,var
        refs={coarse:summarized(torch.stack(allfirst[coarse]),torch.stack(allsecond[coarse])) for coarse in (True,False)}
        reduced=r/'reduced'/f'{pose}.npz';reduced.parent.mkdir(exist_ok=True);assert not reduced.exists();np.savez_compressed(reduced,coarse_first=torch.stack(allfirst[True]).numpy(),coarse_second=torch.stack(allsecond[True]).numpy())
        old=torch.load(orig/'data/training_targets'/f'{pose}_r0.pt')['train'][CFG['fields']].double();oldagg={c:aggregate(old,parent,c) for c in (True,False)};del old
        if pose=='rest':
            rest=dict(refs=refs,old=oldagg)
            oldfull=torch.load(orig.parent/'rest_wl57/data/targets/rest_r0.pt')['train'][CFG['fields']].double();assert torch.equal(aggregate(oldfull,parent),oldagg[True]);del oldfull
            oldr1=torch.load(orig.parent/'rest_wl57/data/targets/rest_r1.pt')['train'][CFG['fields']].double();r1=aggregate(oldr1,parent);del oldr1
            old_pair_noise=((oldagg[True][::2]-oldagg[True][1::2])-(r1[::2]-r1[1::2]))/(2**.5)
            new_pair_var=refs[True]['var'][::2]+refs[True]['var'][1::2]
            rest_gain=new_pair_var.sum((0,1,2)).sqrt()/norm(old_pair_noise.reshape(-1,16,3)).clamp(min=1e-300)
        for coarse,label in ((True,'coarse'),(False,'fine16')):
            q=refs[coarse];mu=q['mean'];se=q['var'].sum((1,2)).sqrt();noise=(q['rep'][0]-q['rep'][1])/(2**.5);ar=se/norm(mu).clamp(min=1e-300)
            _,pc=change(q,'pair')
            for fi,field in enumerate(CFG['fields']):
                absolute.extend(rows3(dict(mean_SE_ratio=ar[fi],single_replica_noise_ratio=ratio(noise,mu)[fi]),pose=pose,field=field,resolution=label))
                discrep.extend(rows3(dict(relative_L2=ratio(oldagg[coarse]-mu,mu)[fi]),pose=pose,field_or_pair=field,kind='absolute',resolution=label))
            oldpair=oldagg[coarse][::2]-oldagg[coarse][1::2];newpair=mu[::2]-mu[1::2]
            for pi in range(8):
                pairs.extend(rows3({k:v[pi] for k,v in pc.items()},pose=pose,pair=pi,field_a=80+2*pi,field_b=81+2*pi,resolution=label))
                discrep.extend(rows3(dict(relative_L2=ratio(oldpair-newpair,newpair)[pi]),pose=pose,field_or_pair=pi,kind='directional',resolution=label))
            if pose!='rest':
                rq=rest['refs'][coarse];d=mu-rq['mean'];v=q['var']+rq['var'];dr=q['rep']-rq['rep'];dn=(dr[0]-dr[1])/(2**.5);sn=norm(d)/norm(dn).clamp(min=1e-300);se=v.sum((1,2)).sqrt();sr=se/norm(d).clamp(min=1e-300);nr=se/loss_scale(d,rq['mean']).clamp(min=1e-300);od=oldagg[coarse]-rest['old'][coarse]
                for fi,field in enumerate(CFG['fields']):
                    deltas.extend(rows3(dict(SNR=sn[fi],mean_SE_ratio=sr[fi],loss_scale_SE=nr[fi],signal_norm=norm(d)[fi],SE_norm=se[fi]),pose=pose,field=field,resolution=label))
                    discrep.extend(rows3(dict(relative_L2=ratio(od-d,d)[fi]),pose=pose,field_or_pair=field,kind='deformation',resolution=label))
                pooled_snr=norm(d.reshape(-1,16,3))/norm(dn.reshape(-1,16,3)).clamp(min=1e-300)
                geometry.extend(rows3(dict(pooled_delta_SNR=pooled_snr,pooled_delta_SE_ratio=v.sum((0,1,2)).sqrt()/norm(d.reshape(-1,16,3)).clamp(min=1e-300)),pose=pose,resolution=label))
        del allfirst,allsecond,oldagg
        if pose!='rest':del refs
        print('ANALYZED',pose,flush=True)
    assert len([v for v in absolute if v['resolution']=='coarse'])==624 and len([v for v in pairs if v['resolution']=='coarse'])==312 and len([v for v in deltas if v['resolution']=='coarse'])==576
    ac=[v for v in absolute if v['resolution']=='coarse'];af=[v for v in absolute if v['resolution']=='fine16'];pc=[v for v in pairs if v['resolution']=='coarse'];dc=[v for v in deltas if v['resolution']=='coarse']
    checks=dict(absolute_coarse=all(v['mean_SE_ratio']<=gates['absolute_coarse_mean_SE_ratio_max'] for v in ac),absolute_fine16=all(v['mean_SE_ratio']<=gates['absolute_fine16_mean_SE_ratio_max'] for v in af),directional_coarse=all(v['SNR']>=3 and v['mean_SE_ratio']<=.10 for v in pc),delta_original_loss_precision=all(v['loss_scale_SE']<=.25 for v in dc),rest_precision_gain=bool((rest_gain<=.5).all()))
    status='READY_FOR_CONTROLLED_TRAINING' if all(checks.values()) else 'STOP_INSUFFICIENT_PRECISION'
    summary=dict(status=status,phase='A',whole_batch_complete=False,phase_B_authorized=False,checks=checks,coverage=dict(actions=208,packets=832,photons=26624000000,geometries=13,fields=16,pairs=8),qualified_direction_cells=sum(v['SNR']>=3 and v['mean_SE_ratio']<=.1 for v in pc),weak_delta_cells=sum(v['SNR']<3 for v in dc),delta_loss_precision_failed_cells=sum(v['loss_scale_SE']>.25 for v in dc),absolute_coarse_failed_cells=sum(v['mean_SE_ratio']>.01 for v in ac),absolute_fine16_failed_cells=sum(v['mean_SE_ratio']>.05 for v in af),rest_pooled_new_SE_over_old_single_noise_RGB=rest_gain.tolist(),max_coarse_absolute_SE_ratio=max(v['mean_SE_ratio'] for v in ac),max_fine16_absolute_SE_ratio=max(v['mean_SE_ratio'] for v in af),minimum_direction_SNR=min(v['SNR'] for v in pc),maximum_direction_SE_ratio=max(v['mean_SE_ratio'] for v in pc),maximum_delta_loss_scale_SE=max(v['loss_scale_SE'] for v in dc),analysis_seconds=time.perf_counter()-t0,CPU_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,note='Readiness is reference precision only. No model quality or architecture success. No Phase B before explicit user decision; complete population retained; weak delta detectability separate from original loss-scale uncertainty.')
    write_new(r/'analysis.json',dict(summary=summary,absolute=absolute,directional=pairs,deformation=deltas,geometry_delta=geometry,historical_discrepancy=discrep,covariance='Independent fields/geometry streams: pair/delta variance adds. Same rest reused across current deltas => Cov(delta_g,delta_h)=Var(rest); their difference cancels rest. Output bins mutually exclusive: aggregate moments BEFORE variance. RGB channels correlated; no iid-cell significance/CI claim. Original deformed-label variance unknown; discrepancies not z-scores.'))
    write_new(r/'labels_manifest.json',dict(status=status,qualified_for_future_controlled_intervention=all(checks.values()),original96_distribution_unchanged=True,replacement_fields_only=CFG['fields'],labels=labels,training_authorized=False));write_new(r/'phaseA_status.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main();sys.stdout.flush();__import__('os')._exit(0)
