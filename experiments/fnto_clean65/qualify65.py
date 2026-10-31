"""Gate 2: complete-population re-qualification of the combined Phase A + top-up bank; no fitting."""
import argparse,json,time,sys,resource
from pathlib import Path
import torch,numpy as np
from ext65 import CFG,CFG1,TOP,PHASE_A,ORIG,HOME,sha,write_new,save_new,packets,Combined
from moments65 import aggregate,norm,ratio,loss_scale
from analyze65 import rows3,change

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;orig=ORIG
    assert not (r/'qualification_status.json').exists()
    top=json.loads((r/'topup_complete.json').read_text());assert top['packets']==TOP['total_packets'] and top['photons']==TOP['total_photons'] and top['physical'] and top['estimator_source_identical_to_phaseA']
    receipts={('phaseA',v['pose'],v['field'],v['replica'],v['half']):v for v in json.loads((PHASE_A/'reference_complete.json').read_text())['records']}
    receipts.update({('topup',v['pose'],v['field'],v['replica'],v['k']):v for v in top['records']})
    phaseA=json.loads((PHASE_A/'analysis.json').read_text())
    torch.set_num_threads(8);torch.set_grad_enabled(False);t0=time.perf_counter()
    absolute=[];pairs=[];deltas=[];discrep=[];geometry=[];labels=[];consistency=[];bank_change=[];photons=dict(phaseA=0,topup=0);gates=CFG['gates'];rest=None
    for pose in CFG1['poses']:
        g=torch.load(orig/'data/geometry'/f'{pose}.pt');parent=g['parent'];refs={True:[],False:[]};means=[];variances=[]
        bank_a=torch.load(PHASE_A/'labels'/f'{pose}.pt');var_a=torch.load(PHASE_A/'label_variance'/f'{pose}.pt');assert bank_a['fields']==CFG1['fields']
        for fi,field in enumerate(CFG1['fields']):
            plist=packets(pose,field,root_t=r);full=Combined([n for *_,n,_ in plist],[rep for _,rep,*_ in plist])
            agg={c:Combined([n for *_,n,_ in plist],[rep for _,rep,*_ in plist]) for c in (True,False)}
            split={s:Combined([n for k,_,_,n,_ in plist if k==s],[rep for k,rep,*_ in plist if k==s]) for s in ('phaseA','topup') if any(k==s for k,*_ in plist)}
            for kind,rep,p,N,seed in plist:
                key=(kind,pose,field,rep,int(p.stem[-1]));receipt=receipts[key];assert sha(p)==receipt['sha256'] and receipt['seed']==seed and receipt['N']==N,p
                packet=torch.load(p,map_location='cpu');assert all(packet['observer_checks'].values()) and packet['N']==N
                f,s=packet['first64'],packet['second'];full.add(f,s);photons[kind]+=N
                fa,sa=aggregate(f,parent,True)[0],aggregate(s,parent,True)[0]
                for c in (True,False):agg[c].add(*((fa,sa) if c else (aggregate(f,parent,False)[0],aggregate(s,parent,False)[0])))
                split[kind].add(fa,sa);del packet,f,s
            fr=full.result();mean=fr['mean'].float()[0];var=fr['var'].float()[0]
            if field in TOP['fields']:
                bank_change.append(dict(pose=pose,field=field,**{f'relative_L2_{c}':float(v) for c,v in zip('RGB',ratio((mean.double()-bank_a['response'][fi].double())[None],mean.double()[None])[0])}))
                sa,st=split['phaseA'].result(),split['topup'].result();d=(sa['mean']-st['mean']).square().sum((0,1));e=(sa['var']+st['var']).sum((0,1))
                consistency.extend(dict(pose=pose,field=field,channel='RGB'[c],D=float(d[c]/e[c]),phaseA_SE_norm=float(sa['var'].sum((0,1))[c].sqrt()),topup_SE_norm=float(st['var'].sum((0,1))[c].sqrt())) for c in range(3))
            else:
                # Unchanged fields must reproduce the Phase-A bank bitwise.
                assert torch.equal(mean,bank_a['response'][fi]) and torch.equal(var,var_a['variance_of_final_mean'][fi]),(pose,field)
            means.append(mean);variances.append(var)
            for c in (True,False):refs[c].append(agg[c].result())
        bank=torch.stack(means);vv=torch.stack(variances);assert bank.shape==(16,g['P'],64,3) and torch.isfinite(bank).all() and (bank>=0).all()
        lp=r/'labels_v2'/f'{pose}.pt';vp=r/'label_variance_v2'/f'{pose}.pt'
        save_new(lp,dict(fields=CFG1['fields'],response=bank,split='train',geometry=pose,photons_per_field=[sum(n for *_,n,_ in packets(pose,f,root_t=r)) for f in CFG1['fields']],phaseA_bank_sha256=sha(PHASE_A/'labels'/f'{pose}.pt'),source_sha256=sha(orig/'data/training_targets'/f'{pose}_r0.pt')))
        save_new(vp,dict(fields=CFG1['fields'],variance_of_final_mean=vv,diagonal_only=True))
        labels.append(dict(pose=pose,path=str(lp),sha256=sha(lp),variance_path=str(vp),variance_sha256=sha(vp),bytes=lp.stat().st_size+vp.stat().st_size));del means,variances,bank,vv
        q2={c:dict(mean=torch.stack([v['mean'] for v in refs[c]]),var=torch.stack([v['var'] for v in refs[c]]),rep=torch.stack([v['rep'] for v in refs[c]],1)) for c in (True,False)}
        old=torch.load(orig/'data/training_targets'/f'{pose}_r0.pt')['train'][CFG1['fields']].double();oldagg={c:aggregate(old,parent,c) for c in (True,False)};del old
        if pose=='rest':
            rest=dict(refs=q2,old=oldagg)
            oldr1=torch.load(HOME/'rest_wl57/data/targets/rest_r1.pt')['train'][CFG1['fields']].double();r1=aggregate(oldr1,parent);del oldr1
            old_pair_noise=((oldagg[True][::2]-oldagg[True][1::2])-(r1[::2]-r1[1::2]))/(2**.5)
            rest_gain=(q2[True]['var'][::2]+q2[True]['var'][1::2]).sum((0,1,2)).sqrt()/norm(old_pair_noise.reshape(-1,16,3)).clamp(min=1e-300)
        for coarse,label in ((True,'coarse'),(False,'fine16')):
            q=q2[coarse];mu=q['mean'];se=q['var'].sum((1,2)).sqrt();noise=(q['rep'][0]-q['rep'][1])/(2**.5);ar=se/norm(mu).clamp(min=1e-300)
            _,pc=change(q,'pair')
            for fi,field in enumerate(CFG1['fields']):
                absolute.extend(rows3(dict(mean_SE_ratio=ar[fi],single_replica_noise_ratio=ratio(noise,mu)[fi]),pose=pose,field=field,resolution=label,photons=photons_of(pose,field,r)))
                discrep.extend(rows3(dict(relative_L2=ratio(oldagg[coarse]-mu,mu)[fi]),pose=pose,field_or_pair=field,kind='absolute',resolution=label))
            oldpair=oldagg[coarse][::2]-oldagg[coarse][1::2];newpair=mu[::2]-mu[1::2]
            for pi in range(8):
                pairs.extend(rows3({k:v[pi] for k,v in pc.items()},pose=pose,pair=pi,field_a=80+2*pi,field_b=81+2*pi,resolution=label))
                discrep.extend(rows3(dict(relative_L2=ratio(oldpair-newpair,newpair)[pi]),pose=pose,field_or_pair=pi,kind='directional',resolution=label))
            if pose!='rest':
                rq=rest['refs'][coarse];d=mu-rq['mean'];v=q['var']+rq['var'];dr=q['rep']-rq['rep'];dn=(dr[0]-dr[1])/(2**.5);sn=norm(d)/norm(dn).clamp(min=1e-300);se=v.sum((1,2)).sqrt();sr=se/norm(d).clamp(min=1e-300);nr=se/loss_scale(d,rq['mean']).clamp(min=1e-300);od=oldagg[coarse]-rest['old'][coarse]
                for fi,field in enumerate(CFG1['fields']):
                    deltas.extend(rows3(dict(SNR=sn[fi],mean_SE_ratio=sr[fi],loss_scale_SE=nr[fi],signal_norm=norm(d)[fi],SE_norm=se[fi]),pose=pose,field=field,resolution=label,topped_up=field in TOP['fields']))
                    discrep.extend(rows3(dict(relative_L2=ratio(od-d,d)[fi]),pose=pose,field_or_pair=field,kind='deformation',resolution=label))
                geometry.extend(rows3(dict(pooled_delta_SNR=norm(d.reshape(-1,16,3))/norm(dn.reshape(-1,16,3)).clamp(min=1e-300),pooled_delta_SE_ratio=v.sum((0,1,2)).sqrt()/norm(d.reshape(-1,16,3)).clamp(min=1e-300)),pose=pose,resolution=label))
        del refs,q2,oldagg
        print('QUALIFIED',pose,round(time.perf_counter()-t0,1),flush=True)
    assert photons['phaseA']==CFG['phase_A']['photons'] and photons['topup']==TOP['total_photons']
    # Unchanged-field regression against the preserved Phase-A analysis (same estimator, same packets).
    drift=regression(phaseA,dict(absolute=absolute,directional=pairs,deformation=deltas))
    ac=[v for v in absolute if v['resolution']=='coarse'];af=[v for v in absolute if v['resolution']=='fine16'];pc=[v for v in pairs if v['resolution']=='coarse'];dc=[v for v in deltas if v['resolution']=='coarse']
    assert len(ac)==624 and len(pc)==312 and len(dc)==576
    Dmed=float(np.median([v['D'] for v in consistency])) if consistency else None
    checks=dict(absolute_coarse=all(v['mean_SE_ratio']<=gates['absolute_coarse_mean_SE_ratio_max'] for v in ac),absolute_fine16=all(v['mean_SE_ratio']<=gates['absolute_fine16_mean_SE_ratio_max'] for v in af),
        directional_coarse=all(v['SNR']>=gates['direction_coarse_SNR_min'] and v['mean_SE_ratio']<=gates['direction_coarse_mean_SE_ratio_max'] for v in pc),delta_original_loss_precision=all(v['loss_scale_SE']<=gates['delta_coarse_loss_scale_SE_max'] for v in dc),
        rest_precision_gain=bool((rest_gain<=gates['rest_pooled_contrast_new_SE_over_old_single_noise_max']).all()))
    validity=dict(identity_and_physical=True,semantic_consistency=Dmed is not None and .67<=Dmed<=1.5,unchanged_fields_bitwise_phaseA=True,unchanged_field_analysis_max_relative_drift=drift)
    if not validity['semantic_consistency']:status='STOP_INVALID_REFERENCE'
    else:status='READY_FOR_CONTROLLED_TRAINING' if all(checks.values()) else 'STOP_INSUFFICIENT_PRECISION'
    summary=dict(status=status,checks=checks,validity=validity,semantic_consistency_median_D=Dmed,semantic_consistency_range_D=[min(v['D'] for v in consistency),max(v['D'] for v in consistency)] if consistency else None,
        coverage=dict(actions=208,topup_actions=52,phaseA_photons=photons['phaseA'],topup_photons=photons['topup'],total_photons=photons['phaseA']+photons['topup']),
        qualified_direction_cells=sum(v['SNR']>=3 and v['mean_SE_ratio']<=.1 for v in pc),weak_delta_cells=sum(v['SNR']<3 for v in dc),delta_loss_precision_failed_cells=sum(v['loss_scale_SE']>.25 for v in dc),
        delta_failed_cells_by_field={str(f):sum(v['loss_scale_SE']>.25 for v in dc if v['field']==f) for f in CFG1['fields']},absolute_coarse_failed_cells=sum(v['mean_SE_ratio']>.01 for v in ac),absolute_fine16_failed_cells=sum(v['mean_SE_ratio']>.05 for v in af),
        rest_pooled_new_SE_over_old_single_noise_RGB=rest_gain.tolist(),max_coarse_absolute_SE_ratio=max(v['mean_SE_ratio'] for v in ac),max_fine16_absolute_SE_ratio=max(v['mean_SE_ratio'] for v in af),
        minimum_direction_SNR=min(v['SNR'] for v in pc),maximum_direction_SE_ratio=max(v['mean_SE_ratio'] for v in pc),maximum_delta_loss_scale_SE=max(v['loss_scale_SE'] for v in dc),
        maximum_delta_loss_scale_SE_topped_up=max((v['loss_scale_SE'] for v in dc if v['topped_up']),default=None),maximum_delta_loss_scale_SE_untouched=max((v['loss_scale_SE'] for v in dc if not v['topped_up']),default=None),
        analysis_seconds=time.perf_counter()-t0,CPU_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Reference precision only; no model quality. Phase-A STOP verdict preserved. READY permits only the prospectively fixed A/B in topup_ab_v2.json.')
    write_new(r/'qualification.json',dict(summary=summary,absolute=absolute,directional=pairs,deformation=deltas,geometry_delta=geometry,historical_discrepancy=discrep,phaseA_topup_consistency=consistency,bank_change_vs_phaseA=bank_change,covariance=CFG['combination']['covariance']))
    write_new(r/'labels_v2_manifest.json',dict(status=status,qualified_for_controlled_intervention=status=='READY_FOR_CONTROLLED_TRAINING',original96_distribution_unchanged=True,replacement_fields_only=CFG1['fields'],topped_up_fields=TOP['fields'],labels=labels,phaseA_bank_unchanged=True))
    write_new(r/'qualification_status.json',summary);print(json.dumps(summary,indent=2),flush=True)

def photons_of(pose,field,r):return sum(n for *_,n,_ in packets(pose,field,root_t=r))

def regression(old,new):
    """Maximum relative change of every preserved Phase-A row whose inputs are unchanged."""
    worst={}
    for key,ident,cols,changed in (('absolute',('pose','field','channel','resolution'),('mean_SE_ratio','single_replica_noise_ratio'),lambda v:v['field'] in TOP['fields']),
                                   ('directional',('pose','pair','channel','resolution'),('SNR','mean_SE_ratio','signal_norm','SE_norm'),lambda v:v['field_a'] in TOP['fields'] or v['field_b'] in TOP['fields']),
                                   ('deformation',('pose','field','channel','resolution'),('SNR','mean_SE_ratio','loss_scale_SE','signal_norm','SE_norm'),lambda v:v['field'] in TOP['fields'])):
        ref={tuple(v[k] for k in ident):v for v in old[key]};m=0.;n=0
        for v in new[key]:
            if changed(v):continue
            o=ref[tuple(v[k] for k in ident)];n+=1
            for c in cols:m=max(m,abs(v[c]-o[c])/max(abs(o[c]),1e-300))
        assert m<=1e-9,(key,m);worst[key]=dict(rows=n,max_relative_drift=m)
    return worst

if __name__=='__main__':main();sys.stdout.flush();__import__('os')._exit(0)
