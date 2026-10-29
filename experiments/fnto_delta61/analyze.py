"""Compare frozen models, historical labels and fresh independent WL61 actions."""
import argparse,hashlib,itertools,json,os,sys,time
from pathlib import Path
import torch,numpy as np
from metrics import aggregate,half_variance,mean_variance,channel_norm,relative,per_field_relative,paired,alignment
from audit_run import sha,write_json,CFG,HERE
sys.path.insert(0,str(HERE/'frozen60'))
from replay import coefficients,action

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def cosine(a,b):return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp(min=1e-30))
def normratio(noise,signal):return (channel_norm(noise)/channel_norm(signal).clamp(min=1e-30)).tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();root=a.root;orig=a.original
    torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter()
    pre=json.loads((root/'preflight.json').read_text());completion=json.loads((root/'reference_complete.json').read_text())
    assert completion['total_photons']==768000000 and completion['physical_passed']
    for rel,r in pre['inputs'].items():assert sha(orig/rel)==r['sha256'],f'Original input changed: {rel}'
    for r in completion['files']:assert sha(r['path'])==r['sha256'],f'Half changed: {r["path"]}'
    lights=load(orig/'data/lights.pt')['train'][CFG['fields']];parents={};old={};new={};var={};reps={};fine_new={};oldfine={};coarse_halves=[];coarse_vars=[]
    for pose in CFG['poses']:
        g=load(orig/'data/geometry'/f'{pose}.pt');parent=g['parent'];parents[pose]=parent
        target=load(orig/'data/training_targets'/f'{pose}_r0.pt')['train'][CFG['fields']]
        old[pose]=aggregate(target,parent);oldfine[pose]=aggregate(target,parent,False)
        cm=[];cv=[];rep_fields=[];fine_fields=[];pose_h=[];pose_v=[]
        for field in CFG['fields']:
            hv=[];vv=[];responses=[]
            for rep in range(2):
                for half in range(2):
                    p=root/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt';obj=load(p)
                    y=aggregate(obj['response'],parent)[0];s2=aggregate(obj['second'],parent)[0]
                    hv.append(y);vv.append(half_variance(y,s2,obj['N']));responses.append(obj['response'][0].double())
            mean,v=mean_variance(hv,vv);cm.append(mean);cv.append(v)
            rep_fields.append(torch.stack([.5*(hv[0]+hv[1]),.5*(hv[2]+hv[3])]))
            fine_fields.append(torch.stack(responses).mean(0));pose_h.append(torch.stack(hv).reshape(2,2,*hv[0].shape));pose_v.append(torch.stack(vv).reshape(2,2,*vv[0].shape))
        new[pose]=torch.stack(cm);var[pose]=torch.stack(cv);reps[pose]=torch.stack(rep_fields,dim=1)
        fine_new[pose]=aggregate(torch.stack(fine_fields),parent,False)
        coarse_halves.append(torch.stack(pose_h));coarse_vars.append(torch.stack(pose_v))
    report=dict(batch_id=CFG['batch_id'],source_commit=a.source_commit,analysis_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol/audit_v1.json'),fields=CFG['fields'],light_split='train',precision={},absolute={},dynamic={},models={},geometry_gap={},limitations=['Two replicas do not establish zero bias or a confidence interval','No independent deformed historical-label uncertainty; discrepancy/new-SE is not a combined significance test','Only fixed train fields 8/9 at rest/tr11/va1; no full primary or directional-pair re-adjudication'])
    for pose in CFG['poses']:
        single=(reps[pose][0]-reps[pose][1])/(2**.5)
        per=normratio(single,new[pose]);perfield=(single.flatten(1,2).norm(dim=1)/new[pose].flatten(1,2).norm(dim=1)).tolist()
        report['precision'][pose]=dict(single_replica_noise_pooled_RGB=per,single_replica_noise_per_field=perfield,mean_reference_SE_norm_relative_RGB=(var[pose].sum((0,1,2)).sqrt()/channel_norm(new[pose])).tolist(),absolute_readiness=all(v<=CFG['coarse_noise_max'] for v in np.mean(perfield,axis=0)))
        report['absolute'][pose]=dict(old_vs_clean_coarse_per_field=per_field_relative(old[pose],new[pose]),old_vs_clean_coarse_pooled=relative(old[pose],new[pose]),old_vs_clean_fine16_per_field=per_field_relative(oldfine[pose],fine_new[pose]),old_discrepancy_over_new_SE=(channel_norm(old[pose]-new[pose])/var[pose].sum((0,1,2)).sqrt().clamp(min=1e-30)).tolist())
    for pose in ('tr11','va1'):
        d=paired(new,pose);od=paired(old,pose);v=var[pose]+var['rest']
        cn=(reps[pose][0]-reps[pose][1])/(2**.5);rn=(reps['rest'][0]-reps['rest'][1])/(2**.5)
        denom=(cn.square().sum((0,1,2))+rn.square().sum((0,1,2))).sqrt()
        snr=channel_norm(d)/denom.clamp(min=1e-30)
        pfden=(cn.flatten(1,2).square().sum(1)+rn.flatten(1,2).square().sum(1)).sqrt()
        pfsnr=d.flatten(1,2).norm(dim=1)/pfden.clamp(min=1e-30)
        report['dynamic'][pose]=dict(delta_reference_norm_RGB=channel_norm(d).tolist(),historical_delta_norm_RGB=channel_norm(od).tolist(),conservative_pooled_SNR=snr.tolist(),conservative_per_field_SNR=pfsnr.tolist(),qualified_RGB=(snr>=3).tolist(),mean_reference_SE_norm_ratio_RGB=(v.sum((0,1,2)).sqrt()/channel_norm(d)).tolist(),historical_vs_clean_delta=relative(od,d),historical_vs_clean_per_field=per_field_relative(od,d),historical_discrepancy_over_new_SE=(channel_norm(od-d)/v.sum((0,1,2)).sqrt().clamp(min=1e-30)).tolist(),caution='The last ratio excludes unknown historical deformed-label error, so is not a z-score or significance test.')
    baseline=json.loads((HERE/'baseline_wl60_replay.json').read_text());preds={}
    for seed in CFG['checkpoints']:
        cp=orig/'checkpoints'/f'P_{seed}.pt';ck=load(cp);assert ck['seed']==seed and ck['steps']==6000
        p={};rows={'absolute':{},'dynamic':{}}
        for pose in CFG['poses']:
            g=load(orig/'data/geometry'/f'{pose}.pt');co=coefficients(ck['model'],g)
            y=action(g,lights,co);p[pose]=aggregate(y,g['parent'])
            old_abs=per_field_relative(p[pose],old[pose]);expected=baseline['seeds'][str(seed)][f'{pose}_train']['absolute_relative_L2'][:2]
            assert np.max(np.abs(np.array(old_abs)-np.array(expected)))<2e-6,'Frozen replay differs from WL60'
            rows['absolute'][pose]=dict(prediction_vs_old_coarse_per_field=old_abs,prediction_vs_clean_coarse_per_field=per_field_relative(p[pose],new[pose]),prediction_vs_clean_coarse_pooled=relative(p[pose],new[pose]),prediction_vs_clean_fine16_per_field=per_field_relative(aggregate(y,g['parent'],False),fine_new[pose]))
        for pose in ('tr11','va1'):
            yd=paired(p,pose);od=paired(old,pose);d=paired(new,pose)
            erold=relative(yd,od);exp=baseline['seeds'][str(seed)][f'{pose}_train']['focused_delta_error_unqualified']
            assert np.max(np.abs(np.array(erold)-np.array(exp)))<2e-6,'Original delta metric changed'
            combos=[]
            for r,s in itertools.product(range(2),repeat=2):combos.append(dict(current_replica=r,rest_replica=s,error=relative(yd,reps[pose][r]-reps['rest'][s])))
            common=channel_norm(d)
            decomposition=[]
            for c in range(3):
                fit=(yd-od)[...,c];lab=(od-d)[...,c];decomposition.append(dict(fit_norm_over_clean=float(fit.norm()/common[c]),label_norm_over_clean=float(lab.norm()/common[c]),cosine=cosine(fit,lab)))
            rows['dynamic'][pose]=dict(prediction_vs_old=erold,prediction_vs_clean=relative(yd,d),prediction_vs_old_common_clean_denominator=(channel_norm(yd-od)/common).tolist(),prediction_vs_clean_per_field=per_field_relative(yd,d),clean_alignment=alignment(yd,d),old_alignment=alignment(yd,od),replica_sensitivity=combos,error_decomposition=decomposition,qualified_RGB=report['dynamic'][pose]['qualified_RGB'])
        gaps=[]
        # Same rest replica in both geometry deltas; do not pretend their errors are independent.
        for rr,rt,rv in itertools.product(range(2),repeat=3):
            et=relative(paired(p,'tr11'),reps['tr11'][rt]-reps['rest'][rr]);ev=relative(paired(p,'va1'),reps['va1'][rv]-reps['rest'][rr])
            gaps.append(dict(rest_replica=rr,train_replica=rt,validation_replica=rv,validation_minus_train_error=(np.array(ev)-np.array(et)).tolist()))
        vals=np.array([g['validation_minus_train_error'] for g in gaps]);report['geometry_gap'][str(seed)]=dict(clean_validation_minus_train_error=(np.array(rows['dynamic']['va1']['prediction_vs_clean'])-np.array(rows['dynamic']['tr11']['prediction_vs_clean'])).tolist(),old_validation_minus_train_error=(np.array(rows['dynamic']['va1']['prediction_vs_old'])-np.array(rows['dynamic']['tr11']['prediction_vs_old'])).tolist(),replica_gap_min=vals.min(0).tolist(),replica_gap_max=vals.max(0).tolist(),coherent_replica_sensitivity=gaps)
        report['models'][str(seed)]=rows;preds[seed]=p
    # Exact cancellation of shared rest in the linear difference of geometry deltas.
    gd=new['va1']-new['tr11'];gvar=var['va1']+var['tr11']
    report['between_delta_reference']=dict(norm_RGB=channel_norm(gd).tolist(),mean_SE_norm_ratio_RGB=(gvar.sum((0,1,2)).sqrt()/channel_norm(gd)).tolist(),variance='var(va1)+var(tr11); shared rest cancels, not +2*var(rest)')
    report['cross_seed']={pose:dict(delta_prediction_difference_relative_to_clean=(channel_norm(paired(preds[5901],pose)-paired(preds[5902],pose))/channel_norm(paired(new,pose)).clamp(min=1e-30)).tolist(),residual_cosine_RGB=[cosine((paired(preds[5901],pose)-paired(new,pose))[...,c],(paired(preds[5902],pose)-paired(new,pose))[...,c]) for c in range(3)]) for pose in ('tr11','va1')}
    # Compatible historical rest-only replicate comparison; not a deformed-label variance estimate.
    oldpath=orig.parent/'rest_wl57/data/targets/rest_r1.pt';r1=aggregate(load(oldpath)['train'][CFG['fields']],parents['rest']);hn=(old['rest']-r1)/(2**.5)
    report['historical_rest_replica']=dict(path=str(oldpath),sha256=sha(oldpath),single_label_noise_relative_to_midpoint=normratio(hn,.5*(old['rest']+r1)),caution='Rest only; no corresponding independent tr11/va1 historical labels')
    report['all_required_delta_channels_qualified']=all(all(r['qualified_RGB']) for r in report['dynamic'].values())
    report['reference_readiness']=all(r['absolute_readiness'] for r in report['precision'].values()) and report['all_required_delta_channels_qualified']
    report['completion_status']='COMPLETE' if report['reference_readiness'] else 'STOP_FIXED_BUDGET_PRECISION'
    report['seconds']=time.perf_counter()-start
    # Save only reviewer-scale arrays locally; full first-exit halves remain original server evidence.
    arrays=dict(half_actions=np.stack([v.numpy() for v in coarse_halves]),half_variances=np.stack([v.numpy() for v in coarse_vars]),reference=np.stack([new[p].numpy() for p in CFG['poses']]),reference_mean_variance=np.stack([var[p].numpy() for p in CFG['poses']]),historical_r0=np.stack([old[p].numpy() for p in CFG['poses']]),predictions=np.stack([[preds[s][p].numpy() for p in CFG['poses']] for s in CFG['checkpoints']]))
    apath=root/'coarse_evidence.npz';assert not apath.exists();np.savez_compressed(apath,**arrays)
    report['coarse_evidence_sha256']=sha(apath)
    for rel,r in pre['inputs'].items():assert sha(orig/rel)==r['sha256'],'Original input was modified'
    write_json(root/'analysis.json',report);print(report['completion_status'],flush=True)
if __name__=='__main__':main()
