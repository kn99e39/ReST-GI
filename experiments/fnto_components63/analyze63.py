"""Physical tagged component precision and frozen-checkpoint Pair7 error accounting."""
import argparse,itertools,json,sys,time
from pathlib import Path
import torch,numpy as np
from common63 import CFG,HERE,sha,dump_new
from metrics63 import aggregate,variance,covariance_BS,norm,error,decompose
sys.path.insert(0,str(HERE/'frozen60'))
from replay import coefficients,action
from analyze_stored import ballistic

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def scalar_relative(a,b):return float((a-b).abs().sum()/b.abs().sum().clamp(min=1e-300))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();root=a.root;orig=a.original;torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter()
    pre=json.loads((root/'preflight.json').read_text());complete=json.loads((root/'reference_complete.json').read_text())
    assert complete['photons']==256000000 and complete['physical_passed'] and complete['gates_A_B_passed']
    for rel,v in pre['inputs'].items():assert sha(orig/rel)==v['sha256'],rel
    g=load(orig/'data/geometry/pt0.pt');x=load(orig/'data/lights.pt')['test'][CFG['fields']];names=['T','B','Scat'];first=[];second=[];gates=[]
    for field in CFG['fields']:
        ff=[];ss=[]
        for rep in range(2):
            for half in range(2):
                p=root/'halves'/f'pt0_f{field}_r{rep}_h{half}.pt';receipt=json.loads(p.with_suffix('.json').read_text());assert sha(p)==receipt['sha256'];r=load(p)
                f=torch.stack([aggregate(r['component_first'][k],g['parent'])[0] for k in names]);s=torch.stack([aggregate(r['component_second'][k],g['parent'])[0] for k in names]);ff.append(f);ss.append(s)
                vv=[variance(f[i],s[i],r['N']) for i in range(3)];cov=covariance_BS(f[1],f[2],r['N'])
                closure=scalar_relative(vv[1]+vv[2]+2*cov,vv[0]);assert closure<=CFG['variance_closure_relative'],'Variance/covariance closure failure'
                gates.append(dict(field=field,replica=rep,half=half,variance_covariance_closure=closure,minimum_variance=min(float(v.min()) for v in vv),raw_BS_cross_moment=r['raw_BS_cross_moment']))
        first.append(torch.stack(ff));second.append(torch.stack(ss))
    first=torch.stack(first);second=torch.stack(second) # field, half4, component, coarseport, bin16, RGB
    mean=first.mean(1);vv=variance(first,second,CFG['N_per_half']).sum(1)/16
    covariance=covariance_BS(first[:,:,1],first[:,:,2],CFG['N_per_half']).sum(1)/16
    ref={k:mean[:,i] for i,k in enumerate(names)};rep={k:torch.stack([first[:,:2,i].mean(1),first[:,2:,i].mean(1)]) for i,k in enumerate(names)}
    d={k:ref[k][0]-ref[k][1] for k in names};dv={k:vv[0,i]+vv[1,i] for i,k in enumerate(names)};dcov=covariance.sum(0)
    result=dict(batch_id=CFG['batch_id'],source_commit=a.source_commit,script_sha256=sha(__file__),fields=CFG['fields'],split='test',pose='pt0',gate_A_B=True,per_half_variance_checks=gates,precision={},models={},old_reference_comparison={})
    for k in names:
        diff=(rep[k][0,0]-rep[k][0,1])-(rep[k][1,0]-rep[k][1,1]);n=diff/(2**.5);snr=norm(d[k])/norm(n).clamp(min=1e-300)
        result['precision'][k]=dict(contrast_norm_RGB=norm(d[k]).tolist(),single_replica_contrast_noise_norm_ratio=(norm(n)/norm(d[k]).clamp(min=1e-300)).tolist(),SNR_RGB=snr.tolist(),qualified_RGB=(snr>=3).tolist(),final_mean_SE_over_component_contrast=(dv[k].sum((0,1)).sqrt()/norm(d[k]).clamp(min=1e-300)).tolist(),final_mean_SE_over_total_contrast=(dv[k].sum((0,1)).sqrt()/norm(d['T']).clamp(min=1e-300)).tolist())
    result['qualified_RGB']=[all(result['precision'][k]['qualified_RGB'][c] for k in names) for c in range(3)]
    result['gate_C_passed']=all(result['qualified_RGB'])
    result['contrast_moment_closure']=scalar_relative(dv['B']+dv['Scat']+2*dcov,dv['T']);assert result['contrast_moment_closure']<=CFG['variance_closure_relative']
    braw=ballistic(g,x).double();bm=aggregate(braw,g['parent']);bd=bm[0]-bm[1];preds=[];allclasses={c:[] for c in range(3)}
    for seed in CFG['checkpoint_seeds']:
        ck=load(orig/'checkpoints'/f'P_{seed}.pt');assert ck['seed']==seed and ck['steps']==6000
        y=action(g,x,coefficients(ck['model'],g));ym=aggregate(y,g['parent']);sm=ym-bm;sd=sm[0]-sm[1];td=ym[0]-ym[1]
        stored=load(orig/'out'/f'actions_{seed}.pt',True);stored_y=stored['predictions']['pt0']['F'][CFG['fields']]
        replay=float((y.double()-stored_y.double()).norm()/stored_y.double().norm());assert replay<2e-5,'Frozen replay mismatch'
        oldref=aggregate(stored['reference']['pt0'][CFG['fields']],g['parent']);result['old_reference_comparison'][str(seed)]=dict(new_vs_old_total_action_relative=error(ref['T'],oldref),new_vs_old_total_contrast_relative=error(d['T'],oldref[0]-oldref[1]))
        rows=decompose(bd,sd,d['B'],d['Scat']);variants=[]
        for ra,rb in itertools.product(range(2),repeat=2):
            rbref=rep['B'][ra,0]-rep['B'][rb,1];rsref=rep['Scat'][ra,0]-rep['Scat'][rb,1]
            vals=decompose(bd,sd,rbref,rsref);variants.append(dict(field62_replica=ra,field63_replica=rb,rows=vals))
        for c,row in enumerate(rows):
            vB=float(dv['B'][...,c].sum());vS=float(dv['Scat'][...,c].sum());cov=float(dcov[...,c].sum());vT=float(dv['T'][...,c].sum());den=float(d['T'][...,c].square().sum())
            row.update(qualified=result['qualified_RGB'][c],B_error_SNR=float((bd-d['B'])[...,c].norm()/max(vB,1e-300)**.5),Scat_error_SNR=float((sd-d['Scat'])[...,c].norm()/max(vS,1e-300)**.5),post_hoc_moment_corrected_B_squared=(row['B_squared_over_total_signal']*den-vB)/den,post_hoc_moment_corrected_Scat_squared=(row['Scat_squared_over_total_signal']*den-vS)/den,post_hoc_moment_corrected_cross=(row['cross_over_total_signal']*den-2*cov)/den,post_hoc_moment_corrected_total_squared=(row['total_squared_over_total_signal']*den-vT)/den,reference_B_Scat_covariance_trace=cov)
            if not row['qualified']:row['classification']='D_PRECISION_OR_SEMANTICS'
            classes=[row['classification']]+[v['rows'][c]['classification'] for v in variants]
            row['classification_stable_across_replicas']=len(set(classes))==1
            allclasses[c].extend(classes)
        result['models'][str(seed)]=dict(replay_relative_L2=replay,rows=rows,replica_variants=variants,total_vs_component_contrast_closure=error(td,bd+sd))
        preds.append(torch.stack([ym,bm,sm]));del stored
    result['classification_RGB']=['D_PRECISION_OR_SEMANTICS' if not result['qualified_RGB'][c] else (allclasses[c][0] if len(set(allclasses[c]))==1 else 'C_INTERACTING_OR_MIXED') for c in range(3)]
    result['status']='COMPLETE' if result['gate_C_passed'] else 'STOP_COMPONENT_PRECISION'
    result['covariance_note']='B*Scat per-photon raw cross moment is zero; Cov(Bmean,Scatmean) is negative and retained. Error cross terms and noise corrections explicitly include it. Ratios are not unbiased and no confidence interval is claimed.'
    result['seconds']=time.perf_counter()-start
    out=root/'coarse_components.npz';assert not out.exists();np.savez_compressed(out,half_first=first.numpy(),half_second=second.numpy(),reference=mean.numpy(),reference_variance=vv.numpy(),reference_B_Scat_covariance=covariance.numpy(),predictions=torch.stack(preds).numpy())
    result['coarse_arrays_sha256']=sha(out)
    for rel,v in pre['inputs'].items():assert sha(orig/rel)==v['sha256']
    dump_new(root/'analysis.json',result);print(result['status'],result['classification_RGB'],flush=True)
if __name__=='__main__':main()
