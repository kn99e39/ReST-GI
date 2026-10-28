"""WL56 all-groups analysis and measured-cost / labelled analytic projections."""
import sys, json, math
from pathlib import Path
import numpy as np
import s56 as S
import s56_fixtures as F

def load(state,arm,rep,suffix=''):
    p=S.OUT/'runs'/f'{state}_{arm}_r{rep}{suffix}.npz'
    return np.load(p) if p.exists() else None

def pred(z,key,mode):
    return z[key+'|nee']+z[key+'|ball'] if mode=='nee_ball' else z[key+'|bin']

def group_stats(ps,u,sel,lab,cam,mode,arm):
    ref=u['seeds'];mean=ref.mean(0);den=mean[u['sel']].sum(0);out=[]
    labels=list(np.unique(lab[sel]));m=F.intrinsic('rest',0)[0]
    disc=np.concatenate([np.hypot(m['port_center'][:,0],m['port_center'][:,1])<=.025,np.zeros(len(lab)-len(m['port_area']),dtype=bool)])
    if cam=='gapdown': labels.append('lower:top@disc25mm')
    for gn in labels:
        mask=sel&(lab==gn) if '@' not in gn else sel&(lab=='lower:top')&disc
        if not mask.any():continue
        rg=ref[:,mask].mean(1);pv=np.stack([p[mask].mean(0) for p in ps]);r=rg.mean(0);p=pv.mean(0)
        sig=mean[mask].sum(0)/np.maximum(den,1e-30)
        sr=rg.std(0,ddof=1)/np.sqrt(len(rg));sk=pv.std(0,ddof=1)/np.sqrt(len(pv)) if len(pv)>1 else np.zeros(3)
        se=np.hypot(sr,sk)/np.maximum(r,1e-30);bias=(p-r)/np.maximum(r,1e-30)
        diff=np.stack(ps).mean(0)[mask]-mean[mask]
        ruled=(mask.sum()>=5)&(sig>=.02)
        if '@' in gn: ruled=np.repeat(mask.sum()>=5,3)  # preregistered localized diagnostic, separate from historical counts
        out.append(dict(group=gn,n_ports=int(mask.sum()),share=sig,reference_mean=r,prediction_mean=p,
            absolute_bias=p-r,relative_bias=bias,se_total=se,se_reference=sr/np.maximum(r,1e-30),se_K=sk/np.maximum(r,1e-30),
            interval_3SE=np.stack([bias-3*se,bias+3*se],1),mean_absolute_radiance_error=np.abs(diff).mean(0),
            relative_L2=np.linalg.norm(diff,axis=0)/np.maximum(np.linalg.norm(mean[mask],axis=0),1e-30),
            absolute_error_p99=np.percentile(np.abs(diff),99,axis=0),
            ruled=ruled,passed=(np.abs(bias)<=.03+3*se)&ruled,primary=cam!='otop' and '@' not in gn,arm=arm,mode=mode))
    return out

def main():
    res=dict(fidelity={},equivalence={},temporal={},cost={},readout_note='Stale upper KC invalid; compare all arms bin for stale loss. F/S physical NEE+KB and bin both reported.')
    for state in ('rest','dent','rigid'):
        cyc=S.extract(state);sd=F.scene(state);lab=S.CM.labels_of(sd);rec={}
        arms=('S',) if state=='rest' else ('F','S','T') if state=='dent' else ('F','S')
        for cam,units in cyc.items():
            key=S.CM.dirkey(sd['cams'][cam]['dir']);rec[cam]={}
            for unit,u in units.items():
                tables=[]
                for arm in arms:
                    zz=[load(state,arm,r) for r in (0,1)];zz=[z for z in zz if z is not None]
                    if not zz:continue
                    for mode in ('nee_ball','bin'):
                        tables+=group_stats([pred(z,key,mode) for z in zz],u,u['sel'],lab,cam,mode,arm)
                rec[cam][unit]=tables
        res['fidelity'][state]=rec
        for r in (0,1):
            a,b=load(state,'F',r),load(state,'S',r)
            if a is None or b is None:continue
            rr={}
            for k in a.files:
                if k not in b.files:continue
                d=a[k]-b[k];sc=max(float(np.abs(a[k]).sum()),1e-30)
                rr[k]=dict(relative_L1=float(np.abs(d).sum()/sc),max_abs=float(np.abs(d).max()),normalized_max=float(np.abs(d).max()/max(float(np.abs(a[k]).max()),1e-30)))
            res['equivalence'][state+f'|r{r}']=dict(fields=rr,passed=all(v['relative_L1']<=1e-6 and v['normalized_max']<=1e-5 for v in rr.values()))
    # Paired signed reference and prediction change on every unchanged-domain face, plus the fixed disc.
    R,D=S.extract('rest'),S.extract('dent');lab=S.CM.labels_of(F.scene('rest'))
    for cam in R:
        key=S.CM.dirkey(F.scene('rest')['cams'][cam]['dir']);rec=[]
        for unit in ('amended','declared'):
            r,d=R[cam][unit],D[cam][unit];sel=r['sel']&d['sel']
            disc=np.concatenate([np.hypot(F.intrinsic('rest',0)[0]['port_center'][:,0],F.intrinsic('rest',0)[0]['port_center'][:,1])<=.025,np.zeros(len(lab)-240,dtype=bool)])
            groups=list(np.unique(lab[sel]));groups+=['lower:top@disc25mm'] if cam=='gapdown' else []
            for gn in groups:
                mask=sel&(lab==gn) if '@' not in gn else sel&(lab=='lower:top')&disc
                if mask.sum()<5:continue
                delta=d['seeds'][:,mask]-r['seeds'][:,mask];dm=delta.mean(0);refbase=r['mean'][mask].mean(0)
                for arm in ('F','S','T'):
                    for mode in ('nee_ball','bin'):
                        pp=[]
                        for rp in (0,1):
                            q,t=load('dent',arm,rp),load('rest','S',rp)
                            if q is not None and t is not None:pp.append(pred(q,key,mode)[mask]-pred(t,key,mode)[mask])
                        if not pp:continue
                        pm=np.mean(pp,0);reg=[]
                        for c in range(3):
                            x,y=dm[:,c],pm[:,c]; xx=x-x.mean();yy=y-y.mean();vr=float(xx@xx)
                            slope=float(xx@yy/vr) if vr>1e-30 else None
                            corr=float(xx@yy/np.sqrt((xx@xx)*(yy@yy))) if vr>1e-30 and yy@yy>1e-30 else None
                            reg.append(dict(slope=slope,corr=corr,relative_mean_change=float(dm[:,c].mean()/max(refbase[c],1e-30)),
                                reference_change_SE=float(delta[:,:,c].mean(1).std(ddof=1)/np.sqrt(len(delta))/max(refbase[c],1e-30)),
                                predicted_mean_change=float(pm[:,c].mean()/max(refbase[c],1e-30)),
                                relative_change_L2_error=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-30))))
                        rec.append(dict(unit=unit,group=gn,arm=arm,mode=mode,n_ports=int(mask.sum()),channels=reg))
        res['temporal'][cam]=rec
    # Actual wall times; full costly builds never synthesized by multiplication.
    for p in sorted((S.OUT/'runs').glob('*.json')):
        d=json.load(open(p));res['cost'][p.stem]=d
    ratios={}
    for state in ('dent','rigid'):
        vals={}
        for arm in ('F','S'):
            q=[v for v in res['cost'].values() if v['state']==state and v['arm']==arm and not v['replay'] and not v['independent_clean']]
            if q: vals[arm]=dict(total_median=np.median([v['total_update_seconds'] for v in q]),total_all=[v['total_update_seconds'] for v in q],
                  package_median=np.median([v['package_seconds'] for v in q]),shared_median=np.median([v['update_shared_seconds'] for v in q]))
        if len(vals)==2:
            vals['speedup']=vals['F']['total_median']/vals['S']['total_median'];vals['saved_fraction']=1-1/vals['speedup']
        ratios[state]=vals
    res['cost_summary']=ratios
    if len(ratios.get('dent',{}))>=2:
        hf=ratios['dent']['F']['shared_median'];hs=ratios['dent']['S']['shared_median']
        # Metadata/load/hash overhead in S is observable separately from actual complete TR builds.
        ov=[]
        for name,v in res['cost'].items():
            if name.startswith('dent_S_') and not v['replay']:
                ov.append(sum(p.get('load_seconds',0)+p.get('identity_check_seconds',0) for p in v['packages'].values()))
        overhead=float(np.median(ov)); hs+=overhead
        res['projection']=dict(label='ANALYTIC ONLY: hypothetical cheaper complete-K update, no neural implementation',H_full=hf,H_selective=hs,
            k_break_even_for_20pct_saving=max(0,(1.25*hs-hf)/.75),
            base_N2_d1=[dict(k=k,full=2*k+hf,selective=k+hs,speedup=(2*k+hf)/(k+hs)) for k in (.001,.01,.1,1,10,100)],
            scaling=[dict(N=n,dirty=d,formula_full=f'{n} k + H_full(N)',formula_selective=f'{d} k + H_selective(N)',note='H(N) unmeasured; cannot infer actual scene scaling') for n in (2,4,8) for d in (0,1,n)])
    # Independently sampled clean K, same current dirty K; compare group differences to the empirical package-seed variation.
    independent=[]
    cyc=S.extract('dent');lab=S.CM.labels_of(F.scene('dent'))
    for cam,units in cyc.items():
        key=S.CM.dirkey(F.scene('dent')['cams'][cam]['dir']);u=units['amended']
        for mode in ('nee_ball','bin'):
            ss=[load('dent','S',r) for r in (0,1)];ii=[load('dent','F',r,'_ind') for r in (0,1)]
            if any(x is None for x in ss+ii):continue
            for gn in np.unique(lab[u['sel']]):
                mask=u['sel']&(lab==gn)
                if mask.sum()<5:continue
                den=u['mean'][mask].mean(0)
                a=np.array([pred(z,key,mode)[mask].mean(0) for z in ss]);b=np.array([pred(z,key,mode)[mask].mean(0) for z in ii])
                diff=(b-a)/np.maximum(den,1e-30);noise=np.hypot(np.abs(a[0]-a[1])/2,np.abs(b[0]-b[1])/2)/np.maximum(den,1e-30)
                independent.append(dict(cam=cam,group=gn,mode=mode,difference_by_seed=diff,empirical_SE=noise,passed=np.abs(diff)<=3*noise+1e-4))
    res['independent_clean']=independent
    sensitivities={}
    for kind in ('field','C'):
        f=S.OUT/f'noise_{kind}.npz'
        if not f.exists():continue
        z=np.load(f);base=load('dent','S',0);rr=[]
        for cam,units in cyc.items():
            key=S.CM.dirkey(F.scene('dent')['cams'][cam]['dir']);u=units['amended']
            for gn in np.unique(lab[u['sel']]):
                mask=u['sel']&(lab==gn)
                if mask.sum()<5:continue
                den=u['mean'][mask].mean(0)
                for mode in ('nee_ball','bin'):
                    diff=(pred(z,key,mode)[mask]-pred(base,key,mode)[mask]).mean(0)/np.maximum(den,1e-30)
                    rr.append(dict(cam=cam,group=gn,mode=mode,relative_group_difference=diff))
        sensitivities[kind]=rr
    res['sampling_sensitivity']=sensitivities
    if (S.OUT/'resident_rigid.json').exists():res['resident_rigid']=json.load(open(S.OUT/'resident_rigid.json'))
    S.dump(res,S.OUT/'analysis.json')
    print(json.dumps(S.CM.jclean(dict(equivalence=res['equivalence'],cost_summary=res['cost_summary'])),indent=1))

if __name__=='__main__':main();sys.stdout.flush();__import__('os')._exit(0)
