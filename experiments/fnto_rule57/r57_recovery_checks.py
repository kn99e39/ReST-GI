"""A2 predeclared bounded estimator controls/resources and independent data gate."""
import os,sys,json,shutil,time
from pathlib import Path
import torch
import r57_geometry as RG
from r57_data import ROOT,DATA,OUT,dump,aggregate,sha
from r57_action_oracle import action
from r57_recovery_data import OLD,CFG,path,checked

def norm(x):return x.square().sum((1,2)).sqrt()
def pilot():
    torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats();g=torch.load(DATA/'geometry/rest.pt');parent=g['parent'].cpu()
    oldlight=torch.load(OLD/'data/lights.pt')['test'];m,V=RG.mesh('rest');fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC)
    rows=[];passed=True
    for fi in (0,48):
        results=[action(dom,oldlight[fi],1_000_000,578800+fi*10+r) for r in range(4)]
        ys=torch.stack([aggregate(z['response'],parent)[0].double() for z in results]);mean=ys.mean(0)
        old=[aggregate(torch.load(OLD/'data/targets'/f'rest_r{r}.pt')['test'][fi:fi+1],parent)[0].double() for r in (0,1)]
        ref=.5*(old[0]+old[1]);vnew=ys.var(0,unbiased=True)/4;vold=(old[0]-old[1]).square()/4
        err=(mean-ref).square().sum((0,1)).sqrt();SE=(vnew+vold).sum((0,1)).sqrt()
        energy_error=(mean-ref).sum((0,1)).abs();ev=ys.sum((1,2)).var(0,unbiased=True)/4+(old[0]-old[1]).sum((0,1)).square()/4
        ok=bool((err<=2*SE+1e-7).all() and (energy_error<=5*ev.sqrt()+1e-6).all() and all(z['physical_passed'] for z in results))
        rows.append(dict(field=fi,L2_error=err.tolist(),combined_norm_SE=SE.tolist(),integrated_error=energy_error.tolist(),integrated_SE=ev.sqrt().tolist(),physical=[z['physical_passed'] for z in results],passed=ok,seconds=[z['seconds'] for z in results]));passed &= ok
    a,b=oldlight[0],oldlight[48];z=action(dom,torch.stack([a,b,a+b]),200_000,578899)
    lin=float((z['response'][2]-z['response'][0]-z['response'][1]).abs().sum()/z['response'][2].abs().sum());zero=action(dom,a*0,200_000,578899)
    passed &= lin<=1e-5 and bool(zero['response'].abs().max()==0) and z['physical_passed']
    disk=shutil.disk_usage(ROOT).free;mem={k:int(v.strip().split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
    free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_reserved();available=free+torch.cuda.memory_reserved()
    resources=dict(disk_free=disk,CPU_available=mem['MemAvailable'],GPU_available_before_this_process_reservations=available,GPU_total=total,pilot_peak_reserved=peak,conservative_primary_peak=2*peak+3_000_000_000,
        feasible=bool(disk>=CFG['resources']['disk_required_bytes'] and mem['MemAvailable']>=CFG['resources']['CPU_available_required_bytes'] and available>=max(CFG['resources']['GPU_free_required_bytes'],2*peak+3_000_000_000)))
    passed &= resources['feasible'];dump(dict(passed=passed,controls=rows,superposition_relative_L1=lin,zero_exact=True,resources=resources,
        strategy='Fixed64M per replica; pilot validates estimator and resource feasibility only, never selects favorable final fields or changes budget.'),OUT/'pilot.json')
    assert passed,'Estimator/resource pilot NO-GO; do not generate confirmation'

def gate():
    parent=torch.load(DATA/'geometry/rest.pt')['parent'].cpu();records=[];all_physics=True;null_ok=True
    def halves(name):
        H=[];var=[];phys=[]
        for fi in range(16):
            ys=[];vs=[]
            for rep in (0,1):
                for half in (0,1):
                    p=path(name,fi,rep,half);assert checked(p);r=torch.load(p)
                    y=aggregate(r['response'],parent).double()[0];s2=aggregate(r['second'],parent).double()[0]
                    ys.append(y);vs.append(((s2-y.square()/r['N'])*r['N']/(r['N']-1)).clamp(min=0).sum((0,1)))
                    phys.append(bool(r['physical_passed'] and abs(r['proposal_sum']-1)<1e-12 and r['importance_ratio_max']<=3+1e-6))
            H.append(torch.stack(ys));var.append(torch.stack(vs))
        return torch.stack(H,dim=1),torch.stack(var,dim=1),all(phys)
    rest,rv,ok=halves('rest');all_physics &= ok
    rest_var=rest.var(0,unbiased=True).sum((1,2))
    all_null=[]
    for name in ['rest']+RG.TEST:
        if name=='rest':cur,cv,ok=rest,rv,ok
        else:cur,cv,ok=halves(name)
        all_physics &= ok
        noise_ratios=torch.cat([norm(cur[i]-cur[j])/(cv[i]+cv[j]).sqrt().clamp(min=1e-30) for i,j in ((0,1),(2,3))])
        fractions=((noise_ratios>=.5)&(noise_ratios<=2)).double().mean(0);cal=bool((fractions>=.9).all());null_ok &= cal
        all_null.append(dict(pose=name,pair_norm_to_photon_SE=noise_ratios.tolist(),within_half_to_double_fraction=fractions.tolist(),passed=cal))
        if name=='rest':continue
        delta=cur.mean(0)-rest.mean(0);variance=(cur.var(0,unbiased=True).sum((1,2))+rest_var)/2;den=variance.sqrt();sn=norm(delta)/den.clamp(min=1e-30)
        counts=(sn>=3).sum(0);meaningful=counts>=4
        records.append(dict(pose=name,SNR=sn.tolist(),qualified_counts=counts.tolist(),meaningful_channels=meaningful.tolist(),mean_delta_norm=norm(delta).tolist(),single_64M_replica_noise=den.tolist(),mean_delta_norm_SE=(den/2**.5).tolist()))
    signal=sum(sum(x['meaningful_channels'])>=2 for x in records)>=3
    pilot_pass=json.load(open(OUT/'pilot.json'))['passed'];go=bool(signal and all_physics and null_ok and pilot_pass)
    decision=dict(proceed=go,verdict='GO' if go else 'NO-GO',signal_pass=signal,physics_pass=all_physics,noise_calibration_pass=null_ok,pilot_pass=pilot_pass,signal=records,noise_controls=all_null,
        commit=(ROOT/'repo_commit.txt').read_text().strip(),protocol_sha256=sha(os.environ['WL57_CONFIRM_PROTOCOL']),old_result='Original dc6a532 STOPPED_GATE1 remains FAILED; no retroactive validation.',automatic_continuation_authorized=True)
    dump(decision,OUT/'confirmation_gate.json');dump(decision,OUT/'gate1.json');print(json.dumps({k:v for k,v in decision.items() if k not in ('signal','noise_controls')},indent=1),flush=True)
    if not go:raise RuntimeError('Scientific NO-GO: frozen single confirmation attempt failed; no adaptive recovery')
if __name__=='__main__':
    {'pilot':pilot,'gate':gate}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
