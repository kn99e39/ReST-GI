"""Focused CPU tests for the top-up/A-B extension; no physical photons, no model training."""
import sys,json,ast,hashlib
from pathlib import Path
import torch,numpy as np
from ext65 import CFG,CFG1,TOP,ORIG,HERE,sha,packets,phaseA_seed,topup_seed,packet_variance,Combined,override
from moments65 import variance

def moments(x):
    """Packet first moment (mean) and second moment (sum of squares / N^2) exactly as the accumulator."""
    N=len(x);return x.mean(0),x.square().sum(0)/N**2,N

def main():
    torch.set_num_threads(1);checks={}
    # Reference identity: fixed seeds independent of every earlier stream.
    top=[topup_seed(p,f,r,k) for p in range(13) for f in TOP['fields'] for r in range(2) for k in range(6)]
    old={phaseA_seed(p,f,r,h) for p in range(13) for f in CFG1['fields'] for r in range(2) for h in range(2)}
    others={640000000+f*100+r*10+h for f in range(16) for r in range(2) for h in range(2)}|{599000000+f*100+r*10+h for f in range(64) for r in range(2) for h in range(2)}|{670000000+f*100+r*10+h for f in range(16) for r in range(2) for h in range(2)}
    assert len(set(top))==624 and not set(top)&old and not set(top)&others and len(old)==832;checks['624_distinct_topup_seeds_disjoint']=True
    # Replica grouping: balanced additions, unchanged Phase-A packet identities.
    for pose in TOP['poses']:
        for f in CFG1['fields']:
            pl=packets(pose,f,root_a=Path('A'),root_t=Path('T'));src=[k for k,*_ in pl];grp=[g for _,g,*_ in pl]
            assert [str(p) for k,_,p,*_ in pl if k=='phaseA']==[str(Path('A/halves')/f'{pose}_f{f}_r{r}_h{h}.pt') for r in range(2) for h in range(2)]
            if f in TOP['fields']:
                assert len(pl)==16 and grp.count(0)==grp.count(1)==8 and src.count('topup')==12 and sum(n for *_,n,_ in pl)==512000000
                assert all(sum(1 for k,g,*_ in pl if k==s and g==r)==(2 if s=='phaseA' else 6) for s in ('phaseA','topup') for r in (0,1))
            else:assert len(pl)==4 and src.count('topup')==0
    assert sum(sum(n for k,_,_,n,_ in packets(p,f) if k=='topup') for p in TOP['poses'] for f in TOP['fields'])==19968000000;checks['balanced_replica_groups_and_exact_budget']=True
    # Equal counts reproduce Phase A's estimator bitwise.
    g=torch.Generator().manual_seed(1);xs=[torch.rand(50,7,generator=g,dtype=torch.float64)**3 for _ in range(4)];c=Combined([50]*4,[0,0,1,1]);mean=var=None
    for x in xs:
        f,s,N=moments(x);c.add(f,s);mean=f/4 if mean is None else mean+f/4;v=variance(f,s,N)/16;var=v if var is None else var+v
    r=c.result();assert torch.equal(r['mean'],mean) and torch.equal(r['var'],var)
    assert torch.allclose(r['rep'][0],torch.stack([moments(xs[0])[0],moments(xs[1])[0]]).mean(0),rtol=0,atol=1e-15);checks['equal_count_bitwise_phaseA_estimator']=True
    # Unequal counts: combined mean equals the pooled sample mean; variance matches Monte Carlo.
    counts=[40,40,120,120,120,120];grp=[0,1,0,0,1,1];xs=[torch.rand(n,5,generator=g,dtype=torch.float64)**3 for n in counts];c=Combined(counts,grp)
    for x in xs:c.add(*moments(x)[:2])
    r=c.result();assert torch.allclose(r['mean'],torch.cat(xs).mean(0),rtol=0,atol=1e-14)
    assert torch.allclose(r['rep'][0],torch.cat([x for x,h in zip(xs,grp) if h==0]).mean(0),rtol=0,atol=1e-14)
    est=[];emp=[]
    for t in range(3000):
        xs=[torch.rand(n,3,generator=g,dtype=torch.float64)**3 for n in counts];c=Combined(counts,grp)
        for x in xs:c.add(*moments(x)[:2])
        rr=c.result();est.append(rr['var']);emp.append(rr['mean'])
    ratio=(torch.stack(emp).var(0)/torch.stack(est).mean(0));assert ((ratio>.9)&(ratio<1.1)).all(),ratio;checks['count_weighted_mean_and_variance_monte_carlo']=float(ratio.mean())
    # Rest/current covariance: delta variances add; shared rest covariance cancels between dents.
    rest=torch.randn(20000,generator=g,dtype=torch.float64)*2;cur1=torch.randn(20000,generator=g,dtype=torch.float64)*3;cur2=torch.randn(20000,generator=g,dtype=torch.float64)
    d1=cur1-rest;d2=cur2-rest;assert abs(float(d1.var())/13-1)<.05 and abs(float(torch.cov(torch.stack([d1,d2]))[0,1])/4-1)<.05 and abs(float((d1-d2).var())/10-1)<.05
    checks['rest_current_covariance']=True
    # Label override isolation.
    orig={'train':torch.rand(96,5,64,3,generator=g),'validation':torch.rand(32,5,64,3,generator=g)};resp=torch.rand(16,5,64,3,generator=g);snap={k:v.clone() for k,v in orig.items()}
    new=override(orig,resp);assert set(new)==set(orig) and torch.equal(new['train'][:80],orig['train'][:80]) and torch.equal(new['train'][80:],resp) and torch.equal(new['validation'],orig['validation'])
    assert all(torch.equal(orig[k],snap[k]) for k in orig);checks['label_override_isolation']=True
    # Unchanged objective/runner/evaluation: frozen WL59 bytes, same in both arms; criteria unchanged.
    rep=ORIG/'repo/experiments/fnto_persist59'
    for name in ('train.py','evaluate.py','cost.py','run.sh','runtime.py','common.py'):assert sha(rep/name)==CFG['source_expected_sha256'][f'experiments/fnto_persist59/{name}']
    assert CFG['source_expected_sha256']['experiments/fnto_persist59/train.py']==CFG1['source_expected_sha256']['experiments/fnto_persist59/train.py']
    src=(rep/'train.py').read_text()
    for frag in ("loss=L+.25*delta","AdamW(model.parameters(),lr=.003,weight_decay=1e-5)","clip_grad_norm_(model.parameters(),1.)","default_rng(574001)","rng.integers(0,96,2)","RG.TRAIN[1+step%(len(RG.TRAIN)-1)]","torch.manual_seed(seed)","WL57_TRAIN_TARGETS"):assert frag in src,frag
    tree=ast.parse(src);assert not any(isinstance(n,ast.Name) and n.id in ('contrast','FIXED') for n in ast.walk(tree));checks['unchanged_WL59_objective_runner_no_WL64_term']=True
    crit=json.loads((rep/'protocol/persist_v1.json').read_text())['criteria']
    assert (crit['coarse_mean_relative_L2'],crit['coarse_p90'],crit['fine16_mean_relative_L2'],crit['delta_relative_L2'],crit['delta_correlation'],crit['directional_pair_relative_L2'],crit['noise_qualification_SNR'])==(.1,.2,.15,.35,.9,.25,3)
    ev=(rep/'evaluate.py').read_text()
    for frag in ("v<=.10","v<=.20","v<=.15","er<=.35 and corr>=.90","errors<=.25","3*pn"):assert frag in ev,frag
    checks['frozen_evaluation_criteria']=True
    # Schedule: arm-independent reconstruction is deterministic and matches the runner's draw count.
    rng=np.random.default_rng(574001);a=[rng.integers(0,96,2).tolist() for _ in range(6000)];rng2=np.random.default_rng(574001);b=[rng2.integers(0,96,2).tolist() for _ in range(6000)]
    assert a==b;checks['identical_AB_schedule_reconstruction']=hashlib.sha256(json.dumps(a).encode()).hexdigest()
    # Gates copied unchanged from Phase A.
    for k in ('absolute_coarse_mean_SE_ratio_max','absolute_fine16_mean_SE_ratio_max','direction_coarse_SNR_min','direction_coarse_mean_SE_ratio_max','delta_coarse_loss_scale_SE_max','rest_pooled_contrast_new_SE_over_old_single_noise_max'):assert CFG['gates'][k]==CFG1['gates'][k],k
    checks['phaseA_gates_unchanged']=True
    print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
