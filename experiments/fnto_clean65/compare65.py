"""Frozen A/B adjudication: original WL59 evaluation, uninspected U65, historically inspected WL64/Pair7, controls."""
import os,sys,json
from pathlib import Path
HOME=Path.home();ROOT=Path(os.environ.get('C65X_ROOT',HOME/'rest_clean65_ext'));ORIG=HOME/'rest_persist_v1';HERE=Path(__file__).resolve().parent
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'))
import torch,numpy as np
from common import RG
from runtime import Transport
from r57_data import aggregate,dump,sha
CFG=json.loads((HERE/'protocol/topup_ab_v2.json').read_text());TOL=2e-3;SEEDS=(5901,5902);STEPS=int(os.environ.get('C65X_SMOKE_STEPS','6000'))  # smoke harness only; the chain never sets it

def directional(y,t):
    norm=lambda z:z.flatten(1,2).norm(dim=1)
    d=y[::2]-y[1::2];q=t[::2]-t[1::2]
    yc=d-d.mean((1,2),keepdim=True);tc=q-q.mean((1,2),keepdim=True)
    return dict(error=(norm(d-q)/norm(q).clamp(min=1e-30)).tolist(),amplitude=(norm(d)/norm(q).clamp(min=1e-30)).tolist(),correlation=((yc*tc).sum((1,2))/(norm(yc)*norm(tc)).clamp(min=1e-30)).tolist())
def ballistic(g,x):
    return torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(len(x),-1).T).T.reshape(len(x),g['P'],64) for c in range(3)],-1)
def absolute(y,t):
    return (((y-t).flatten(1,2).norm(dim=1))/t.flatten(1,2).norm(dim=1).clamp(min=1e-30)).tolist()

def metrics_of(row):
    """Flatten every scalar quality metric of one evaluate.py model row (for reproduction drift)."""
    out={}
    for pose,p in row['poses'].items():
        for policy in ('F','S'):
            for res in ('coarse','fine16'):
                for stat in ('mean','p90'):
                    for c,v in enumerate(p['metrics'][policy][res][stat]):out[(pose,policy,res,stat,c)]=v
            if 'dynamic' in p:
                for c,d in enumerate(p['dynamic'][policy]):
                    if d['detectable']:
                        for k in ('error','corr','slope'):out[(pose,policy,'dynamic',k,c)]=d[k]
                for i,e in enumerate(p['directional_pairs'][policy]['errors']):
                    for c,v in enumerate(e):out[(pose,policy,'pair',i,c)]=v
    return out

def pair_failures(row):
    d=row['poses']['pt0']['directional_pairs']['F'];e=np.array(d['errors']);m=np.array(d['detectable'])
    return int(((e>.25)&m).sum())

def main():
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    for rel,expected in CFG['external_expected_sha256'].items():assert sha(Path(rel.replace('~',str(HOME),1)))==expected,rel
    hist=json.loads((ORIG/'out/heldout.json').read_text());assert sha(ORIG/'out/heldout.json')==CFG['input_expected_sha256']['out/heldout.json']
    pops={}
    u=np.load(ROOT/'fresh/independent_reference.npz');pops['U65_uninspected']=(torch.from_numpy(u['reference']),torch.load(ROOT/'fresh/independent_lights.pt'))
    w=np.load(HOME/'rest_direction64/independent_reference.npz');pops['WL64_historically_inspected']=(torch.from_numpy(w['reference']),torch.load(HOME/'rest_direction64/independent_lights.pt'))
    p7=np.load(HOME/'rest_components63/coarse_components.npz');pops['Pair7_historically_inspected']=(torch.from_numpy(p7['reference']),torch.load(ORIG/'data/lights.pt')['test'][62:64])
    uq=json.loads((ROOT/'fresh/independent_complete.json').read_text())
    g=RG.device(torch.load(ORIG/'data/geometry/pt0.pt'));balls={k:aggregate(ballistic(g,x.cuda()),g['parent']).cpu().double() for k,(_,x) in pops.items()}
    arms_manifest=json.loads((ROOT/'arms_manifest.json').read_text());results={};arrays={};controls={}
    for arm in ('A','B'):
        root=ROOT/'arms'/arm;held=json.loads((root/'out/heldout.json').read_text())
        for seed in SEEDS:
            ck=torch.load(root/'checkpoints'/f'P_{seed}.pt',map_location='cpu');assert ck['seed']==seed and ck['steps']==STEPS
            pr=torch.load(root/'checkpoints'/f'P_{seed}_progress.pt',map_location='cpu',weights_only=False);assert pr['step']==STEPS
            m=Transport().cuda();m.load_state_dict(ck['model']);m.eval();co=m.coefficients(g);params=sum(p.numel() for p in m.parameters())
            row={'checkpoint_sha256':sha(root/'checkpoints'/f'P_{seed}.pt'),'parameters':params,'original':held[f'P_{seed}'],'populations':{}}
            for name,(ref,x) in pops.items():
                x=x.cuda();y=torch.cat([m.apply(g,x[i:i+2],co) for i in range(0,len(x),2)]);ym=aggregate(y,g['parent']).cpu().double()
                pred=torch.stack([ym,balls[name],ym-balls[name]],1);row['populations'][name]={c:directional(pred[:,i],ref[:,i]) for i,c in enumerate(('T','B','Scat'))}
                row['populations'][name]['absolute_T']=absolute(ym,ref[:,0]);arrays[f'{arm}_{seed}_{name}']=pred.numpy();del y
            tr=json.loads((root/'out/training'/f'P_{seed}.json').read_text())
            row['training']={k:tr[k] for k in ('seconds','peak_GPU_bytes','parameters','steps','history')}
            controls[f'{arm}_{seed}']=dict(numpy_rng=pr['numpy_rng'],optimizer={k:v for k,v in pr['optimizer']['param_groups'][0].items() if k!='params'},parameters=params)
            results[f'{arm}_{seed}']=row;del m,co;torch.cuda.empty_cache()
    # Controls: identical schedule/optimizer, A reproduces preserved WL59.
    sched=all(controls[f'A_{s}']['numpy_rng']==controls[f'B_{s}']['numpy_rng']==arms_manifest['schedule_final_numpy_state'] for s in SEEDS)
    optim=all({k:v for k,v in controls[f'A_{s}']['optimizer'].items() if k!='lr'}=={k:v for k,v in controls[f'B_{s}']['optimizer'].items() if k!='lr'} and abs(controls[f'A_{s}']['optimizer']['lr']-controls[f'B_{s}']['optimizer']['lr'])==0 for s in SEEDS)
    reproduction={}
    for s in SEEDS:
        a=metrics_of(results[f'A_{s}']['original']);h=metrics_of(hist[f'P_{s}']);assert a.keys()==h.keys()
        drift=max(abs(a[k]-h[k]) for k in a);worst=max(a,key=lambda k:abs(a[k]-h[k]))
        ra=results[f'A_{s}']['original'];rh=hist[f'P_{s}']
        same=ra['eligible']==rh['eligible'] and ra['useful_and_physical']==rh['useful_and_physical'] and pair_failures(ra)==pair_failures(rh)
        reproduction[str(s)]=dict(max_abs_drift=drift,worst_metric=list(map(str,worst)),flags_identical=same,passed=drift<=TOL and same,eligible_A=ra['eligible'],eligible_WL59=rh['eligible'])
    physical={k:all(p['physics']['passed'] and p['F_S_passed'] for p in v['original']['poses'].values()) and v['parameters']==157024 for k,v in results.items()}
    controls_ok=sched and optim and all(v['passed'] for v in reproduction.values()) and all(physical.values())
    decisions={}
    for s in SEEDS:
        a,b=results[f'A_{s}'],results[f'B_{s}'];pa=a['original']['poses']['pt0']['directional_pairs']['F'];pb=b['original']['poses']['pt0']['directional_pairs']['F']
        mask=np.array(pa['detectable'])&np.array(pb['detectable']);ea=np.array(pa['errors']);eb=np.array(pb['errors'])
        d=dict(primary_mean_A=float(ea[mask].mean()),primary_mean_B=float(eb[mask].mean()),primary_ratio=float(eb[mask].mean()/ea[mask].mean()),primary_qualified_cells=int(mask.sum()),
               primary_failures_A=int(((ea>.25)&mask).sum()),primary_failures_B=int(((eb>.25)&mask).sum()),old_pairs0_6_ratio=float(eb[:7][mask[:7]].mean()/ea[:7][mask[:7]].mean()),
               primary_cells_improved=int(((eb<ea)&mask).sum()),primary_cells_regressed=int(((eb>ea)&mask).sum()),
               primary_channel_ratio={c:float(eb[:,i][mask[:,i]].mean()/ea[:,i][mask[:,i]].mean()) for i,c in enumerate('RGB')})
        for name in pops:
            fa=np.array(a['populations'][name]['T']['error']);fb=np.array(b['populations'][name]['T']['error'])
            d[name]=dict(mean_A=float(fa.mean()),mean_B=float(fb.mean()),ratio=float(fb.mean()/fa.mean()),failures_A=int((fa>.25).sum()),failures_B=int((fb>.25).sum()),cells_improved=int((fb<fa).sum()),
                         scat_mean_A=float(np.mean(a['populations'][name]['Scat']['error'])),scat_mean_B=float(np.mean(b['populations'][name]['Scat']['error'])))
        d['primary_improvement']=d['primary_ratio']<=.90 and d['primary_failures_B']<=d['primary_failures_A']
        d['uninspected_improvement']=d['U65_uninspected']['ratio']<=.90 if uq['total_all24_qualified'] else None
        d['regression']=d['primary_ratio']>=1.10 or d['U65_uninspected']['ratio']>=1.10
        d['complete_K_A']=a['original']['useful_and_physical'];d['complete_K_B']=b['original']['useful_and_physical']
        decisions[str(s)]=d
    if not controls_ok:outcome='D'
    elif all(v['primary_improvement'] and v['uninspected_improvement'] for v in decisions.values()):outcome='A'
    else:outcome='B'
    sub=None if outcome!='B' else ('regression' if any(v['regression'] for v in decisions.values()) else 'no_meaningful_or_mixed_benefit')
    dump(dict(models=results,decisions=decisions,outcome=outcome,outcome_B_sublabel=sub,controls=dict(schedule_identical=sched,optimizer_identical=optim,A_reproduction=reproduction,physical_and_FS=physical,valid=controls_ok,detail=controls),
        U65_qualified=uq['total_all24_qualified'],U65_qualified_cells=uq['total_qualified_cells'],complete_K_B_both_seeds=all(v['complete_K_B'] for v in decisions.values()),
        coupled_GI_eligible=all(v['complete_K_B'] for v in decisions.values()),note='Partial clean supervision (16/96 fields, train geometries only); 2 seeds; U65 fresh sources on known pt0 only; Pair7/WL64/original pairs historically inspected.'),ROOT/'comparison.json')
    np.savez_compressed(ROOT/'comparison_components.npz',**arrays)
    print(json.dumps(dict(outcome=outcome,sub=sub,decisions=decisions,reproduction=reproduction,schedule=sched,optimizer=optim),indent=1),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
