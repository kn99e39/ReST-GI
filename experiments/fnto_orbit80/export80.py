"""WL80 export: final/summary.json (Stage 0 / 1 summaries, tests, fit gate, mechanism gate, Stage 5 decision, Stage 6 flags, cost, run records),
group_metrics.csv (every model / stage / group / metric / channel), per_matrix.csv (Stage 5), copies of the stage records, manifest."""
import sys,os,json,csv,shutil,hashlib
from common80 import CFG,ROOT,write_new

def rd(p):
    p=ROOT/p;return json.loads(p.read_text()) if p.exists() else None
def main():
    F=ROOT/'final';F.mkdir(exist_ok=True);rows=[];pm=[]
    s3=rd('stage3/analysis.json');s5=rd('stage5/analysis.json');mech=rd('mech/mech.json')
    srcs=[('stage3',s3['results'] if s3 else None),('stage5',s5['results'] if s5 else None),('part0_dev',mech.get('part0_dev') if mech else None)]
    for st,data in srcs:
        if not data:continue
        for n,r in data.items():
            for g,v in r['groups'].items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
            for dn,v in r.get('deformation',{}).items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=f'deformation:{dn}',metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
    if s5:
        for n,r in s5['results'].items():
            for split,key in (('r0123','per_matrix'),('r123','per_matrix_r123')):
                for k,v in r.get(key,{}).items():
                    for m,ch in v.items():
                        for c,x in enumerate(ch):pm.append(dict(model=n,replicas=split,matrix=k,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
    for name,data in (('group_metrics.csv',rows),('per_matrix.csv',pm)):
        if data:
            with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for p in ('stage0/stage0.json','basis/select.json','stage2/gate.json','stage2/G1/stage1/oracle.json','stage2/G1/stage1/closure.json','tests/tests.json','probe/lanes.json',
              'stage3/gate.json','mech/mech.json','mech/gate.json','stage6/deformation.json','cost/final.json','ports80.json','refs/complete.json')+tuple(
              f'stage2/{d.name}/stage1/oracle.json' for d in sorted((ROOT/'stage2').glob('B80_r*'))):
        if (ROOT/p).exists():shutil.copy(ROOT/p,F/p.replace('/','_'))
    runs={}
    for n in CFG['runs']:
        p=ROOT/'runs'/f'{n}.pt'
        if p.exists():
            import torch;x=torch.load(p,map_location='cpu',weights_only=False)
            runs[n]={k:x.get(k) for k in ('J_initial','J_final','L_sym_final','converged','last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','parameters')}
    s0=rd('stage0/stage0.json');sel=rd('basis/select.json')
    write_new(F/'summary.json',json.loads(json.dumps(dict(stage0=dict(valid=s0['valid'],C2=s0['C2']) if s0 else None,basis=dict(selected=sel['selected_restart'],J=sel['J_by_restart'],spread=sel['restart_spread']) if sel else None,
        stage2=rd('stage2/gate.json'),closure=(rd('stage2/G1/stage1/closure.json') or {}).get('summary'),tests=rd('tests/tests.json'),fit_gate=rd('stage3/gate.json'),
        mechanism=rd('mech/gate.json'),stage5=s5['decision'] if s5 else None,stage6=rd('stage6/deformation.json'),cost=rd('cost/final.json'),runs=runs),default=float)))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
