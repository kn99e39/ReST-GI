"""WL78 export: final/summary.json (tests, cost, fit gate, mechanism criterion, Stage 4 decision, run records), group_metrics.csv
(every model / stage / group / metric / channel), per_matrix.csv (Stage 4), copies of the stage records, manifest."""
import sys,os,json,csv,shutil,hashlib
from common78 import CFG,ROOT,write_new

def rd(p):
    p=ROOT/p;return json.loads(p.read_text()) if p.exists() else None
def main():
    F=ROOT/'final';F.mkdir(exist_ok=True);s3=rd('stage3/analysis.json');s4=rd('stage4/analysis.json');rows=[];pm=[]
    for st,data in (('stage3',s3),('stage4',s4)):
        if not data:continue
        for n,r in data['results'].items():
            for g,v in r['groups'].items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=g,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],Q=v['Q']['passed']))
            for dn,v in r.get('deformation',{}).items():
                if v.get('unmeasured'):continue
                for m,ch in v['pooled'].items():
                    for c,x in enumerate(ch):rows.append(dict(stage=st,model=n,group=f'deformation:{dn}',metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi'],Q=v['Q']['passed'] if isinstance(v['Q'],dict) and 'passed' in v['Q'] else None))
    if s4:
        for n,r in s4['results'].items():
            for split,key in (('r0123','per_matrix'),('r123','per_matrix_r123')):
                for k,v in r.get(key,{}).items():
                    for m,ch in v.items():
                        for c,x in enumerate(ch):pm.append(dict(model=n,replicas=split,matrix=k,metric=m,channel='RGB'[c],rho=x['rho'],lo=x['lo'],hi=x['hi']))
    for name,data in (('group_metrics.csv',rows),('per_matrix.csv',pm)):
        if data:
            with open(F/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for p in ('tests/tests.json','cost/gate.json','cost/final.json','probe/lanes.json','stage3/gate.json','mech/mech.json'):
        if (ROOT/p).exists():shutil.copy(ROOT/p,F/p.replace('/','_'))
    runs={}
    for n in CFG['runs']:
        p=ROOT/'runs'/f'{n}.pt'
        if p.exists():
            import torch;x=torch.load(p,map_location='cpu',weights_only=False)
            runs[n]={k:x.get(k) for k in ('J_initial','J_final','L_sym_final','converged','last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','parameters')}
    mech=rd('mech/mech.json')
    write_new(F/'summary.json',json.loads(json.dumps(dict(tests=rd('tests/tests.json'),cost_gate=rd('cost/gate.json'),cost_final=rd('cost/final.json'),
        fit_gate=s3['gate'] if s3 else None,mechanism=mech['criterion'] if mech else None,stage4=s4['decision'] if s4 else None,runs=runs),default=float)))
    write_new(F/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*')) if p.name!='manifest.json'})
    print('EXPORT_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
