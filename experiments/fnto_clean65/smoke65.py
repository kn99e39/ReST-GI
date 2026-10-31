"""Non-evidence smoke harness in a scratch root: replays Phase-A-only qualification, one tiny top-up action, and a
100-step A/B runner/evaluation/compare/export pass. Nothing here is a scientific result; the scratch root is discarded."""
import os,sys,json,subprocess,shutil
from pathlib import Path
HOME=Path.home();S=Path(os.environ.get('C65X_ROOT',HOME/'rest_clean65_smoke'));HERE=Path(__file__).resolve().parent;ORIG=HOME/'rest_persist_v1';PY=sys.executable
assert S.name.endswith('smoke');os.environ['C65X_ROOT']=str(S)
from chain65 import RUNENV
os.environ.update(RUNENV,TMPDIR=str(S/'tmp'),P53_TMPDIR=str(S/'tmp'))

def sub(name,cmd,env=None):
    with open(S/f'{name}.log','w') as f:subprocess.run(list(map(str,cmd)),env=env or os.environ.copy(),cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('SMOKE_OK',name,flush=True)

def main():
    S.mkdir(parents=True,exist_ok=False);(S/'tmp').mkdir();(S/'source_commit.txt').write_text('smoke\n')
    sub('tests_phaseA',[PY,HERE/'test65.py']);sub('tests_ext',[PY,HERE/'test65b.py'])
    # 1. Phase-A-only replay through the combined estimator (no top-up fields): all rows/banks must reproduce.
    (S/'topup_complete.json').write_text(json.dumps(dict(packets=0,photons=0,physical=True,estimator_source_identical_to_phaseA=True,records=[])))
    code=("import ext65,sys;ext65.TOP['fields']=[];ext65.TOP['total_packets']=0;ext65.TOP['total_photons']=0;"
          "import qualify65;sys.argv=['qualify65','--root',r'%s'];qualify65.main()"%S)
    sub('qualify_replay',[PY,'-c',code]);q=json.loads((S/'qualification.json').read_text())['summary']
    assert q['validity']['unchanged_fields_bitwise_phaseA'];print('REPLAY_DRIFT',json.dumps(q['validity']['unchanged_field_analysis_max_relative_drift']),q['delta_loss_precision_failed_cells'],flush=True)
    assert q['delta_loss_precision_failed_cells']==107
    sub('label_noise',[PY,HERE/'labelnoise65.py'])
    # 2. One tiny real top-up action through the unchanged estimator (2M photons, smoke seed).
    code=("import torch;from common65 import reference;from ext65 import ORIG;from moments65 import observed_action;RG,AO=reference(ORIG);"
          "L=torch.load(ORIG/'data/lights.pt')['train'];m,V=RG.mesh('rest');fm,VC,par=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC);"
          "v=observed_action(AO,dom,L[84],2000000,1);print('PACKET_SMOKE',v['physical_passed'],v['observer_checks'],v['seconds']);assert v['physical_passed'] and all(v['observer_checks'].values())")
    sub('packet',[PY,'-c',code])
    # 3. Arm construction + 100-step runner, unchanged evaluation/cost, invariants, compare, export.
    st=json.loads((S/'qualification_status.json').read_text());st['status']='READY_FOR_CONTROLLED_TRAINING';(S/'qualification_status.json').write_text(json.dumps(st))
    m=json.loads((S/'labels_v2_manifest.json').read_text());m['qualified_for_controlled_intervention']=True;(S/'labels_v2_manifest.json').write_text(json.dumps(m))
    (S/'fresh').mkdir()
    for name in ('independent_reference.npz','independent_lights.pt'):(S/'fresh'/name).symlink_to(HOME/'rest_direction64'/name)
    (S/'fresh/independent_complete.json').write_text(json.dumps(dict(evaluation_only=True,total_all24_qualified=True,total_qualified_cells=24)))
    sub('arms',[PY,HERE/'arms65.py'])
    run=ORIG/'repo/experiments/fnto_persist59/run.sh';pol=dict(WL57_ACTION_CHUNK='32000000',WL57_EDGE_CHUNK='65536',WL57_EVAL_BATCH='4',P59_ATTEMPT='0')
    for seed in (5901,5902):
        for arm in ('A','B'):sub(f'train_{arm}_{seed}',['bash',run,'train.py','P',seed,100],dict(os.environ,P59=str(S/'arms'/arm),**pol))
    for arm in ('A','B'):sub(f'evaluate_{arm}',['bash',run,'evaluate.py'],dict(os.environ,P59=str(S/'arms'/arm),**pol))
    for arm in ('A','B'):sub(f'cost_{arm}',['bash',run,'cost.py'],dict(os.environ,P59=str(S/'arms'/arm),**pol))
    sub('invariants',[PY,HERE/'invariants65.py']);sub('compare',[PY,HERE/'compare65.py'],dict(os.environ,C65X_SMOKE_STEPS='100'))
    sub('export',[PY,HERE/'export65.py',S,S/'review'])
    print('SMOKE_COMPLETE',flush=True)
if __name__=='__main__':main()
