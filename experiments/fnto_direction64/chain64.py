"""Durable serial execution; any failed stage stops, no automatic scientific repair."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
ROOT=Path.home()/'rest_direction64';HERE=Path(__file__).resolve().parent;OLD=Path.home()/'rest_persist_v1/repo/experiments/fnto_persist59'
def run(name,args,env):
    occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    started=time.time()
    status={'stage':name,'state':'RUNNING','time':time.time()};(ROOT/'status.json').write_text(json.dumps(status))
    with open(ROOT/f'{name}.log','w') as f:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    receipt={'stage':name,'started':started,'ended':time.time(),'seconds':time.time()-started,'GPU_processes_before':occupancy,'GPU_processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)}
    (ROOT/f'{name}_stage.json').write_text(json.dumps(receipt,indent=2))
    print('DONE',name,flush=True)
def main():
    assert (HERE/'protocol/fixed_objective.json').read_bytes()==(ROOT/'fixed_objective.json').read_bytes()
    env=os.environ.copy();env['PYTHONPATH']=str(OLD);env['WL57_TRAIN_TARGETS']=str(Path.home()/'rest_persist_v1/data/training_targets')
    # Independent evaluation population qualified before any model training/outcomes.
    run('fresh',[HERE/'fresh64.py'],env)
    # Alternated arm order across seeds, same original budget.
    for seed,order in ((5901,('A','B')),(5902,('B','A'))):
        for arm in order:
            ar=ROOT/'arms'/arm;e=env|{'D64_ARM':arm,'P59':str(ar),'WL57':str(ar)}
            run(f'train_{arm}_{seed}',[HERE/'train64.py','P',seed,6000],e)
    for arm in ('A','B'):
        ar=ROOT/'arms'/arm;e=env|{'D64_ARM':arm,'P59':str(ar),'WL57':str(ar)}
        run(f'evaluate_{arm}',[OLD/'evaluate.py'],e)
        run(f'cost_{arm}',[OLD/'cost.py'],e)
    run('compare',[HERE/'compare64.py'],env)
    c=json.loads((ROOT/'comparison.json').read_text())
    (ROOT/'status.json').write_text(json.dumps({'state':'COMPLETE','K_eligible':c['coupled_GI_eligible'],'coupled_GI':'not started; requires original eligibility and separate review of applicable coupled protocol','time':time.time()}))
if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());(ROOT/'status.json').write_text(json.dumps({'state':'FAILED','time':time.time()}));raise
