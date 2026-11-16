"""Execution-only OOM recovery: resume saved model+Adam at step 155648."""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

BASE=Path.home()/'rest_8dna_ck21b'
UP=Path.home()/'NeuralShaderDemo/external/8dna26'
# Reuse exactly the admitted packed dataset/loader and scene registration.
definitions=runpy.run_path(str(BASE/'scripts/train_packed21b.py'),run_name='packed_recovery_definitions')
import torch
import lightning.pytorch.callbacks as callbacks
NativeCheckpoint=callbacks.ModelCheckpoint
class DurableEpochCheckpoint(NativeCheckpoint):
    def __init__(self,*args,**kwargs):
        # Retention only. Fixed final checkpoint remains the evaluation choice.
        kwargs['save_top_k']=-1
        kwargs['save_last']=True
        super().__init__(*args,**kwargs)
callbacks.ModelCheckpoint=DurableEpochCheckpoint
source=BASE/'training_packed/CK2_bx38/epoch=18-step=155648.ckpt'
root=BASE/'training_recovery19'
ckdir=root/'CK2_bx38'
ckdir.mkdir(parents=True,exist_ok=True)
initial=ckdir/'last.ckpt'
if initial.exists():
    raise RuntimeError('Recovery already initialized; inspect progress rather than resetting it')
saved=torch.load(source,map_location='cpu')
assert int(saved['epoch'])==18 and int(saved['global_step'])==155648
assert len(saved['optimizer_states'])==1
assert all(torch.isfinite(v).all().item() for v in saved['state_dict'].values() if torch.is_floating_point(v))
del saved
shutil.copy2(source,initial)
seed=21119
sys.argv=[str(UP/'train.py'),'--config',str(UP/'configs/default.yaml'),
    '--experiment_name','CK2_bx38','--log_path',str(root),'--max_epochs','30',
    '--device','0','--seed',str(seed),'--resume','scene=ck21b_bx38','dataset.SPP=32','val_spp=16']
record={'reason':'Original run OOM in validation after epoch21 updates while high-spp rendering overlapped.',
    'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'resume_epoch_zero_based':18,'resume_global_step':155648,'optimizer_preserved':True,
    'target_epochs':30,'target_global_step':245760,'argv':sys.argv,
    'rng':f'Fresh continuation seed {seed}; original checkpoint has no RNG state. Not bit-identical.',
    'lost_unsaved_optimizer_steps':24576,
    'retention':'Keep every completed epoch; native model/loss/Adam/batch/data/validation unchanged.',
    'evaluation':'Fixed final.ckpt, never best-by-image selection.'}
(BASE/'recovery19.json').write_text(json.dumps(record,indent=2)+'\n')
print('RECOVERY19',json.dumps(record),flush=True)
runpy.run_path(str(UP/'train.py'),run_name='__main__')
from figure import make
make(ckdir/'final.ckpt','final_epoch30',256)
