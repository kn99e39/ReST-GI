"""Register the CK2 asset and run unchanged upstream train.py/default model."""
import gc, json, os, runpy, sys, time, types
from pathlib import Path
UP=Path.home()/'NeuralShaderDemo'/'external'/'8dna26'
sys.path.insert(0,str(UP)); os.chdir(UP)
import bridge as B
import torch
import lightning as pl
from lightning.pytorch.callbacks import Callback
B.dr.set_flag(B.dr.JitFlag.LoopRecord,True); B.dr.set_flag(B.dr.JitFlag.VCallRecord,True)
name='ck21b_bx38'
module=types.ModuleType('scenes.'+name); module.get_scene=B.training_scene
sys.modules['scenes.'+name]=module
import scenes
setattr(scenes,name,module)

# 128^4 int64 indices fit, but CUDA randperm's extra sorting workspace does not
# fit this 12GB device alongside the persistent index array. Preserve the full
# buffer and uniform global permutation; generate that permutation on the host.
from utils.dataset import PathSamplingDataset
def host_resample(self):
    B.dr.sync_thread(); gc.collect(); B.dr.flush_malloc_cache(); torch.cuda.empty_cache()
    count=self.H*self.H*self.N*self.N
    start=time.perf_counter(); permutation=torch.randperm(count,device='cpu')
    import hashlib
    fingerprint=hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    chunk=1<<20
    for first in range(0,count,chunk):
        last=min(first+chunk,count); self.inds[first:last].copy_(permutation[first:last])
    print('HOST_PERMUTATION',json.dumps({'count':count,'sha256_int64_bytes':fingerprint,'seconds':time.perf_counter()-start}),flush=True)
    del permutation; gc.collect(); torch.cuda.empty_cache()
PathSamplingDataset.resample=host_resample

class Preview(Callback):
    def on_train_epoch_end(self,trainer,model):
        if trainer.current_epoch!=0: return
        checkpoint=B.BASE/'results'/'epoch01_preview.ckpt'
        trainer.save_checkpoint(str(checkpoint))
        was_training=model.training
        model.eval(); torch.cuda.empty_cache()
        from figure import make
        with torch.no_grad(): make(checkpoint,'epoch01_preview',128)
        gc.collect(); torch.cuda.empty_cache()
        B.dr.set_flag(B.dr.JitFlag.LoopRecord,True); B.dr.set_flag(B.dr.JitFlag.VCallRecord,True)
        model.train(was_training)

OriginalTrainer=pl.Trainer
class CKTrainer(OriginalTrainer):
    def __init__(self,*args,**kwargs):
        kwargs['callbacks']=list(kwargs.get('callbacks',[]))+[Preview()]
        super().__init__(*args,**kwargs)
pl.Trainer=CKTrainer

def main():
    if json.loads((B.BASE/'results'/'preflight.json').read_text())['status']!='pass': raise RuntimeError('Preflight not passed')
    # Pure physical path sampling can use upstream's default recorded loops;
    # qualify the flags before constructing the large native training buffer.
    from preflight import render
    import numpy as np
    B.dr.set_flag(B.dr.JitFlag.LoopRecord,False); B.dr.set_flag(B.dr.JitFlag.VCallRecord,False)
    small=B.mi.load_dict(B.make_scene(64,max_depth=7)[0])
    wave,_=render(small,small.integrator(),16,21180)
    B.dr.set_flag(B.dr.JitFlag.LoopRecord,True); B.dr.set_flag(B.dr.JitFlag.VCallRecord,True)
    recorded,_=render(small,small.integrator(),16,21180)
    probe={'res':64,'spp':16,'seed':21180,'max_abs':float(np.abs(recorded-wave).max()),'relative_l1':float(np.abs(recorded-wave).sum()/max(np.abs(wave).sum(),1e-30))}
    (B.BASE/'results'/'training_jit_probe.json').write_text(json.dumps(probe,indent=2)+'\n')
    print('TRAIN_JIT',json.dumps(probe),flush=True)
    if probe['relative_l1']>1e-4: raise RuntimeError('Recorded native training runtime disagrees with wavefront physics')
    del small,wave,recorded; gc.collect(); torch.cuda.empty_cache()
    sys.argv=[str(UP/'train.py'),'--config',str(UP/'configs'/'default.yaml'),'--experiment_name','CK2_bx38','--log_path',str(B.BASE/'training'),'--max_epochs','30','--device','0','--seed','21101',f'scene={name}','dataset.SPP=32','val_spp=16']
    (B.BASE/'training_command.json').write_text(json.dumps({'argv':sys.argv,'target_epochs':30,'native_steps_per_epoch':8192,'final_selection':'fixed final.ckpt; no image-based selection'},indent=2)+'\n')
    print('NATIVE_TRAIN',sys.argv,flush=True)
    runpy.run_path(str(UP/'train.py'),run_name='__main__')
    final=B.BASE/'training'/'CK2_bx38'/'final.ckpt'
    from figure import make
    make(final,'final_epoch30',256)

if __name__=='__main__': main()
