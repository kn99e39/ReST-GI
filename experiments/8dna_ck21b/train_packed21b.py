"""WL21B Amendment 1 migration: resume the native CK2 8DNA fit from the paused epoch-4 anchor with the admitted
execution-only data pipeline (packed CPU tuples, pinned staging, asynchronous CUDA prefetch).

Unchanged: upstream train.py / ModelTrainer / EightDNA (669K, FP32), objectives, Adam + clip 5, batch 32768,
128^4 fresh physical tuples per epoch with one uniform global permutation, dataset_reload 1, 30-epoch fixed schedule
(245760 optimizer steps in total, 40960 of them already in the anchor), scene / ER states / reference / figure rules.
Changed (execution only, disclosed): host tuple layout, host permutation kept on CPU, batch fetch by a producer thread
and a separate CUDA copy stream; resumed in a distinct log directory; a fresh seed for the RNG stream after the
execution boundary (the native checkpoint holds no RNG state, and reusing seed 21101 would replay the epoch-0 data stream).
"""
import gc, hashlib, json, os, runpy, shutil, sys, time, types
from pathlib import Path
UP = Path.home() / 'NeuralShaderDemo' / 'external' / '8dna26'
sys.path.insert(0, str(UP)); os.chdir(UP)
import bridge as B
import torch
import torch.utils.data as tud
B.dr.set_flag(B.dr.JitFlag.LoopRecord, True); B.dr.set_flag(B.dr.JitFlag.VCallRecord, True)
name = 'ck21b_bx38'
module = types.ModuleType('scenes.' + name); module.get_scene = B.training_scene
sys.modules['scenes.' + name] = module
import scenes
setattr(scenes, name, module)

import utils.dataset as UD
from packed_pipeline import PrefetchLoader, packed_dataset_class

Native = UD.PathSamplingDataset
Packed = packed_dataset_class(Native)


def host_resample(self):
    """One uniform global permutation per epoch, kept on the host (int64 CPU), hashed and logged."""
    B.dr.sync_thread(); gc.collect(); B.dr.flush_malloc_cache(); torch.cuda.empty_cache()
    count = self.H * self.H * self.N * self.N
    start = time.perf_counter()
    self.inds = torch.randperm(count, device='cpu')
    fingerprint = hashlib.sha256(self.inds.numpy().tobytes()).hexdigest()
    print('HOST_PERMUTATION', json.dumps({'count': count, 'sha256_int64_bytes': fingerprint,
                                          'seconds': time.perf_counter() - start}), flush=True)


Packed.resample = host_resample
UD.PathSamplingDataset = Packed                       # train.py does `from utils.dataset import PathSamplingDataset`

_OrigDL = tud.DataLoader


class _SwitchDL:
    """Return the prefetch loader for the packed training dataset; every other DataLoader is unchanged."""
    def __new__(cls, dataset, *args, **kwargs):
        if isinstance(dataset, Packed):
            return PrefetchLoader(dataset)
        return _OrigDL(dataset, *args, **kwargs)


tud.DataLoader = _SwitchDL                            # train.py does `from torch.utils.data import DataLoader`

BASE = B.BASE
ANCHOR = json.loads((BASE / 'pause_boundary.json').read_text())
LOG = BASE / 'training_packed'
EXP = 'CK2_bx38'
SEED = 21105


def main():
    anchor = Path(ANCHOR['checkpoint'])
    digest = hashlib.sha256(anchor.read_bytes()).hexdigest()
    if digest != ANCHOR['sha256']:
        raise RuntimeError('Anchor checkpoint hash mismatch')
    bench = json.loads((BASE / 'results' / 'pipeline_benchmark.json').read_text())
    if bench.get('status') != 'admitted':
        raise RuntimeError('Pipeline not admitted by the predeclared benchmark rule')
    ckdir = LOG / EXP
    ckdir.mkdir(parents=True, exist_ok=True)
    if not (ckdir / 'last.ckpt').exists():
        shutil.copy2(anchor, ckdir / 'last.ckpt')       # train.py --resume reads <log>/<exp>/last.ckpt
    sys.argv = [str(UP / 'train.py'), '--config', str(UP / 'configs' / 'default.yaml'), '--experiment_name', EXP,
                '--log_path', str(LOG), '--max_epochs', '30', '--device', '0', '--seed', str(SEED), '--resume',
                f'scene={name}', 'dataset.SPP=32', 'val_spp=16']
    (BASE / 'training_command_packed.json').write_text(json.dumps({
        'argv': sys.argv, 'anchor': ANCHOR, 'anchor_sha256_verified': digest, 'benchmark_speedup': bench.get('speedup'),
        'target_epochs': 30, 'native_steps_per_epoch': 8192, 'final_selection': 'fixed final.ckpt; no image-based selection',
        'rng': f'fresh seed {SEED} after the execution boundary (native checkpoint has no RNG state); not bit-identical'},
        indent=2) + '\n')
    print('PACKED_TRAIN', sys.argv, flush=True)
    runpy.run_path(str(UP / 'train.py'), run_name='__main__')
    final = ckdir / 'final.ckpt'
    from figure import make
    make(final, 'final_epoch30', 256)


if __name__ == '__main__':
    main()
