"""Backend selection for the ECRO live runtime, by shims only (no frozen source is modified).

The frozen sources hard-code CUDA in exactly three places: pt86 (`layered81.init('cuda_ad_rgb')`), rt101
(`mi.set_variant('cuda_ad_rgb')`, `dev = torch.device('cuda')`) and the `torch.cuda.synchronize()` calls of rt101 / live110.
For the LLVM (CPU) backend, which is what macOS offers (Dr.Jit has no Metal backend), the shims are:
  * mitsuba.set_variant: 'cuda_*' -> 'llvm_*'           (same Dr.Jit program, CPU code generation through LLVM + Embree)
  * rt101.dev = torch.device('cpu')                     (torch tensors in host memory; DLPack hand-off to Dr.Jit LLVM arrays)
  * torch.cuda.synchronize -> no-op (or torch.mps.synchronize when the generator runs on MPS)
  * kernel replay off by default on LLVM (rt101.FREEZE = False, i.e. WL101's B_eager execution; see configure)
Must be called before anything imports mitsuba / rt101.
"""
import glob, os, platform, sys
from dataclasses import dataclass, field

LLVM_CANDIDATES = {
    'Darwin': ['/opt/homebrew/opt/llvm/lib/libLLVM.dylib', '/usr/local/opt/llvm/lib/libLLVM.dylib',
               '/opt/homebrew/opt/llvm@*/lib/libLLVM.dylib', '/usr/local/opt/llvm@*/lib/libLLVM.dylib'],
    'Linux': ['/usr/lib64/libLLVM*.so', '/usr/lib/x86_64-linux-gnu/libLLVM*.so*', '/usr/lib/llvm-*/lib/libLLVM*.so*'],
}


@dataclass
class Backend:
    name: str                     # 'cuda' | 'llvm'
    variant: str
    generator_device: str         # 'cuda' | 'cpu' | 'mps'
    R: object = None              # the rt101 module
    info: dict = field(default_factory=dict)


def _find_libllvm():
    if os.environ.get('DRJIT_LIBLLVM_PATH'):
        return os.environ['DRJIT_LIBLLVM_PATH']
    if getattr(sys, '_MEIPASS', None):                                  # bundled copy (build_macos.sh)
        for n in ('libLLVM.dylib', 'libLLVM.so'):
            p = os.path.join(sys._MEIPASS, n)
            if os.path.exists(p):
                return p
    for pat in LLVM_CANDIDATES.get(platform.system(), []):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def configure(backend='auto', generator_device='auto', mi_dir=None, freeze='auto'):
    if 'rt101' in sys.modules:
        raise RuntimeError('backend.configure() must run before rt101 is imported')
    mi_dir = mi_dir or os.environ.get('DYN101_MI_DIR')
    if mi_dir:
        sys.path.insert(0, mi_dir)
    if backend in ('auto', 'llvm'):
        lib = _find_libllvm()
        if lib:
            os.environ['DRJIT_LIBLLVM_PATH'] = lib
    import mitsuba as mi
    import torch
    variants = set(mi.variants())
    cuda_ok = torch.cuda.is_available() and 'cuda_ad_rgb' in variants
    if backend == 'auto':
        backend = 'cuda' if cuda_ok else 'llvm'
    if backend == 'cuda' and not cuda_ok:
        raise RuntimeError('CUDA backend requested but torch CUDA or the cuda_ad_rgb variant is unavailable')
    if backend == 'llvm' and 'llvm_ad_rgb' not in variants:
        raise RuntimeError(f'llvm_ad_rgb variant missing from this Mitsuba build ({sorted(variants)})')
    if generator_device == 'auto':
        generator_device = 'cuda' if backend == 'cuda' else 'cpu'
    if generator_device == 'mps' and not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
        raise RuntimeError('MPS generator requested but torch.backends.mps is unavailable')
    if backend == 'cuda' and generator_device != 'cuda':
        raise RuntimeError('with the CUDA backend the generator runs on CUDA (shared device memory)')
    if backend == 'llvm':
        _sv = mi.set_variant
        mi.set_variant = lambda v, *a, **k: _sv(v.replace('cuda_', 'llvm_'), *a, **k)
        if generator_device == 'mps':
            torch.cuda.synchronize = lambda *a, **k: torch.mps.synchronize()
        else:
            torch.cuda.synchronize = lambda *a, **k: None
    # rt101 asserts its Mitsuba version against DYN101_MI_VERSION (WL101 validated 3.8.0 on LabServer69)
    os.environ.setdefault('DYN101_MI_VERSION', mi.__version__)
    os.environ.setdefault('FREEZE', '1')
    import rt101 as R
    import live110  # noqa: F401  (its import-time `assert R.FREEZE` guards the CUDA evidence configuration)
    if backend == 'llvm':
        R.dev = torch.device('cpu')
    if freeze == 'auto':
        freeze = 'on' if backend == 'cuda' else 'off'
    # Dr.Jit 1.3.1 kernel replay (dr.freeze) is NOT reliable on the LLVM backend for this program: re-recordings with failed
    # dry runs and intermittently corrupted relation state (up to 24k of 650k rays, table rel L1 up to 8e-2 vs CUDA), while the
    # same code run eagerly matches CUDA (<= 2 tie rays, table rel L1 <= 3e-6). WL110A diagnosis; the LLVM default is eager.
    # rt101.maybe_freeze / update_state read the module global at call time, so this takes effect for every RT built later.
    R.FREEZE = (freeze == 'on')
    info = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
            'mitsuba': mi.__version__, 'drjit': R.dr.__version__, 'torch': torch.__version__, 'variant': mi.variant(),
            'libllvm': os.environ.get('DRJIT_LIBLLVM_PATH') if backend == 'llvm' else None,
            'torch_threads': torch.get_num_threads(),
            'device_name': torch.cuda.get_device_name() if backend == 'cuda' else platform.processor() or platform.machine(),
            'wl101_validated_stack': mi.__version__ == '3.8.0' and R.dr.__version__ == '1.3.1'}
    info['kernel_replay'] = R.FREEZE
    return Backend(backend, mi.variant(), generator_device, R, info)


def sync(be):
    be.R.dr.sync_thread()
    import torch
    if be.name == 'cuda':
        torch.cuda.synchronize()
    elif be.generator_device == 'mps':
        torch.mps.synchronize()
