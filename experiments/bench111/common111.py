"""WL111 common helpers: scene construction for the ECRO fixtures, seeds, timing, denoiser, metrics. No method-specific tuning."""
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LUMA = np.array([0.2126, 0.7152, 0.0722])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def splitmix64(x):
    """Independent launch seeds (consecutive seed*1000003 streams of the ECRO PCG are correlated; memory/WL109A)."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def seed_of(*parts):
    """Deterministic, well-mixed seed < 2^40 from a tuple of ints/strings (seed*1000003 + launch stays < 2^64)."""
    h = int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16)
    return splitmix64(h) % (1 << 40)


# ------------------------------------------------------------------ GPU exclusivity / snapshots
def foreign_gpu_pids():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    pids = []
    for row in out.strip().splitlines():
        p = row.split(',')[0].strip()
        if p.isdigit() and int(p) != os.getpid():
            pids.append(int(p))
    return pids


def gpu_snapshot():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used,clocks.sm,clocks.max.sm,'
                            'temperature.gpu,power.draw', '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        q = f'nvidia-smi failed: {e}'
    return {'query': q, 'foreign_compute_pids': foreign_gpu_pids(), 't': time.time()}


def versions(mi=None, dr=None, torch=None):
    v = {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(), 'host': platform.node()}
    if mi is not None:
        v['mitsuba'] = mi.__version__
        v['variant'] = mi.variant()
    if dr is not None:
        v['drjit'] = dr.__version__
    if torch is not None:
        v['torch'] = torch.__version__
        v['cuda'] = torch.version.cuda
        v['gpu'] = torch.cuda.get_device_name(0)
    return v


# ------------------------------------------------------------------ images and metrics
def ldr(x):
    """Display transform used by WL102 / TransGI reproduction: clip(x, 0, 1)^(1/2.2) (exposure 1, no tone curve)."""
    return np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)


def metrics(x, ref, mask=None, ref_noise_var=None):
    """x, ref: [H, W, 3] linear HDR. ref_noise_var: per-pixel-channel variance of the reference mean (from halves) or None.
    Returns full-precision floats. RMSE_db = sqrt(max(MSE - mean(ref_noise_var), 0)) (reference-noise debiased)."""
    if mask is None:
        mask = np.ones(ref.shape[:2], bool)
    d = (x - ref)[mask]
    r = ref[mask]
    mse = float(np.mean(d ** 2))
    nv = float(np.mean(ref_noise_var[mask])) if ref_noise_var is not None else 0.0
    mr = float(np.mean(r))
    out = {
        'n_px': int(mask.sum()),
        'mean_ref': mr,
        'mse': mse,
        'ref_noise_mse': nv,
        'rmse': math.sqrt(mse),
        'rmse_db': math.sqrt(max(mse - nv, 0.0)),
        'rel_rmse': math.sqrt(mse) / mr,
        'rel_rmse_db': math.sqrt(max(mse - nv, 0.0)) / mr,
        'rel_l1': float(np.mean(np.abs(d))) / mr,
        'rel_bias': (float(np.mean(x[mask])) - mr) / mr,
    }
    lx, lr = ldr(x)[mask], ldr(ref)[mask]
    out['ldr_rmse'] = math.sqrt(float(np.mean((lx - lr) ** 2)))
    out['ldr_l1'] = float(np.mean(np.abs(lx - lr)))
    out['psnr_ldr'] = 10 * math.log10(1.0 / max(out['ldr_rmse'] ** 2, 1e-20))
    return out


def ssim_ldr(x, ref):
    """Global SSIM of LDR images (gaussian sigma 1.5, as skimage); secondary evidence only."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(ldr(ref), ldr(x), channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                          use_sample_covariance=False))
    except Exception:
        return float('nan')


def luminance(x):
    return x @ LUMA
