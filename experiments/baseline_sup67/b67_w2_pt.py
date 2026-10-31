"""WL67 W2 (teaset common light): quality-matched Mitsuba path-traced Full Recompute and native 8DNA inference re-execution.

python b67_w2_pt.py <out> ladder     # PT spp ladder T0/T3 x 2 realizations + 4096-spp cross-machine check
python b67_w2_pt.py <out> eightdna   # released 8DNA, WL22 'attached' contract, frozen T0/T3 at the WL22 evaluation spp + 1-spp timing
Scene, lighting, integrator and chunked rendering are the unchanged WL22 code (ednalib, teaset_parts); seeds are disjoint from GT.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '8dna_deformation_replication'))
import numpy as np  # noqa: E402

import ednalib as L  # noqa: E402
import teaset_parts as T  # noqa: E402

PROTO22 = HERE.parent / '8dna_deformation_replication/protocol/teaset_cross_backbone_locked.json'
LADDER = (1, 4, 16, 64, 256, 1024)
STATES = ('T0', 'T3')


def seed_of(state, li, r):
    return 6700000 + 100000 * STATES.index(state) + 1000 * li + 100 * r


def gpu_peers():
    import subprocess
    return subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()


def log(out, rec):
    with open(out / 'timings.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(time.strftime('%H:%M:%S'), json.dumps({k: v for k, v in rec.items() if k != 'peers'}), flush=True)


def load(mi, proto, state):
    import torch
    torch.cuda.synchronize(); a = time.perf_counter()
    scene = mi.load_dict(T.scene_dict(proto['res'], proto['states'][state], proto.get('lighting')))
    torch.cuda.synchronize(); return scene, time.perf_counter() - a


def ladder(out, mi, proto):
    for state in STATES:
        scene, _ = load(mi, proto, state)
        _, tw = L.render_chunked(scene, scene.integrator(), 1, 1, 6799999)       # warm-up, excluded
        log(out, dict(kind='warmup', state=state, seconds=tw))
        for li, spp in enumerate(LADDER):
            for r in (0, 1):
                peers = gpu_peers(); scene, tl = load(mi, proto, state)            # geometry update = fresh scene load
                seed = seed_of(state, li, r); img, tr = L.render_chunked(scene, scene.integrator(), spp, min(spp, 32), seed)
                np.save(out / f'pt_{state}_spp{spp}_r{r}.npy', img.astype(np.float32))
                log(out, dict(kind='pt', state=state, spp=spp, r=r, seed=seed, load_seconds=tl, render_seconds=tr,
                              update_seconds=tl + tr, peers=peers))
    scene, tl = load(mi, proto, 'T3'); img, tr = L.render_chunked(scene, scene.integrator(), 4096, 32, 6790000)
    np.save(out / 'pt_T3_spp4096_xcheck.npy', img.astype(np.float32))
    log(out, dict(kind='xcheck', state='T3', spp=4096, seed=6790000, load_seconds=tl, render_seconds=tr))


def eightdna(out, mi, proto):
    ck = L.sha256(L.checkpoint_path(proto['asset']))
    assert ck == proto['checkpoint_sha256'], 'released 8DNA checkpoint differs from the WL22 locked protocol'
    nr = proto['neural']; scene0, _ = load(mi, proto, 'T0'); bbox0 = scene0.shapes()[0].bbox()
    import torch
    torch.cuda.synchronize(); a = time.perf_counter()
    integ = L.load_neural_integrator(proto['asset']); base = integ.asset_models[0]
    torch.cuda.synchronize(); log(out, dict(kind='8dna_load', seconds=time.perf_counter() - a, checkpoint_sha256=ck))
    for state in STATES:
        tr_ = proto['states'][state]; scene, tl = load(mi, proto, state)
        T.configure(integ, base, scene, 'attached', tr_, bbox0)
        one = [L.render_chunked(scene, integ, 1, 1, 6800000 + k)[1] for k in range(6)]       # first = warm-up
        log(out, dict(kind='8dna_1spp', state=state, warmup_seconds=one[0], seconds=one[1:], load_seconds=tl))
        peers = gpu_peers(); img, t = L.render_chunked(scene, integ, nr['spp'], nr['chunk'], nr['seed'])
        np.save(out / f'8dna_attached_{state}.npy', img.astype(np.float32))
        log(out, dict(kind='8dna_eval', state=state, spp=nr['spp'], chunk=nr['chunk'], seed=nr['seed'], seconds=t, load_seconds=tl, peers=peers))


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    proto = json.loads(PROTO22.read_text(encoding='utf-8'))
    mi, dr = L.init_upstream()
    log(out, dict(kind='env', mode=sys.argv[2], environment=L.environment_record(), peers_at_start=gpu_peers()))
    {'ladder': ladder, 'eightdna': eightdna}[sys.argv[2]](out, mi, proto)
    sys.stdout.flush(); __import__('os')._exit(0)
