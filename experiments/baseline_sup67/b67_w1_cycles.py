"""WL67 W1 quality-matched physical Full Recompute: Cycles volume re-render after the edit (protocol baseline_sup67_v1).

Runs inside the official Blender 3.5.1 (CUDA), with the WL55/WL56 renderer code and WL56 fixture unchanged:
    blender -b --factory-startup --python-use-system-env --python-exit-code 1 --python b67_w1_cycles.py -- <out> ladder <state>
    blender ... --python b67_w1_cycles.py -- <out> bounce
Every update = scene reset + build (current geometry/material/emitter) + four camera renders, each timed separately.
Seeds are disjoint from the WL56 references (5601-5614). Settings are read back from bpy for every render.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fnto_sel56')); sys.path.insert(0, str(HERE.parent / 'fnto_k55'))
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import s56_fixtures as SF  # noqa: E402
import k55_cycles as KC  # noqa: E402

PRIMARY = ('top', 'gapdown', 'gapup', 'bot')
LADDER = (1, 4, 16, 64, 256, 1024)
STATES = ('rest', 'dent', 'rigid')


def seed_of(state, li, r):
    return 670000 + 1000 * STATES.index(state) + 10 * li + r


def settings(sc):
    c = sc.cycles
    keys = ('device', 'samples', 'seed', 'use_denoising', 'use_adaptive_sampling', 'max_bounces', 'diffuse_bounces', 'glossy_bounces',
            'transmission_bounces', 'volume_bounces', 'transparent_max_bounces', 'sample_clamp_direct', 'sample_clamp_indirect',
            'caustics_reflective', 'caustics_refractive', 'blur_glossy', 'pixel_filter_type', 'filter_width', 'use_light_tree')
    out = {k: getattr(c, k) for k in keys}
    out.update(film_transparent=sc.render.film_transparent, view_transform=sc.view_settings.view_transform,
               compute_device_type=bpy.context.preferences.addons['cycles'].preferences.compute_device_type,
               blender=bpy.app.version_string, build_hash=bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash)
    return {k: (float(v) if isinstance(v, float) else v) for k, v in out.items()}


def one_update(out, state, spp, seed, cams, tag, mutate=None):
    """Complete physical recompute of the current scene for the fixed query workload; returns timing record."""
    sd = SF.scene(state); rec = dict(state=state, spp=spp, seed=seed, tag=tag, cams={})
    a = time.perf_counter(); sc = KC.setup(spp, seed, 16); rec['reset_seconds'] = time.perf_counter() - a
    if mutate: mutate(sc)
    a = time.perf_counter(); KC.build(sd); rec['build_seconds'] = time.perf_counter() - a
    rec['settings'] = settings(sc); geo = {}
    for cam in cams:
        c = dict(sd['cams'][cam])
        a = time.perf_counter(); ob, g = KC.PC.ortho(cam, c, c['res']); tc = time.perf_counter() - a; geo[cam] = g
        a = time.perf_counter(); img = KC.PC.render(sc, ob, c['res'], spp, seed); dt = time.perf_counter() - a
        np.save(out / f'{tag}_{state}_{cam}_spp{spp}_s{seed}.npy', img.astype(np.float32))
        rec['cams'][cam] = dict(render_seconds=dt, camera_setup_seconds=tc, res=c['res'])
    rec['update_seconds'] = rec['reset_seconds'] + rec['build_seconds'] + sum(v['render_seconds'] + v['camera_setup_seconds'] for v in rec['cams'].values())
    json.dump(geo, open(out / f'ccam_{tag}_{state}.json', 'w'), indent=1)
    with open(out / 'timings.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(time.strftime('%H:%M:%S'), tag, state, spp, seed, round(rec['update_seconds'], 2), flush=True)
    return rec


def warmup(out, state):
    """Device/kernel initialisation, excluded from update timing and reported."""
    a = time.perf_counter(); one_update(out, state, 1, 679999, ('top',), 'warmup'); return time.perf_counter() - a


def main():
    a = sys.argv[sys.argv.index('--') + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    KC.DEVICE = 'CUDA'; KC.PC.OUT = out; KC.PC.LOG = open(out / 'cycles.log', 'a'); KC.KF.scene_def = SF.scene
    proc = dict(process_start=time.strftime('%Y-%m-%dT%H:%M:%S'), mode=a[1])
    if a[1] == 'ladder':
        state = a[2]; proc['warmup_seconds'] = warmup(out, state)
        for li, spp in enumerate(LADDER):
            for r in (0, 1):
                one_update(out, state, spp, seed_of(state, li, r), PRIMARY, 'ladder')
    elif a[1] == 'bounce':
        proc['warmup_seconds'] = warmup(out, 'dent')

        def rna_native(sc):
            # RNA training_dataset/blender_render_dataset.py: max_bounces 32, sample_clamp_direct 20 (all else = reference).
            sc.cycles.max_bounces = 32; sc.cycles.sample_clamp_direct = 20
        one_update(out, 'dent', 1024, 679001, ('top', 'bot'), 'bounce', mutate=rna_native)
        # Same seed / settings except the two RNA-native values: isolates their effect from sampling noise.
        one_update(out, 'dent', 1024, 679001, ('top', 'bot'), 'bouncectl')
    with open(out / 'process.jsonl', 'a') as f:
        f.write(json.dumps(proc) + '\n')


if __name__ == '__main__':
    main()
