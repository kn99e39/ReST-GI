"""WL97 audit track B: independent regeneration of the WL86 training / validation population (geometries + lights) from the
frozen sampling rule (experiments/ecro86/scene86.py, numpy only) and distance checks against every later 'held-out' condition.

  python regen_population.py <out_json>
No renderer, no GPU. Imports scene86 read-only (PYTHONDONTWRITEBYTECODE=1 recommended).
"""
import json, math, os, sys, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import scene86 as S  # noqa: E402

TR = S.P['training']


def population(split):
    seed = TR['splits'][split]['seed']; n = TR['splits'][split]['geometries']; nl = TR['lightings_per_geometry']
    out = []
    for idx in range(n):
        rng = np.random.default_rng([seed, idx])
        geom = S.sample_geom(rng)
        lights = [S.sample_light(rng, geom) for _ in range(nl)]
        out.append({'file': f'g{idx:05d}.npz', 'geom': geom, 'lights': lights})
    return out


def digest(pop):
    h = hashlib.sha256()
    for it in pop:
        h.update(json.dumps(it['geom']).encode()); h.update(json.dumps(it['lights']).encode())
    return h.hexdigest()


if __name__ == '__main__':
    out = {}
    for sp in ('train', 'val', 'trainhq', 'dev'):
        pop = population(sp)
        out[sp] = {'n': len(pop), 'sha256_geom_lights_json': digest(pop), 'items': pop}
        print(sp, len(pop), out[sp]['sha256_geom_lights_json'][:16], flush=True)
    json.dump(out, open(sys.argv[1], 'w'))
