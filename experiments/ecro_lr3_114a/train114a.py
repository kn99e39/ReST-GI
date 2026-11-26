"""WL114A LR3 training: the unchanged WL102 recipe (train102: ECRO v0 735,664 parameters, WL92A OWN loss, Adam 1e-3 -> 1e-5 cosine,
warm-up 500, clip 10, 24,000 steps, final step, one item = 4 lights, no area floor) on the WL114A dataset with D6 (fh) or full-depth
(fhF) global targets. The ONLY difference between the two arms of a seed is the global target key.

  python train114a.py DATA_ROOT OUT_DIR DEPTH SEED [STEPS]      DEPTH in D6, F
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_lr3_102'))
os.environ.pop('WL102_AREA_FLOOR', None)
import train102 as T  # noqa: E402

DEPTH = 'D6'


class Item(T.Item):
    def __init__(self, path):
        z = np.load(path)
        gk = 'fh' if DEPTH == 'D6' else 'fhF'
        self.s, self.f2, self.fh = z['s'], z['f2'], z[gk]
        if 'f2b' in z.files:
            self.f2 = .5 * (self.f2 + z['f2b']); self.fh = .5 * (self.fh + z[gk + 'b'])


if __name__ == '__main__':
    root, out, DEPTH, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    assert DEPTH in ('D6', 'F')
    T.Item = Item
    T.TR['seed'] = seed
    steps = sys.argv[5] if len(sys.argv) > 5 else None
    T.main(root, out, steps)
    j = Path(out) / 'training.json'
    d = json.loads(j.read_text()); d['depth'] = DEPTH; d['wl114a'] = 'train102 recipe unchanged; global target key ' + ('fh' if DEPTH == 'D6' else 'fhF')
    j.write_text(json.dumps(d, indent=1))
