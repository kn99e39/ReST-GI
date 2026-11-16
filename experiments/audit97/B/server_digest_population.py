"""WL97 audit track B (server side, read-only): digest of the geom / lights JSON actually stored in the WL86 data files, plus file
sha256 prefixes, to compare with the local regeneration (regen_population.py).

  python server_digest_population.py <R86> <out_json>
"""
import glob, hashlib, json, os, sys
import numpy as np

R86, outfn = sys.argv[1], sys.argv[2]
out = {}
for sp in ('train', 'val', 'trainhq'):
    fns = sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz')))
    h = hashlib.sha256(); items = []
    for fn in fns:
        z = np.load(fn)
        g = json.loads(str(z['geom'])); ls = json.loads(str(z['lights']))
        h.update(json.dumps(g).encode()); h.update(json.dumps(ls).encode())
        items.append({'file': os.path.basename(fn), 'geom': g, 'lights': ls, 'keys': sorted(z.files), 'f_shape': list(z['f'].shape),
                      'has_fb': 'fb' in z.files})
    out[sp] = {'n': len(fns), 'sha256_geom_lights_json': h.hexdigest(), 'items': items}
    print(sp, len(fns), h.hexdigest()[:16], flush=True)
json.dump(out, open(outfn, 'w'))
