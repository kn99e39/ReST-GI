"""Arm A (frozen stack) reference dumps for equivalence tests: full state, source, table B and same-seed frames.
  python dump_a101.py <case> <out.npz> <spec>   spec: 'test:0,1,2' (WL100 test configs with their light 0)
Each state is reached by the frozen selective edit sequence from the first one (so B is the frozen lifecycle output).
"""
import json, sys
import numpy as np
import common101 as K

case, out, spec = sys.argv[1], sys.argv[2], sys.argv[3]
rows = K.configs(case)['test']
idx = [int(v) for v in spec.split(':')[1].split(',')]
model, ck = K.load_model(case)
A = K.FrozenECRO(case, rows[idx[0]]['geom'], model)
cam = K.cam_of(case, 512)
res = {}
for n, i in enumerate(idx):
    T = K.Timer()
    d = A.edit(rows[i]['geom'], rows[i]['lights'][0], T)
    stf = K.EC.build_state(A.ctx)
    assert np.array_equal(stf.hit, A.st.hit) and np.array_equal(stf.dbin, A.st.dbin)
    res[f's{n}_hit'] = A.st.hit; res[f's{n}_dbin'] = A.st.dbin; res[f's{n}_seg'] = A.st.seg
    res[f's{n}_src'] = d['s']; res[f's{n}_B'] = d['B']; res[f's{n}_dirty'] = d['dirty_rays']
    for spp in (1, 4):
        acc, _ = A.frame(d['lt'], d['Bd'], cam, 512, spp, 7000 + n, K.Timer())
        for k, v in acc.items():
            res[f's{n}_f{spp}_{k}'] = v.astype(np.float32)
    print('state', n, 'config', i, 'dirty', d['dirty_rays'], flush=True)
res['meta'] = json.dumps({'case': case, 'spec': spec, 'checkpoint_sha256': ck, 'frozen': K.frozen_hashes(),
                          'versions': {'mitsuba': K.mi.__version__, 'drjit': K.dr.__version__, 'torch': K.torch.__version__}})
np.savez_compressed(out, **res)
K.os._exit(0)
