"""WL86 material table (frozen): ER states of the WL85 fixture materials B, S1, S2 (bit-identical copies) + Lambert-only ER states.

  python mats86.py     (reads results/gm1_char85/results/census/fixture.npz, writes data/mats86.npz + data/mats86.json)
ids: 0 B, 1 S1, 2 S2, 3 Lambert walls / ceiling, 4 Lambert box A, 5 Lambert box B.
"""
import hashlib, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'results', 'gm1_char85', 'results', 'census', 'fixture.npz')
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86.json')))


def lambert_state(alb):
    st = np.zeros(69, np.float32)
    for c in range(3):
        st[c * 8 + 0] = alb
    st[24:31] = 0.3; st[31:38] = 0.3; st[38:45] = 0.5
    st[45] = 1.0
    st[53:68] = 1.0; st[68] = 1.0
    return st


z = np.load(SRC)
la = P['fixture']['lambert_albedo']
albs = [la['walls_ceiling'], la['boxA'], la['boxB']]
states = np.concatenate([z['states'][:3], np.stack([lambert_state(a) for a in albs])]).astype(np.float32)
desc = np.concatenate([z['desc'][:3], np.tile(z['desc'][3:4], (3, 1))]).astype(np.float32)
os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
np.savez(os.path.join(HERE, 'data', 'mats86.npz'), states=states, desc=desc, lambert=np.array([0, 0, 0, 1, 1, 1], np.int32),
         albedo=np.array([0, 0, 0] + albs, np.float32))
json.dump({'source': 'results/gm1_char85/results/census/fixture.npz', 'source_sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest(),
           'er_states_sha256': hashlib.sha256(states[:3].tobytes()).hexdigest(),
           'er_states_equal_wl85': bool(np.array_equal(states[:3], z['states'][:3]))}, open(os.path.join(HERE, 'data', 'mats86.json'), 'w'), indent=1)
print(open(os.path.join(HERE, 'data', 'mats86.json')).read())
