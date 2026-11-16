"""WL100 dataset/configuration definitions. No GPU imports in this module."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'ecro86'
sys.path.insert(0,str(BASE))
import scene86 as S
P=json.loads((HERE/'protocol.json').read_text())
CK=json.loads((HERE.parent/'ecro_nearmirror93e/scene.json').read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def jd(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2),encoding='utf8')


def lambert(albedo):
    s=np.zeros(69,np.float32)
    for c in range(3):s[c*8]=np.broadcast_to(albedo,(3,))[c]
    s[24:38]=.3;s[38:45]=.5;s[45]=1;s[53:]=1
    return s


def materials(case):
    st=np.load(BASE/'data/mats86.npz')['states'].copy()
    if case=='near':
        z=lambert(0);z[24:38]=.01;z[38:45]=1;z[45:53]=0;z[46]=1
        for c in range(3):z[c*8+1]=.85
        st[1]=z
    else:
        st[0]=lambert(.55);st[1]=lambert(.55)
        st[2]=lambert([.92,.80,.65])
        # Typed fixed-medium feature row, consumed only by the neural encoder.
        # The physical dispatcher does not evaluate this as a surface BSDF.
        st[2,24:38]=1/12;st[2,68]=1.
    return np.concatenate([st,np.stack([lambert(.3),lambert(.8)])])


def checker(g):
    for r in range(6,12):
        nu,nv=g.RG[r]
        for v in range(nv):
            for u in range(nu):g.PMAT[g.RBASE[r]+u+nu*v]=6 if (u+v)%2==0 else 7
    return g


def config(case,split,index):
    rng=np.random.default_rng([P['splits'][split]['seed'],index])
    if case=='near':
        geom=json.loads(json.dumps(CK['geom']))
        geom['A']['c']=(np.array(geom['A']['c'])+rng.uniform([- .035,-.07,-.04],[.035,.07,.04])).tolist()
        geom['A']['yaw']=float(rng.uniform(.08,.22))
        geom['B']['c']=[float(rng.uniform(1.2,1.5)),float(rng.uniform(1.2,1.6)),.5]
        geom['B']['yaw']=float(rng.uniform(0,.3))
        lights=[]
        for _ in range(P['splits'][split]['lights_per_geometry']):
            pos=[float(rng.uniform(-1.48,-1.42)),float(rng.uniform(-.95,-.35)),float(rng.uniform(1.55,2.05))]
            strength=float(rng.uniform(24,36));lights.append([{'pos':pos,'rgb':[strength]*3}])
    else:
        geom={'A':{'c':[float(rng.uniform(-.35,.05)),float(rng.uniform(.25,.75)),.75],
                   'h':[.22,.35,.50],'yaw':float(rng.uniform(0,.4))},
              'B':{'c':[float(rng.uniform(1.5,1.7)),float(rng.uniform(.8,1.3)),.35],
                   'h':[.15,.18,.35],'yaw':float(rng.uniform(0,.3))}}
        lights=[]
        for li in range(P['splits'][split]['lights_per_geometry']):
            # Back / side lighting illuminates both sphere and its neighbour.
            pos=[float(rng.uniform(.35,1.5)),float(rng.uniform(.85,1.5)),float(rng.uniform(1.0,1.7))]
            strength=float(rng.uniform(20,32));lights.append([{'pos':pos,'rgb':[strength]*3}])
    return {'case':case,'split':split,'index':index,'geom':geom,'lights':lights,
            'camera':P['datasets'][case]['camera']}


def freeze(root):
    cfg={case:{split:[config(case,split,i) for i in range(p['geometries'])]
               for split,p in P['splits'].items()} for case in ['near','sss']}
    hashes={case:{split:{digest({'geom':c['geom'],'lights':c['lights']}) for c in rows}
                  for split,rows in splits.items()} for case,splits in cfg.items()}
    for case,h in hashes.items():
        assert not (h['train']&h['val'] or h['train']&h['test'] or h['val']&h['test'])
    jd(cfg,Path(root)/'configs.json')
    deps=['ecro86/model86.py','ecro86/pt86.py','ecro86/scene86.py','ecro_own92a/own92a.py','fnto_compiled83/closure83_dr.py']
    jd({'base_commit':P['base_commit'],'configs_sha256':sha(Path(root)/'configs.json'),
        'source_hashes':{f:sha(HERE.parent/f) for f in deps},
        'new_source_hashes':{p.name:sha(p) for p in HERE.iterdir() if p.is_file()},
        'split_disjoint':True,'protocol':P},Path(root)/'freeze_manifest.json')
    return cfg
