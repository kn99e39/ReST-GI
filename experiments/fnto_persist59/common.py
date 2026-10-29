import os, sys, json, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('P59', Path.home()/'rest_persist_v1'))
OLD = Path.home()/'rest_wl57_recovery'
sys.path.insert(0, str(HERE.parent/'fnto_rule57'))
os.environ.setdefault('WL57', str(ROOT))
os.environ.setdefault('WL57_CONFIRM_PROTOCOL', str(HERE.parent/'fnto_rule57/protocol/rule57_recovery_a2.json'))
import r57_geometry as RG
from r57_data import dump, save, sha

CFG = json.loads((HERE/'protocol/persist_v1.json').read_text())
RG.POSES.update({k:tuple(v) for k,v in CFG['primary_pose'].items()})
PRIMARY = list(CFG['primary_pose'])
DIAGNOSTIC = CFG['diagnostic_poses']
OUT, DATA = ROOT/'out', ROOT/'data'
CONTRACT = hashlib.sha256(json.dumps(dict(interface=CFG['representation'],
    material=dict(sigma_t=float(RG.FX.SIGMA_T),albedo=list(map(float,RG.FX.ALBEDO)),phase_g=float(RG.FX.G_PHASE)),
    units='metres/flux',frame='supplied rigid material frame; current triangle normal/tangent'),sort_keys=True).encode()).hexdigest()

def geometry(name, **kw):
    g=RG.geometry(name, **kw)
    g['cache_contract']=CONTRACT
    return g

def loaded_geometry(name):
    import torch
    g=RG.device(torch.load(DATA/'geometry'/f'{name}.pt'))
    g['cache_contract']=CONTRACT
    return g

def rigid_geometry(ballistic=True):
    import numpy as np
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);angle=np.deg2rad(37.)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross
    t=np.array([.03,-.02,.011]);_,V=RG.mesh('rest')
    return geometry('rest',ballistic=ballistic,V_world=V@R.T+t,rigid_R=R,rigid_t=t)
