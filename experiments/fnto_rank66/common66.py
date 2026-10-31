"""WL66 shared paths/IO. Reference and model code are imported from the immutable WL59 repo copy."""
import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;HOME=Path.home()
CFG=json.loads((HERE/'protocol/rank66_v1.json').read_text())
ROOT=Path(os.environ.get('R66_ROOT',HOME/'rest_rank66'));ORIG=HOME/'rest_persist_v1'
GEOMS=CFG['geometries'];SLOTS=CFG['port_selection']['slots'];NBIN=16
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def tensor_sha(t):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()
def write_new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(f'Preserve evidence: {p}')
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def save_new(p,obj):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(f'Preserve evidence: {p}')
    q=p.with_suffix(p.suffix+'.partial');torch.save(obj,q);os.replace(q,p)
def seed(g,slot,a,rep):return 680000000+g*100000+slot*10000+a*100+rep
def environment():
    """WL59 runtime + reference modules from the executed original repository copy."""
    os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG);os.environ['WL57_WALKER']='fast'
    os.environ['WL57_ACTION_CHUNK']=str(CFG['probes']['action_chunk'])
    os.environ['WL57_CONFIRM_PROTOCOL']=str(ORIG/'repo/experiments/fnto_rule57/protocol/rule57_recovery_a2.json')
    for d in ('fnto_persist59','fnto_rule57'):sys.path.insert(0,str(ORIG/'repo/experiments'/d))
    sys.path.insert(0,str(HERE.parent/'fnto_components63'))
    import common as C59
    import r57_geometry as RG
    RG.POSES['pt0']=tuple(CFG['pt0_pose'])
    return C59,RG
def verify_sources():
    out={}
    for rel,h in CFG['source_expected_sha256'].items():
        assert sha(ORIG/'repo'/rel)==h,rel;out[rel]=h
    t=HERE.parent/'fnto_components63/tag_reference.py';assert sha(t)==CFG['tag_reference_sha256'];out['tag_reference.py']=CFG['tag_reference_sha256']
    for rel,h in CFG['input_expected_sha256'].items():assert sha(ORIG/rel)==h,rel;out[rel]=h
    for name,rel in {**CFG['checkpoints']['primary'],**CFG['checkpoints']['secondary_descriptive']}.items():
        assert sha(HOME/rel)==CFG['checkpoints']['sha256'][name],name;out[name]=CFG['checkpoints']['sha256'][name]
    return out
def ports():return json.loads((ROOT/'ports.json').read_text())
def budget():return json.loads((ROOT/'budget.json').read_text())
