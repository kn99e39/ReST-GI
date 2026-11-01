"""WL68 Track A shared paths/IO. WL59 model/reference code is imported from the deployed, hash-verified repository copy."""
import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;HOME=Path.home()
CFG=json.loads((HERE/'protocol/repr68_v1.json').read_text()) if (HERE/'protocol/repr68_v1.json').exists() else {}
ROOT=Path(os.environ.get('R68_ROOT',HOME/'rest_repr68'));INP=ROOT/'inputs'
NBIN=16;FIT_GEOMS=('rest','pt0');FIT_SLOTS=(0,1,2,3)
PT0=(0.00575,0.0225,0.005,-0.005)
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
def heldout_seed(g,slot,a,rep):return 683000000+g*100000+slot*10000+a*100+rep
def environment(repo=None):
    """WL59 runtime/reference modules from the deployed repository copy (hashes verified against WL66's protocol)."""
    repo=Path(repo or ROOT/'repo')/'experiments'
    os.environ.setdefault('P59',str(ROOT));os.environ.setdefault('WL57',str(ROOT));os.environ['WL57_WALKER']='fast'
    os.environ['WL57_CONFIRM_PROTOCOL']=str(repo/'fnto_rule57/protocol/rule57_recovery_a2.json')
    for d in ('fnto_persist59','fnto_rule57','fnto_components63','fnto_rank66'):
        if str(repo/d) not in sys.path:sys.path.insert(0,str(repo/d))
    import r57_geometry as RG
    RG.POSES['pt0']=PT0
    return RG
def verify_sources(repo=None):
    """Same WL59 source files and hashes that WL66 verified (LF checkout)."""
    repo=Path(repo or ROOT/'repo');c66=json.loads((repo/'experiments/fnto_rank66/protocol/rank66_v1.json').read_text());out={}
    for rel,h in c66['source_expected_sha256'].items():assert sha(repo/rel)==h,rel;out[rel]=h
    t=repo/'experiments/fnto_components63/tag_reference.py';assert sha(t)==c66['tag_reference_sha256'];out['tag_reference.py']=c66['tag_reference_sha256']
    return out
def verify_inputs():
    m=json.loads((INP/'manifest.json').read_text());bad=[k for k,v in m['files'].items() if sha(INP/k)!=v['sha256']]
    assert not bad,('input hash mismatch',bad[:5]);return m
