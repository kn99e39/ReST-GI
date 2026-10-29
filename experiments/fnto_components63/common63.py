import hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol/components_v1.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump_new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(f'Preserve evidence: {p}')
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def reference(original):
    repo=Path(original)/'repo';os.environ['WL57']=str(original)
    os.environ['WL57_CONFIRM_PROTOCOL']=str(repo/'experiments/fnto_rule57/protocol/rule57_recovery_a2.json')
    os.environ['WL57_WALKER']='fast';os.environ['WL57_ACTION_CHUNK']=str(CFG['action_chunk'])
    sys.path.insert(0,str(repo/'experiments/fnto_rule57'))
    import r57_geometry as RG,r57_action_oracle as AO,r57_fastwalk as FW
    RG.POSES['pt0']=tuple(CFG['pose_parameters'])
    return RG,AO,FW
