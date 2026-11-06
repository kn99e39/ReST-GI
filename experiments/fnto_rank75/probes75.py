"""WL75 Stage 3 references: the frozen, never-generated WL74 Stage 4 population (ports4.json, cf1) with WL74's own reference code
(probes74.generate / bundle: unchanged WL57 fast walker and gates, 32M photons with the 16M OOM fallback, K = 4, seeds 743/744xxxxxx
predeclared in ato74_v1), written into the WL75 root. Only the module's root (and its dev flag) is redirected; nothing else changes.
    python probes75.py stage | generate | bundle"""
import sys,os,json,shutil
from common75 import CFG,ROOT,R74,sha,write_new,dev,environment
import probes74 as P74

P74.ROOT=ROOT;P74.dev=dev;P74.environment=environment   # redirect output root / repo copy; seeds, packets, estimator, gates unchanged
if dev():   # dev smoke only: a small slot subset (packet overhead ~2 s each) to exercise the generate / bundle / Stage 3 path
    _parts=P74.parts;DEV=CFG['execution']['dev_slots']
    P74.parts=lambda:[(g,gi,[s for s in sl if s in DEV[g]]) for g,gi,sl in _parts()]

def stage():
    """Copy the frozen WL74 inputs (geometry-only) into the WL75 root after verifying their hashes against the protocol."""
    fz=CFG['stage3']['inputs']
    for rel,h in fz.items():
        src=R74/rel;dst=ROOT/rel;assert sha(src)==h,('input hash',rel)
        if dst.exists():assert sha(dst)==h;continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
    assert not (R74/'refs').exists(),'WL74 Stage 4 references must never have been generated'
    got={int(k):v['port'] for k,v in json.loads((ROOT/'ports4.json').read_text())['ports'].items()}
    assert got=={int(k):v for k,v in CFG['stage3']['ports'].items()},'Stage 4 port list'
    print('STAGED',json.dumps({k:v for k,v in fz.items()}),flush=True)

if __name__=='__main__':
    {'stage':stage,'generate':P74.generate,'bundle':P74.bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
