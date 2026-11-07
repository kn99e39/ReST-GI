"""WL77A Gate 4 references (Part 0, frozen before any candidate exists): the WL76 Stage 4 population (ports76.json, te1; geometry-only,
never scored by WL76 or WL77) with the unchanged WL74/73/72 reference code (probes74.generate / bundle: WL57 fast walker, gates, 32M
photons with the 16M OOM fallback, K = 4) and the seeds WL76 predeclared for it. Only root, ports, parts and seeds are redirected.
    python probes77a.py stage | generate | bundle"""
import sys,os,json,shutil
from common77a import CFG,ROOT,R76,dev,environment,seed,sha
import probes74 as P74
def rows():return json.loads((ROOT/'ports76.json').read_text())['ports']
def ports():return {int(k):v['port'] for k,v in rows().items() if v['port'] is not None}
def parts():
    r=rows();fp=sorted(int(k) for k,v in r.items() if v['port'] is not None and 'face' in v['spec']);st=sorted(int(k) for k,v in r.items() if v['port'] is not None and int(k) not in fp)
    out=[('rest',0,st+fp),(CFG['part0']['pose'],1,fp)]
    if dev():D=CFG['part0']['dev_slots'];out=[(g,gi,[s for s in sl if s in D[g]]) for g,gi,sl in out]
    return out
def stage():
    for rel,h in CFG['part0']['inputs'].items():
        src=R76/rel;dst=ROOT/rel;assert sha(src)==h,('hash',rel)
        if dst.exists():assert sha(dst)==h;continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
    assert not (R76/'refs').exists() and not (R76/'stage4').exists(),'WL76 population must be unscored'
    assert not (Path_home()/'rest_fia77'/'refs').exists() and not (Path_home()/'rest_fia77'/'stage4').exists(),'WL77 must not have scored it'
    print('STAGED',len(ports()),'ports',flush=True)
def Path_home():
    from pathlib import Path;return Path.home()
P74.ROOT=ROOT;P74.dev=dev;P74.environment=environment;P74.ports=ports;P74.parts=parts;P74.seed=seed
if __name__=='__main__':
    {'stage':stage,'generate':P74.generate,'bundle':P74.bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
