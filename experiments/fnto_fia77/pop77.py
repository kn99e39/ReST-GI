"""WL77 confirmation population (brief Stage 5): reuse of the WL76 Stage 4 population, frozen geometry-only in fia76_v1 and never scored.
Verifies: ports76.json and inputs/geometry_te1.pt hashes equal the protocol; the WL76 evidence root has no Stage 4 references or analysis;
the port list equals the WL76 protocol's frozen list; the ports are disjoint from every previously scored port and its images (re-asserted).
Copies both files into the WL77 root.   python pop77.py"""
import sys,os,json,shutil
from common77 import CFG,ROOT,R76,R73,R74,INP68,sha,environment
import torch

def main():
    fz=CFG['stage4']['inputs']
    for rel,h in fz.items():
        src=R76/rel;dst=ROOT/rel;assert sha(src)==h,('hash',rel)
        if dst.exists():assert sha(dst)==h;continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
    assert not (R76/'refs').exists() and not (R76/'stage4').exists(),'the WL76 Stage 4 population must be unscored'
    got={k:v['port'] for k,v in json.loads((ROOT/'ports76.json').read_text())['ports'].items()};assert got==CFG['stage4']['ports'],'port list'
    environment();import fit73,ports73 as P73
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    s4={v for v in got.values() if v is not None};w=json.loads((R73/'ports.json').read_text());w74=json.loads((R74/'ports4.json').read_text())
    prev=set(p for _,p in fit73.populations()['S16R'])|{v['port'] for v in w['confirmation'].values()}|{v['port'] for v in w74['ports'].values()}
    p71,p72=P73.referenced();prev|=set(p71.values())|set(p72.values());sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    assert not (s4&set(sl.images(list(prev)))),'leakage';print('POP77 ok',len(s4),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
