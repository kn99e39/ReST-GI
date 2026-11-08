"""WL80 Stage 5 population freeze: the WL79 frozen, never-referenced population (sym79_v1.json stage5.ports; ports79.design, geometry
only) re-derived here and checked: identical port list; no WL79 reference packet / bundle / ports79.json exists (never scored); disjoint from
the S16-R training orbit (= the images exposed by any B1 arm; the orbit does not depend on the angular basis) and from every earlier
referenced / scored port and all its images; one port per orbit.  python ports80.py freeze"""
import sys,os,json
from common80 import CFG,ROOT,R79,INP68,environment,write_new,sha,load
environment()
import ports79 as P9,ports73 as P73
def main():
    out=ROOT/'ports80.json'
    if out.exists():raise FileExistsError('immutable')
    w79=json.loads((ROOT/'repo/experiments/fnto_sym79/protocol/sym79_v1.json').read_text())['stage5']['ports'];rows,prev,orbit=P9.design();got={k:v['port'] for k,v in rows.items()}
    assert got==w79==CFG['stage5']['ports'],('population differs',got)
    assert not (R79/'refs').exists() and not (R79/'ports79.json').exists() and not (R79/'stage5').exists(),'WL79 population was referenced or scored'
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));pts=[v for v in got.values() if v is not None]
    assert not set(pts)&set(orbit) and not set(pts)&set(sl.images(prev)) and len({sl.orbit_key(p) for p in pts})==len(pts)
    write_new(out,dict(ports=rows,excluded_previous=len(prev),training_orbit=orbit,source='WL79 sym79_v1 stage5 (frozen 2026-11-07, never referenced)',
                       seeds='79000000 + slot*1e4 + bin*100 + rep (WL79 predeclared, never generated)',information_used='geometry only'))
    print('PORTS80',json.dumps(got),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
