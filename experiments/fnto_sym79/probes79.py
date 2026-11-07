"""WL79 Stage 5 references (generated only after the Stage 3 fit gate and the Stage 4 mechanism gate pass): the frozen ports79.json
population on the rest geometry with the unchanged WL74/77A reference code (probes74.generate / bundle: WL57 fast walker, physical / identity
gates, 32M photons with the 16M fallback, K = 4) and new predeclared seeds 79000000 + slot*1e4 + bin*100 + rep.
    python probes79.py generate | bundle"""
import sys,os,json
from common79 import CFG,ROOT,environment,dev
import probes74 as P74
def seed(g,slot,a,rep):return 79000000+g*1000000+slot*10000+a*100+rep
def ports():return {int(k):v['port'] for k,v in json.loads((ROOT/'ports79.json').read_text())['ports'].items() if v['port'] is not None}
def parts():
    sl=sorted(ports())
    if dev():sl=[s for s in sl if s in CFG['execution']['dev_ref_slots']]
    return [('rest',0,sl)]
P74.ROOT=ROOT;P74.dev=dev;P74.environment=environment;P74.ports=ports;P74.parts=parts;P74.seed=seed
if __name__=='__main__':
    {'generate':P74.generate,'bundle':P74.bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
