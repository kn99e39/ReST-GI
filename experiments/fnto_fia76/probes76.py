"""WL76 Stage 4 references: the frozen WL76 population (ports76.json, te1) with the unchanged WL74/73/72 reference code (probes74.generate /
bundle: WL57 fast walker, gates, 32M photons with the 16M OOM fallback, K = 4); only the root, the port list, the parts and the seeds
(766/767xxxxxx, protocol) are redirected.   python probes76.py generate | bundle"""
import sys,os,json
from common76 import CFG,ROOT,dev,environment,seed
import probes74 as P74
S4=CFG['stage4']
def ports():return {int(k):v['port'] for k,v in json.loads((ROOT/'ports76.json').read_text())['ports'].items() if v['port'] is not None}
def parts():
    P=ports();fp=[int(k) for k,v in S4['slots'].items() if v['context']=='FP' and int(k) in P];st=[s for s in sorted(P) if s not in fp]
    out=[('rest',0,st+fp),(S4['pose'],1,fp)]
    if dev():D=CFG['execution']['dev_slots'];out=[(g,gi,[s for s in sl if s in D[g]]) for g,gi,sl in out]
    return out
P74.ROOT=ROOT;P74.dev=dev;P74.environment=environment;P74.ports=ports;P74.parts=parts;P74.seed=lambda g,s,a,k:seed('conf',g,s,a,k)
if __name__=='__main__':
    {'generate':P74.generate,'bundle':P74.bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
