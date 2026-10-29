import os,sys,time,shutil,json
import torch
from common import *
from runtime import full,selective
from evaluate import model_of

def main():
    held=json.loads((OUT/'heldout.json').read_text())
    if not any(row['useful_and_physical'] for row in held.values()):
        dump(dict(executed=False,reason='No model passed primary physical, absolute, directional and signed-deformation fidelity. No fast-inaccurate S claim; historical WL56 blue floor retained.'),OUT/'composition.json')
        return
    import r57_compose as C
    C.model_of=model_of
    records={}
    for policy in ('F','S'):
        class StateAdapter:
            def __init__(self,m,g):
                self.m,self.g=m,g;g['cache_contract']=CONTRACT
                if policy=='S' and g['name']!='rest':
                    c=full(m,loaded_geometry('rest'));c,audit=selective(m,c,g)
                else:c=full(m,g)
                self.co=c['co']
            def __getitem__(self,k):return C.Channel(self,k)
        C.Adapter=StateAdapter
        torch.cuda.synchronize();t=time.perf_counter();C.main();torch.cuda.synchronize()
        records[policy]=dict(total_stage_seconds=time.perf_counter()-t,output=json.loads((OUT/'composition.json').read_text()))
        (OUT/'composition.json').rename(OUT/f'composition_{policy}.json')
        (OUT/'composition').rename(OUT/f'composition_{policy}')
    dump(dict(executed=True,policies=records,scope='conditional WL56 canonical transfer, separate from fresh pt0 intrinsic primary; full stage includes common C, source, physical controls, solves and readout'),OUT/'composition.json')

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
