"""POST HOC: stored WL59 flux -> face-on outgoing-bin radiance, no model run.

Run with the completed server snapshot's common module on PYTHONPATH.
This is not a Cycles image, continuous angular readout, or coupled GI solve.
"""
import json,inspect
from pathlib import Path
import numpy as np
import torch
from common import ROOT,DATA,RG

def main():
    torch.set_num_threads(8)
    destination=ROOT/'visual_review';destination.mkdir(exist_ok=True)
    refs={name:sum(torch.load(DATA/'targets'/f'{name}_r{r}.pt')['test'] for r in (0,1))*.5 for name in ('rest','pt0')}
    delta=(refs['pt0'][:16,:,:,2]-refs['rest'][:16,:,:,2]).sum(2)
    strongest=int(delta.norm(dim=1).argmax())
    fields=list(dict.fromkeys([0,strongest,62,63]));payload={'fields':np.array(fields)}
    geometry={}
    for name in ('rest','pt0'):
        m,V=RG.mesh(name);fm,VC,_=RG.KO.refined(m,V,4)
        fr=RG.G.tri_frames(fm['V'],VC,fm['F'],fm['tri_port'],fm['port_tangent'])
        area=np.bincount(fm['tri_port'],weights=fr['area'],minlength=len(fm['port_area']))
        geometry[name]=(fm,VC,fr,area)
        payload[name+'_triangles']=VC[fm['F']]
    lights=torch.load(DATA/'lights.pt')['test'][fields].sum(2).numpy()
    def export(name,tag,values):
        fm,V,fr,area=geometry[name]
        for view,d in [('top',np.array([0.,0.,1.])),('bottom',np.array([0.,0.,-1.]))]:
            mask=(fr['n']@d)>1e-6;ports=fm['tri_port'][mask]
            local=np.column_stack([fr['t'][mask]@d,fr['b'][mask]@d,fr['n'][mask]@d])
            bins=RG.G.local_to_bin(local,8)
            radiance=values[fields][:,ports,bins,:].numpy()/(area[ports][None,:,None]*(np.pi/64))
            payload[f'{name}_{view}_{tag}']=radiance
            payload[f'{name}_{view}_triangles']=V[fm['F'][mask]]
            payload[f'{name}_{view}_incident']=lights[:,ports]/area[ports][None,:,None]
    for name in ('rest','pt0'):export(name,'oracle',refs[name])
    export('pt0','stale_physical',refs['rest'])
    for seed in (5901,5902):
        kw={'map_location':'cpu'}
        if 'mmap' in inspect.signature(torch.load).parameters:kw['mmap']=True
        stored=torch.load(ROOT/'out'/f'actions_{seed}.pt',**kw)
        for name in ('rest','pt0'):
            for policy in ('F','S'):export(name,f'{policy}_{seed}',stored['predictions'][name][policy])
        del stored
    np.savez_compressed(destination/'boundary_readouts.npz',**payload)
    metadata=dict(kind='POST_HOC_STORED_BOUNDARY_BIN_READOUT',fields=fields,
        field_selection={'field0':'fixed first focused field','strongest_blue_field':strongest,
                         'strongest_rule':'reference-only max blue integrated-flux delta L2 among first16; post hoc display selection',
                         '62_63':'both fields of primary failing directional pair7'},
        readout='Lo=outgoing cell/bin flux / (current port area*pi/64); per-triangle current material normal/tangent; face-on +-z',
        scope='K-only prescribed incident fields; includes stored scattered+ballistic flux. Not monolithic Cycles or coupled GI. Current/stale evaluated in same current bin interface.',
        source='completed ~/rest_persist_v1 stored actions and independently seeded references; no fitting/inference/acceptance changes')
    (destination/'readout_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
