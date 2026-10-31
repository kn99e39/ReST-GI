"""Deterministic geometry-only port selection and identity record, written before any probe reference exists."""
import sys,json
import numpy as np
from common66 import CFG,ROOT,ORIG,GEOMS,SLOTS,write_new,environment,tensor_sha

def select(geoms):
    """geoms: name -> dict(rest_xyz, node, beta) as numpy. Returns one port per slot by the protocol rule."""
    rule=CFG['port_selection'];chosen=[]
    for s in SLOTS:
        ref=geoms[GEOMS[0]];face=np.rint(ref['node'][:,15]*5).astype(int);xyz0=ref['rest_xyz']
        for g in geoms.values():
            assert np.array_equal(np.rint(g['node'][:,15]*5).astype(int),face) and np.array_equal(g['rest_xyz'],xyz0),'material correspondence'
        m=np.where(face==s['face'])[0];tx,ty=s['target_rest_xy']
        dist=np.round(np.hypot(xyz0[m,0]-tx,xyz0[m,1]-ty),9);order=m[np.lexsort((m,dist))]
        for i in order:
            side=min(.05-abs(xyz0[i,0]),.05-abs(xyz0[i,1]))
            budget=min(float((1-g['beta'][i]).min()) for g in geoms.values())
            if budget>=.5 and side>=.01-1e-12:chosen.append(int(i));break
        else:raise RuntimeError('No admissible port: STOP, no reselection')
    assert len(set(chosen))==len(chosen)
    return chosen

def main():
    import torch
    C59,RG=environment();torch.set_grad_enabled(False)
    if (ROOT/'ports.json').exists():raise FileExistsError('Port identities are immutable')
    arch={};fresh={}
    for name in GEOMS:
        arch[name]=torch.load(ORIG/CFG['geometry_files'][name],map_location='cpu',weights_only=False)
        fresh[name]=RG.geometry(name,ballistic=False)
        for k in ('xyz','rest_xyz','area','node','pair','prior','parent'):assert torch.equal(fresh[name][k].cpu(),arch[name][k]),(name,k)
    np_g={n:dict(rest_xyz=g['rest_xyz'].numpy(),node=g['node'].numpy(),beta=g['beta'].numpy()) for n,g in arch.items()}
    chosen=select(np_g);rec=[]
    for s,i in zip(SLOTS,chosen):
        per={}
        for name in GEOMS:
            g=arch[name];fr=fresh[name]['frame'];tp=np.asarray(fresh[name]['mesh']['tri_port']);tris=np.where(tp==i)[0];w=fr['area'][tris]
            n=(fr['n'][tris]*w[:,None]).sum(0);t=(fr['t'][tris]*w[:,None]).sum(0)
            per[name]=dict(xyz=g['xyz'][i].tolist(),area=float(g['area'][i]),parent=int(g['parent'][i]),face=int(round(float(g['node'][i,15])*5)),
                triangles=tris.tolist(),mean_normal=(n/np.linalg.norm(n)).tolist(),mean_tangent=(t/np.linalg.norm(t)).tolist(),
                triangle_normals=fr['n'][tris].tolist(),beta=g['beta'][i].tolist(),d=(1-g['beta'][i]).tolist(),signature=g['signature'],
                beta_row_sha256=tensor_sha(g['beta'][i]))
        assert per[GEOMS[0]]['parent']==per[GEOMS[1]]['parent'] and per[GEOMS[0]]['face']==per[GEOMS[1]]['face']
        rec.append(dict(slot=s['slot'],name=s['name'],port=i,rest_xyz=arch[GEOMS[0]]['rest_xyz'][i].tolist(),target_rest_xy=s['target_rest_xy'],geometries=per))
    dmin=min(min(r['geometries'][n]['d']) for r in rec for n in GEOMS)
    write_new(ROOT/'ports.json',dict(rule=CFG['port_selection'],ports=rec,min_collided_budget_d=dmin,geometry_features_bitwise_equal=True,
        information_used='archived geometry only; no probe, model output or error read',bin_semantics='incoming bin a = direction the light comes FROM in each triangle current frame (t,b,n), concentric-square 4x4 equal projected solid angle'))
    print('PORTS',chosen,dmin,flush=True)

if __name__=='__main__':main();sys.stdout.flush()
