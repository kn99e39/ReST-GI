"""Geometry-only identity gate and held-out port selection (written before any held-out reference or fit exists).
Uses rest material coordinates, face index and model beta only; reads no probe response, model output or error."""
import sys,json
import numpy as np
from common68 import CFG,ROOT,INP,write_new,environment,tensor_sha,sha

GEOMS=('rest','pt0','cf3')
def select(geoms,slots,fit_ports,min_sep):
    ref=geoms['rest'];face=np.rint(ref['node'][:,15]*5).astype(int);xyz0=ref['rest_xyz'];chosen=[]
    for g in geoms.values():assert np.array_equal(np.rint(g['node'][:,15]*5).astype(int),face) and np.array_equal(g['rest_xyz'],xyz0),'material correspondence'
    for s in slots:
        m=np.where(face==s['face'])[0];tx,ty=s['target_rest_xy']
        dist=np.round(np.hypot(xyz0[m,0]-tx,xyz0[m,1]-ty),9);order=m[np.lexsort((m,dist))]
        for i in order:
            side=min(.05-abs(xyz0[i,0]),.05-abs(xyz0[i,1]));budget=min(float((1-g['beta'][i]).min()) for g in geoms.values())
            sep=min(float(np.hypot(*(xyz0[i,:2]-xyz0[j,:2]))) for j in fit_ports+chosen if face[j]==face[i]) if any(face[j]==face[i] for j in fit_ports+chosen) else 1.
            if budget>=.5 and side>=.01-1e-12 and sep>=min_sep-1e-12:chosen.append(int(i));break
        else:raise RuntimeError('No admissible held-out port: STOP, no reselection')
    return chosen

def main():
    import torch
    RG=environment();torch.set_grad_enabled(False)
    if (ROOT/'heldout/ports.json').exists():raise FileExistsError('Port identities are immutable')
    arch={};fresh={};ident={}
    for g in GEOMS:
        arch[g]=torch.load(INP/'wl59'/f'geometry_{g}.pt',map_location='cpu',weights_only=False)
        fresh[g]=RG.geometry(g,ballistic=True)
        for k in ('xyz','rest_xyz','area','node','pair','prior','parent','src','dst','graph'):assert torch.equal(fresh[g][k].cpu(),arch[g][k]),('geometry identity',g,k)
        bf,ba=fresh[g]['beta'].cpu().double(),arch[g]['beta'].double()
        ident[g]=dict(features_bitwise_equal=True,signature_equal=fresh[g]['signature']==arch[g]['signature'],
                      beta_max_abs_diff=float((bf-ba).abs().max()),beta_rel_L2=float((bf-ba).norm()/ba.norm()),B_nnz=(fresh[g]['B']._nnz(),arch[g]['B']._nnz()))
    fit=json.loads((INP/'wl66_ports.json').read_text());fit_ports=[s['port'] for s in sorted(fit['ports'],key=lambda s:s['slot'])]
    np_g={n:dict(rest_xyz=g['rest_xyz'].numpy(),node=g['node'].numpy(),beta=g['beta'].numpy()) for n,g in arch.items()}
    rule=CFG['heldout_ports'];chosen=select(np_g,rule['slots'],fit_ports,rule['min_separation_m'])
    fit_cf3={s:dict(port=i,min_d=float((1-arch['cf3']['beta'][i]).min()),admissible=float((1-arch['cf3']['beta'][i]).min())>=.5) for s,i in enumerate(fit_ports)}
    rec=[]
    for s,i in zip(rule['slots'],chosen):
        per={}
        for g in GEOMS:
            a=arch[g];fr=fresh[g]['frame'];tp=np.asarray(fresh[g]['mesh']['tri_port']);tris=np.where(tp==i)[0];w=fr['area'][tris]
            n=(fr['n'][tris]*w[:,None]).sum(0)
            per[g]=dict(xyz=a['xyz'][i].tolist(),area=float(a['area'][i]),parent=int(a['parent'][i]),face=int(round(float(a['node'][i,15])*5)),
                        mean_normal=(n/np.linalg.norm(n)).tolist(),beta=a['beta'][i].tolist(),d=(1-a['beta'][i]).tolist(),signature=a['signature'],beta_row_sha256=tensor_sha(a['beta'][i]))
        rec.append(dict(slot=s['slot'],name=s['name'],port=i,rest_xyz=arch['rest']['rest_xyz'][i].tolist(),target_rest_xy=s['target_rest_xy'],geometries=per))
    write_new(ROOT/'heldout/ports.json',dict(rule=rule,ports=rec,fitting_ports=fit_ports,fitting_ports_at_cf3=fit_cf3,geometry_identity=ident,
        information_used='archived geometry (rest material coordinates, face, beta) only; no probe, model output or error read'))
    print('HELDOUT_PORTS',chosen,json.dumps(ident),flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
