"""Geometry identity gate (cf2) and port selection by the predeclared geometry-only rule (final ports first, then new
training ports). Reads archived geometry only; no probe response, model output or error. Writes ROOT/ports.json (immutable)."""
import sys,json
import numpy as np
import torch
from common71 import CFG,ROOT,INP68,R68,write_new,tensor_sha,sha,environment

KEYS=('xyz','rest_xyz','area','node','pair','prior','parent','src','dst','graph')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def geometries():
    return {'rest':load(INP68/'wl59/geometry_rest.pt'),'pt0':load(INP68/'wl59/geometry_pt0.pt'),'cf2':load(ROOT/'inputs/geometry_cf2.pt')}

def existing_ports():
    fit=json.loads((INP68/'wl66_ports.json').read_text())['ports'];ho=json.loads((R68/'heldout/ports.json').read_text())['ports']
    return {s['slot']:s['port'] for s in fit}|{s['slot']:s['port'] for s in ho}

def select(slots,geoms,blockers,face,xyz0):
    chosen=[]
    for s in slots:
        m=np.where(face==s['face'])[0];tx,ty=s['target_rest_xy']
        dist=np.round(np.hypot(xyz0[m,0]-tx,xyz0[m,1]-ty),9);order=m[np.lexsort((m,dist))]
        for i in order:
            side=min(.05-abs(xyz0[i,0]),.05-abs(xyz0[i,1]));budget=min(float((1-g['beta'][i]).min()) for g in geoms)
            others=[j for j in blockers+chosen if face[j]==face[i]]
            sep=min(float(np.hypot(*(xyz0[i,:2]-xyz0[j,:2]))) for j in others) if others else 1.
            if i in blockers or i in chosen:continue
            if budget>=.5 and side>=.01-1e-12 and sep>=.01-1e-12:chosen.append(int(i));break
        else:raise RuntimeError(f"STOP: no admissible port for slot {s['slot']} (no reselection)")
    return chosen

def main():
    RG=environment();torch.set_grad_enabled(False)
    if (ROOT/'ports.json').exists():raise FileExistsError('port identities are immutable')
    G=geometries();fresh=RG.geometry('cf2',ballistic=True);ident={}
    for k in KEYS:assert torch.equal(fresh[k].cpu(),G['cf2'][k]),('cf2 geometry identity',k)
    bf,ba=fresh['beta'].cpu().double(),G['cf2']['beta'].double()
    ident['cf2']=dict(features_bitwise_equal=True,signature_equal=fresh['signature']==G['cf2']['signature'],beta_max_abs_diff=float((bf-ba).abs().max()),
                      beta_rel_L2=float((bf-ba).norm()/ba.norm()),archived_sha256=sha(ROOT/'inputs/geometry_cf2.pt'))
    for g in G.values():assert torch.equal(g['rest_xyz'],G['rest']['rest_xyz']) and torch.equal(g['node'][:,15],G['rest']['node'][:,15]),'material correspondence'
    np_g={n:dict(beta=g['beta'].numpy()) for n,g in G.items()};xyz0=G['rest']['rest_xyz'].numpy();face=np.rint(G['rest']['node'][:,15].numpy()*5).astype(int)
    ex=existing_ports();prev=[ex[s] for s in sorted(ex)]
    final=select(CFG['ports']['final_slots'],[np_g['rest'],np_g['cf2']],prev,face,xyz0)
    new=select(CFG['ports']['training_new_slots'],[np_g['rest'],np_g['pt0']],prev+final,face,xyz0)
    slots={**{s:ex[s] for s in range(8)},**{s['slot']:i for s,i in zip(CFG['ports']['training_new_slots'],new)},**{s['slot']:i for s,i in zip(CFG['ports']['final_slots'],final)}}
    assert len(set(slots.values()))==len(slots),'distinct ports'
    rec={}
    for s,i in sorted(slots.items()):
        rec[str(s)]=dict(port=i,face=int(face[i]),rest_xyz=xyz0[i].tolist(),
            d_min={n:float((1-g['beta'][i]).min()) for n,g in G.items()},beta_row_sha256={n:tensor_sha(g['beta'][i]) for n,g in G.items()},
            dz_mm={n:float((g['xyz'][i,2]-g['rest_xyz'][i,2])*1e3) for n,g in G.items()})
    write_new(ROOT/'ports.json',dict(slots=rec,final=final,training_new=new,existing=ex,geometry_identity=ident,
        information_used='archived geometry (rest material coordinates, face, beta) only'))
    print('PORTS final',final,'training_new',new,json.dumps(ident),flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
