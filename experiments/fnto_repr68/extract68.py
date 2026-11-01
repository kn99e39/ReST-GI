"""Runs on LabServer69 (where WL59/WL66 evidence lives). Hash-verifies every WL66 packet against its receipt and writes
compact WL68 input bundles; copies the WL59 inputs needed on LabServer63. Reads no model fit or WL68 result; writes only
under --out (separate execution root). Physical probe values are transcribed, never filtered or reweighted."""
import sys,json,shutil,argparse,time,os,hashlib
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bin_parent(nf=8,nc=4):
    f=nf//nc;b=np.arange(nf*nf);return (b%nf)//f+nc*((b//nf)//f)
# Predeclared cross-machine identity packets (WL66 seeds; protocol fixes these before any WL68 evidence).
CROSS=[('rest',0,0,0),('rest',2,7,3),('pt0',1,13,5),('pt0',3,8,2)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank66',required=True);ap.add_argument('--persist',required=True);ap.add_argument('--wl57',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    import torch
    R66,P59,W57,OUT=map(lambda s:Path(s).expanduser(),(a.rank66,a.persist,a.wl57,a.out))
    if OUT.exists():raise FileExistsError('export root is write-once')
    (OUT/'fit').mkdir(parents=True);(OUT/'cross').mkdir();(OUT/'wl59').mkdir();t0=time.time()
    done=json.loads((R66/'probes_complete.json').read_text());assert done['all_passed'] and done['packets']==1024
    rec={r['path']:r for r in done['records']};K=json.loads((R66/'budget.json').read_text())['replicas'];assert K==8
    bm=torch.as_tensor(bin_parent());files={};sources={}
    for name in ('rest','pt0'):
        for slot in range(4):
            S=[];B=[];seeds=[]
            for k in range(K):
                rowS=[];rowB=[]
                for b in range(16):
                    p=R66/'probes'/f'{name}_s{slot}_a{b:02d}_k{k}.pt';r=rec[str(p)];h=sha(p);assert h==r['sha256'] and r['passed'],p
                    sources[str(p)]=h;x=torch.load(p,map_location='cpu',weights_only=False)
                    assert x['N']==32_000_000 and x['seed']==r['seed'];seeds.append(r['seed'])
                    rowS.append(x['component_first']['Scat'][0].float())
                    bb=x['component_first']['B'][0];rowB.append(torch.zeros(bb.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,bb).float());del x
                S.append(torch.stack(rowS));B.append(torch.stack(rowB))
            out=OUT/'fit'/f'{name}_s{slot}.pt'
            torch.save(dict(S=torch.stack(S),B16=torch.stack(B),seeds=seeds,geometry=name,slot=slot,K=K,N=32_000_000,
                            layout='S:(replica,bin,port,64,RGB) collided first-exit flux, float32 of float64 packet means; B16:(replica,bin,port,16,RGB) zero-collision'),out)
            files[f'fit/{out.name}']=dict(sha256=sha(out),bytes=out.stat().st_size);print('BUNDLE',name,slot,round(time.time()-t0),flush=True)
    for name,slot,b,k in CROSS:
        p=R66/'probes'/f'{name}_s{slot}_a{b:02d}_k{k}.pt';assert sha(p)==rec[str(p)]['sha256'];q=OUT/'cross'/p.name;shutil.copy2(p,q)
        files[f'cross/{q.name}']=dict(sha256=sha(q),bytes=q.stat().st_size,seed=rec[str(p)]['seed'],source=str(p))
    copies={'wl59/geometry_rest.pt':P59/'data/geometry/rest.pt','wl59/geometry_pt0.pt':P59/'data/geometry/pt0.pt','wl59/geometry_cf3.pt':W57/'data/geometry/cf3.pt',
            'wl59/P_5901.pt':P59/'checkpoints/P_5901.pt','wl59/P_5902.pt':P59/'checkpoints/P_5902.pt','wl59/lights.pt':P59/'data/lights.pt',
            'wl66_ports.json':R66/'ports.json','wl66_budget.json':R66/'budget.json','wl66_model_complete.json':R66/'model_complete.json'}
    for n in ('rest','pt0','cf3'):
        for r in (0,1):copies[f'wl59/targets_{n}_r{r}.pt']=P59/'data/targets'/f'{n}_r{r}.pt'
    for dst,src in copies.items():
        src=src.resolve();shutil.copy2(src,OUT/dst);files[dst]=dict(sha256=sha(OUT/dst),bytes=(OUT/dst).stat().st_size,source=str(src),source_sha256=sha(src))
    # Frozen replay subset (fields 8,9,62,63 at pt0, as WL66) and coarse fingerprints of WL66 frozen-model probe outputs.
    for seed in (5901,5902):
        src=P59/'out'/f'actions_{seed}.pt';y=torch.load(src,map_location='cpu')['predictions']['pt0']['F'][[8,9,62,63]].clone()
        q=OUT/'wl59'/f'replay_{seed}.pt';torch.save(dict(F=y,fields=[8,9,62,63],source=str(src)),q);files[f'wl59/{q.name}']=dict(sha256=sha(q),bytes=q.stat().st_size,source=str(src),source_sha256=sha(src))
    g0=torch.load(P59/'data/geometry/rest.pt',map_location='cpu');parent=g0['parent']
    fp={}
    for ck in ('WL59_P_5901','WL59_P_5902'):
        for n in ('rest','pt0'):
            m=torch.load(R66/'model'/f'{ck}_{n}.pt',map_location='cpu')
            for slot,y in m['scatter'].items():
                z=torch.zeros(16,y.shape[1],16,3,dtype=torch.float64).index_add_(2,bm,y.double())
                fp[f'{ck}/{n}/s{slot}']=torch.zeros(16,240,16,3,dtype=torch.float64).index_add_(1,parent,z)
    q=OUT/'wl66_frozen_coarse.pt';torch.save(fp,q);files[q.name]=dict(sha256=sha(q),bytes=q.stat().st_size)
    (OUT/'manifest.json').write_text(json.dumps(dict(created=time.strftime('%Y-%m-%d %H:%M:%S %z'),host=os.uname().nodename,files=files,
        wl66_packet_sha256=sources,cross_machine_packets=CROSS,seconds=time.time()-t0),indent=1)+'\n')
    print('EXPORT_DONE',round(time.time()-t0),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
