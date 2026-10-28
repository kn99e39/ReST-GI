"""Conditional G5 only: structured learned K adapter, unchanged historical C and solve.
Complete outgoing bins include scattering AND zero collision; source radiance is known.
No target K is loaded in the candidate solve. Oracle comparison is a separate solve.
"""
import sys,os,json,time
import numpy as np
import torch
import r57_geometry as RG
from r57_data import ROOT,OUT,DATA,dump,log
from r57_eval import model_of
import s56 as S
import s56_fixtures as SF
import s56_analyze as SA
import p53_compose as CP
import k55_compose as KP
import c54_casp as CA
import k55_ops as KO
import p53_geom as G

class Channel:
    def __init__(self,owner,k):self.owner,self.k=owner,k
    def __matmul__(self,v):
        a=self.owner;x=torch.zeros(1,a.g['P'],16,3,device='cuda');x[0,:,:,self.k]=v.view(a.g['P'],16)
        return a.m.apply(a.g,x,a.co)[0,:,:,self.k].reshape(-1)
class Adapter:
    def __init__(self,m,g):self.m,self.g=m,g;self.co=m.coefficients(g)
    def __getitem__(self,k):return Channel(self,k)

def shared(state):
    sd=SF.scene(state);ss,parent,_=KP.build_ss(sd,[4,4,1]);cnt=np.bincount(parent);mp=np.maximum(1,np.round(128/cnt[parent])).astype(int)
    for ci,c in enumerate(ss.comps):
        if c.kind=='black':mp[ss.comp_ports(ci)]=1
    C,esc,stats=CA.cast_adaptive(ss,mp,5681)
    gen=torch.Generator(device='cuda').manual_seed(5661);N=64_000_000;v=torch.zeros(ss.Ptot*16,device='cuda',dtype=torch.float64)
    for i in range(0,N,8_000_000):
        n=min(8_000_000,N-i);x,d,w=KP.emitter_rays(ss,sd,n,gen);w=w*n/N;t,p=ss.isect(x,d);j=torch.isfinite(t).nonzero().squeeze(1);j=j[(d[j]*ss.n[p[j]]).sum(1)<0];pj=p[j]
        b=G.local_to_bin(G.to_local(-d[j],ss.n[pj],ss.t[pj],ss.b[pj]),ss.Nin);v.index_add_(0,ss.tri_port[pj]*16+b,w[j].double())
    return sd,ss,parent,C,v[:,None].expand(-1,3).float(),dict(C_stats=stats,closure_max=float((torch.sparse.sum(C,0).to_dense()+esc-1).abs().max()))

def readout(ss,parent,phi):
    Pc=int(parent.max())+1;par=torch.as_tensor(parent,device='cuda');out={}
    for key in SF.KEYS[:2]:
        d=torch.tensor(SF.FX.DIRS[key],device='cuda',dtype=torch.float32)
        cell=ss.tri_port;ob=G.local_to_bin(G.to_local(d.expand(len(cell),3),ss.n,ss.t,ss.b),ss.Nout)
        lt=phi[cell*64+ob].double()/(ss.port_area[cell].double()*np.pi/64)[:,None]
        # The emitter is absorbing for recurrence but its declared unit emission is visible to cameras.
        lt[ss.tri_comp==2]=1.
        w=(ss.area*(ss.n*d).sum(1).clamp(min=0)).double();den=torch.zeros(Pc,device='cuda',dtype=torch.float64).index_add_(0,par[cell],w)
        num=torch.zeros(Pc,3,device='cuda',dtype=torch.float64).index_add_(0,par[cell],lt*w[:,None]);out[key+'|bin']=(num/den.clamp(min=1e-30)[:,None]).cpu().numpy()
    out['phi_out']=phi.cpu().numpy();return out

def main():
    held=json.load(open(OUT/'heldout.json'));eligible=[k for k,v in held.items() if v['useful_and_physical']]
    if not eligible:
        dump(dict(executed=False,reason='No learned model passed preregistered operator fidelity+physics; G5 skipped. WL56 physical blue floor remains.',models=list(held)),OUT/'composition.json');return
    records={};(OUT/'composition').mkdir(exist_ok=True)
    with torch.no_grad():
        for state in ('rest','dent'):
            sd,ss,parent,C,ext,meta=shared(state);refs=S.extract(state);labels=S.CM.labels_of(sd);pose={};oracle_predictions=[]
            for rep in (0,1):
                ks={}
                for ci,name in ((0,'rest'),(1,'te0' if state=='dent' else 'rest')):
                    r=torch.load(ROOT/'ops'/f'{name}_r{rep}.pt');ks[ci]=KO.KOp(r['idx'],r['val'],r['shape']);del r
                sol=CP.solve(ss,ks,C,CP.apply_K(ss,ks,ext));assert sol['converged'];op=readout(ss,parent,sol['phi_out']);oracle_predictions.append(op);del ks;torch.cuda.empty_cache()
            for tag in eligible:
                arm,seed=tag.split('_');m=model_of(arm,int(seed));ks={}
                for ci,name in ((0,'rest'),(1,'te0' if state=='dent' else 'rest')):ks[ci]=Adapter(m,RG.device(torch.load(DATA/'geometry'/f'{name}.pt')))
                t0=time.perf_counter();sol=CP.solve(ss,ks,C,CP.apply_K(ss,ks,ext));torch.cuda.synchronize();sec=time.perf_counter()-t0
                pred=readout(ss,parent,sol['phi_out']);np.savez(OUT/'composition'/f'{state}_{tag}.npz',**pred)
                rows=[]
                for cam,units in refs.items():
                    if cam=='otop':continue
                    key=S.CM.dirkey(sd['cams'][cam]['dir'])
                    for mask,u in units.items():
                        for row in SA.group_stats([pred[key+'|bin']],u,u['sel'],labels,cam,'bin',tag):
                            sel=u['sel']&(labels==row['group'])
                            if '@' in row['group']:continue
                            o=np.mean([p[key+'|bin'][sel].mean(0) for p in oracle_predictions],0);p=pred[key+'|bin'][sel].mean(0)
                            row.update(cam=cam,mask=mask,candidate_vs_oracle=(p-o)/np.maximum(o,1e-30),oracle_vs_cycles=(o-row['reference_mean'])/np.maximum(row['reference_mean'],1e-30));rows.append(row)
                pose[tag]=dict(converged=sol['converged'],iterations=sol['iters'],residual=sol['last_rel'],solve_seconds=sec,comparisons=rows)
                del m,ks;torch.cuda.empty_cache()
            records[state]=dict(shared=meta,models=pose);dump(dict(executed=True,states=records,readout='Common full outgoing-bin readout; no oracle KC head at model runtime. Separate from historical NEE+KB.'),OUT/'composition.json')
if __name__=='__main__':torch.set_num_threads(8);main();sys.stdout.flush();os._exit(0)
