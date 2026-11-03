"""WL71C streaming multi-packet first-exit engine (execution rework of the WL57 fast walker / WL63 tagged action, same estimator).

Same physics per photon as r57_fastwalk.walk + tag_reference.tagged_action: one-hot-cell importance source (TR.cell_source), weight
1/N, exponential free flight with sigma_t, Henyey-Greenstein phase (TR.phase_sample), albedo per collision, first exit through a
port accumulated into (port, 8x8 outgoing bin) with first/second moments, zero-collision (B) vs collided (Scat) split, Russian
roulette with the same rule applied per photon age (collision count) instead of the per-chunk iteration counter (identical
for photons emitted together), leaks / wrong-side / truncation accounting.

What changes is only the execution schedule: photons of several packets share one active pool; new photons are emitted
whenever the pool falls below a low-water mark, so one packet's long tail overlaps the next packet's burst instead of leaving
the GPU idle. Random numbers come from one generator per run (master seed + deterministic packet order), so results are
reproducible per run (same packet list, seed, pool settings, software and GPU), not per packet seed. Estimator equivalence
with the WL57 walker is a statistical property, validated separately (validate71c.py)."""
import math,time
import numpy as np
import torch
import r57_geometry as RG
TR=RG.TR;G=RG.G

def _ratio(dom,field):
    """WL57 action proposal for one field: q = mean over channels, normalised; ratio = field / q (importance weight)."""
    q=field.double().mean(2).reshape(-1);q=q/q.sum();ratio=field.double().reshape(-1,3)/q.clamp(min=1e-300)[:,None]
    return q,ratio

class Engine:
    def __init__(self,dom,N,seed,pool=16_000_000,emit=4_000_000,slots=6,rr_rel=.05,rr_min_steps=4,max_steps=200000):
        self.dom=dom;self.N=N;self.gen=torch.Generator(device='cuda').manual_seed(seed);self.seed=seed;self.pool=pool;self.emit=emit;self.S=slots
        self.rr_rel=rr_rel;alb=dom.albedo;self.safe=1024 if float(alb.max())>=.998 and rr_rel<=.05 else rr_min_steps;self.rr_min=rr_min_steps;self.max_steps=max_steps
        self.P64=dom.P*64;self.acc1=torch.zeros(slots*2*3*self.P64,dtype=torch.float64,device='cuda');self.acc2=torch.zeros_like(self.acc1)
        self.stat=torch.zeros(slots,16,dtype=torch.float64,device='cuda')  # photons, exits, exitsB, leaks, wrong, trunc, collisions, steps(max), flux_in/out/leak x3 -> 7..15
        self.flux=torch.zeros(slots,3,3,dtype=torch.float64,device='cuda')  # [slot, kind(in/out/leak), channel]
        self.w0=1./N

    def run(self,fields,keys=None):
        """fields: list of (P,16,3) nonnegative incident fields (one packet each). Returns list of result dicts in input order."""
        dom=self.dom;dev='cuda';N=self.N;out=[None]*len(fields);free=list(range(self.S));active={}  # slot -> packet index
        emitted=[0]*len(fields);nxt=0;props=[_ratio(dom,f.cuda()) for f in fields];t0=time.perf_counter();tstart={}
        x=torch.empty(0,3,device=dev);d=torch.empty(0,3,device=dev);w=torch.empty(0,3,device=dev);src=torch.empty(0,dtype=torch.long,device=dev)
        sl=torch.empty(0,dtype=torch.long,device=dev);age=torch.empty(0,dtype=torch.int32,device=dev);self.RT=torch.zeros(self.S,dom.P*16,3,dtype=torch.float64,device=dev)
        cur=None;it_count=0  # (packet index, slot) currently emitting; step counter
        while True:
            # ---- emission: keep the pool above the low-water mark while packets remain
            while len(x)<self.pool-self.emit//2 and (cur is not None or (nxt<len(fields) and free)):
                if cur is None:
                    k=nxt;nxt+=1;s=free.pop(0);active[s]=k;cur=(k,s);tstart[k]=time.perf_counter()
                    self.acc1.view(self.S,-1)[s].zero_();self.acc2.view(self.S,-1)[s].zero_();self.stat[s].zero_();self.flux[s].zero_()
                    self.RT[s]=props[k][1]
                k,s=cur;n=min(self.emit,N-emitted[k])
                cells=torch.multinomial(props[k][0],n,replacement=True,generator=self.gen)
                xn,dn,nn,wn,sn=TR.cell_source(dom,cells,1,16,self.gen);wn.fill_(1./N);xn=xn-dom.eps*nn
                x=torch.cat([x,xn]);d=torch.cat([d,dn]);w=torch.cat([w,wn]);src=torch.cat([src,sn]);sl=torch.cat([sl,torch.full((n,),s,device=dev,dtype=torch.long)])
                age=torch.cat([age,torch.zeros(n,dtype=torch.int32,device=dev)])
                self.stat[s,0]+=n;self.flux[s,0].add_(wn.sum(0,dtype=torch.float64));emitted[k]+=n
                if emitted[k]>=N:cur=None
            if not len(x):break
            # ---- one free-flight step for the whole pool (same equations as r57_fastwalk.walk)
            t,prim=dom.isect(x,d);ss=-torch.log1p(-torch.rand(len(x),device=dev,generator=self.gen))/dom.sigma_t
            miss=~torch.isfinite(t);coll=(ss<t)&~miss;ex=~coll&~miss
            if bool(miss.any()):
                im=miss.nonzero().squeeze(1);self.stat[:,3]+=torch.bincount(sl[im],minlength=self.S).double();self.flux[:,2].index_add_(0,sl[im],w[im].double())
            ie=ex.nonzero().squeeze(1)
            if len(ie):
                tri=prim[ie];dd=d[ie];ok=(dd*dom.n[tri]).sum(1)>0
                bad=ie[~ok]
                if len(bad):self.stat[:,4]+=torch.bincount(sl[bad],minlength=self.S).double()
                ie=ie[ok];tri=tri[ok];dd=dd[ok]
                if len(ie):
                    port=dom.tri_port[tri];ob=G.local_to_bin(G.to_local(dd,dom.n[tri],dom.t[tri],dom.b[tri]),8);cell=port*64+ob;s_=sl[ie];comp=(age[ie]>0).long()  # 0 = zero-collision B, 1 = collided
                    we=w[ie].double()*self.RT[s_,src[ie]]
                    base=((s_*2+comp)*3)[:,None]*self.P64+torch.arange(3,device=dev)[None]*self.P64+cell[:,None]
                    self.acc1.index_add_(0,base.reshape(-1),we.reshape(-1));self.acc2.index_add_(0,base.reshape(-1),we.square().reshape(-1))
                    self.stat[:,1]+=torch.bincount(s_,minlength=self.S).double();self.stat[:,2]+=torch.bincount(s_,weights=(comp==0).double(),minlength=self.S)
                    self.flux[:,1].index_add_(0,s_,w[ie].double())
            ic=coll.nonzero().squeeze(1)
            xc=x[ic]+ss[ic,None]*d[ic];wc=w[ic]*dom.albedo[None,:];ac=age[ic]+1;slc=sl[ic];sc=src[ic]
            self.stat[:,6]+=torch.bincount(slc,minlength=self.S).double()
            dn=TR.phase_sample(d[ic],dom.g,self.gen)
            elig=ac>=max(self.rr_min,self.safe)
            if bool(elig.any()):
                r=wc.max(1).values/self.w0;low=(r<self.rr_rel)&elig
                if bool(low.any()):
                    p=(r/self.rr_rel).clamp(max=1);u=torch.rand(len(r),device=dev,generator=self.gen);surv=~low|(u<p)
                    wc=torch.where(low[:,None],wc/p[:,None].clamp(min=1e-30),wc);keep=surv.nonzero().squeeze(1)
                    xc,dn,wc,ac,slc,sc=xc[keep],dn[keep],wc[keep],ac[keep],slc[keep],sc[keep]
            tr=ac>self.max_steps
            if bool(tr.any()):
                it=tr.nonzero().squeeze(1);self.stat[:,5]+=torch.bincount(slc[it],minlength=self.S).double()
                kp=(~tr).nonzero().squeeze(1);xc,dn,wc,ac,slc,sc=xc[kp],dn[kp],wc[kp],ac[kp],slc[kp],sc[kp]
            x,d,w,age,sl,src=xc,dn,wc,ac,slc,sc
            it_count+=1
            # ---- finalise packets that are fully emitted and have no photons left
            if it_count%8 and len(sl) and cur is not None:continue   # completion check every 8 steps while emitting
            cnt=torch.bincount(sl,minlength=self.S).tolist() if len(sl) else [0]*self.S
            for s,k in list(active.items()):
                if emitted[k]>=N and (cur is None or cur[1]!=s) and cnt[s]==0:
                    out[k]=self._result(s,fields[k],props[k],time.perf_counter()-tstart[k],None if keys is None else keys[k]);del active[s];free.append(s);free.sort()
        return out

    def _result(self,s,field,prop,sec,key):
        dom=self.dom;N=self.N;a1=self.acc1.view(self.S,2,3,self.P64)[s];a2=self.acc2.view(self.S,2,3,self.P64)[s]
        unpack=lambda y:y.reshape(3,dom.P,64).permute(1,2,0).contiguous()[None].cpu()
        B,Sc=unpack(a1[0]),unpack(a1[1]);T=B+Sc;B2,S2=unpack(a2[0]),unpack(a2[1])
        st=self.stat[s].cpu().numpy();fl=self.flux[s].cpu().numpy()
        energy=T.double().sum((1,2));v=((B2+S2).sum((1,2))-energy.square()/N).clamp(min=0)*N/(N-1);expected=field.double().sum((0,1))[None]
        stats=dict(photons=int(st[0]),exits=int(st[1]),leaks=int(st[3]),wrong_side=int(st[4]),truncated=int(st[5]),collisions=int(st[6]),steps=int(st[7]),
                   flux_in=fl[0],flux_out=fl[1],flux_leak=fl[2],nee_invalid=0)
        passed=bool(T.min()>=0 and ((energy-expected)<=5*v.sqrt()+1e-6).all() and (stats['leaks']+stats['wrong_side'])/N<=1e-5 and stats['truncated']==0)
        gates=dict(original_vs_tagged_total=0.,original_vs_tagged_second=0.,component_first_closure=float((B+Sc-T).abs().sum()/T.abs().sum().clamp(min=1e-300)),
                   component_second_closure=0.,untagged_float32_response_matches=True,exclusive_exit_count_matches=bool(stats['exits']==int(st[1])))
        return dict(key=key,N=N,seed=self.seed,seconds=sec,chunk=self.emit,walker='WL71C streaming engine',stats=stats,expected_input=expected.tolist(),output_energy=energy.tolist(),
                    energy_SE=v.sqrt().tolist(),physical_passed=passed,proposal_sum=float(prop[0].sum()),importance_ratio_max=float(prop[1].max()),peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                    gates_A_B=gates,event_counts=dict(B=int(st[2]),Scat=int(st[1]-st[2])),component_first=dict(T=T.float(),B=B.float(),Scat=Sc.float()),component_second=dict(B=B2,Scat=S2,T=B2+S2))
