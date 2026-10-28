"""Execution-only specialization of immutable p53_tracer.walk for A2 unit RGB paths.

Same intersections, distances, frames, phase_sample, absorption, RNG calls and RR.
No NEE/air continuation (A2 contract). Reuse compaction indices; defer counters
to GPU; omit unused exit points/S2 copies/single-element cat. For unit RGB and
max albedo>=.998, RR cannot activate in the first1024 collisions: .998**1024>.05.
Float32 rounding is far below that margin. Thus skip only a provably false RR test.
"""
import torch
import r57_geometry as RG
TR=RG.TR;G=RG.G

def walk(acc,x,d,n_entry,w,src,gen,rr_rel=.05,rr_min_steps=4,max_steps=200000):
    assert acc.KC is None and torch.equal(w,w[0,0].expand_as(w))
    dom=acc.dom;sig=dom.sigma_t;alb=dom.albedo;st=acc.stats
    x=x-dom.eps*n_entry;w0=w[0,0].clone().clamp(min=1e-30)
    safe_skip=1024 if float(alb.max())>=.998 and rr_rel<=.05 else rr_min_steps
    if not hasattr(acc,'_fast_flux'):acc._fast_flux=torch.zeros(3,3,device=x.device,dtype=torch.float64)
    flux=acc._fast_flux;flux[0].add_(w.sum(0,dtype=torch.float64));st['photons']+=len(x);step=0
    while len(x):
        step+=1
        if step>max_steps:st['truncated']+=len(x);break
        t,prim=dom.isect(x,d)
        s=-torch.log1p(-torch.rand(len(x),device=x.device,generator=gen))/sig
        miss=~torch.isfinite(t);coll=(s<t)&~miss;ex=~coll&~miss
        # Exactly one dynamic compaction per mask; its shape already supplies counts.
        ie=ex.nonzero().squeeze(1);ic=coll.nonzero().squeeze(1)
        missing=len(x)-len(ie)-len(ic)
        if missing:
            st['leaks']+=missing;flux[2].add_(w[miss].sum(0,dtype=torch.float64))
        if len(ie):
            tri=prim[ie];dd=d[ie];ok=(dd*dom.n[tri]).sum(1)>0;good=ok.nonzero().squeeze(1)
            st['wrong_side']+=len(ie)-len(good)
            if len(good)!=len(ie):ie=ie[good];tri=tri[good];dd=dd[good]
            if len(ie):
                port=dom.tri_port[tri];ob=G.local_to_bin(G.to_local(dd,dom.n[tri],dom.t[tri],dom.b[tri]),acc.Nout)
                we=w[ie];acc.T.index_add_(1,(port*acc.Dout+ob)*acc.nsrc+src[ie],we.T.double())
                # Action sink already records weighted second moments; canonical S2 is unused.
                st['exits']+=len(ie);flux[1].add_(we.sum(0,dtype=torch.float64))
        if not len(ic):break
        xc=x[ic]+s[ic,None]*d[ic];dc=d[ic];wc=w[ic]*alb[None,:];sc=src[ic]
        st['collisions']+=len(ic);dn=TR.phase_sample(dc,dom.g,gen)
        if step>=max(rr_min_steps,safe_skip):
            r=wc.max(1).values/w0;low=r<rr_rel
            if low.any():
                p=(r/rr_rel).clamp(max=1);u=torch.rand(len(r),device=x.device,generator=gen);surv=~low|(u<p)
                wc=torch.where(low[:,None],wc/p[:,None].clamp(min=1e-30),wc)
                # Reuse the survivor index rather than repeat boolean compaction.
                si=surv.nonzero().squeeze(1);xc,dn,wc,sc=xc[si],dn[si],wc[si],sc[si]
        x,d,w,src=xc,dn,wc,sc
    st['steps']=max(st['steps'],step)

def flush(acc):
    if hasattr(acc,'_fast_flux'):
        values=acc._fast_flux.cpu().numpy()
        for i,key in enumerate(('flux_in','flux_out','flux_leak')):acc.stats[key]+=values[i]
        del acc._fast_flux
