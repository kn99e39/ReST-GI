import sys, json, math, numpy as np
sys.path.insert(0, r'c:/Projects/ReST-GI/experiments/ecro86')
import scene86 as S
ck = json.load(open(r'c:/Projects/ReST-GI/experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
G = S.Geom(ck['geom'])
N = G.N
# S1 cells 5 cm
pid = np.where(G.PMAT==1)[0]
cells=[]
for p in pid:
    for a in (np.arange(4)+.5)/4:
        for b in (np.arange(4)+.5)/4:
            cells.append(G.PO[p]+a*G.PE1[p]+b*G.PE2[p])
cells=np.array(cells); pn=G.PN[pid[0]]
# receiver points 4x4 on rect patches (skip sphere), excluding panel
rp=[];rn=[];rpid=[]
for p in range(G.SBASE):
    if G.PMAT[p]==1: continue
    for a in (np.arange(4)+.5)/4:
        for b in (np.arange(4)+.5)/4:
            rp.append(G.PO[p]+a*G.PE1[p]+b*G.PE2[p]); rn.append(G.PN[p]); rpid.append(p)
rp=np.array(rp); rn=np.array(rn); rpid=np.array(rpid)
print('cells',len(cells),'recv pts',len(rp),'pairs',len(cells)*len(rp))
def obb_hit(o,d,box,tmax):
    c=np.array(box['c']);h=np.array(box['h']);ax=np.array(S.box_axes(box['yaw']))
    lo=np.zeros(o.shape[:-1]);hi=np.full(o.shape[:-1],1.0)
    for k in range(3):
        oo=(o-c)@ax[k]; dd=d@ax[k]
        with np.errstate(divide='ignore',invalid='ignore'):
            t1=(-h[k]-oo)/dd; t2=(h[k]-oo)/dd
        tn=np.minimum(t1,t2); tf=np.maximum(t1,t2)
        par=np.abs(dd)<1e-12
        tn=np.where(par,np.where(np.abs(oo)<=h[k],-np.inf,np.inf),tn)
        tf=np.where(par,np.where(np.abs(oo)<=h[k],np.inf,-np.inf),tf)
        lo=np.maximum(lo,tn);hi=np.minimum(hi,tf)
    return (lo<hi-1e-9)&(hi>1e-6)&(lo<1-1e-6)
A=ck['geom']['A']
tot=0;front=0;dirtyA=0
recvA=(G.PSID[rpid]==S.SID_BOXA) if hasattr(G,'PSID') else None
for i in range(0,len(cells),32):
    c=cells[i:i+32][:,None,:]
    d=rp[None]-c
    ff=((d@pn)>1e-9)&((np.einsum('ijk,jk->ij',-d,rn))>1e-9)
    hit=obb_hit(c+0*d,d,A,1.0)
    tot+=d.shape[0]*d.shape[1]; front+=ff.sum(); dirtyA+=(ff&hit).sum()
onA=np.isin(rpid,np.where(G.PSID==S.SID_BOXA)[0])
print('front pairs',front,'frac',front/tot)
print('front pairs blocked by box A (current pose)',dirtyA,'frac of front',dirtyA/front)
# pairs landing on box A receivers that are front-facing
ffA=0
for i in range(0,len(cells),32):
    c=cells[i:i+32][:,None,:]; d=rp[onA][None]-c
    ff=((d@pn)>1e-9)&((np.einsum('ijk,jk->ij',-d,rn[onA]))>1e-9); ffA+=ff.sum()
print('front pairs to box A receivers',ffA,'frac',ffA/front)
# moved pose: shift box A by 0.3 m in y as a typical edit
A2=dict(A); A2['c']=[A['c'][0],A['c'][1]-0.3,A['c'][2]]
d2=0
for i in range(0,len(cells),32):
    c=cells[i:i+32][:,None,:]; d=rp[None]-c
    ff=((d@pn)>1e-9)&((np.einsum('ijk,jk->ij',-d,rn))>1e-9)
    d2+=(ff&(obb_hit(c+0*d,d,A,1.0)|obb_hit(c+0*d,d,A2,1.0))).sum()
print('dirty (old or new pose, 0.3 m y-shift) frac of front, excl. receivers on A',d2/front)
