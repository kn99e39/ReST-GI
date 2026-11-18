"""Independent numpy model of the sphere/floor tangency events (spawn-offset geometry only).
Predicts per-replica counts of (a) WL101G 'tangency' misses and (b) cap back-receiver hits,
using Mitsuba 3 offset_p: mag=(1+max|p|)*2^-24*1500 along n (sign of n.w)."""
import math, numpy as np
RAYEPS=2.0**-24*1500
SC=np.array([0.9,0.4,0.4]); R=0.4
NZ,NPHI=4,16
rng=np.random.default_rng(1)
def frame(iz,ip):
    z=(iz+.5)/4*2-1; phi=(ip+.5)/8*2*math.pi-math.pi
    n=np.array([math.sqrt(1-z*z)*math.cos(phi),math.sqrt(1-z*z)*math.sin(phi),z])
    t=np.cross([0,0,1],n); t/=np.linalg.norm(t); b=np.cross(n,t); return t,b,n
tot={'valid':0,'tangency':0,'cap_back':0,'cap_valid':0}
reps=8
for rep in range(reps):
  for ip in range(8):
    t,b,nc=frame(0,ip)
    for bn in range(64):
        m=4096
        u0,u1,u2,u3=rng.random((4,m))
        z=(0+u1)*(2/4)-1; phi=(ip+u0)*(2*math.pi/8)-math.pi
        r=np.sqrt(np.maximum(1-z*z,0)); ns=np.stack([r*np.cos(phi),r*np.sin(phi),z],1)
        p=SC+R*ns
        if bn<16:
            lz=1.25*u2-1; ph=((bn%16)+u3)*(2*math.pi/16)-math.pi
        else:
            lz=((bn//16)+u2)/NZ; ph=((bn%16)+u3)*(2*math.pi/16)-math.pi
        rr=np.sqrt(np.maximum(1-lz*lz,0)); loc=np.stack([rr*np.cos(ph),rr*np.sin(ph),lz],1)
        w=loc[:,:1]*t+loc[:,1:2]*b+loc[:,2:]*nc
        cand=(np.einsum('ij,ij->i',w,ns)>1e-6)&(p[:,2]>0)
        mag=(1+np.abs(p).max(1))*RAYEPS
        o=p+mag[:,None]*ns          # n.w>0 for candidates
        below=o[:,2]<0
        tot['valid']+=cand.sum()
        tot['cap_valid']+=(cand&below).sum()
        tot['tangency']+=(cand&below&(w[:,2]<0)).sum()
        tot['cap_back']+=(cand&below&(w[:,2]>=0)).sum()
print({k:v/reps for k,v in tot.items()})
mag0=(1+0.9)*RAYEPS
print('floor-side contact ring radius m', math.sqrt(2*R*mag0), 'area m2', math.pi*2*R*mag0,
      'fraction of floor patch', math.pi*2*R*mag0/0.0625, 'of subcell', math.pi*2*R*mag0/0.015625)
print('sphere cap height m', mag0, 'box spawn offset at |p|=1.3757:', (1+1.3757323)*RAYEPS, ' at 1.5530683:', (1+1.5530683)*RAYEPS)
