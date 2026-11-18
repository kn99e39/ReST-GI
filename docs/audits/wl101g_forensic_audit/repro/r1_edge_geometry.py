"""Reconstruct the failing Box B samples from (pid, u0, u1) with the frozen scene86 layout; metric edge distances."""
import math, numpy as np
c=np.array([1.5236285335112942,1.0627971190791698,0.35]); h=[0.15,0.18,0.35]; yaw=0.2917575507983794
ax=[np.array([math.cos(yaw),math.sin(yaw),0]),np.array([-math.sin(yaw),math.cos(yaw),0]),np.array([0,0,1.])]
pairs={(0,1):(1,2),(0,0):(2,1),(1,1):(2,0),(1,0):(0,2),(2,1):(0,1),(2,0):(1,0)}
faces=[]
for a in range(3):
    for pos in (0,1):
        i1,i2=pairs[(a,pos)]; faces.append((c+(1 if pos else -1)*h[a]*ax[a],ax[i1],ax[i2],h[i1],h[i2],a,pos))
NRECT_BEFORE=256+256+192+192+192+42+54   # floor, ceiling, back, left, right, panel, box A
def patch(pid):
    k=pid-NRECT_BEFORE; f,cell=divmod(k,9); iv,iu=divmod(cell,3)
    C,A1,A2,H1,H2,a,pos=faces[f]
    e1=A1*2*H1/3; e2=A2*2*H2/3; o0=C-A1*H1-A2*H2
    return f,iu,iv,o0+iu*e1+iv*e2,e1,e2,np.cross(A1,A2)
names=['-x\'','+x\'','-y\'','+y\'','-z','+z']
for pid,u0,u1,label in ((1211,0.7080796957015991,5.960464477539062e-07,'A1 membership miss'),):
    f,iu,iv,po,e1,e2,n=patch(pid)
    p=po+u0*e1+u1*e2
    print(label,'face',names[f],'cell',(iu,iv),'p',p,'normal',n,'|e1|,|e2|',np.linalg.norm(e1),np.linalg.norm(e2))
    print('  metric distance to the u1=0 boundary (m):',u1*np.linalg.norm(e2),' boundary is face edge:',iv==0,
          ' float32 ulp of max coordinate:',np.spacing(np.float32(np.abs(p).max())))
    print('  WL101E 4-eps band on this axis (m):',4*np.finfo(np.float32).eps*np.linalg.norm(e2),
          ' Amendment-1 8e-6 band (m):',8e-6*np.linalg.norm(e2))
for pid in (1188,1202,1203):
    f,iu,iv,po,e1,e2,n=patch(pid); print(pid,'face',names[f],'cell',(iu,iv),'normal',np.round(n,6),'po',np.round(po,4))
# parametric 4-eps and 8e-6 bands in metres for every planar patch dimension in the scene
dims={'room 0.25':0.25,'panel 0.2':0.2,'boxA x 0.1467':2*0.22/3,'boxA y 0.2333':2*0.35/3,'boxA z 0.3333':1/3,
      'boxB x 0.1':0.1,'boxB y 0.12':0.12,'boxB z 0.2333':0.7/3}
for k,d in dims.items():
    print(f'{k:16s} 4eps {4*np.finfo(np.float32).eps*d:.2e} m   8e-6 {8e-6*d:.2e} m')
