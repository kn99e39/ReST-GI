import torch

def aggregate(y,parent):
    y=y.double();bm=torch.arange(64,device=y.device);bm=(bm%8)//2+4*((bm//8)//2)
    z=torch.zeros(*y.shape[:-2],16,3,device=y.device,dtype=torch.float64).index_add_(-2,bm,y)
    return torch.zeros(len(y),int(parent.max())+1,16,3,device=y.device,dtype=torch.float64).index_add_(1,parent,z)
def variance(first,second,N):return (second-first.square()/N)*(N/(N-1))
def covariance_BS(B,S,N):return -B*S/(N-1)
def norm(x):return x.reshape(-1,3).norm(dim=0)
def ratios(a,b):return (norm(a)/norm(b).clamp(min=1e-300)).tolist()
def error(a,b):return ratios(a-b,b)
def classify(bb,ss,cross,bt,st):
    if bb>=4*ss and bt>0 and abs(cross)<=.5*bb:return 'A_B_DOMINATED'
    if ss>=4*bb and st>0 and abs(cross)<=.5*ss:return 'B_SCATTERING_DOMINATED'
    return 'C_INTERACTING_OR_MIXED'
def decompose(Bmodel,Smodel,Bref,Sref):
    eb=Bmodel-Bref;es=Smodel-Sref;et=eb+es;t=Bref+Sref;rows=[]
    for c in range(3):
        b=eb[...,c].flatten();s=es[...,c].flatten();e=et[...,c].flatten();den=float(t[...,c].square().sum());bb=float(b@b);ss=float(s@s);cross=float(2*(b@s));ee=float(e@e)
        bt=float(b@e);st=float(s@e)
        rows.append(dict(channel=c,total_relative_L2=(ee/den)**.5,B_error_on_total_norm=(bb/den)**.5,Scat_error_on_total_norm=(ss/den)**.5,B_error_relative_to_B=float(b.norm()/Bref[...,c].norm().clamp(min=1e-300)),Scat_error_relative_to_Scat=float(s.norm()/Sref[...,c].norm().clamp(min=1e-300)),B_squared_over_total_signal=bb/den,Scat_squared_over_total_signal=ss/den,cross_over_total_signal=cross/den,total_squared_over_total_signal=ee/den,error_energy_identity_abs=abs(ee-bb-ss-cross),error_cosine=float((b@s)/(b.norm()*s.norm()).clamp(min=1e-300)),B_alignment_with_total=float(bt/(b.norm()*e.norm()).clamp(min=1e-300)),Scat_alignment_with_total=float(st/(s.norm()*e.norm()).clamp(min=1e-300)),B_projection_share=bt/max(ee,1e-300),Scat_projection_share=st/max(ee,1e-300),classification=classify(bb,ss,cross,bt,st)))
    return rows
