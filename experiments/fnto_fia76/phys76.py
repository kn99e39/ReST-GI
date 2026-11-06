"""WL76 Stage 0/1: first-interaction factorization of the collided first-exit operator and its physical correctness gate.

Medium (p53_fixtures): homogeneous, sigma_t = 1000 /m, albedo (0.999, 0.99, 0.9), isotropic phase (g = 0), index-matched boundary
(a photon reaching the boundary exits). Input cell (port i, bin a): unit flux, uniform on the port, cosine-weighted inside the bin,
travelling along -w (p53_tracer.cell_source). Output: flux by exit port j and outgoing bin o (64), RGB.

For one entering photon at x0 (on the port) with direction d, chord t = distance to the boundary along d:
  zero-collision exit with probability exp(-sigma t)                                    -> B (ballistic, existing)
  first collision with probability c = 1 - exp(-sigma t), at s ~ sigma e^{-sigma s} / c on [0, t)    -> p1(x | i, a)
  at the collision the weight is multiplied by albedo and the direction is redrawn ISOTROPICALLY, independent of d (g = 0).
Hence  K_scat(i,a -> j,o) = (1 - beta(i,a)) * E_{x ~ p1(.|i,a)} [ Q(x -> j,o) ]  with  1 - beta(i,a) = E[c]  and
  Q(x -> j,o) = albedo * [ E1(x -> j,o) + M(x -> j,o) ]
  E1 = (1/4pi) int over directions w that leave through (j, o) of exp(-sigma |exit - x|)       (single scatter: exact, geometry only)
  M  = int over directions of (1 - exp(-sigma t_w)) E_{y ~ next collision} [ albedo E1(y) + ... ]  (orders >= 2)
Q depends on x and the current geometry only (Markov property of the walk + isotropic phase): not on i, a or d. With a forward-
peaked phase (g != 0) Q would depend on d; with surface (BSDF) scattering the 'first interaction' is on the boundary and the
redirection depends on the incident direction and the local frame - neither factorization holds unchanged.
Energy: sum_{j,o} E1 <= 1, sum_{j,o} Q <= albedo, sum_{j,o} K_scat <= 1 - beta: passive.

Estimators here (validation only; never inference):
  decomposed(): forced-first-collision analog MC: weight c * albedo at x ~ p1, isotropic restart, the next segment sampled exactly
     ('single' if it exits, else a second collision), then the UNCHANGED reference walker p53_tracer.walk from the second collision ('multi').
     Unbiased for K_scat; independent random streams from every archived reference.
  single_quadrature(): expected-value single-scatter estimator (exp(-sigma t2) weight instead of a sampled exit) = the deployable E1 term.
  forward_control(): NEGATIVE CONTROL - the same estimator with the restart direction kept equal to d (a g -> 1 assumption); must fail.
    python phys76.py stage1"""
import sys,os,json,time,math
import torch
from common76 import CFG,ROOT,R73,R68,R71,INP68,write_new,environment,seed,dev

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

class Acc:
    """Accumulator for p53_tracer.walk: dense first/second moments of exit flux by (port*64 + bin), single source."""
    def __init__(self,dom):
        self.dom=dom;self.Dout=64;self.Din=16;self.Nout=8;self.nsrc=1;self.psrc=1;self.KC=None
        self.m1=torch.zeros(3,dom.P*64,device='cuda',dtype=torch.float64);self.m2=torch.zeros_like(self.m1);self.m0=torch.zeros(dom.P*64,device='cuda',dtype=torch.float64)
        outer=self
        class Sink:
            def index_add_(self,dim,idx,we):outer.m1.index_add_(1,idx,we);outer.m2.index_add_(1,idx,we.square());outer.m0.index_add_(0,idx,torch.ones_like(idx,dtype=torch.float64))
        class Nop:
            def index_add_(self,*a):pass
        self.T=Sink();self.S2=Nop()
        import numpy as np
        self.stats=dict(photons=0,exits=0,leaks=0,wrong_side=0,truncated=0,nee_invalid=0,steps=0,collisions=0,flux_in=np.zeros(3),flux_out=np.zeros(3),flux_leak=np.zeros(3))
    def tally(self,x,d,w):
        """Record exits of photons at boundary points leaving along d (outward), weight w (n,3); same port/bin mapping as the walker."""
        dom=self.dom;import r57_geometry as RG
        t,tri=dom.isect(x,d);ok=torch.isfinite(t);ok[ok.clone()]=(d[ok]*dom.n[tri[ok]]).sum(1)>0;i=ok.nonzero().squeeze(1)
        G=RG.G;port=dom.tri_port[tri[i]];ob=G.local_to_bin(G.to_local(d[i],dom.n[tri[i]],dom.t[tri[i]],dom.b[tri[i]]),8)
        idx=port*64+ob;we=w[i].T.double();self.m1.index_add_(1,idx,we);self.m2.index_add_(1,idx,we.square());self.m0.index_add_(0,idx,torch.ones_like(idx,dtype=torch.float64));return int(len(x)-len(i))

def first_collision(dom,i,a,n,gen):
    """Entry photons of cell (i, a) and their first-collision points: x0, d, chord t, c = 1 - exp(-sigma t), s, x1."""
    import r57_geometry as RG
    TR=RG.TR;cells=torch.full((n,),i*16+a,device='cuda',dtype=torch.long)
    x0,d,nrm,_,_=TR.cell_source(dom,cells,1,16,gen);x0=x0-dom.eps*nrm
    t,_=dom.isect(x0,d);ok=torch.isfinite(t);c=torch.zeros_like(t);c[ok]=-torch.expm1(-dom.sigma_t*t[ok])
    u=torch.rand(n,device='cuda',generator=gen);s=torch.zeros_like(t);s[ok]=-torch.log1p(-u[ok]*c[ok])/dom.sigma_t
    s=torch.minimum(s,t.nan_to_num(posinf=0.)*(1-1e-7))
    return x0,d,t,c,s,x0+s[:,None]*d,ok

def iso(n,gen):
    import r57_geometry as RG
    z=torch.zeros(n,3,device='cuda');z[:,2]=1.;return RG.TR.phase_sample(z,0.,gen)

def decomposed(dom,i,a,N,sd,mode='iso',chunk=2_000_000):
    """Forced-first-collision analog MC (mode 'iso'), or the forward negative control (mode 'fwd'). Returns single / multi / total
    first and second moments (per photon, so mean = m1/N, var of mean = (m2/N - mean^2)/(N-1)), E[c] and its SE, depth statistics."""
    import r57_geometry as RG
    TR=RG.TR;gen=torch.Generator(device='cuda').manual_seed(sd);single=Acc(dom);multi=Acc(dom);sc=sc2=0.;dep=0.;leak=0;t0=time.perf_counter()
    alb=dom.albedo[None,:]
    for start in range(0,N,chunk):
        n=min(chunk,N-start);x0,d,t,c,s,x1,ok=first_collision(dom,i,a,n,gen);leak+=int((~ok).sum())
        sc+=float(c.double().sum());sc2+=float(c.double().square().sum());dep+=float((c*s).double().sum())
        w=(c[:,None]*alb)/N;d1=iso(n,gen) if mode=='iso' else d
        t2,_=dom.isect(x1,d1);s2=-torch.log1p(-torch.rand(n,device='cuda',generator=gen))/dom.sigma_t
        ex=(s2>=t2)&torch.isfinite(t2);co=(s2<t2)&torch.isfinite(t2);leak+=int((~torch.isfinite(t2)).sum())
        if ex.any():single.tally(x1[ex],d1[ex],w[ex])
        if co.any():
            j=co.nonzero().squeeze(1);x2=x1[j]+s2[j,None]*d1[j];w2=w[j]*alb;d2=iso(len(j),gen) if mode=='iso' else d1[j]
            TR.walk(multi,x2,d2,torch.zeros_like(x2),w2,torch.zeros(len(j),dtype=torch.long,device='cuda'),gen,continue_air=False)
    Ec=sc/N;se=math.sqrt(max(sc2/N-Ec*Ec,0)/(N-1))
    return dict(single=(single.m1,single.m2),multi=(multi.m1,multi.m2),hits=dict(single=single.m0,multi=multi.m0),N=N,Ec=Ec,Ec_SE=se,mean_depth_m=dep/max(sc,1e-300),leaks=leak+multi.stats['leaks'],
                wrong_side=multi.stats['wrong_side'],truncated=multi.stats['truncated'],seconds=time.perf_counter()-t0,seed=sd,mode=mode)

def single_quadrature(dom,i,a,N,sd,chunk=2_000_000):
    """Expected-value single-scatter term (the deployable E1 composition): weight c * albedo * exp(-sigma t2) at the exit of the isotropic ray."""
    gen=torch.Generator(device='cuda').manual_seed(sd);acc=Acc(dom);alb=dom.albedo[None,:]
    for start in range(0,N,chunk):
        n=min(chunk,N-start);x0,d,t,c,s,x1,ok=first_collision(dom,i,a,n,gen);d1=iso(n,gen);t2,_=dom.isect(x1,d1);f=torch.isfinite(t2)
        w=(c[:,None]*alb*torch.exp(-dom.sigma_t*t2.nan_to_num(posinf=1e9))[:,None])/N;acc.tally(x1[f],d1[f],w[f])
    return acc.m1,acc.m2,acc.m0

def coarse(y,parent,bm):
    """(3, P*64) -> (240, 16, 3)."""
    z=y.T.reshape(-1,64,3).double().cpu();f=torch.zeros(z.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,z)
    return torch.zeros(240,16,3,dtype=torch.float64).index_add_(0,parent.cpu(),f)

def mv(m,N):
    """Per-photon tallies already carry 1/N: mean = m1, var(mean) = (N m2 - m1^2) / (N - 1)."""
    m1,m2=m;return m1,((N*m2-m1.square())/(N-1)).clamp(min=0)

def compare(ca,va,cb,vb,sel,N,min_hits):
    """Coarse (240,16,3) means and variances of two independent estimators: energy z per channel, coarse-cell mean z^2, relative L1."""
    e=ca.sum((0,1))-cb.sum((0,1));se=(va.sum((0,1))+vb.sum((0,1))).sqrt();z=(e/se.clamp(min=1e-300)).tolist()
    dm=ca-cb;dv=va+vb;mask=(dv>0)&(N*sel>=min_hits)   # cells chosen from an INDEPENDENT estimate 'sel': expected events N*mean >= min_hits (per-event weight <= 1),
                                                      # never from the tested estimator's own counts (that selection biases z^2 upward)
    return dict(energy_z=z,coarse_mean_z2=float((dm[mask].square()/dv[mask]).mean()) if mask.any() else float('nan'),cells=int(mask.sum()),
                coarse_rel_L1=float(dm.abs().sum()/cb.abs().sum().clamp(min=1e-300)),energy_a=ca.sum((0,1)).tolist(),energy_b=cb.sum((0,1)).tolist())

def stage1():
    RG=environment();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'stage1/physics.json'
    if out.exists():raise FileExistsError('immutable')
    import probes68 as P68
    sc=CFG['stage1'];N=sc['photons'] if 'R76_SMOKE_N' not in os.environ else int(os.environ['R76_SMOKE_N'])
    if 'R76_SMOKE_N' in os.environ:assert dev()
    bm=torch.as_tensor(RG.G.bin_parent(8,4));rows=[];doms={};t0=time.perf_counter();gi={'rest':0,'pt0':1,'te0':2}
    for pk in sc['packets']:
        g,slot,port=pk['geometry'],pk['slot'],pk['port']
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=(P68.domain(RG,g),torch.as_tensor(RG.KO.refined(*RG.mesh(g),4)[2]))
        dom,parent=doms[g];b=load(Path_(pk['bundle']));assert b.get('port',port)==port;K=b['S'].shape[0]
        C=lambda y:coarse(y,parent,bm)   # (3, P*64) -> (240,16,3)
        for a in sc['bins']:
            Rc=torch.stack([C(b['S'][k,a].double().reshape(-1,3).T) for k in range(K)]);rc,rv=Rc.mean(0),Rc.var(0,unbiased=True)/K
            Bref=b['B16'][:,a].double().sum((1,2));beta_ref=float(Bref.mean());beta_se=float(Bref.mean(1).std()/math.sqrt(K))
            r=decomposed(dom,port,a,N,seed('phys',gi[g],slot,a,0))
            tm,tv=mv((r['single'][0]+r['multi'][0],r['single'][1]+r['multi'][1]),N);sm,sv=mv(r['single'],N)   # one exit per photon: exclusive events
            q=single_quadrature(dom,port,a,N,seed('quad',gi[g],slot,a,0));qm,qv=mv(q[:2],N);H=lambda h:C(h[None].expand(3,-1))
            mh=sc['rules']['min_hits']
            bse=max(beta_se,math.sqrt(max(1-r['Ec'],0)*min(r['Ec'],1)/(K*sc['reference_photons'])))   # binomial floor: a replica SE of 0 when no ballistic exit was observed
            row=dict(geometry=g,slot=slot,port=port,context=pk['context'],bin=a,K_ref=K,
                     total_vs_reference=compare(C(tm),C(tv),rc,rv,rc,N,mh),single_quadrature_vs_mc=compare(C(qm),C(qv),C(sm),C(sv),C(qm),N,mh),
                     first_collision=dict(Ec=r['Ec'],Ec_SE=r['Ec_SE'],one_minus_beta_reference=1-beta_ref,beta_ref_SE=bse,beta_ref_replica_SE=beta_se,
                         z=(r['Ec']-(1-beta_ref))/math.sqrt(r['Ec_SE']**2+bse**2+1e-300),mean_depth_mm=1e3*r['mean_depth_m']),
                     single_fraction=(sm.sum(1)/tm.sum(1).clamp(min=1e-300)).tolist(),leaks=r['leaks'],wrong_side=r['wrong_side'],truncated=r['truncated'],seconds=r['seconds'])
            if a in sc['control_bins'] and pk['context'] in sc['control_contexts']:
                f=decomposed(dom,port,a,N,seed('phys',gi[g],slot,a,1),mode='fwd');fm,fv=mv((f['single'][0]+f['multi'][0],f['single'][1]+f['multi'][1]),N)
                row['forward_control_vs_reference']=compare(C(fm),C(fv),rc,rv,rc,N,mh)
            rows.append(row);print('S1',g,slot,a,[round(z,2) for z in row['total_vs_reference']['energy_z']],round(row['total_vs_reference']['coarse_mean_z2'],3),
                                   round(row['first_collision']['z'],2),row.get('forward_control_vs_reference',{}).get('coarse_mean_z2'),round(r['seconds'],1),flush=True)
    res=judge(rows);write_new(out,json.loads(json.dumps(dict(packets=rows,gate=res,photons=N,seconds=time.perf_counter()-t0,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)),default=float)))
    print('STAGE1',res['passed'],flush=True)
    if not res['passed']:raise SystemExit('STOP: first-interaction factorization not validated (protocol stage1)')

def Path_(s):
    from pathlib import Path
    return Path(os.path.expanduser(s))

def judge(rows):
    """Protocol stage1.rules (frozen)."""
    r=CFG['stage1']['rules'];fails=[]
    def ok(c):return max(abs(z) for z in c['energy_z'])<=r['energy_z_max'] and (c['cells']<r['min_cells'] or r['z2_lo']<=c['coarse_mean_z2']<=r['z2_hi'])
    for x in rows:
        k=f"{x['geometry']}_s{x['slot']}_a{x['bin']}"
        if not ok(x['total_vs_reference']):fails.append((k,'total'))
        if not ok(x['single_quadrature_vs_mc']):fails.append((k,'single quadrature'))
        if abs(x['first_collision']['z'])>r['energy_z_max']:fails.append((k,'first-collision probability'))
        if x['leaks']+x['wrong_side']>r['max_leaks'] or x['truncated']:fails.append((k,'leaks/truncation'))
    ctrl=[x for x in rows if 'forward_control_vs_reference' in x];rej=[x for x in ctrl if not ok(x['forward_control_vs_reference'])]
    power=len(ctrl)>0 and len(rej)==len(ctrl)
    if not power:fails.append(('forward control','not rejected in every control packet'))
    return dict(passed=not fails,failures=fails,packets=len(rows),controls=len(ctrl),controls_rejected=len(rej))

if __name__=='__main__':
    {'stage1':stage1}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
