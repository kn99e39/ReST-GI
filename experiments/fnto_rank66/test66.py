"""Focused WL66 tests: source/normalisation/correspondence, factorisation, bound algebra, estimator bias/uncertainty, replay."""
import sys,math,unittest,itertools
import numpy as np
import torch
from common66 import CFG,ROOT,ORIG,HOME,GEOMS,NBIN,environment
import estimators66 as E
C59,RG=environment()
import r57_action_oracle as AO,r57_fastwalk as FW
from tag_reference import tagged_action
from probes66 import one_hot,packet_gates,domain
from ports66 import select
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False

def rel(a,b):return float((a-b).norm()/b.norm())

class Source(unittest.TestCase):
    def test_one_hot_source_and_tag_closure(self):
        """Every sampled photon starts in the probe cell, on port i, travelling inward, from bin a; unit flux; tags close."""
        dom=domain(RG,'rest');i,a=2085,13;x=one_hot(dom.P,i,a);seen=[]
        orig=RG.TR.cell_source
        def spy(dom_,cells,n,Din,gen):
            out=orig(dom_,cells,n,Din,gen);seen.append([t.clone() for t in out]);return out
        RG.TR.cell_source=spy
        try:r=tagged_action(AO,FW,dom,x,200000,689999001)
        finally:RG.TR.cell_source=orig
        ok,det=packet_gates(r,x,200000);self.assertTrue(ok,det)
        pos,d,n,w,src=[torch.cat([s[k] for s in seen]) for k in range(5)]
        self.assertTrue(bool((src==i*16+a).all()))
        tris=dom.port_tris[i][dom.port_tris[i]>=0];V=dom.V[dom.F[tris]].reshape(-1,3);lo,hi=V.min(0).values,V.max(0).values
        self.assertTrue(bool(((pos>=lo-1e-6)&(pos<=hi+1e-6)).all()))
        t,b=dom.t[tris[0]],dom.b[tris[0]];nn=dom.n[tris[0]]
        self.assertTrue(bool((dom.n[tris]-nn).abs().max()<1e-6))  # flat rest port: one frame
        wl=RG.G.to_local(-d,nn[None],t[None],b[None]);self.assertTrue(bool((wl[:,2]>0).all()))
        self.assertTrue(bool((RG.G.local_to_bin(wl,4)==a).all()))
        self.assertEqual(r['expected_input'][0],[1.,1.,1.])
        self.assertLessEqual(max(r['gates_A_B'][k] for k in r['gates_A_B'] if not isinstance(r['gates_A_B'][k],bool)),1e-12)

class Normalisation(unittest.TestCase):
    def test_equal_projected_solid_angle_bins(self):
        g=torch.Generator(device='cuda').manual_seed(1);u1,u2=torch.rand(2,4_000_000,device='cuda',generator=g)
        r=torch.sqrt(u1);ph=2*math.pi*u2;w=torch.stack([r*torch.cos(ph),r*torch.sin(ph),torch.sqrt(1-u1)],1)  # cosine-weighted
        f=torch.bincount(RG.G.local_to_bin(w,4),minlength=16).double()/len(w);self.assertLess(float((f-1/16).abs().max()),5e-4)
    def test_aggregation_conserves_flux(self):
        from analyze66 import aggregate
        parent=torch.load(ORIG/'data/geometry/rest.pt',map_location='cpu',weights_only=False)['parent'];bm=torch.as_tensor(RG.G.bin_parent(8,4))
        y=torch.rand(2,3840,64,3,dtype=torch.float64)
        for sp in ('fine16','coarse'):self.assertLess(float((aggregate(y,sp,parent,bm).sum((1,2))-y.sum((1,2))).abs().max()),1e-9)

class Correspondence(unittest.TestCase):
    def test_ports_fixed_and_reproducible(self):
        import json
        pr=json.loads((ROOT/'ports.json').read_text())
        g={n:torch.load(ORIG/CFG['geometry_files'][n],map_location='cpu',weights_only=False) for n in GEOMS}
        sel=select({n:dict(rest_xyz=v['rest_xyz'].numpy(),node=v['node'].numpy(),beta=v['beta'].numpy()) for n,v in g.items()})
        self.assertEqual(sel,[p['port'] for p in pr['ports']])
        for p in pr['ports']:
            i=p['port'];self.assertTrue(torch.equal(g['rest']['rest_xyz'][i],g['pt0']['rest_xyz'][i]))
            self.assertEqual(int(g['rest']['parent'][i]),int(g['pt0']['parent'][i]))
            for n in GEOMS:self.assertEqual(p['geometries'][n]['d'],(1-g[n]['beta'][i]).tolist())

class Factorisation(unittest.TestCase):
    def test_beta_aware_factorisation_random_parameters(self):
        from runtime import Transport,full,apply
        from model66 import direct_scatter
        torch.manual_seed(66);m=Transport().cuda()
        for p in m.parameters():p.add_(torch.randn_like(p)*.3)
        g=C59.loaded_geometry('pt0');c=full(m,g);i=2085;d=1-g['beta'][i].double()
        x=torch.zeros(NBIN,g['P'],NBIN,3,device='cuda');x[torch.arange(NBIN),i,torch.arange(NBIN)]=1.
        tot=torch.cat([apply(m,c,x[k:k+4]) for k in range(0,16,4)])
        ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,cc].reshape(NBIN,-1).T).T.reshape(NBIN,g['P'],64) for cc in range(3)],-1)
        self.assertLess(rel((tot-ball).double(),direct_scatter(g,c['co'],i,d)),1e-5)
        A=c['co']['A'].double()
        for cc in range(3):
            At=A[:,:,cc].T;one=torch.ones(16,dtype=torch.float64,device=At.device);self.assertLess(float((At@torch.linalg.pinv(At)@one-one).norm()),1e-6)  # float32 softmax column sums
            Z=((tot-ball)[...,cc].double()/d[:,None,None]).reshape(16,-1);Pi=At@torch.linalg.pinv(At);self.assertLess(rel(Pi@Z,Z),1e-5)

class Bounds(unittest.TestCase):
    def test_tail_is_minimum_over_projectors(self):
        g=torch.Generator().manual_seed(3);X=torch.randn(16,300,dtype=torch.float64,generator=g);G=X@X.T;tl=E.tail(G,7)
        for _ in range(300):
            Q=torch.linalg.qr(torch.randn(16,7,dtype=torch.float64,generator=g))[0];self.assertGreaterEqual(float(torch.trace(G)-torch.trace(Q.T@G@Q))+1e-9,tl)
    def test_family_error_never_below_contrast_bound(self):
        """Any column-stochastic A (any G) has squared Z error >= tail_7(H Z Z^T H); pooled sharing >= sum of per-port bounds."""
        g=torch.Generator().manual_seed(4);Zs=[torch.randn(16,200,dtype=torch.float64,generator=g)+3 for _ in range(3)];Hm=E.H()
        per=sum(E.tail(Hm@Z@Z.T@Hm,7) for Z in Zs);pool=E.tail(sum(Hm@Z@Z.T@Hm for Z in Zs),7);self.assertGreaterEqual(pool+1e-9,per)
        for _ in range(200):
            A=torch.softmax(torch.randn(8,16,dtype=torch.float64,generator=g)*3,0);At=A.T;Pi=At@torch.linalg.pinv(At)
            err=sum(float(((Z-Pi@Z)**2).sum()) for Z in Zs);self.assertGreaterEqual(err+1e-9,pool)
    def test_frozen_contrast_basis_is_architecture_exact(self):
        """float32 column-stochastic A: TA has rank 7 and the contrast residual equals the absolute range(A^T) residual."""
        g=torch.Generator().manual_seed(8)
        for _ in range(20):
            A=torch.softmax(torch.randn(8,16,3,generator=g)*2,0).float();fb=E.frozen_bases(A,torch.softmax(torch.randn(8,64,3,generator=g),1).float())
            Z=torch.randn(16,300,dtype=torch.float64,generator=g)+2;Hm=E.H();G=Z@Z.T;GC=Hm@G@Hm
            for b in fb:
                self.assertEqual(b['TA'].shape[1],7)
                fa=float(torch.trace(GC)-torch.trace(b['TA'].T@GC@b['TA']));fz=float(torch.trace(G)-torch.trace(b['Pi']@G@b['Pi'].T))
                self.assertLess(abs(fa-fz)/fz,1e-6)
    def test_lower_low_biased_upper_high_biased_and_unbiased_fixed_projector(self):
        g=torch.Generator().manual_seed(5);K=8;Q=400;U=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64,generator=g))[0]
        sv=torch.tensor([10,5,3,2,1.5,1,.8,.5,.3,.2,.1,.05,.03,.02,.01,0.],dtype=torch.float64)
        Zs=U@torch.diag(sv)@torch.linalg.qr(torch.randn(Q,16,dtype=torch.float64,generator=g))[0].T;Hm=E.H();Cs=Hm@Zs
        true=E.tail(Cs@Cs.T,7);P=torch.linalg.qr(torch.randn(16,7,dtype=torch.float64,generator=g))[0];truefix=float(torch.trace(Cs@Cs.T)-torch.trace(P.T@Cs@Cs.T@P))
        L=[];Up=[];F=[];S=[];truthS=float((Cs**2).sum())
        for trial in range(150):
            Zk=Zs[None]+.15*torch.randn(K,16,Q,dtype=torch.float64,generator=g);C=torch.einsum('ab,kbq->kaq',Hm,Zk);M=torch.einsum('kaq,lbq->klab',C,C)
            G=E.ust(M,range(K));L.append(E.tail(G,7));Up.append(E.upper(M,list(range(K)),7));F.append(float(torch.trace(G)-torch.trace(P.T@G@P)));S.append(float(torch.trace(G)))
        L,Up,F,S=map(np.array,(L,Up,F,S))
        self.assertLess(L.mean(),true);self.assertGreater(Up.mean(),true)
        self.assertLess(abs(F.mean()-truefix),4*F.std()/np.sqrt(len(F)));self.assertLess(abs(S.mean()-truthS),4*S.std()/np.sqrt(len(S)))

class Uncertainty(unittest.TestCase):
    def test_packet_moment_variance_and_jackknife_coverage(self):
        rng=np.random.default_rng(6);n,N=20,2000;p=rng.dirichlet(np.ones(n));val=rng.uniform(.5,2,n)
        firsts=[];seconds=[]
        for _ in range(400):
            e=rng.choice(n,N,p=p);x=np.zeros(n);s=np.zeros(n);np.add.at(x,e,val[e]/N);np.add.at(s,e,(val[e]/N)**2);firsts.append(x);seconds.append(s)
        firsts=np.array(firsts);seconds=np.array(seconds);ratio=((N*seconds-firsts**2)/(N-1)).mean(0)/firsts.var(0,ddof=1)
        self.assertLess(abs(np.median(ratio)-1),.1);self.assertLess(np.abs(ratio-1).max(),.4)
        g=torch.Generator().manual_seed(7);truth=torch.randn(16,50,dtype=torch.float64,generator=g);P=torch.linalg.qr(torch.randn(16,7,dtype=torch.float64,generator=g))[0];Hm=E.H()
        tf=float(((Hm@truth-P@P.T@Hm@truth)**2).sum());cover=0
        for _ in range(200):
            Zk=truth[None]+.3*torch.randn(8,16,50,dtype=torch.float64,generator=g);C=torch.einsum('ab,kbq->kaq',Hm,Zk);M=torch.einsum('kaq,lbq->klab',C,C)
            j=E.jackknife(lambda S:{'F':float(torch.trace(E.ust(M,S))-torch.trace(P.T@E.ust(M,S)@P))},8,2.364624)['F'];cover+=j['lo']<=tf<=j['hi']
        self.assertGreaterEqual(cover/200,.88)

class Replay(unittest.TestCase):
    def test_frozen_checkpoint_reproducibility(self):
        from runtime import Transport,full,apply
        x=torch.load(ORIG/'data/lights.pt',map_location='cpu')['test'][[62]];g=C59.loaded_geometry('pt0')
        for seed in (5901,5902):
            m=Transport().cuda();m.load_state_dict(torch.load(ORIG/'checkpoints'/f'P_{seed}.pt',map_location='cpu')['model']);m.eval()
            y=apply(m,full(m,g),x.cuda()).cpu();st=torch.load(ORIG/'out'/f'actions_{seed}.pt',map_location='cpu')['predictions']['pt0']['F'][[62]]
            self.assertLess(rel(y.double(),st.double()),2e-5)

if __name__=='__main__':
    r=unittest.main(exit=False,verbosity=2).result;sys.stdout.flush();import os;os._exit(0 if r.wasSuccessful() else 1)
