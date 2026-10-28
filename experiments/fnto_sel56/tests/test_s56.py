"""Focused WL56 tests; failures return nonzero and are saved with measured evidence."""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import s56 as S
import s56_fixtures as F
import numpy as np
import torch

def run(out):
    res=[]
    def test(name,fn):
        try:
            v=fn(); res.append(dict(name=name,passed=True,measured=v))
        except Exception: res.append(dict(name=name,passed=False,error=traceback.format_exc()))
        print(res[-1],flush=True)
    def ownership():
        assert F.dirty('rest')==[] and F.dirty('rigid')==[] and F.dirty('dent')==[1]
        a,b=F.scene('rest'),F.scene('dent')
        assert np.array_equal(a['comps'][0][3],b['comps'][0][3])
        assert not np.array_equal(a['comps'][1][3],b['comps'][1][3])
        assert np.array_equal(a['comps'][1][2]['F'],b['comps'][1][2]['F'])
        return dict(dirty_dent=F.dirty('dent'),dirty_rigid=F.dirty('rigid'),lower_unchanged=True)
    test('optical_domain_ownership_and_material_correspondence',ownership)
    def source():
        sd=F.scene('rest'); n,k,m,V=sd['comps'][2]
        fr=S.G.tri_frames(m['V'],V,m['F'],m['tri_port'],m['port_tangent'])
        assert np.all(fr['n'][:,2]<-.99)
        assert abs(fr['area'].sum()-.04)<1e-12
        assert np.linalg.norm(sd['comps'][1][3].max(0)-sd['comps'][1][3].min(0))>0
        return dict(area=fr['area'].sum(),normal_z=float(fr['n'][:,2].max()))
    test('physical_emitter_area_and_direction',source)
    # Small exact dense algebra control runs the unchanged global solver, including stationary receiver response.
    def fixedpoint():
        import p53_compose as CP
        m=S.G.voxel_boundary(np.ones((1,1,1),bool),(.01,)*3,sub=1)
        comps=[CP.Comp('a','lambert',m,m['V'],rho=.5),CP.Comp('b','lambert',m,m['V']+(0,0,.018),rho=.5)]
        ss=CP.SceneSet(comps,1,1); n=ss.Ptot
        C=torch.zeros(n,n,device='cuda'); C[:6,6:]=.08;C[6:,:6]=.08
        C=C.to_sparse(); x=torch.zeros(n,3,device='cuda');x[0]=1
        Fsol=CP.solve(ss,{},C,x); Ssol=CP.solve(ss,{},C,x)
        dense=torch.linalg.solve(torch.eye(n,device='cuda')-.5*C.to_dense(),x)
        err=float((Fsol['phi_out']-dense).abs().max())
        assert err<1e-6 and torch.equal(Fsol['phi_out'],Ssol['phi_out']) and float(Fsol['phi_out'][6:].sum())>0
        zero=CP.solve(ss,{},C,x*0);twice=CP.solve(ss,{},C,x*2)
        assert float(zero['phi_out'].abs().max())==0 and float((twice['phi_out']-2*Fsol['phi_out']).abs().max())<1e-6
        return dict(dense_max_error=err,stationary_indirect=float(Fsol['phi_out'][6:].sum()),zero=True,linearity=True)
    test('shared_F_S_global_fixedpoint_zero_and_superposition',fixedpoint)
    def mapping():
        for state in ('rest','dent','rigid'):
            m,V=F.intrinsic(state,1); fm,vc,par=S.KO.refined(m,V,4)
            ar=S.G.tri_frames(fm['V'],vc,fm['F'],fm['tri_port'],fm['port_tangent'])['area'].sum()
            orig=S.G.tri_frames(m['V'],V,m['F'],m['tri_port'],m['port_tangent'])['area'].sum()
            assert abs(ar/orig-1)<1e-12 and np.bincount(par).min()==16
        return dict(max_area_tolerance=1e-12,nested_cells_per_port=16)
    test('uniform_boundary_conservative_mapping_and_frames',mapping)
    def currentC():
        a=F.scene('rest');b=F.scene('rigid')
        s0,p,_=S.KP.build_ss(a,[1,1,1]);s1,_,_=S.KP.build_ss(b,[1,1,1])
        c0,e0,t0=S.CA.cast_adaptive(s0,np.full(s0.Ptot,2),5691)
        c1,e1,t1=S.CA.cast_adaptive(s1,np.full(s1.Ptot,2),5691)
        col=torch.sparse.sum(c1,dim=0).to_dense()+e1
        assert t1['back_hits']==0 and float((col-1).abs().max())<1e-6
        assert not torch.equal(c0._indices(),c1._indices())
        kb=S.KP.ballistic_KB(s1,1,p,F.FX.DIRS['otop'],n_per_port=128,seed=5692)
        assert kb._nnz()>0 and bool((kb.values()>=0).all())
        return dict(C_changed=True,closure=float((col-1).abs().max()),back_hits=t1['back_hits'],ballistic_nnz=kb._nnz())
    test('current_geometry_C_and_ballistic_ownership',currentC)
    def masking():
        # Run the actual renderer geometry -> GPU classifier twice at the same camera.
        import k55_common as CM
        import p53_compose as CP
        import c54_common as CC
        import json
        geo_path=S.HIST/'out/cycles_cuda/ccam_gap8.json'
        g=json.load(open(geo_path))['gapdown'];ss=CM.classify_sceneset(F.scene('rest'))
        rays=CC.blender_ortho_rays(g,CM.V1_OFF)
        a=CP.classify(ss,rays);b=CP.classify(ss,rays)
        assert torch.equal(a,b)
        return dict(identical=True,pixels=int((a>=0).sum()))
    test('deterministic_physical_comparison_masks',masking)
    S.dump(dict(tests=res,passed=sum(r['passed'] for r in res),total=len(res)),out)
    assert all(r['passed'] for r in res)

if __name__=='__main__':run(Path(sys.argv[1]));sys.stdout.flush();__import__('os')._exit(0)
