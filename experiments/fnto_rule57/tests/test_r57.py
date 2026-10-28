"""WL57 focused implementation tests before primary supervision/training."""
import sys,os,time,traceback,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import r57_geometry as RG
from r57_model import Transport,parameter_count
from r57_data import dump,light_fields,aggregate

def main(out):
    results=[]
    def test(name,fn):
        try:r=fn();results.append(dict(name=name,passed=True,measured=r))
        except Exception:results.append(dict(name=name,passed=False,error=traceback.format_exc()))
        print(results[-1],flush=True)
    def splits():
        assert not set(RG.TRAIN)&set(RG.VAL) and not set(RG.TRAIN)&set(RG.TEST) and not set(RG.VAL)&set(RG.TEST)
        assert len(set(RG.POSES.values()))==len(RG.POSES)
        assert abs(parameter_count('D')/parameter_count('R')-1)<.05
        return dict(D_parameters=parameter_count('D'),R_parameters=parameter_count('R'),train=RG.TRAIN,val=RG.VAL,test=RG.TEST)
    test('disjoint_geometry_and_parameter_budgets',splits)
    g=RG.geometry('rest',q=8)
    def fields():
        x,meta=light_fields(g,64,571003,'test');y,_=light_fields(g,96,571001,'train')
        import hashlib
        hx={hashlib.sha256(v.numpy().tobytes()).digest() for v in x};hy={hashlib.sha256(v.numpy().tobytes()).digest() for v in y}
        assert not hx&hy
        assert float((x.sum((1,2))-1).abs().max())<1e-5
        for i in range(48,64,2):assert torch.allclose(x[i].sum(1),x[i+1].sum(1),rtol=1e-6,atol=1e-8) and not torch.equal(x[i],x[i+1])
        return dict(fields=64,train_fields=96,unit_flux=True,equal_irradiance_pairs=8)
    test('disjoint_lights_flux_and_directional_pairs',fields)
    def runtime_inputs():
        import r57_model
        src=inspect.getsource(r57_model)
        assert not any(s in src for s in ('torch.load','raw_oracle','targets/','Cycles','pose_id'))
        # Exact passive construction only uses current geometry; deliberately remove name/signature metadata.
        q=dict(g);q.pop('name');q.pop('signature');model=Transport('R').cuda();co=model.coefficients(q)
        assert float(model.column_sums(q,co).max())<=1+1e-6
        return dict(no_target_IO=True,metadata_not_model_input=True,max_column_sum=float(model.column_sums(q,co).max()))
    test('runtime_oracle_independence_and_passivity',runtime_inputs)
    def algebra():
        torch.manual_seed(5701);x=torch.rand(2,g['P'],16,3,device='cuda');x/=x.sum((1,2),keepdim=True);out={}
        for arm in ('D','R'):
            model=Transport(arm).cuda();st=model.state(g);co=model.coefficients(g,st)
            y=model.apply(g,x,co);zz=model.apply(g,torch.zeros_like(x),co)
            sup=model.apply(g,x[:1]+x[1:],co);rel=float((sup-y.sum(0,keepdim=True)).abs().sum()/sup.abs().sum())
            lazy=model(g,x,st);err=float((y-lazy).abs().sum()/y.abs().sum())
            assert float(zz.abs().max())==0 and float(y.min())>=0 and rel<1e-5 and err<1e-6
            assert float((y.sum((1,2))-x.sum((1,2))).max())<=1e-6
            out[arm]=dict(linearity=rel,cache_equivalence=err,max_colsum=float(model.column_sums(g,co).max()))
        return out
    test('zero_linearity_nonnegative_passive_and_fair_cache',algebra)
    def grads():
        model=Transport('R').cuda();x=torch.rand(1,g['P'],16,3,device='cuda');x/=x.sum((1,2),keepdim=True)
        y=model(g,x);loss=y.square().sum();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        return dict(finite_loss=float(loss),finite_gradients=True)
    test('differentiable_full_spatial_angular_operator',grads)
    def correspondence():
        m,v=RG.mesh('te0');fm,vc,par=RG.KO.refined(m,v,4)
        assert np.array_equal(par,g['parent'].cpu().numpy())
        R=RG.G.rot((1,2,3),37);w=RG.G.map_rigid(vc,R,(.03,-.02,.01));back=(w-np.array((.03,-.02,.01)))@R
        assert np.max(np.abs(back-vc))<1e-12
        assert not np.array_equal(vc,g['Vcur'])
        return dict(nested_material_cells=True,rigid_frame_inverse_max=float(np.max(np.abs(back-vc))),dent_changes_geometry=True)
    test('boundary_identity_rigid_frame_and_invalidation',correspondence)
    dump(dict(passed=sum(r['passed'] for r in results),total=len(results),tests=results),out)
    assert all(r['passed'] for r in results)

if __name__=='__main__':main(Path(sys.argv[1]));sys.stdout.flush();os._exit(0)
