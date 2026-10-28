"""Bounded GPU preflight: real rigid operator, unique memory, atomic resume."""
import sys,os,json,uuid,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import r57_geometry as RG
from r57_model import Transport
from r57_data import dump,save,sha
from r57_memory import bytes_of,cache_record

def main(out):
    torch.set_num_threads(8);torch.manual_seed(5701);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    def test(name,fn):
        try:rows.append(dict(name=name,passed=True,measurement=fn()))
        except Exception:rows.append(dict(name=name,passed=False,error=traceback.format_exc()))
        print(rows[-1],flush=True)
    def memory():
        x=torch.zeros(1024,device='cuda');g=dict(node=x);st=dict(z=x,s=x[:10],d=x.view(256,4));co=dict(W=torch.ones(37,device='cuda'))
        m=cache_record(g,st,co);assert m['geometry_bytes']==4096 and m['compact_incremental_bytes']==0 and m['coefficient_incremental_bytes']==148 and m['full_total_bytes']==4244
        parameter=torch.nn.Parameter(torch.ones(13,device='cuda'));pm=cache_record(g,st,co,iter([parameter]))
        assert pm['parameters_bytes']==52 and pm['all_retained_bytes']==4296
        sparse=torch.sparse_coo_tensor(torch.tensor([[0,1],[1,2]],device='cuda'),torch.ones(2,device='cuda'),(3,3)).coalesce()
        assert bytes_of(sparse)==40
        return dict(aliased_D_storage_counted_once=m,parameter_iterator_counted_in_total=pm,sparse_indices_values_bytes=bytes_of(sparse))
    test('unique_storage_identity_alias_views_and_sparse',memory)
    def rigid():
        _,v=RG.mesh('rest');R=RG.G.rot((1,2,3),37);t=np.array([.03,-.02,.011]);world=RG.G.map_rigid(v,R,t)
        a=RG.geometry('rest',q=32);b=RG.geometry('rest',q=32,V_world=world,rigid_R=R,rigid_t=t)
        assert a['signature']==b['signature']
        x=torch.rand(3,a['P'],16,3,device='cuda');x/=x.sum((1,2),keepdim=True);res={}
        with torch.no_grad():
            for arm in ('D','R'):
                torch.manual_seed(5701);m=Transport(arm).cuda();ca=m.coefficients(a);cb=m.coefficients(b)
                reused=m.apply(a,x,ca);current=m.apply(b,x,cb);er=float((reused-current).abs().sum()/reused.abs().sum())
                assert er<=1e-6;res[arm]=dict(reused_vs_rebuilt_operator_action_relative_L1=er,max_column_sum=float(m.column_sums(b,cb).max()))
        assert RG.geometry('te0',ballistic=False)['signature']!=a['signature']
        return dict(controls=res,signature_rigid_invariant=True,dent_invalidates=True,fields=3,units='same material cell/bin flux',quadrature=32,geometry='actual world rigid transform pulled back through declared supplied rigid frame; current ray intersections recomputed')
    test('rigid_operator_reuse_vs_current_geometry_rebuild',rigid)
    def atomic():
        root=out.parent/('atomic_'+uuid.uuid4().hex[:8]);root.mkdir();p=root/'cache.pt'
        save(dict(complete=True,value=torch.arange(4)),p);h=sha(p)
        # An interrupted writer leaves only a PID-labelled partial; readers still see old complete cache.
        p.with_name(p.name+'.999.partial').write_bytes(b'incomplete intentionally')
        assert torch.load(p)['complete'] and sha(p)==h
        import r57_data as RD
        old=RD.ROOT;RD.ROOT=root
        try:
            receipt=root/'receipt.json';dump(dict(sha256=h),receipt);assert RD.cache_ready(p,receipt)
            p.write_bytes(b'broken completed filename');assert not RD.cache_ready(p,receipt) and not p.exists()
            assert list((root/'quarantined_partial').glob('*.reason.json'))
            orphan=root/'orphan.pt';save(dict(no_receipt=True),orphan);assert not RD.cache_ready(orphan)
        finally:RD.ROOT=old
        npg=np.random.default_rng(123);torch.manual_seed(123)
        state=dict(np=npg.bit_generator.state,cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        save(state,root/'rng.pt');a=npg.random(4);b=torch.rand(4);c=torch.rand(4,device='cuda');r=torch.load(root/'rng.pt')
        npg.bit_generator.state=r['np'];torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
        assert np.array_equal(a,npg.random(4)) and torch.equal(b,torch.rand(4)) and torch.equal(c,torch.rand(4,device='cuda'))
        return dict(partials_not_read_as_complete=True,checksum_corruption_and_missing_receipt_quarantined=True,complete_hash_unchanged=True,NumPy_Torch_CPU_CUDA_RNG_exact_resume=True)
    test('atomic_cache_and_RNG_resume',atomic)
    dump(dict(passed=sum(r['passed'] for r in rows),total=len(rows),tests=rows),out);assert all(r['passed'] for r in rows)
if __name__=='__main__':main(Path(sys.argv[1]));sys.stdout.flush();os._exit(0)
