import os,sys,json
import numpy as np
import torch
from common import *
from runtime import Transport, full, selective, apply

def relative(a,b):return float((a-b).abs().sum()/b.abs().sum().clamp(min=1e-30))

def main():
    torch.set_num_threads(8);torch.manual_seed(59101)
    torch.backends.cuda.matmul.allow_tf32=False
    m=Transport().cuda().eval()
    rest=geometry('rest');current=geometry(PRIMARY[0]);current_s=geometry(PRIMARY[0]);x=torch.rand(2,rest['P'],16,3,device='cuda');x/=x.sum((1,2),keepdim=True)
    a=full(m,rest);raw=a['raw'].clone();head=a['head'].clone();f=full(m,current)
    s,audit=selective(m,a,current_s);mask=audit.pop('masks')
    checks=dict(FS=relative(apply(m,s,x),apply(m,f,x)),
                independent_B_indices=bool(torch.equal(current['B'].indices(),current_s['B'].indices())),
                independent_B_values=relative(current['B'].values(),current_s['B'].values()),
                retained_raw_bitwise=bool(torch.equal(raw[~mask['edges']],s['raw'][~mask['edges']])),
                retained_head_bitwise=bool(torch.equal(head[~mask['nodes']],s['head'][~mask['nodes']])),
                raw_reconstruction=relative(s['raw'],f['raw']),head_reconstruction=relative(s['head'],f['head']))
    from r57_eval import physics
    checks['physics']=physics(m,current,x,f['co'])
    y=apply(m,s,x)
    checks['independent_source_output_difference']=relative(y[:1],y[1:])
    # Same-state no-op, no light-specific cache update.
    co_identity=id(s['co'])
    noop,_=selective(m,s,current)
    checks['noop']=relative(apply(m,noop,x),y)
    checks['noop_cache_identity']=id(noop['co'])==co_identity
    # Bounded 37-degree rigid material-frame pullback; S retains B and all coefficients.
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);a0=np.deg2rad(37.)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)*np.cos(a0)+(1-np.cos(a0))*np.outer(axis,axis)+np.sin(a0)*cross;t=np.array([.03,-.02,.011])
    mesh,V=RG.mesh('rest');rigid=geometry('rest',ballistic=False,V_world=V@R.T+t,rigid_R=R,rigid_t=t)
    rc=full(m,rest);before=apply(m,rc,x);r,ra=selective(m,rc,rigid)
    checks['rigid']=relative(apply(m,r,x),before);checks['rigid_all_retained']=ra['raw_retained_fraction']==1.
    # A stale checkpoint or correspondence must not be silently reused.
    rejected=False
    with torch.no_grad():next(m.parameters()).add_(.001)
    try:selective(m,r,rigid)
    except ValueError:rejected=True
    checks['weight_change_rejected']=rejected
    m=Transport().cuda().eval();c=full(m,rest);bad=dict(rest,graph=rest['graph'].roll(1,0))
    rejected=False
    try:selective(m,c,bad)
    except ValueError:rejected=True
    checks['correspondence_change_rejected']=rejected
    # Explicit third-party metric changes must update untouched-endpoint edges.
    alt=dict(rest);alt['pair']=rest['pair'].clone();alt['pair'][0,0]+=.001;alt['signature']='synthetic-metric-change'
    c=full(m,rest);ff=full(m,alt);ss,aa=selective(m,c,alt)
    checks['pair_dependency']=relative(apply(m,ss,x),apply(m,ff,x));checks['pair_edge_marked']=bool(aa['masks']['edges'][0])
    ok=checks['FS']<=1e-5 and checks['raw_reconstruction']<=1e-5 and checks['head_reconstruction']<=1e-5
    ok &= checks['retained_raw_bitwise'] and checks['retained_head_bitwise'] and checks['physics']['passed']
    ok &= checks['independent_B_indices'] and checks['independent_B_values']<=1e-6
    # Repeated CUDA index_add can differ in summation order with unchanged state.
    # Check exact cache reuse, and the SAME preregistered F/S output tolerance.
    ok &= checks['rigid']<=1e-6 and checks['rigid_all_retained'] and checks['noop_cache_identity'] and checks['noop']<=CFG['criteria']['FS_relative_L1']
    ok &= checks['weight_change_rejected'] and checks['correspondence_change_rejected'] and checks['pair_edge_marked'] and checks['pair_dependency']<=1e-5
    dump(dict(passed=bool(ok),checks=checks,audit=audit,commit=(ROOT/'repo_commit.txt').read_text().strip()),OUT/'preflight.json')
    assert ok,checks
    # This is a primary-physics gate only; no optional glossy test is run.
    dump(dict(proceed=True,scope='actual primary F/S/identity/physics controls'),OUT/'gate1.json')

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
