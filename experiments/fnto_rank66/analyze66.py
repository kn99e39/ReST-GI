"""WL66 analysis: physical probe matrices x frozen model outputs -> bounds, intervals, precision gate, predeclared decision."""
import sys,json,time,os
import numpy as np
from common66 import CFG,ROOT,ORIG,GEOMS,SLOTS,NBIN,sha,write_new,environment,ports,budget
import estimators66 as E

SPACES=('native','fine16','coarse');PRIMARY_CK=('WL59_P_5901','WL59_P_5902')

def aggregate(y,space,parent,bm):
    """y (n,P,64,3) -> space; flux sums (r57 aggregate semantics)."""
    import torch
    if space=='native':return y
    z=torch.zeros(y.shape[0],y.shape[1],16,3,dtype=y.dtype,device=y.device).index_add_(2,bm,y)
    if space=='fine16':return z
    return torch.zeros(y.shape[0],240,16,3,dtype=y.dtype,device=y.device).index_add_(1,parent,z)

def load_probe(name,slot,K,receipts):
    import torch
    out={k:[] for k in ('R','B','R2','B2')}
    for a in range(NBIN):
        row={k:[] for k in out}
        for k in range(K):
            p=ROOT/'probes'/f'{name}_s{slot}_a{a:02d}_k{k}.pt';assert sha(p)==receipts[str(p)],p
            r=torch.load(p,map_location='cpu',weights_only=False)
            row['R'].append(r['component_first']['Scat'][0]);row['B'].append(r['component_first']['B'][0])
            row['R2'].append(r['component_second']['Scat'][0]);row['B2'].append(r['component_second']['B'][0]);del r
        for k_ in out:out[k_].append(torch.stack(row[k_]))
    return {k:torch.stack(v,1) for k,v in out.items()}  # (K,16,P,64,3)

def decide(pool,t):
    th=CFG['decision']['thresholds'];per={}
    for c,ch in enumerate('RGB'):
        j=pool[('coarse',c)];sq=lambda x:float(np.sqrt(max(x,0.)))
        L=sq(j['L7_rel']['lo']);U=sq(j['U7_rel']['hi'])
        fa_lo={ck:sq(j[ck+'/F_A_rel']['lo']) for ck in PRIMARY_CK};fa_hi={ck:sq(j[ck+'/F_A_rel']['hi']) for ck in PRIMARY_CK}
        share={ck:j[ck+'/F_A']['hi']<=th['share_of_model_error_energy']*j[ck+'/E_Mc']['lo'] for ck in PRIMARY_CK}
        if L>=th['significant']:cat='A'
        elif U<th['small']:
            if all(fa_lo[ck]>=th['significant'] for ck in PRIMARY_CK):cat='B'
            elif all(share[ck] and fa_hi[ck]<th['significant'] for ck in PRIMARY_CK):cat='C'
            else:cat='nondecisive'
        else:cat='nondecisive'
        attr={ck:j[ck+'/PAOc']['hi']<=th['share_of_model_error_energy']*j[ck+'/E_Mc']['lo'] for ck in PRIMARY_CK}
        per[ch]=dict(category=cat,relaxed_pooled_rho_lower=L,relaxed_pooled_rho_upper=U,frozen_A_rho_lower=fa_lo,frozen_A_rho_upper=fa_hi,
                     frozen_A_share_rule=share,model_rho={ck:sq(j[ck+'/E_Mc_rel']['est']) for ck in PRIMARY_CK},
                     frozen_AO_rho={ck:sq(j[ck+'/PAOc_rel']['est']) for ck in PRIMARY_CK},frozen_AO_attribution_rule=attr)
    cats=[v['category'] for v in per.values()]
    overall='A' if 'A' in cats else 'B' if 'B' in cats else 'C' if all(c=='C' for c in cats) else 'D'
    return overall,per

def main():
    import torch
    C59,RG=environment();torch.set_grad_enabled(False);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    if (ROOT/'analysis.json').exists():raise FileExistsError('analysis is immutable')
    pr=ports();b=budget();K=b['replicas'];t=b['t_quantile'];N=b['packet_photons'];t0=time.perf_counter()
    comp=json.loads((ROOT/'probes_complete.json').read_text());assert comp['all_passed'] and comp['packets']==b['packets']
    receipts={r['path']:r['sha256'] for r in comp['records']}
    mc=json.loads((ROOT/'model_complete.json').read_text())
    for f,h in mc['files'].items():assert sha(ROOT/'model'/f)==h,f
    bm=torch.as_tensor(RG.G.bin_parent(8,4),device='cuda');cks=list(mc['checkpoints'])
    parent=torch.load(ORIG/CFG['geometry_files']['rest'],map_location='cpu',weights_only=False)['parent'].cuda()
    model={(ck,g):torch.load(ROOT/'model'/f'{ck}_{g}.pt',map_location='cpu',weights_only=False) for ck in cks for g in GEOMS}
    ball={g:torch.load(ROOT/'model'/f'ball_{g}.pt',map_location='cpu',weights_only=False)['ball'] for g in GEOMS}
    fb={sp:{ck:E.frozen_bases(model[(ck,GEOMS[0])]['A'],model[(ck,GEOMS[0])]['O'],None if sp=='native' else bm.cpu()) for ck in cks} for sp in SPACES}
    for ck in cks:
        for g in GEOMS:assert torch.equal(model[(ck,g)]['A'],model[(ck,GEOMS[0])]['A']) and torch.equal(model[(ck,g)]['O'],model[(ck,GEOMS[0])]['O'])
    summ={};cells={'native':3840,'fine16':3840,'coarse':240};nout={'native':64,'fine16':16,'coarse':16};invariants={}
    for name in GEOMS:
        for s in pr['ports']:
            ph=load_probe(name,s['slot'],K,receipts);d=torch.tensor(s['geometries'][name]['d'],dtype=torch.float64,device='cuda')
            # Physical invariants per probe: zero-collision fraction vs model beta; energy; locality.
            Rn=ph['R'].sum((2,3));Bn=ph['B'].sum((2,3))
            invariants[f'{name}/s{s["slot"]}']=dict(zero_collision_fraction_phys=Bn.mean(0)[:,0].tolist(),beta_model=s['geometries'][name]['beta'],
                collided_energy_RGB=Rn.mean(0).tolist(),total_energy_RGB=(Rn+Bn).mean(0).tolist(),
                source_cell_collided_share_RGB=(ph['R'][:,:,s['port']].sum(2)/ph['R'].sum((2,3)).clamp(min=1e-300)).mean(0).tolist())
            for sp in SPACES:
                X={k:aggregate(v.reshape(-1,3840,64,3).cuda(),sp,parent,bm).reshape(K,NBIN,cells[sp]*nout[sp],3) for k,v in ph.items()}
                zm={ck:aggregate(model[(ck,name)]['scatter'][s['slot']].double().cuda(),sp,parent,bm).reshape(NBIN,-1,3) for ck in cks}
                bh=aggregate(ball[name][s['slot']].double().cuda(),sp,parent,bm).reshape(NBIN,-1,3)
                for c in range(3):
                    R=X['R'][...,c];Z=R/d[None,:,None]
                    summ[(name,s['slot'],c,sp)]=E.summary(Z,d,X['B'][...,c],R,{ck:zm[ck][...,c]/d[:,None] for ck in cks},bh[...,c],cells[sp],nout[sp],
                        {ck:fb[sp][ck][c] for ck in cks},second=X['R2'][...,c],secondB=X['B2'][...,c],N=N)
                del X,zm,bh;torch.cuda.empty_cache()
            del ph;print('LOADED',name,s['slot'],round(time.perf_counter()-t0,1),flush=True)
    per={};pool={};pool_geom={};share_cost={}
    for sp in SPACES:
        for c in range(3):
            for name in GEOMS:
                for s in pr['ports']:
                    one=E.add([summ[(name,s['slot'],c,sp)]])
                    per[(name,s['slot'],c,sp)]=E.jackknife(lambda S:E.stats(one,S,fb[sp],c,pairs=sp=='coarse'),K,t)
            mats=[summ[(n,s['slot'],c,sp)] for n in GEOMS for s in pr['ports']];allp=E.add(mats)
            pool[(sp,c)]=E.jackknife(lambda S:E.stats(allp,S,fb[sp],c,pairs=sp=='coarse'),K,t)
            for n in GEOMS:
                gp=E.add([summ[(n,s['slot'],c,sp)] for s in pr['ports']])
                pool_geom[(sp,c,n)]=E.jackknife(lambda S:E.stats(gp,S,fb[sp],c,pairs=False),K,t)
            share_cost[(sp,c)]=dict(sum_per_port_L7=sum(per[(n,s['slot'],c,sp)]['L7']['est'] for n in GEOMS for s in pr['ports']),pooled_L7=pool[(sp,c)]['L7']['est'],
                                   sum_per_port_U7=sum(per[(n,s['slot'],c,sp)]['U7']['est'] for n in GEOMS for s in pr['ports']),pooled_U7=pool[(sp,c)]['U7']['est'])
        print('STATS',sp,round(time.perf_counter()-t0,1),flush=True)
    th=CFG['gates']['precision'];prec=dict(per_matrix={},pooled={})
    for name in GEOMS:
        for s in pr['ports']:
            for c in range(3):
                j=per[(name,s['slot'],c,'coarse')]['S_C'];prec['per_matrix'][f'{name}/s{s["slot"]}/{"RGB"[c]}']=j['est']/max(j['se'],1e-300)
    for c in range(3):j=pool[('coarse',c)]['S_C'];prec['pooled']['RGB'[c]]=j['se']/j['est']
    precision_ok=min(prec['per_matrix'].values())>=10 and max(prec['pooled'].values())<=.05
    if precision_ok:overall,per_channel=decide(pool,t)
    else:overall,per_channel='D',{'reason':'STOP_INSUFFICIENT_PRECISION'}
    sanity={f'{n}/s{s["slot"]}/{"RGB"[c]}/{sp}/{ck}':dict(A=summ[(n,s['slot'],c,sp)]['ck'][ck]['model_A_span_residual'],AO=summ[(n,s['slot'],c,sp)]['ck'][ck]['model_AO_span_residual'])
            for n in GEOMS for s in pr['ports'] for c in range(3) for sp in SPACES for ck in cks}
    basis_info={ck:[dict(one_in_range=fb['native'][ck][c]['one_in_range'],TA_rank=int(fb['native'][ck][c]['TA'].shape[1]),
        At_singular_values=torch.linalg.svdvals(fb['native'][ck][c]['At']).tolist(),HAt_singular_values=torch.linalg.svdvals(E.H()@fb['native'][ck][c]['At']).tolist()) for c in range(3)] for ck in cks}
    # Principal-angle cosines between frozen TA and the pooled physical top-7 contrast subspace (coarse, descriptive).
    angles={}
    for c in range(3):
        allp=E.add([summ[(n,s['slot'],c,'coarse')] for n in GEOMS for s in pr['ports']]);G=E.ust(allp['MC'],list(range(K)));V=torch.linalg.eigh(G)[1][:,-7:]
        for ck in cks:angles[f'{ck}/{"RGB"[c]}']=torch.linalg.svdvals(fb['coarse'][ck][c]['TA'].T@V).tolist()
    noise={f'{n}/s{s["slot"]}/{"RGB"[c]}/{sp}':dict(packet_rows_mean=summ[(n,s['slot'],c,sp)]['noise_packet_rows'].mean(0).tolist(),sample_rows=summ[(n,s['slot'],c,sp)]['noise_sample_rows'].tolist(),
        packet_C_mean=float(summ[(n,s['slot'],c,sp)]['noise_packet_C'].mean()),sample_C=float(summ[(n,s['slot'],c,sp)]['noise_sample_C']))
        for n in GEOMS for s in pr['ports'] for c in range(3) for sp in SPACES}
    key=lambda k:'/'.join(map(str,k))
    import resource
    write_new(ROOT/'analysis.json',dict(outcome=overall,per_channel=per_channel,precision=dict(values=prec,passed=precision_ok),t_quantile=t,replicas=K,
        per_matrix={key(k):v for k,v in per.items()},pooled={key(k):v for k,v in pool.items()},pooled_by_geometry={key(k):v for k,v in pool_geom.items()},
        sharing={key(k):v for k,v in share_cost.items()},model_span_sanity=sanity,frozen_basis=basis_info,principal_cosines_TA_vs_physical_top7=angles,
        noise=noise,invariants=invariants,checkpoints=cks,primary_checkpoints=PRIMARY_CK,seconds=time.perf_counter()-t0,
        peak_GPU_allocated=torch.cuda.max_memory_allocated(),CPU_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        inputs=dict(probes_complete=sha(ROOT/'probes_complete.json'),model_complete=sha(ROOT/'model_complete.json'),ports=sha(ROOT/'ports.json'),budget=sha(ROOT/'budget.json'))))
    print('ANALYSIS',overall,json.dumps(per_channel)[:2000],flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
