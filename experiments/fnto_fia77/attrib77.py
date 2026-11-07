"""WL77 brief Stage 4 - bounded failure attribution, run ONLY if the Stage 3 fit gate fails (training sources; labelled diagnostics).
A  outgoing angular residual: native error of the learned multiple-scatter part (prediction - exact SS vs reference - exact SS) for F77 and
   the preserved WL76 F76, beside the Stage 1 projection floors (ring / FIA-AZ span).
B  spatial footprint: per incoming bin a, the coarse spatial centroid of the collided response relative to the source port (mm, tangent
   plane) - reference vs model, total and multiple-scatter part; the incoming-bin spread of that centroid (the incidence-dependent lateral
   shift) and the coarse incoming contrast of the multiple-scatter part.
C  optimisation: last-decile relative change of J, J at 2k / 6k / 10k / final, final gradient norms.
    python attrib77.py"""
import sys,os,json,torch
from common77 import CFG,ROOT,R76,INP68,write_new,environment
import family68 as FM,fia77 as FI,fia76 as FI76
import gate77 as GT

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    RG=environment();torch.set_grad_enabled(False);out=ROOT/'attrib/attribution.json'
    if out.exists():raise FileExistsError('immutable')
    import fit71,fit73,probes68 as P68
    gs=fit71.geometries(RG);g=gs['rest'];G=FI.attach_frames(FI.Geo(g,P68.domain(RG,'rest')));bm=FM.bin_map('cuda');H=FM.Hmat('cuda',torch.float64)
    models={}
    for n in CFG['runs']:x=load(ROOT/'runs'/f'{n}.pt');m=FI.FIA(CFG['candidate']['hidden']).cuda();m.load_state_dict(x['model']);m.eval();models[n]=(m,FI)
    for n in ('F76_7601','F76_7602'):x=load(R76/'runs'/f'{n}.pt');m=FI76.FIA(CFG['candidate']['hidden']).cuda();m.load_state_dict(x['model']);m.eval();models[n]=(m,FI76)
    xt=G.x.double();rows={}
    for s,p in fit73.populations()['S16R']:
        bb=load(GT.bundle(s));ref=bb['S'].double().mean(0).cuda();q=FI.quad_ports(G,[p],0,CFG['candidate']['quadrature_q']);cells=torch.arange(16,device='cuda')+16*p
        r,c,v=q['SS'];ss=torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
        A=lambda y,sp:FM.aggregate(y,sp,g['parent'],bm);rel=lambda a,b:float((a-b).norm()/b.norm().clamp(min=1e-30))
        d=xt-xt[p];n0=G.n[p].double();d=d-(d@n0)[:,None]*n0                                                # tangent-plane offsets (P,3)
        def centroid(y):   # per incoming bin, energy-weighted tangent offset (mm), (16,3)
            e=y.sum((2,3));return 1e3*(e@d)/e.sum(1,keepdim=True)
        def con(y):
            z=A(y,'coarse')/((1-g['beta'][p]).double()[:,None,None,None]);return torch.einsum('ab,bqoc->aqoc',H,z)
        rr=dict(ref_shift_spread_mm=float(centroid(ref).std(0).norm()),ref_ms_shift_spread_mm=float(centroid(ref-ss).std(0).norm()),ss_share=float(ss.sum()/ref.sum()))
        for n,(m,mod) in models.items():
            h=mod.predict(m,G,q['W'],q['SS'],cells,G.dom.albedo).double();ms=h-ss;msr=ref-ss
            rr[n]=dict(err_ms={sp:rel(A(ms,sp),A(msr,sp)) for sp in ('native','fine16','coarse')},err_total_native=rel(h,ref),
                centroid_err_mm=float((centroid(h)-centroid(ref)).norm(dim=1).mean()),shift_spread_mm=float(centroid(h).std(0).norm()),
                ms_shift_spread_mm=float(centroid(ms).std(0).norm()),con_ms=rel(con(ms),con(msr)),con_total=rel(con(h),con(ref)))
        rows[f'rest_s{s}']=rr;print('ATTR',s,json.dumps({k:(v if not isinstance(v,dict) else {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()}) for k,v in rr.items()},default=float)[:400],flush=True)
        del ref,ss;torch.cuda.empty_cache()
    opt={}
    for n in CFG['runs']:
        x=load(ROOT/'runs'/f'{n}.pt');h=x['history'];J={st:j for st,j in [(e['step'],e['J']) for e in h]}
        opt[n]=dict(J_initial=x['J_initial'],J_final=x['J_final'],last_decile_relative_change=x['last_decile_relative_change'],
            J_at={k:min(J.items(),key=lambda t:abs(t[0]-k))[1] for k in (2000,6000,10000,19999)},grad_norm_last=[e['grad_norm'] for e in h[-5:]])
    s1=json.loads((ROOT/'stage1/expressivity.json').read_text())['mean']
    write_new(out,json.loads(json.dumps(dict(per_source=rows,optimisation=opt,stage1_floors=s1,label='POST-FAILURE ATTRIBUTION (brief Stage 4), training sources, labelled diagnostics; no model change'),default=float)))
    print('ATTRIB_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
