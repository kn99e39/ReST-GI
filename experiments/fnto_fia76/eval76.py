"""WL76 evaluation (eval68 estimators, eval73 groups / qualification unchanged).
stage3: training-source fidelity (S16-R held-out replicas 4-7) + WL73 development groups for FIA (F76_*), the preserved learned baselines
        AR_5901/5902 (rank 8) and A75_5901/5902 (rank 16), labelled G0 / G2 source-specific oracles; FIA runtime physics (rest, pt0);
        measured cost (same process); the predeclared fit and computational gates (protocol stage3.gate).
stage4: (only after the gates) the WL76 confirmation population (rest slots + te1 footprint, DEF = te1 - rest); the same six finals
        + FIA checkpoints; labelled oracles; mechanism data (exact single-scatter share, learned-part error); the predeclared decision.
    python eval76.py stage3 | stage4"""
import sys,os,json,time
import torch
from common76 import CFG,ROOT,R73,R74,R75,INP68,write_new,environment,dev
import family68 as FM
import eval68 as E
import eval72 as E72
import eval73 as E73
import eval75 as E75   # patches E73.candidate for A75_* (rank-16 W+UV, ~/rest_rank75/runs)
import fia76 as FI

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
AR=['AR_5901','AR_5902'];A75=['A75_5901','A75_5902'];F76=list(CFG.get('runs',{}));CK=CFG.get('optimizer',{}).get('checkpoints',[])
W=lambda res,n,g,m:max(c['hi'] for c in res[n]['groups'][g]['pooled'][m])
Lo=lambda res,n,g,m:max(c['lo'] for c in res[n]['groups'][g]['pooled'][m])

class FIAPred:
    """FIA predictions h(g, port) -> (16, P, 64, 3); Geo and per-port quadrature cached per geometry (deterministic per-port seeds)."""
    def __init__(self,path,G,RG):
        x=load(path);self.model=FI.FIA(CFG['candidate']['hidden']).cuda();self.model.load_state_dict(x['model']);self.model.eval();self.G=G;self.RG=RG;self.geo={};self.q={}
    def geo_(self,g):
        if g not in self.geo:
            import probes68 as P68
            self.geo[g]=FI.Geo(self.G[g],P68.domain(self.RG,g))
        return self.geo[g]
    def __call__(self,g,port):
        Gg=self.geo_(g);key=(g,port)
        if key not in self.q:self.q[key]=FI.quad_ports(Gg,[port],CFG['geometry_index'][g],CFG['candidate']['quadrature_q'])
        Q=self.q[key];cells=torch.arange(16,device='cuda')+16*port
        with torch.no_grad():return FI.predict(self.model,Gg,Q['W'],Q['SS'],cells,Gg.dom.albedo)
    def ss(self,g,port):
        """The exact single-scatter part alone (mechanism analysis)."""
        Gg=self.geo_(g);Q=self.q[(g,port)];cells=torch.arange(16,device='cuda')+16*port;r,c,v=Q['SS'];out=torch.zeros(16,Gg.P*64,3,device='cuda')
        sel=(r>=cells[0])&(r<=cells[-1]);out=out.index_put((r[sel]-cells[0],c[sel]),v[sel].float()[:,None]*Gg.dom.albedo[None],accumulate=True);return out.view(16,Gg.P,64,3)

def baseline_h(name):
    m,cols,kind=E73.candidate(name)
    def h(G,g,port):
        T,A,O=cols(G[g],[port]);return FM.predict(T,(1-G[g]['beta'][[port]]).float(),A,O,'native')[0]
    return m,h

def score(hfun,G,P,R,D,qual,dq,GR,defmap,subs=None):
    """Same estimators / grouping / qualification as eval73.evaluate; hfun(g, port) -> native prediction."""
    row=dict(groups={},per_matrix={},per_matrix_r123={},deformation={});hs={};cg={}
    with torch.no_grad():
        for (g,s),r in R.items():
            h=hfun(g,P[s]);hs[(g,s)]=h;Sg,Bg=r.gpu();cg[(g,s)]=E.cross(r,Sg,Bg,h);del Sg,Bg
            row['per_matrix'][f'{g}_s{s}']=E.pooled([r],[cg[(g,s)]],E.STD)
            if subs is not None:sb=subs[(g,s)];Sg,Bg=sb.gpu();row['per_matrix_r123'][f'{g}_s{s}']=E.pooled([sb],[E.cross(sb,Sg,Bg,h)],E.STD);del Sg,Bg
        for gn,keys in GR.items():
            keys=[k for k in keys if k in R and qual[k]]
            if len(keys)<4:row['groups'][gn]=dict(unmeasured=True,matrices=len(keys));continue
            p=E.pooled([R[k] for k in keys],[cg[k] for k in keys],E.STD)
            row['groups'][gn]=dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys],corr_slope=E.corr_slope([R[k] for k in keys],[cg[k] for k in keys]))
        for dn,pose in defmap.items():
            keys=[k for k in D if k[0]==dn and dq[k] and (pose,k[1]) in hs and ('rest',k[1]) in hs]
            if len(keys)<4:row['deformation'][dn]=dict(unmeasured=True,matrices=len(keys));continue
            dcg=[]
            for k in keys:x=D[k];Sg,Bg=x.gpu();dcg.append(E.cross(x,Sg,Bg,hs[(pose,k[1])]-hs[('rest',k[1])]));del Sg,Bg
            p=E.pooled([D[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D[k] for k in keys],dcg)
            row['deformation'][dn]=dict(pooled=p,corr_slope=cs,Q=E73.Qdef(p,cs),matrices=[f's{k[1]}' for k in keys])
    return row,hs

def costs(G,lights,f76):
    """Same-process cost of the three models on the rest geometry: rank-8 / rank-16 coefficients + apply; FIA full runtime build + apply."""
    def med(f,n=5):
        f();ts=[]
        for _ in range(n):torch.cuda.synchronize();t=time.perf_counter();f();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
        return sorted(ts)[n//2]
    g=G['rest'];out={}
    with torch.no_grad():
        for n in (AR[0],A75[0]):
            m,_=baseline_h(n);torch.cuda.reset_peak_memory_stats();co=m.coefficients(g)
            out[n]=dict(rebuild_s=med(lambda:m.coefficients(g)),apply64_s=med(lambda:[m.apply(g,lights[i:i+4],co) for i in range(0,64,4)]),apply1_s=med(lambda:m.apply(g,lights[:1],co)),
                state_floats=sum(v.numel() for k,v in co.items() if k in ('W','U','V')),parameters=sum(p.numel() for p in m.parameters()),peak_GPU=torch.cuda.max_memory_allocated());del m,co
        import probes68 as P68
        x=load(ROOT/'runs'/f'{f76}.pt');mdl=FI.FIA(CFG['candidate']['hidden']).cuda();mdl.load_state_dict(x['model']);mdl.eval()
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();Gg=FI.Geo(g,P68.domain(environment(),'rest'));geo_s=time.perf_counter()-t
        rt=FI.Runtime(mdl,Gg,CFG['geometry_index']['rest'],CFG['candidate']['quadrature_q']);alb=Gg.dom.albedo
        out[f76]=dict(geo_s=geo_s,build=rt.seconds,bytes=rt.bytes,state_bytes=sum(rt.bytes.values()),apply64_s=med(lambda:[rt.apply(g,lights[i:i+4],alb) for i in range(0,64,4)]),
            apply1_s=med(lambda:rt.apply(g,lights[:1],alb)),parameters=sum(p.numel() for p in mdl.parameters()),peak_GPU=torch.cuda.max_memory_allocated())
    return out

def fia_physics(name,G,RG,lights,geoms):
    import probes68 as P68
    x=load(ROOT/'runs'/f'{name}.pt');mdl=FI.FIA(CFG['candidate']['hidden']).cuda();mdl.load_state_dict(x['model']);mdl.eval();out={}
    with torch.no_grad():
        for g in geoms:
            Gg=FI.Geo(G[g],P68.domain(RG,g));rt=FI.Runtime(mdl,Gg,CFG['geometry_index'][g],CFG['candidate']['quadrature_q']);out[g]=FI.physics(rt,G[g],lights[:2].cuda(),Gg.dom.albedo)
            rt2=FI.Runtime(mdl,Gg,CFG['geometry_index'][g],CFG['candidate']['quadrature_q']);y1=rt.apply(G[g],lights[:2].cuda(),Gg.dom.albedo);y2=rt2.apply(G[g],lights[:2].cuda(),Gg.dom.albedo)
            out[g]['rebuild_identity']=float((y1-y2).abs().max());out[g]['passed']=out[g]['passed'] and out[g]['rebuild_identity']==0.;del rt,rt2;torch.cuda.empty_cache()
    return out

# ------------------------------------------------------------------ Stage 3
def gate3(res,cost):
    gc=CFG['stage3']['gate'];per={}
    for n in F76:
        fit={f'{g}:{m}':dict(new=W(res,n,g,m),AR_max=max(W(res,a,g,m) for a in AR)) for g in gc['groups'] for m in gc['metrics']}
        for v in fit.values():v['passed']=v['new']<=gc['factor']*v['AR_max']
        valid=all(v['passed'] for v in res[n]['physics'].values())
        per[n]=dict(fit=fit,valid=valid,passed=valid and all(v['passed'] for v in fit.values()))
    c=cost[F76[0]];comp=dict(state_bytes=c['state_bytes'],state_max=gc['state_bytes_max'],apply64_ratio_to_A75=c['apply64_s']/cost[A75[0]]['apply64_s'],apply_ratio_max=gc['apply64_ratio_max'])
    comp['passed']=comp['state_bytes']<=comp['state_max'] and comp['apply64_ratio_to_A75']<=comp['apply_ratio_max']
    valid=all(per[n]['valid'] for n in F76);fit=all(per[n]['passed'] for n in F76)
    cls='PHYSICAL VALIDITY FAILURE' if not valid else 'INSUFFICIENT FIT' if not fit else 'COMPUTATIONAL INFEASIBILITY' if not comp['passed'] else 'GATES PASSED'
    return dict(per_candidate=per,computational=comp,passed=cls=='GATES PASSED',classification=cls)

def stage3():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage3/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=E73.geoms(RG);P=E73.slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=E73.load_refs(G,P);GR=E73.groups()
    pc=E73.CFG['precision'];prec={k:E72.sigSE(r) for k,r in R.items()};dprec={k:E72.sigSE(r) for k,r in D.items()}
    qual={k:min(v)>=pc['abs_min'] for k,v in prec.items()};dq={k:min(v)>=pc['deformation_min'] for k,v in dprec.items()};res={}
    print('REFS',round(time.perf_counter()-t0),flush=True)
    for n in AR+A75:res[n]=E73.evaluate(n,G,P,R,D,qual,dq,None,GR,full=False);torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    defmap={'DEF71':'cf2','CONFDEF72':'te2','DEF73':E73.POSE}
    for n in F76+([] if dev() else [f'{f}@{c}' for f in F76 for c in CK]):
        pr=FIAPred(ROOT/'runs'/f'{n}.pt',G,RG);res[n],_=score(pr,G,P,R,D,qual,dq,GR,defmap);del pr;torch.cuda.empty_cache()
        if '@' not in n:
            res[n]['physics']=fia_physics(n,G,RG,lights,('rest','pt0'));x=load(ROOT/'runs'/f'{n}.pt')
            res[n]['run']={k:x.get(k) for k in ('J_initial','J_final','converged','last_decile_relative_change','seconds','prep_seconds','peak_GPU_allocated','trainable','concurrent')}
            res[n]['run']['history']=[(h['step'],h['J']) for h in x['history'] if h['step']%500==0]
        print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    orc=E75.oracle_train(G,P,R,GR,['G0_r0','G2_r0']);cost=costs(G,lights.cuda(),F76[0]);gate=gate3(res,cost)
    write_new(out,json.loads(json.dumps(dict(results=res,oracle=orc,cost=cost,gate=gate,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    write_new(ROOT/'stage3/gate.json',json.loads(json.dumps(gate,default=float)));print('STAGE3_DONE',gate['classification'],round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ Stage 4
S4=CFG.get('stage4',{});POSE=S4.get('pose')
def s4slots(pred):return [int(k) for k,v in S4.get('slots',{}).items() if S4['ports'].get(k) is not None and pred(v)]
def groups4():
    r=lambda sl:[('rest',s) for s in sl];B=('TBE','TBC','SM','SR')
    G={'BND_FAR':r(s4slots(lambda v:v['context'] in B and v['stratum']=='far')),'BND_NEAR':r(s4slots(lambda v:v['context'] in B and v['stratum']=='near')),
       'INT':r(s4slots(lambda v:v['context']=='INT')),'FP_rest':r(s4slots(lambda v:v['context']=='FP')),f'FP_{POSE}':[(POSE,s) for s in s4slots(lambda v:v['context']=='FP')],
       'UNS':r(s4slots(lambda v:v['context']=='SV')),'BND_HI':r(s4slots(lambda v:v['context'] in B and v['subcell']=='hi')),'BND_LO':r(s4slots(lambda v:v['context'] in B and v['subcell']=='lo')),
       'TOPBOT':r(s4slots(lambda v:v['context'] in ('TBE','TBC'))),'SIDE':r(s4slots(lambda v:v['context'] in ('SM','SR'))),'ALL_rest':r(s4slots(lambda v:True))}
    return G

def refs4(RG,G,P):
    bm=FM.bin_map('cuda');R={};fp=s4slots(lambda v:v['context']=='FP')
    for g,slots in (('rest',s4slots(lambda v:True)),(POSE,fp)):
        for s in slots:
            f=ROOT/'refs/bundles'/f'{g}_s{s}.pt'
            if dev() and not f.exists():continue
            b=load(f);assert b['port']==P[s] and b['K']==4
            R[(g,s)]=E.Ref((g,s),b['S'],b['B16'],(1-G[g]['beta'][P[s]]).float(),G[g]['parent'],bm,E.ball16(G[g],P[s],bm))
    D={}
    for s in fp:
        if (POSE,s) in R and ('rest',s) in R:a,b=R[(POSE,s)],R[('rest',s)];D[('DEF',s)]=E.Ref(('DEF',s),a.S-b.S,a.B16-b.B16,a.d,a.parent,bm,a.ball16-b.ball16)
    return R,D

def decide4(res):
    dc=S4['decision'];f,g_=dc['improve_factor'],dc['regression_factor']
    meas={gn:all(not res[n]['groups'][gn].get('unmeasured') for n in AR+A75+F76) for gn in dc['primary_groups']}
    if not all(meas.values()):return dict(measured=meas,classes=['INCONCLUSIVE (primary group unmeasured)'])
    mean=lambda ns,gn,m:sum(W(res,n,gn,m) for n in ns)/len(ns)
    strong={gn:(AR if mean(AR,gn,'con_coarse')<=mean(A75,gn,'con_coarse') else A75) for gn in res[F76[0]]['groups'] if all(not res[n]['groups'][gn].get('unmeasured') for n in AR+A75+F76)}
    def better(gn,m):
        b=strong[gn];r=dict(new=[W(res,n,gn,m) for n in F76],baseline=b,old=[W(res,n,gn,m) for n in b],old_lo=[Lo(res,n,gn,m) for n in b])
        v='YES' if max(r['new'])<=f*min(r['old']) and max(r['new'])<min(r['old_lo']) else 'NO' if min(r['new'])>f*max(r['old']) else 'OVERLAP';return v,r
    def worse(gn,m):
        best=min((AR,A75),key=lambda ns:mean(ns,gn,m));r=dict(new=[W(res,n,gn,m) for n in F76],baseline=best,old=[W(res,n,gn,m) for n in best])
        v='YES' if min(r['new'])>=g_*max(r['old']) else 'NO' if max(r['new'])<g_*min(r['old']) else 'OVERLAP';return v,r
    far,rf=better('BND_FAR','con_coarse');near,rn=better('BND_NEAR','con_coarse')
    reg={f'{gn}:{m}':worse(gn,m) for gn,ms in dc['regression'].items() for m in ms if gn in strong}
    sec={f'BND_FAR:{m}':better('BND_FAR',m)[0] for m in dc['secondary_metrics']}
    valid=all(all(v['passed'] for v in res[n]['physics'].values()) for n in F76) and all(all(v['passed'] for v in res[n]['passivity'].values()) for n in AR+A75)
    regY=[k for k,(v,_) in reg.items() if v=='YES'];regO=[k for k,(v,_) in reg.items() if v=='OVERLAP']
    if not valid:c=['PHYSICAL VALIDITY FAILURE']
    elif far=='OVERLAP':c=['INCONCLUSIVE (seed / statistical overlap on BND_FAR contrast)']
    elif far=='YES':c=(['TRADE-OFF (far-boundary benefit with regression)'] if regY else ['TRANSFERABLE BENEFIT (independent far boundary)'])+(['possible regression (overlap)'] if regO and not regY else [])
    else:c=['NO INDEPENDENT BOUNDARY-TRANSFER BENEFIT']+(['LOCAL BENEFIT (BND_NEAR contrast)'] if near=='YES' else [])+(['with regression'] if regY else [])
    gates={n:{gn:res[n]['groups'][gn]['Q']['passed'] for gn in res[n]['groups'] if not res[n]['groups'][gn].get('unmeasured')} for n in AR+A75+F76}
    return dict(measured=meas,classes=c,far=far,near=near,secondary=sec,regression={k:v for k,(v,_) in reg.items()},valid=valid,stronger_baseline={k:v[0][:3] for k,v in strong.items()},
                WL59_gates=gates,values=dict(BND_FAR=rf,BND_NEAR=rn,regression={k:r for k,(_,r) in reg.items()}))

def stage4():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage4/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    gate=json.loads((ROOT/'stage3/gate.json').read_text())
    if not gate['passed']:assert dev() and os.environ.get('R76_DEV_FORCE_GATE')=='1','Stage 4 not authorised by the Stage 3 gates'
    t0=time.perf_counter();G=E73.geoms(RG);G[POSE]=RG.device(load(ROOT/'inputs'/f'geometry_{POSE}.pt'))
    P={int(k):v['port'] for k,v in json.loads((ROOT/'ports76.json').read_text())['ports'].items() if v['port'] is not None}
    R,D=refs4(RG,G,P);bm=FM.bin_map('cuda');GR={k:[x for x in v if x in R] for k,v in groups4().items()}
    if dev():GR['DEV']=[k for k in R if k[0]=='rest']
    subs={k:E.Ref(k,r.S[1:],r.B16[1:],r.d,r.parent,bm,r.ball16) for k,r in R.items()}
    pc=E73.CFG['precision'];prec={k:E72.sigSE(r) for k,r in R.items()};dprec={k:E72.sigSE(r) for k,r in D.items()}
    qual={k:min(v)>=pc['abs_min'] for k,v in prec.items()};dq={k:min(v)>=pc['deformation_min'] for k,v in dprec.items()}
    lights=load(INP68/'wl59/lights.pt')['test'];res={};defmap={'DEF':POSE};from r57_eval import physics
    for n in AR+A75:
        m,h=baseline_h(n);res[n],_=score(lambda g,p:h(G,g,p),G,P,R,D,qual,dq,GR,defmap,subs)
        res[n]['passivity']={POSE:physics(m,G[POSE],lights[:2].cuda(),m.coefficients(G[POSE]))};del m;torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    mech={}
    for n in F76+([] if dev() else [f'{f}@{c}' for f in F76 for c in CK]):
        pr=FIAPred(ROOT/'runs'/f'{n}.pt',G,RG);res[n],hs=score(pr,G,P,R,D,qual,dq,GR,defmap,subs if '@' not in n else None)
        if '@' not in n:
            res[n]['physics']=fia_physics(n,G,RG,lights,(POSE,))
            for k,r in R.items():   # mechanism: exact single-scatter share of the reference and the error of the learned multiple-scatter part
                ss=pr.ss(k[0],P[k[1]]);ref=r.S.double().mean(0).cuda();h=hs[k].double()
                cs=lambda y:FM.aggregate(y,'coarse',r.parent,bm)
                mech.setdefault(n,{})[f'{k[0]}_s{k[1]}']=dict(ss_share=float(ss.sum()/ref.sum()),err_coarse_rel=float((cs(h)-cs(ref)).norm()/cs(ref).norm()),
                    err_rel_to_ms=float((cs(h)-cs(ref)).norm()/cs(ref-ss.double()).norm()),ss_coarse_rel_err_vs_ms_ref=float((cs(h-ss.double())-cs(ref-ss.double())).norm()/cs(ref-ss.double()).norm()))
        del pr,hs;torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    keys=[k for k in R if qual[k]];orc={}
    for bid in ('G0_r0','G2_r0'):orc[bid]=E75.oracle4(bid,G,P,R,keys,subs,GR,qual);print('ORACLE',bid,round(time.perf_counter()-t0),flush=True)
    dec=decide4(res)
    write_new(out,json.loads(json.dumps(dict(results=res,oracle=orc,mechanism=mech,decision=dec,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs={f'{a}_s{b}':v for (a,b),v in prec.items()},deformation={f'{a}_s{b}':v for (a,b),v in dprec.items()}),
        seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('STAGE4_DONE',json.dumps(dec['classes']),round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':
    {'stage3':stage3,'stage4':stage4}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
