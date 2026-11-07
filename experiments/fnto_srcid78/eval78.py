"""WL78 evaluation (eval68 estimators, eval73 groups / qualification and the WL76 Stage 4 decision structure unchanged).
stage3: supervised fidelity on S16-R held-out replicas 4-7 (eval73.evaluate, the WL75 Stage 2 path) for the preserved A75 finals and the
        W+UV-Q finals (+ checkpoints, descriptive); passivity on rest/pt0/cf3/cf2/te2/te0; the frozen fit gate (protocol stage3.gate).
stage4: (only after the fit gate) the WL77A Part 0 references (WL76 population: rest slots + te1 footprint, DEF = te1 - rest) for AR, A75,
        Q78 finals (+ checkpoints of A75 and Q78, descriptive); the frozen transfer decision (protocol stage4.decision) with the mechanism
        criterion from mech78 (computed on archived development data only); WL59 gates separately.
    python eval78.py stage3 | stage4"""
import sys,os,json,time
import torch
from common78 import CFG,ROOT,R75,R73,R77A,INP68,write_new,environment,dev,load
RG=environment()
import family68 as FM
import eval68 as E
import eval72 as E72
import eval73 as E73
import rank75 as RK
import q78 as Q

A75=['A75_5901','A75_5902'];AR=['AR_5901','AR_5902'];Q78=list(CFG.get('runs',{})) or ['Q78_5901','Q78_5902'];CK=CFG.get('optimizer',{}).get('checkpoints',[])
W=lambda res,n,g,m:max(c['hi'] for c in res[n]['groups'][g]['pooled'][m])
Lo=lambda res,n,g,m:max(c['lo'] for c in res[n]['groups'][g]['pooled'][m])
_cand=E73.candidate
def candidate(name):
    if name.startswith('A75_'):
        m=RK.load_run(R75/'runs'/f'{name}.pt');return m,(lambda g,ports:RK.columns(m,g,ports)),f'W+UV rank {m.M}'
    if name.startswith('Q78_'):
        m=Q.load_run(RG,ROOT/'runs'/f'{name}.pt');return m,(lambda g,ports:Q.columns(m,g,ports)),f'W+UV-Q rank {m.M}'
    return _cand(name)
E73.candidate=candidate

def run_info(n):
    x=load(ROOT/'runs'/f'{n}.pt');r={k:x.get(k) for k in ('J_initial','J_final','loss_initial','loss_final','L_sym_initial','L_sym_final','converged',
        'last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','trainable','parameters','basis','rank')}
    r['history']=[(h['step'],h['J'],h['L_sym']) for h in x['history'] if h['step']%500==0];a,o,_=RK.basis()
    r['AO_identity']=bool(torch.equal(x['model']['in_logits'],a) and torch.equal(x['model']['out_logits'],o));return r

# ------------------------------------------------------------------ Stage 3
def fit_gate(res):
    """Protocol stage3.gate (frozen before any candidate metric): each Q78 seed, every listed group/metric hi <= factor x max over A75 seeds;
    validity: every passivity check passed and A/O bitwise the frozen basis."""
    gc=CFG['stage3']['gate'] if CFG else dict(factor=1.25,groups=['TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0'],metrics=['abs_coarse','abs_fine16','abs_native','con_coarse'])
    per={}
    for n in Q78:
        reg={}
        for g in gc['groups']:
            for m in gc['metrics']:v=dict(new=W(res,n,g,m),A75_max=max(W(res,o,g,m) for o in A75));v['passed']=v['new']<=gc['factor']*v['A75_max'];reg[f'{g}:{m}']=v
        valid=all(v['passed'] for v in res[n]['passivity'].values()) and res[n]['run']['AO_identity']
        per[n]=dict(fidelity=reg,valid=valid,passed=valid and all(v['passed'] for v in reg.values()))
    valid=all(per[n]['valid'] for n in Q78);passed=all(per[n]['passed'] for n in Q78)
    return dict(per_candidate=per,passed=passed,classification='PHYSICAL VALIDITY FAILURE' if not valid else 'FIT GATE PASSED' if passed else 'INSUFFICIENT FIT')

def stage3():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage3/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=E73.geoms(RG);P=E73.slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=E73.load_refs(G,P);GR=E73.groups()
    pc=E73.CFG['precision'];prec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=pc['abs_min'] for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=pc['deformation_min'] for k in D}
    print('REFS',round(time.perf_counter()-t0),flush=True);res={}
    names=A75+Q78+([] if dev() else [f'{n}@{c}' for n in Q78 for c in CK])
    for n in names:
        res[n]=E73.evaluate(n,G,P,R,D,qual,dq,lights,GR,full='@' not in n)
        if n in Q78:res[n]['run']=run_info(n);res[n]['parameters']=RK.parameter_counts(candidate(n)[0])
        if n in A75:
            x=load(R75/'runs'/f'{n}.pt');a,o,_=RK.basis();res[n]['run']=dict(AO_identity=bool(torch.equal(x['model']['in_logits'],a) and torch.equal(x['model']['out_logits'],o)),
                seconds=x['seconds'],concurrent=x['concurrent'],J_final=x['J_final'],L_sym_final=x['L_sym_final'])
        torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    gate=fit_gate(res)
    write_new(out,json.loads(json.dumps(dict(results=res,gate=gate,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs=prec,deformation=dprec),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    write_new(ROOT/'stage3/gate.json',json.loads(json.dumps(gate,default=float)))
    print('STAGE3_DONE',gate['classification'],round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ Stage 4
S4=CFG.get('stage4',{});POSE=S4.get('pose','te1')
def s4slots(pred):return [int(k) for k,v in S4.get('slots',{}).items() if S4['ports'].get(k) is not None and pred(v)]
def groups4():
    r=lambda sl:[('rest',s) for s in sl];B=('TBE','TBC','SM','SR')
    return {'BND_FAR':r(s4slots(lambda v:v['context'] in B and v['stratum']=='far')),'BND_NEAR':r(s4slots(lambda v:v['context'] in B and v['stratum']=='near')),
       'INT':r(s4slots(lambda v:v['context']=='INT')),'FP_rest':r(s4slots(lambda v:v['context']=='FP')),f'FP_{POSE}':[(POSE,s) for s in s4slots(lambda v:v['context']=='FP')],
       'UNS':r(s4slots(lambda v:v['context']=='SV')),'BND_HI':r(s4slots(lambda v:v['context'] in B and v['subcell']=='hi')),'BND_LO':r(s4slots(lambda v:v['context'] in B and v['subcell']=='lo')),
       'TOPBOT':r(s4slots(lambda v:v['context'] in ('TBE','TBC'))),'SIDE':r(s4slots(lambda v:v['context'] in ('SM','SR'))),'ALL_rest':r(s4slots(lambda v:True))}

def refs4(G,P):
    bm=FM.bin_map('cuda');R={};fp=s4slots(lambda v:v['context']=='FP')
    for g,slots in (('rest',s4slots(lambda v:True)),(POSE,fp)):
        for s in slots:
            f=R77A/'refs/bundles'/f'{g}_s{s}.pt'
            if dev() and s not in CFG['execution']['dev_slots'][g]:continue
            b=load(f);assert b['port']==P[s] and b['K']==4
            R[(g,s)]=E.Ref((g,s),b['S'],b['B16'],(1-G[g]['beta'][P[s]]).float(),G[g]['parent'],bm,E.ball16(G[g],P[s],bm))
    D={}
    for s in fp:
        if (POSE,s) in R and ('rest',s) in R:a,b=R[(POSE,s)],R[('rest',s)];D[('DEF',s)]=E.Ref(('DEF',s),a.S-b.S,a.B16-b.B16,a.d,a.parent,bm,a.ball16-b.ball16)
    return R,D

def decide4(res,mech):
    """Protocol stage4.decision: WL76 far / regression rules with W+UV-Q in place of FIA; verdict mapping with the mechanism criterion."""
    dc=S4['decision'] if S4 else dict(primary_groups=['BND_FAR','BND_NEAR','INT'],improve_factor=.8,regression_factor=1.25,
        regression={'BND_FAR':['abs_coarse','abs_fine16'],'BND_NEAR':['abs_coarse','abs_fine16'],'INT':['abs_coarse','abs_fine16','con_coarse'],'FP_rest':['abs_coarse','abs_fine16'],'FP_te1':['abs_coarse','abs_fine16']},
        secondary_metrics=['con_native','con_fine16','abs_coarse','abs_fine16','abs_native'])
    f,g_=dc['improve_factor'],dc['regression_factor']
    meas={gn:all(not res[n]['groups'][gn].get('unmeasured') for n in AR+A75+Q78) for gn in dc['primary_groups']}
    mean=lambda ns,gn,m:sum(W(res,n,gn,m) for n in ns)/len(ns)
    valid=all(all(v['passed'] for v in res[n]['physics'].values()) for n in Q78)
    if not valid:return dict(verdict='PHYSICAL VALIDITY FAILURE',qualifiers=[],valid=False,measured=meas)
    if not all(meas.values()):return dict(verdict='INCONCLUSIVE',qualifiers=['primary group unmeasured'],valid=True,measured=meas)
    ok=[gn for gn in res[Q78[0]]['groups'] if all(not res[n]['groups'][gn].get('unmeasured') for n in AR+A75+Q78)]
    strong={gn:(AR if mean(AR,gn,'con_coarse')<mean(A75,gn,'con_coarse') else A75) for gn in ok}
    def better(gn,m):
        b=strong[gn];r=dict(new=[W(res,n,gn,m) for n in Q78],baseline=b,old=[W(res,n,gn,m) for n in b],old_lo=[Lo(res,n,gn,m) for n in b])
        v='YES' if max(r['new'])<=f*min(r['old']) and max(r['new'])<min(r['old_lo']) else 'NO' if min(r['new'])>f*max(r['old']) else 'OVERLAP';return v,r
    def worse(gn,m):
        best=min((AR,A75),key=lambda ns:mean(ns,gn,m));r=dict(new=[W(res,n,gn,m) for n in Q78],baseline=best,old=[W(res,n,gn,m) for n in best])
        v='YES' if min(r['new'])>=g_*max(r['old']) else 'NO' if max(r['new'])<g_*min(r['old']) else 'OVERLAP';return v,r
    far,rf=better('BND_FAR','con_coarse');near,rn=better('BND_NEAR','con_coarse')
    reg={f'{gn}:{m}':worse(gn,m) for gn,ms in dc['regression'].items() for m in ms if gn in strong}
    sec={f'BND_FAR:{m}':better('BND_FAR',m)[0] for m in dc['secondary_metrics']}
    regY=[k for k,(v,_) in reg.items() if v=='YES'];regO=[k for k,(v,_) in reg.items() if v=='OVERLAP'];imp=bool(mech.get('improved'))
    if far=='OVERLAP':v='INCONCLUSIVE'
    elif far=='YES':v='SOURCE-IDENTITY LEAKAGE SUPPORTED' if imp else 'CONDITIONING-ONLY TRANSFER BENEFIT'
    else:v='INVARIANCE IMPROVED, TRANSFER UNCHANGED' if imp else 'NO TRANSFER BENEFIT'
    q=(['TRADE-OFF (regression)'] if regY and far=='YES' else [])+(['with regression'] if regY and far!='YES' else [])+(['possible regression (overlap)'] if regO and not regY else [])+(['LOCAL BENEFIT (BND_NEAR contrast)'] if near=='YES' else [])
    gates={n:{gn:res[n]['groups'][gn]['Q']['passed'] for gn in res[n]['groups'] if not res[n]['groups'][gn].get('unmeasured')} for n in AR+A75+Q78}
    return dict(verdict=v,qualifiers=q,far=far,near=near,mechanism_improved=imp,secondary=sec,regression={k:v_ for k,(v_,_) in reg.items()},valid=True,measured=meas,
                stronger_baseline={k:v_[0][:3] for k,v_ in strong.items()},WL59_gates=gates,values=dict(BND_FAR=rf,BND_NEAR=rn,regression={k:r for k,(_,r) in reg.items()}))

def score(name,G,P,R,D,qual,dq,GR,subs=None):
    m,cols,kind=candidate(name);row=dict(kind=kind,groups={},per_matrix={},per_matrix_r123={},deformation={});hs={};cg={}
    with torch.no_grad():
        for (g,s),r in R.items():
            T,A,O=cols(G[g],[P[s]]);h=FM.predict(T,(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0];hs[(g,s)]=h.cpu()
            Sg,Bg=r.gpu();cg[(g,s)]=E.cross(r,Sg,Bg,h);del Sg,Bg
            row['per_matrix'][f'{g}_s{s}']=E.pooled([r],[cg[(g,s)]],E.STD)
            if subs is not None:sb=subs[(g,s)];Sg,Bg=sb.gpu();row['per_matrix_r123'][f'{g}_s{s}']=E.pooled([sb],[E.cross(sb,Sg,Bg,h)],E.STD);del Sg,Bg
            del h;torch.cuda.empty_cache()
        for gn,keys in GR.items():
            keys=[k for k in keys if k in R and qual[k]]
            if len(keys)<4:row['groups'][gn]=dict(unmeasured=True,matrices=len(keys));continue
            p=E.pooled([R[k] for k in keys],[cg[k] for k in keys],E.STD)
            row['groups'][gn]=dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys],corr_slope=E.corr_slope([R[k] for k in keys],[cg[k] for k in keys]))
        keys=[k for k in D if dq[k] and (POSE,k[1]) in hs and ('rest',k[1]) in hs]
        if len(keys)<4:row['deformation']['DEF']=dict(unmeasured=True,matrices=len(keys))
        else:
            dcg=[]
            for k in keys:x=D[k];Sg,Bg=x.gpu();dcg.append(E.cross(x,Sg,Bg,(hs[(POSE,k[1])]-hs[('rest',k[1])]).cuda()));del Sg,Bg
            p=E.pooled([D[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D[k] for k in keys],dcg)
            row['deformation']['DEF']=dict(pooled=p,corr_slope=cs,Q=E73.Qdef(p,cs),matrices=[f's{k[1]}' for k in keys])
    return m,row

def stage4():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage4/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    gate=json.loads((ROOT/'stage3/gate.json').read_text())
    if not gate['passed']:assert dev() and os.environ.get('R78_DEV_FORCE_GATE')=='1','Stage 4 not authorised by the fit gate'
    mech=json.loads((ROOT/'mech/mech.json').read_text())['criterion']
    t0=time.perf_counter();G=E73.geoms(RG);G[POSE]=RG.device(load(R77A/'inputs'/f'geometry_{POSE}.pt'))
    P={int(k):v['port'] for k,v in json.loads((R77A/'ports76.json').read_text())['ports'].items() if v['port'] is not None}
    R,D=refs4(G,P);bm=FM.bin_map('cuda');GR={k:[x for x in v if x in R] for k,v in groups4().items()}
    if dev():GR['DEV']=[k for k in R if k[0]=='rest']
    subs={k:E.Ref(k,r.S[1:],r.B16[1:],r.d,r.parent,bm,r.ball16) for k,r in R.items()}
    pc=E73.CFG['precision'];prec={k:E72.sigSE(r) for k,r in R.items()};dprec={k:E72.sigSE(r) for k,r in D.items()}
    qual={k:min(v)>=pc['abs_min'] for k,v in prec.items()};dq={k:min(v)>=pc['deformation_min'] for k,v in dprec.items()}
    lights=load(INP68/'wl59/lights.pt')['test'];res={};from r57_eval import physics
    names=AR+A75+Q78+([] if dev() else [f'{n}@{c}' for n in A75+Q78 for c in CK])
    for n in names:
        m,res[n]=score(n,G,P,R,D,qual,dq,GR,subs if '@' not in n else None)
        if '@' not in n:res[n]['physics']={POSE:physics(m,G[POSE],lights[:2].cuda(),m.coefficients(G[POSE]))}
        del m;torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    dec=decide4(res,mech)
    write_new(out,json.loads(json.dumps(dict(results=res,decision=dec,mechanism=mech,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs={f'{a}_s{b}':v for (a,b),v in prec.items()},deformation={f'{a}_s{b}':v for (a,b),v in dprec.items()}),
        seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('STAGE4_DONE',dec['verdict'],json.dumps(dec.get('qualifiers')),round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':
    {'stage3':stage3,'stage4':stage4}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
