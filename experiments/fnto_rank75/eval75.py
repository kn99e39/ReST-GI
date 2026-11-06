"""WL75 evaluation (eval68 estimators and eval73 groups/qualification unchanged).
stage2: training-source fidelity (S16-R held-out replicas 4-7) and WL73 development groups for the rank-8 controls AR_5901/5902
        (recomputed; provenance vs the archived WL73 analysis) and the rank-16 candidates A75_5901/5902 (+ checkpoints, descriptive);
        labelled source-specific oracles G0_r0 / G2 on the same sources (WL74 Stage 1 path; provenance vs WL74 scores); passivity on
        rest/pt0/cf3/cf2/te2/te0; the predeclared training-fit gate (protocol stage2.gate).
stage3: (only after the gate) the WL74 Stage 4 population (rest slots 100-115, cf1 footprint 110-115, DEF4 = cf1 - rest): the same
        four finals + all checkpoints; per-matrix all eight metrics (replicas 0-3, and 1-3 for the oracle comparison); cf1 passivity;
        labelled G0_r0 / G2 oracles (T fitted on replica 0, scored on 1-3); the predeclared three-valued decision (protocol stage3.decision).
    python eval75.py stage2 | stage3"""
import sys,os,json,time
import torch
from common75 import CFG,ROOT,R74,R73,INP68,write_new,environment,dev
import family68 as FM
import eval68 as E
import eval72 as E72
import eval73 as E73
import ato74 as AT
import rank75 as RK

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
OLD=['AR_5901','AR_5902'];NEW=['A75_5901','A75_5902'];CK=CFG['optimizer']['checkpoints']
_cand=E73.candidate
def candidate(name):
    if name.startswith('A75_'):
        m=RK.load_run(ROOT/'runs'/f'{name}.pt');return m,(lambda g,ports:RK.columns(m,g,ports)),f'W+UV rank {m.M}'
    return _cand(name)
E73.candidate=candidate
def W(res,n,g,m):return max(c['hi'] for c in res[n]['groups'][g]['pooled'][m])
def Lo(res,n,g,m):return max(c['lo'] for c in res[n]['groups'][g]['pooled'][m])

def run_info(n):
    x=load(ROOT/'runs'/f'{n}.pt');r={k:x.get(k) for k in ('J_initial','J_final','loss_initial','loss_final','L_sym_initial','L_sym_final','converged',
        'last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','trainable','parameters','basis','rank')}
    r['history']=[(h['step'],h['J'],h['L_sym']) for h in x['history'] if h['step']%500==0];a,o,_=RK.basis()
    r['AO_identity']=bool(torch.equal(x['model']['in_logits'],a) and torch.equal(x['model']['out_logits'],o));return r

# ------------------------------------------------------------------ Stage 2
def oracle_train(G,P,R,GR,bids):
    """WL74 ato74.stage1 arithmetic for the given bases (per-source T fitted on fitting replicas 0-3, scored on held-out 4-7)."""
    out={};slots=[s for s,_ in __import__('fit73').populations()['S16R']];keys={g:[(g,s) for s in slots] for g in ('rest','pt0')}
    for bid in bids:
        b=load(R74/'bases'/f'{bid}.pt');assert b['slots']==slots;A,O=(x.float().cuda() for x in AT.AO(b['in_logits'].double(),b['out_logits'].double()));hs={}
        for g in ('rest','pt0'):
            for i,k in enumerate(keys[g]):hs[k]=AT.predict_m(b['T'][g][i:i+1].float().cuda(),(1-G[g]['beta'][[P[k[1]]]]).float(),A,O,'native')[0]
        kk=keys['rest']+keys['pt0'];cg,per=AT.score(E,R,kk,hs)
        out[bid]=dict(per_matrix=per,groups={gn:AT.pooled(E,R,cg,GR[gn]) for gn in ('TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0')},
                      J_final=b['J_final'],fit_seconds=b['seconds']);del hs,cg;torch.cuda.empty_cache()
    return out

def provenance(res,orc,GR):
    """Recomputed AR controls vs the archived WL73 analysis; recomputed oracles vs the archived WL74 Stage 1 scores (max rel diff of hi)."""
    arc=json.loads((R73/'eval/analysis.json').read_text())['results'];s1=json.loads((R74/'stage1/scores.json').read_text());d={}
    for n in OLD:
        x=0.
        for g,v in res[n]['groups'].items():
            if v.get('unmeasured') or g not in arc[n]['groups'] or arc[n]['groups'][g].get('unmeasured'):continue
            assert v['matrices']==arc[n]['groups'][g]['matrices'],('matrix set',n,g)
            for m,ch in v['pooled'].items():
                for c in range(3):a=arc[n]['groups'][g]['pooled'][m][c]['hi'];x=max(x,abs(ch[c]['hi']-a)/max(abs(a),1e-12))
        d[n]=x
    for b,v in orc.items():
        x=0.
        for g,gv in v['groups'].items():
            for m,ch in gv['pooled'].items():
                for c in range(3):a=s1[b]['groups'][g]['pooled'][m][c]['hi'];x=max(x,abs(ch[c]['hi']-a)/max(abs(a),1e-12))
        d[b]=x
    tol=CFG['stage2']['provenance_rel_tol'];return dict(max_rel_diff=d,tolerance=tol,passed=all(v<=tol for v in d.values()))

def fit_gate(res,orc,prov):
    """Protocol stage2.gate (frozen before any candidate metric)."""
    gc=CFG['stage2']['gate'];per={}
    for n in NEW:
        cap={g:dict(new=W(res,n,g,'con_coarse'),old_min=min(W(res,o,g,'con_coarse') for o in OLD)) for g in gc['capacity_groups']}
        for v in cap.values():v['passed']=v['new']<=gc['capacity_factor']*v['old_min']
        reg={}
        for g,ms in gc['regression'].items():
            for m in ms:v=dict(new=W(res,n,g,m),old_max=max(W(res,o,g,m) for o in OLD));v['passed']=v['new']<=gc['regression_factor']*v['old_max'];reg[f'{g}:{m}']=v
        valid=all(v['passed'] for v in res[n]['passivity'].values()) and res[n]['run']['AO_identity']
        per[n]=dict(capacity=cap,regression=reg,valid=valid,passed=valid and all(v['passed'] for v in cap.values()) and all(v['passed'] for v in reg.values()))
    valid=all(per[n]['valid'] for n in NEW);passed=prov['passed'] and all(per[n]['passed'] for n in NEW)
    cls='STOP (provenance)' if not prov['passed'] else 'PHYSICAL VALIDITY FAILURE' if not valid else 'FIT GATE PASSED' if passed else 'INSUFFICIENT FIT'
    # descriptive (not part of the gate): oracle values and the fraction of the AR -> G2-oracle gap closed on the gate groups
    desc={}
    for g in gc['capacity_groups']+['TRAIN_KEEP_rest','TRAIN_KEEP_pt0']:
        for m in ('con_coarse','abs_coarse','abs_fine16','abs_native'):
            o0=max(c['hi'] for c in orc['G0_r0']['groups'][g]['pooled'][m]);o2=max(c['hi'] for c in orc[CFG['basis']['selected']]['groups'][g]['pooled'][m])
            ar=sum(W(res,o,g,m) for o in OLD)/2
            desc[f'{g}:{m}']=dict(G0=o0,G2=o2,AR_mean=ar,A75={n:W(res,n,g,m) for n in NEW},gap_closed={n:(ar-W(res,n,g,m))/(ar-o2) if ar!=o2 else None for n in NEW})
    return dict(per_candidate=per,provenance=prov,passed=passed,classification=cls,descriptive=desc)

def stage2():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage2/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=E73.geoms(RG);P=E73.slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=E73.load_refs(G,P);GR=E73.groups()
    pc=E73.CFG['precision'];prec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=pc['abs_min'] for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=pc['deformation_min'] for k in D}
    print('REFS',round(time.perf_counter()-t0),flush=True);res={}
    names=OLD+NEW+([] if dev() else [f'{n}@{c}' for n in NEW for c in CK])
    for n in names:
        res[n]=E73.evaluate(n,G,P,R,D,qual,dq,lights,GR,full='@' not in n)
        if n in NEW:res[n]['run']=run_info(n);res[n]['parameters']=RK.parameter_counts(candidate(n)[0])
        torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    orc=oracle_train(G,P,R,GR,['G0_r0',CFG['basis']['selected']]);prov=provenance(res,orc,GR);gate=fit_gate(res,orc,prov)
    write_new(out,json.loads(json.dumps(dict(results=res,oracle=orc,gate=gate,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs=prec,deformation=dprec),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    write_new(ROOT/'stage2/gate.json',json.loads(json.dumps(gate,default=float)))
    print('STAGE2_DONE',gate['classification'],round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ Stage 3
S3=CFG['stage3'];POSE=S3['pose']
def s4(kind=None,st=None):return [int(s) for s,v in S3['slots'].items() if (kind is None or v['context'] in kind) and (st is None or v['stratum']==st)]
FP=s4(('FP',))
def groups4():
    r=lambda sl:[('rest',s) for s in sl]
    return {'BND4_NEAR':r(s4(('TBE','TBC','SM','SR'),'near')),'BND4_FAR':r(s4(('TBE','TBC','SM','SR'),'far')),'INT4':r(s4(('INT',))+FP),
            f'FP4_{POSE}':[(POSE,s) for s in FP],'ALL4_rest':r(sorted(int(s) for s in S3['slots'])),'BND4':r(s4(('TBE','TBC','SM','SR'))),
            'TOPBOT4':r(s4(('TBE','TBC'))),'SIDE4':r(s4(('SM','SR')))}

def refs4(RG,G,P):
    bm=FM.bin_map('cuda');R={}
    for g,slots in (('rest',sorted(int(s) for s in S3['slots'])),(POSE,FP)):
        for s in slots:
            if dev() and s not in CFG['execution']['dev_slots'][g]:continue   # dev smoke: subset only
            b=load(ROOT/'refs/bundles'/f'{g}_s{s}.pt');assert b['port']==P[s] and b['K']==4
            R[(g,s)]=E.Ref((g,s),b['S'],b['B16'],(1-G[g]['beta'][P[s]]).float(),G[g]['parent'],bm,E.ball16(G[g],P[s],bm))
    D={}
    for s in FP:
        if (POSE,s) not in R or ('rest',s) not in R:continue
        a,b=R[(POSE,s)],R[('rest',s)];D[('DEF4',s)]=E.Ref(('DEF4',s),a.S-b.S,a.B16-b.B16,a.d,a.parent,bm,a.ball16-b.ball16)
    return R,D

def score_model(name,G,P,R,D,qual,dq,GR,subs,lights,final):
    m,cols,kind=candidate(name);row=dict(kind=kind,groups={},per_matrix={},per_matrix_r123={},deformation={})
    with torch.no_grad():
        hs={};cg={}
        for (g,s),r in R.items():
            T,A,O=cols(G[g],[P[s]]);h=FM.predict(T,(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0];hs[(g,s)]=h
            Sg,Bg=r.gpu();cg[(g,s)]=E.cross(r,Sg,Bg,h);del Sg,Bg
            row['per_matrix'][f'{g}_s{s}']=E.pooled([r],[cg[(g,s)]],E.STD)
            if final:
                sb=subs[(g,s)];Sg,Bg=sb.gpu();c1=E.cross(sb,Sg,Bg,h);del Sg,Bg;row['per_matrix_r123'][f'{g}_s{s}']=E.pooled([sb],[c1],E.STD)
        for gn,keys in GR.items():
            keys=[k for k in keys if qual[k]]
            if len(keys)<4:row['groups'][gn]=dict(unmeasured=True,matrices=len(keys));continue
            p=E.pooled([R[k] for k in keys],[cg[k] for k in keys],E.STD)
            row['groups'][gn]=dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys],corr_slope=E.corr_slope([R[k] for k in keys],[cg[k] for k in keys]))
        keys=[k for k in D if dq[k]]
        if len(keys)>=4:
            dcg=[]
            for k in keys:x=D[k];Sg,Bg=x.gpu();dcg.append(E.cross(x,Sg,Bg,hs[(POSE,k[1])]-hs[('rest',k[1])]));del Sg,Bg
            p=E.pooled([D[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D[k] for k in keys],dcg)
            row['deformation']['DEF4']=dict(pooled=p,corr_slope=cs,Q=E73.Qdef(p,cs),matrices=[f's{k[1]}' for k in keys],
                per_matrix={f's{k[1]}':dict(err=[c['hi'] for c in E.pooled([D[k]],[c_],E.STD[2:3])['abs_coarse']],corr=[c['corr'] for c in E.corr_slope([D[k]],[c_])]) for k,c_ in zip(keys,dcg)})
        else:row['deformation']['DEF4']=dict(unmeasured=True,matrices=len(keys))
        if final:
            from r57_eval import physics
            row['passivity']={POSE:physics(m,G[POSE],lights[:2].cuda(),m.coefficients(G[POSE]))}
    del hs,cg;torch.cuda.empty_cache();return row

def oracle4(bid,G,P,R,keys,subs,GR,qual):
    """Labelled oracle with a frozen basis (WL74 ato74.stage2 arithmetic): per-source T on replica 0 (float64 FISTA from 0.5/P, iters, then the
    same budget again from the solution as a convergence check), prediction from the first solve, scored on replicas 1-3."""
    b=load(R74/'bases'/f'{bid}.pt');A,O=(x.detach().cuda() for x in AT.AO(b['in_logits'].double(),b['out_logits'].double()));M=A.shape[0];bm=FM.bin_map('cuda')
    iters=S3['oracle_iters'] if 'R75_DEV_ITERS' not in os.environ else int(os.environ['R75_DEV_ITERS'])
    if 'R75_DEV_ITERS' in os.environ:assert dev()
    hs={};conv={};t0=time.perf_counter()
    for g in sorted({k[0] for k in keys}):
        kg=[k for k in keys if k[0]==g]
        for c0 in range(0,len(kg),S3['oracle_chunk']):
            ch=kg[c0:c0+S3['oracle_chunk']];S=torch.stack([R[k].S[0] for k in ch]).double().cuda();d=torch.stack([(1-G[g]['beta'][P[s]]).double() for _,s in ch])
            tg=AT.target(S,d,G[g]['parent'],bm);del S
            T0=torch.full((len(ch),G[g]['P'],M,3),.5/G[g]['P'],dtype=torch.float64,device='cuda');T,_=AT.fista_T(tg,A,O,T0,iters)
            with torch.no_grad():Ja=float(AT.objective_m(T,A,O,tg))
            T2,_=AT.fista_T(tg,A,O,T,iters)
            with torch.no_grad():Jb=float(AT.objective_m(T2,A,O,tg))
            conv[f'{g}_{ch[0][1]}-{ch[-1][1]}']=dict(J=Ja,J_2x=Jb,rel_change=(Ja-Jb)/Ja)
            for i,k in enumerate(ch):hs[k]=AT.predict_m(T[i:i+1].float(),d[i:i+1].float(),A.float(),O.float(),'native')[0].detach()
            del tg,T,T2;torch.cuda.empty_cache()
    with torch.no_grad():
        cg={};per={}
        for k in keys:
            sb=subs[k];Sg,Bg=sb.gpu();cg[k]=E.cross(sb,Sg,Bg,hs[k]);del Sg,Bg;per[f'{k[0]}_s{k[1]}']=E.pooled([sb],[cg[k]],E.STD)
        grp={}
        for gn,kk in GR.items():
            kk=[k for k in kk if qual[k] and k in keys]
            grp[gn]=dict(unmeasured=True,matrices=len(kk)) if len(kk)<4 else dict(pooled=E.pooled([subs[k] for k in kk],[cg[k] for k in kk],E.STD),matrices=[f'{a}_s{b}' for a,b in kk])
    return dict(basis=bid,rank=M,per_matrix=per,groups=grp,convergence=conv,seconds=time.perf_counter()-t0)

def decide3(res,orc):
    """Protocol stage3.decision (frozen before any Stage 3 evidence)."""
    dc=S3['decision'];f,g_=dc['improve_factor'],dc['regression_factor']
    meas={gn:all(not res[n]['groups'][gn].get('unmeasured') for n in OLD+NEW) for gn in dc['primary_groups']}
    if not all(meas.values()):return dict(measured=meas,classes=['INCONCLUSIVE (primary group unmeasured)'])
    def better(gn,m):
        r=dict(new=[W(res,n,gn,m) for n in NEW],old=[W(res,n,gn,m) for n in OLD],old_lo=[Lo(res,n,gn,m) for n in OLD])
        if max(r['new'])<=f*min(r['old']) and max(r['new'])<min(r['old_lo']):v='YES'
        elif min(r['new'])>f*max(r['old']):v='NO'
        else:v='OVERLAP'
        return v,r
    def worse(gn,m):
        r=dict(new=[W(res,n,gn,m) for n in NEW],old=[W(res,n,gn,m) for n in OLD])
        if min(r['new'])>=g_*max(r['old']):v='YES'
        elif max(r['new'])<g_*min(r['old']):v='NO'
        else:v='OVERLAP'
        return v,r
    far,rf=better('BND4_FAR','con_coarse');near,rn=better('BND4_NEAR','con_coarse')
    reg={f'{gn}:{m}':worse(gn,m) for gn,ms in dc['regression'].items() for m in ms if all(not res[n]['groups'][gn].get('unmeasured') for n in OLD+NEW)}
    sec={f'BND4_FAR:{m}':better('BND4_FAR',m)[0] for m in dc['secondary_metrics']}
    valid=all(all(v['passed'] for v in res[n]['passivity'].values()) for n in NEW+OLD)
    regYES=[k for k,(v,_) in reg.items() if v=='YES'];regOV=[k for k,(v,_) in reg.items() if v=='OVERLAP']
    if not valid:c=['PHYSICAL VALIDITY FAILURE']
    elif far=='OVERLAP':c=['INCONCLUSIVE (seed / statistical overlap on BND4_FAR contrast)']
    elif far=='YES':c=(['TRADE-OFF (far-boundary contrast benefit with regression)'] if regYES else ['TRANSFERABLE RANK-16 BENEFIT (far boundary)'])+(['possible regression (overlap)'] if regOV and not regYES else [])
    else:c=['NO INDEPENDENT GENERALIZATION BENEFIT (training-only benefit)']+(['LOCAL BENEFIT (BND4_NEAR contrast)'] if near=='YES' else [])+(['with regression'] if regYES else [])
    gates={n:{gn:dict(Q=res[n]['groups'][gn]['Q']['passed']) for gn in dc['primary_groups']+[f'FP4_{POSE}']} for n in OLD+NEW}
    for n in OLD+NEW:gates[n]['DEF4']=res[n]['deformation']['DEF4'].get('Q',{}).get('passed') if not res[n]['deformation']['DEF4'].get('unmeasured') else None
    return dict(measured=meas,classes=c,far=far,near=near,secondary=sec,regression={k:v for k,(v,_) in reg.items()},valid=valid,WL59_gates=gates,
                values=dict(BND4_FAR=rf,BND4_NEAR=rn,regression={k:r for k,(_,r) in reg.items()}))

def stage3():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage3/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    gate=json.loads((ROOT/'stage2/gate.json').read_text())
    if not gate['passed']:assert dev() and os.environ.get('R75_DEV_FORCE_GATE')=='1','Stage 3 not authorised by the training-fit gate'
    t0=time.perf_counter();G=E73.geoms(RG);G[POSE]=RG.device(load(ROOT/'inputs'/f'geometry_{POSE}.pt'))
    P={int(k):v['port'] for k,v in json.loads((ROOT/'ports4.json').read_text())['ports'].items()};R,D=refs4(RG,G,P);bm=FM.bin_map('cuda')
    GR={k:[x for x in v if x in R] for k,v in groups4().items()}
    if dev():GR['DEV4']=[k for k in R if k[0]=='rest']   # dev smoke: one group large enough to exercise the pooled path
    subs={k:E.Ref(k,r.S[1:],r.B16[1:],r.d,r.parent,bm,r.ball16) for k,r in R.items()}
    pc=E73.CFG['precision'];prec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=pc['abs_min'] for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=pc['deformation_min'] for k in D}
    lights=load(INP68/'wl59/lights.pt')['test'];res={}
    names=OLD+NEW+([] if dev() else [f'{n}@{c}' for n in OLD+NEW for c in CK])
    for n in names:res[n]=score_model(n,G,P,R,D,qual,dq,GR,subs,lights,'@' not in n);print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    keys=[k for k in R if qual[k]];orc={}
    for bid in ('G0_r0',CFG['basis']['selected']):orc[bid]=oracle4(bid,G,P,R,keys,subs,GR,qual);print('ORACLE',bid,round(time.perf_counter()-t0),flush=True)
    dec=decide3(res,orc)
    write_new(out,json.loads(json.dumps(dict(results=res,oracle=orc,decision=dec,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs=prec,deformation=dprec,qualified={f'{a}_s{b}':v for (a,b),v in qual.items()},deformation_qualified={f'{a}_s{b}':v for (a,b),v in dq.items()}),
        seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('STAGE3_DONE',json.dumps(dec['classes']),round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':
    {'stage2':stage2,'stage3':stage3}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
