"""WL79 evaluation (eval68 estimators, eval73 / eval78 group machinery unchanged; WL78 modules imported read-only).
 stage3  supervised fidelity on S16-R held-out replicas 4-7 (eval73.evaluate, the WL75/WL78 path) for Q78 (primary control), A75 and the
         WL79 candidates S79 (+ checkpoints, descriptive); Q78 reproduction against WL78's recorded Stage 3; transformed training references
         (exact images of the held-out replicas under the Stage 1 operations) for Q78 / A75 / S79; the frozen fit gate (protocol stage3.gate).
 mech    Stage 4 development mechanism check: M1 (GATE) the historical equivalent-context pairs 101<-50, 105<-52, 107<-53 (exact op + in-face
         translation, exact port / bin maps): near-field (0-5 mm) difference between the far-source prediction and the mapped training-source
         prediction, native and fine16 contrast, worst channel, mean over pairs; S79 / Q78 per seed; M2 learned equivariance (no reference):
         prediction at the image port vs the exact image of the prediction, all 15 operations, training sources and out-of-orbit development
         sources; M3 WL77A Part 0 (scored in WL78; development only) groups incl. DEF (te1 - rest).
 stage5  (only after the fit and mechanism gates) the frozen ports79 population: groups and the frozen transfer decision (protocol stage5).
 stage6  geometry change: WL73 deformation groups (DEF71 cf2, CONFDEF72 te2, DEF73 te0; from stage3) and WL77A Part 0 DEF (from mech M3):
         S79 vs Q78 vs A75, relative error / correlation per channel; frozen deformation-regression flag.
    python eval79.py stage3 | mech | stage5 | stage6"""
import sys,os,json,time
import numpy as np
import torch
from common79 import CFG,ROOT,R78,R75,R73,R77A,R75L,INP68,R71,R68,write_new,environment,dev,load
import common68 as C68
import common78
common78.environment=lambda:C68.environment(ROOT/'repo')        # WL78 modules use this repository copy
RG=environment()
import family68 as FM
import eval68 as E
import eval72 as E72
import eval73 as E73
import eval78 as E78
import rank75 as RK
import q78 as Q
import sym72 as SY
import sym79 as S9
import fit71,fit73,ato74 as AT

Q78=['Q78_5901','Q78_5902'];A75=E78.A75;AR=E78.AR;S79=list(CFG.get('runs',{})) or ['S79_5901','S79_5902'];CK=CFG.get('optimizer',{}).get('checkpoints',[])
W=E78.W;Lo=E78.Lo
_c78=E78.candidate
def candidate(name):
    if name.startswith('S79_'):
        m=Q.load_run(RG,ROOT/'runs'/f'{name}.pt');return m,(lambda g,ports:Q.columns(m,g,ports)),f'W+UV-Q rank {m.M} (symmetry presentation)'
    return _c78(name)
E73.candidate=candidate;E78.candidate=candidate
def seed(n):return n.split('_')[1].split('@')[0]

def run_info(n):
    x=load(ROOT/'runs'/f'{n}.pt');r={k:x.get(k) for k in ('J_initial','J_final','loss_initial','loss_final','L_sym_initial','L_sym_final','converged',
        'last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','trainable','parameters','basis','rank')}
    r['history']=[(h['step'],h['J'],h['L_sym']) for h in x['history'] if h['step']%500==0];a,o,_=RK.basis()
    r['AO_identity']=bool(torch.equal(x['model']['in_logits'],a) and torch.equal(x['model']['out_logits'],o));return r

# ------------------------------------------------------------------ transformed training references (Stage 1 operations)
def transformed_refs(names):
    g=RG.device(fit71.guarded('geometry_rest'));P=g['P'];x0=g['rest_xyz'].double().cpu().numpy();bm=FM.bin_map('cuda');pops=fit73.populations()['S16R']
    import oracle79 as O9
    ops=[O9.op_of(n) for n in CFG['stage1']['ops']][:1 if dev() else None];out={n:{} for n in names};models={n:candidate(n) for n in names}
    for op in ops:
        mp=S9.Map(RG,'rest',op,x0);mp.check();refs=[];cgs={n:[] for n in names}
        for s,p in pops:
            b=load(O9.bundle_path(s));Se,Be=b['S'][[4,5,6,7]],b['B16'][[4,5,6,7]];del b;q=int(mp.perm[p]);d=(1-g['beta'][q]).float().cpu()
            ref=E.Ref(('rest',s,O9.opname(op)),mp.response(Se.cuda(),p).cpu(),mp.fine16(Be.cuda(),p).cpu(),d,g['parent'],bm,E.ball16(g,q,bm));Sg,Bg=ref.gpu()
            with torch.no_grad():
                for n in names:
                    m,cols,_=models[n];T,A,O=cols(g,[q]);h=FM.predict(T,(1-g['beta'][[q]]).float(),A,O,'native')[0];cgs[n].append(E.cross(ref,Sg,Bg,h))
            del Sg,Bg;ref.S=None;ref.B16=None;refs.append(ref);torch.cuda.empty_cache()
        for n in names:out[n][O9.opname(op)]=E.pooled(refs,cgs[n],E.STD)
        print('TRANSFORMED',O9.opname(op),flush=True)
    return out

def fit_gate(res):
    gc=CFG['stage3']['gate'];per={}
    for n in S79:
        reg={}
        for g in gc['groups']:
            for m in gc['metrics']:v=dict(new=W(res,n,g,m),Q78_max=max(W(res,o,g,m) for o in Q78));v['passed']=v['new']<=gc['factor']*v['Q78_max'];reg[f'{g}:{m}']=v
        valid=all(v['passed'] for v in res[n]['passivity'].values()) and res[n]['run']['AO_identity']
        per[n]=dict(fidelity=reg,valid=valid,passed=valid and all(v['passed'] for v in reg.values()))
    valid=all(per[n]['valid'] for n in S79);passed=all(per[n]['passed'] for n in S79)
    return dict(per_candidate=per,passed=passed,classification='PHYSICAL VALIDITY FAILURE' if not valid else 'FIT GATE PASSED' if passed else 'INSUFFICIENT FIT')

def stage3():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage3/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=E73.geoms(RG);P=E73.slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=E73.load_refs(G,P);GR=E73.groups()
    pc=E73.CFG['precision'];prec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=pc['abs_min'] for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=pc['deformation_min'] for k in D}
    res={};names=Q78+A75+S79+([] if dev() else [f'{n}@{c}' for n in S79 for c in CK])
    for n in names:
        res[n]=E73.evaluate(n,G,P,R,D,qual,dq,lights,GR,full='@' not in n)
        if n in S79:res[n]['run']=run_info(n);res[n]['parameters']=RK.parameter_counts(candidate(n)[0])
        if n in Q78+A75:
            x=load((R78 if n in Q78 else R75)/'runs'/f'{n}.pt');a,o,_=RK.basis()
            res[n]['run']=dict(AO_identity=bool(torch.equal(x['model']['in_logits'],a) and torch.equal(x['model']['out_logits'],o)),seconds=x['seconds'],J_final=x['J_final'],L_sym_final=x['L_sym_final'],
                               history=[(h['step'],h['J'],h['L_sym']) for h in x['history'] if h['step']%500==0])
        torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    # Q78 reproduction vs WL78's recorded Stage 3 (same code, data and checkpoints)
    w78=json.loads((R78/'stage3/analysis.json').read_text())['results'];rep={}
    for n in Q78:
        diffs=[abs(c1['hi']-c0['hi']) for gn,gr in res[n]['groups'].items() if not gr.get('unmeasured') for m,v in gr['pooled'].items() for c1,c0 in zip(v,w78[n]['groups'][gn]['pooled'][m])]
        rep[n]=dict(max_abs_diff_hi=max(diffs),values=len(diffs))
    gate=fit_gate(res);tr=transformed_refs(Q78+A75+S79)
    write_new(out,json.loads(json.dumps(dict(results=res,gate=gate,q78_reproduction=rep,transformed_training=tr,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs=prec,deformation=dprec),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    write_new(ROOT/'stage3/gate.json',json.loads(json.dumps(dict(gate,q78_reproduction=rep),default=float)))
    print('STAGE3_DONE',gate['classification'],json.dumps(rep),round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ Stage 4 mechanism (development data)
EDGES=[0,5,10,20,1e9]
def mech():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'mech/mech.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();g=RG.device(fit71.guarded('geometry_rest'));P=g['P'];x=g['rest_xyz'].double().cpu().numpy();parent=g['parent'];bm=FM.bin_map('cpu')
    nn=torch.nn.functional.normalize(g['node'][:,9:12].double(),dim=1).cpu().numpy()
    names=Q78+S79+A75;models={n:candidate(n) for n in names}
    stc={}
    def predict(n,p):
        m=models[n][0];gq=Q.QG(RG,g) if isinstance(m,Q.TransportQ) else g
        if n not in stc:stc[n]=RK.TransportM.state(m,gq)           # geometry state once per model (exactly what columns() recomputes)
        T,A,O=RK.columns(m,gq,[p],state=stc[n]);return FM.predict(T,(1-g['beta'][[p]]).float(),A,O,'native')[0].double().cpu()
    def contrast(Y,d):Yf=FM.aggregate(Y,'fine16',parent.cpu(),bm);return (Yf-Yf.mean(-4,keepdim=True))/d.view(*([1]*(Yf.dim()-4)),16,1,1,1)
    def nrm(X,mask):return X[...,mask,:,:].square().sum(dim=tuple(i for i in range(X.dim()) if i!=X.dim()-1)).sqrt()
    def strata(q):r=np.linalg.norm(x-x[q],axis=1)*1e3;return [(f'{a}-{b if b<1e8 else "inf"}mm',torch.as_tensor((r>=a)&(r<b))) for a,b in zip(EDGES[:-1],EDGES[1:])]
    PORTS={s:p for s,p in fit73.populations()['S16R']};PORTS.update({int(k):v['port'] for k,v in json.loads((R75L/'ports4.json').read_text())['ports'].items()})
    def bundle(s):
        f=R75L/'refs/bundles'/f'rest_s{s}.pt' if s>=100 else R73/'refs/bundles'/f'rest_s{s}.pt';b=load(f);assert int(b['port'])==PORTS[s];return b['S'].double()
    res=dict(note='WL79 Stage 4 mechanism diagnostics on archived development data; non-confirmatory',models=names,pairs={})
    # ---- M1 historical equivalent-context pairs (GATE)
    for st,sf in ((50,101),(52,105),(53,107)):
        ps,pf=PORTS[st],PORTS[sf];cand=[]
        for op in SY.ops(True):
            M=SY.matrix(op);d=x[pf]-M@x[ps]
            if (np.abs(d)>1e-9).sum()<=1 and abs(d@nn[pf])<1e-9:cand.append((float(np.abs(d).sum()),op,d))
        _,op,delta=sorted(cand,key=lambda t:t[0])[0];mp=S9.Map(RG,'rest',op,x,delta=delta)
        Sf=bundle(sf);Ss=bundle(st);A=Sf.mean(0);df=1-g['beta'][pf].double().cpu();cA=contrast(Sf,df).mean(0);Sm,mapped=mp.response_partial(Ss,ps);B=Sm.mean(0);cB=contrast(Sm,df).mean(0)
        row=dict(op=list(map(str,op)),delta_mm=(delta*1e3).round(3).tolist(),models={})
        for n in names:
            hs=predict(n,ps);hf=predict(n,pf);hm,_=mp.response_partial(hs[None],ps);hm=hm[0];chf=contrast(hf[None],df)[0];chm=contrast(hm[None],df)[0];r={}
            for nm,mask in strata(pf):
                mk=mask&mapped;a=nrm(A,mk);ca=nrm(cA,mk)
                r[nm]=dict(model_diff=(nrm(hf-hm,mk)/a).tolist(),con_model_diff=(nrm(chf-chm,mk)/ca).tolist(),phys_diff=(nrm(A-B,mk)/a).tolist(),phys_con_diff=(nrm(cA-cB,mk)/ca).tolist(),
                    err_far=(nrm(hf-A,mk)/a).tolist(),err_train_mapped=(nrm(hm-B,mk)/a).tolist(),con_err_far=(nrm(chf-cA,mk)/ca).tolist(),con_err_train_mapped=(nrm(chm-cB,mk)/ca).tolist())
            row['models'][n]=r
        res['pairs'][f'{st}->{sf}']=row;print('PAIR',st,sf,{n:[round(max(v['0-5mm']['model_diff']),3),round(max(v['0-5mm']['con_model_diff']),3)] for n,v in row['models'].items()},flush=True)
        del Ss,Sf,Sm
    mc=CFG['stage4']['gate'];crit={}
    mn=lambda n,k:float(np.mean([max(res['pairs'][p]['models'][n]['0-5mm'][k]) for p in res['pairs']]))
    for c in S79:
        q=f'Q78_{seed(c)}';crit[c]=dict(native_ratio=mn(c,'model_diff')/mn(q,'model_diff'),contrast_ratio=mn(c,'con_model_diff')/mn(q,'con_model_diff'),
            S79=[mn(c,'model_diff'),mn(c,'con_model_diff')],Q78=[mn(q,'model_diff'),mn(q,'con_model_diff')])
    crit['improved']=all(crit[c]['native_ratio']<=mc['factor'] and crit[c]['contrast_ratio']<=mc['factor'] for c in S79)
    crit['q78_reproduction_vs_WL78']={q:dict(wl79=[mn(q,'model_diff'),mn(q,'con_model_diff')],wl78=json.loads((R78/'mech/mech.json').read_text())['criterion'][seed(q)]['Q78']) for q in Q78}
    res['criterion']=crit;json.dump(res,open(ROOT/'mech/mech_partial.json','w'),default=float)
    # ---- M2 learned equivariance (prediction at the image vs image of the prediction), every non-identity operation
    w76=json.loads((R77A/'ports76.json').read_text())['ports']
    sets=dict(train=[p for _,p in fit73.populations()['S16R']],dev_out_of_orbit=sorted({v for k,v in PORTS.items() if k>=100}|{v['port'] for v in w76.values() if v['port'] is not None and v['spec']['context']!='FP'}))
    if dev():sets={k:v[:2] for k,v in sets.items()}
    maps=[S9.Map(RG,'rest',op,x) for op in SY.ops(True)[1:]];eq={}
    for sn,ports in sets.items():
        for n in names:
            vals=dict(native_total=[],native_near=[],con_total=[],con_near=[])
            for p in ports:
                h=predict(n,p);d=1-g['beta'][p].double().cpu();ch=contrast(h[None],d)[0];al=torch.ones(P,dtype=torch.bool)
                for mp in maps:
                    q=int(mp.perm[p]);hq=predict(n,q);hm=mp.response(h,p);dq=1-g['beta'][q].double().cpu();chq=contrast(hq[None],dq)[0];chm=contrast(hm[None],dq)[0]
                    near=torch.as_tensor(np.linalg.norm(x-x[q],axis=1)*1e3<5)
                    vals['native_total'].append(float((nrm(hq-hm,al)/nrm(hm,al)).max()));vals['native_near'].append(float((nrm(hq-hm,near)/nrm(hm,near)).max()))
                    vals['con_total'].append(float((nrm(chq-chm,al)/nrm(chm,al)).max()));vals['con_near'].append(float((nrm(chq-chm,near)/nrm(chm,near)).max()))
            eq[f'{sn}:{n}']={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),max=float(np.max(v))) for k,v in vals.items()};eq[f'{sn}:{n}']['pairs']=len(vals['native_total'])
            print('EQUIV',sn,n,json.dumps({k:round(v['mean'],4) for k,v in eq[f'{sn}:{n}'].items() if isinstance(v,dict)}),flush=True)
    res['equivariance']=eq;json.dump(res,open(ROOT/'mech/mech_partial.json','w'),default=float)
    # ---- M3 WL77A Part 0 (development; scored in WL78) with the WL78 Stage 4 machinery, incl. DEF (te1 - rest)
    G=E73.geoms(RG);G[E78.POSE]=RG.device(load(R77A/'inputs'/f'geometry_{E78.POSE}.pt'))
    P0={int(k):v['port'] for k,v in w76.items() if v['port'] is not None}
    R,D=E78.refs4(G,P0);GR={k:[x_ for x_ in v if x_ in R] for k,v in E78.groups4().items()}
    pc=E73.CFG['precision'];qual={k:min(E72.sigSE(r))>=pc['abs_min'] for k,r in R.items()};dq={k:min(E72.sigSE(r))>=pc['deformation_min'] for k,r in D.items()}
    part0={}
    for n in Q78+S79+A75+AR:
        m,part0[n]=E78.score(n,G,P0,R,D,qual,dq,GR);del m;torch.cuda.empty_cache();print('PART0',n,flush=True)
    res['part0_dev']=part0;res['seconds']=time.perf_counter()-t0
    write_new(out,json.loads(json.dumps(res,default=float)));write_new(ROOT/'mech/gate.json',json.loads(json.dumps(crit,default=float)))
    print('MECH_DONE',json.dumps(crit,default=float)[:600],flush=True)

# ------------------------------------------------------------------ Stage 5 (prospective population)
def groups5(rows):
    B=('TBE','TBC','SM','SR');sl=lambda pred:[('rest',int(k)) for k,v in rows.items() if v['port'] is not None and pred(v['spec'])]
    return {'BND_FAR':sl(lambda v:v['context'] in B and v['stratum']=='far'),'BND_NEAR':sl(lambda v:v['context'] in B and v['stratum']=='near'),
            'INT':sl(lambda v:v['context']=='INT'),'UNS':sl(lambda v:v['context']=='SV'),'BND_HI':sl(lambda v:v['context'] in B and v['subcell']=='hi'),
            'BND_LO':sl(lambda v:v['context'] in B and v['subcell']=='lo'),'TOPBOT':sl(lambda v:v['context'] in ('TBE','TBC')),'SIDE':sl(lambda v:v['context'] in ('SM','SR')),
            'ALL_rest':sl(lambda v:True)}

def decide5(res):
    dc=CFG['stage5']['decision'];f,g_=dc['improve_factor'],dc['regression_factor']
    meas={gn:all(not res[n]['groups'][gn].get('unmeasured') for n in Q78+A75+S79) for gn in dc['primary_groups']}
    mean=lambda ns,gn,m:sum(W(res,n,gn,m) for n in ns)/len(ns)
    valid=all(all(v['passed'] for v in res[n]['physics'].values()) for n in S79)
    if not valid:return dict(verdict='PHYSICAL VALIDITY FAILURE',qualifiers=[],valid=False,measured=meas)
    if not all(meas.values()):return dict(verdict='INCONCLUSIVE',qualifiers=['primary group unmeasured'],valid=True,measured=meas)
    ok=[gn for gn in res[S79[0]]['groups'] if all(not res[n]['groups'][gn].get('unmeasured') for n in Q78+A75+S79)]
    strong={gn:(Q78 if mean(Q78,gn,'con_coarse')<=mean(A75,gn,'con_coarse') else A75) for gn in ok}
    def better(gn,m):
        b=strong[gn];r=dict(new=[W(res,n,gn,m) for n in S79],baseline=b,old=[W(res,n,gn,m) for n in b],old_lo=[Lo(res,n,gn,m) for n in b])
        v='YES' if max(r['new'])<=f*min(r['old']) and max(r['new'])<min(r['old_lo']) else 'NO' if min(r['new'])>f*max(r['old']) else 'OVERLAP';return v,r
    def worse(gn,m):
        best=min((Q78,A75),key=lambda ns:mean(ns,gn,m));r=dict(new=[W(res,n,gn,m) for n in S79],baseline=best,old=[W(res,n,gn,m) for n in best])
        v='YES' if min(r['new'])>=g_*max(r['old']) else 'NO' if max(r['new'])<g_*min(r['old']) else 'OVERLAP';return v,r
    far,rf=better('BND_FAR','con_coarse');near,rn=better('BND_NEAR','con_coarse')
    reg={f'{gn}:{m}':worse(gn,m) for gn,ms in dc['regression'].items() for m in ms if gn in strong}
    sec={f'BND_FAR:{m}':better('BND_FAR',m)[0] for m in dc['secondary_metrics']}
    regY=[k for k,(v,_) in reg.items() if v=='YES'];regO=[k for k,(v,_) in reg.items() if v=='OVERLAP']
    if far=='YES' and not regY:v='SYMMETRY TRAINING BENEFIT'
    elif far=='YES':v='INCONCLUSIVE'
    elif far=='NO':v='EQUIVARIANCE IMPROVED, TRANSFER UNCHANGED'
    else:v='INCONCLUSIVE'
    q=(['far benefit with material regression: '+', '.join(regY)] if regY and far=='YES' else [])+(['with regression: '+', '.join(regY)] if regY and far!='YES' else [])+ \
      (['possible regression (overlap): '+', '.join(regO)] if regO and not regY else [])+(['LOCAL BENEFIT (BND_NEAR contrast)'] if near=='YES' else [])+(['BND_FAR seed overlap'] if far=='OVERLAP' else [])
    gates={n:{gn:res[n]['groups'][gn]['Q']['passed'] for gn in res[n]['groups'] if not res[n]['groups'][gn].get('unmeasured')} for n in Q78+A75+AR+S79}
    return dict(verdict=v,qualifiers=q,far=far,near=near,secondary=sec,regression={k:v_ for k,(v_,_) in reg.items()},valid=True,measured=meas,
                stronger_control={k:v_[0][:3] for k,v_ in strong.items()},WL59_gates=gates,values=dict(BND_FAR=rf,BND_NEAR=rn,regression={k:r for k,(_,r) in reg.items()}))

def stage5():
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage5/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    g3=json.loads((ROOT/'stage3/gate.json').read_text());g4=json.loads((ROOT/'mech/gate.json').read_text())
    assert (g3['passed'] and g4['improved']) or (dev() and os.environ.get('R79_DEV_FORCE')=='1'),'Stage 5 not authorised by the fit / mechanism gates'
    t0=time.perf_counter();G=E73.geoms(RG);rows=json.loads((ROOT/'ports79.json').read_text())['ports'];P={int(k):v['port'] for k,v in rows.items() if v['port'] is not None}
    bm=FM.bin_map('cuda');R={}
    for s,p in P.items():
        f=ROOT/'refs/bundles'/f'rest_s{s}.pt'
        if not f.exists():assert dev();continue
        b=load(f);assert b['port']==p and b['K']==4;R[('rest',s)]=E.Ref(('rest',s),b['S'],b['B16'],(1-G['rest']['beta'][p]).float(),G['rest']['parent'],bm,E.ball16(G['rest'],p,bm));del b
    GR={k:[x_ for x_ in v if x_ in R] for k,v in groups5(rows).items()}
    if dev():GR={k:(v if len(v)>=4 else list(R)) for k,v in GR.items()}
    subs={k:E.Ref(k,r.S[1:],r.B16[1:],r.d,r.parent,bm,r.ball16) for k,r in R.items()}
    pc=E73.CFG['precision'];prec={k:E72.sigSE(r) for k,r in R.items()};qual={k:min(v)>=pc['abs_min'] for k,v in prec.items()}
    lights=load(INP68/'wl59/lights.pt')['test'];res={};from r57_eval import physics
    for n in Q78+A75+AR+S79:
        m,res[n]=E78.score(n,G,P,R,{},qual,{},GR,subs)
        if n in S79:res[n]['physics']={'rest':physics(m,G['rest'],lights[:2].cuda(),m.coefficients(G['rest']))}
        del m;torch.cuda.empty_cache();print('EVAL5',n,round(time.perf_counter()-t0),flush=True)
    dec=decide5(res)
    write_new(out,json.loads(json.dumps(dict(results=res,decision=dec,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},qualified={f'{a}_s{b}':v for (a,b),v in qual.items()},
        precision={f'{a}_s{b}':v for (a,b),v in prec.items()},seconds=time.perf_counter()-t0),default=float)))
    print('STAGE5_DONE',dec['verdict'],json.dumps(dec.get('qualifiers')),round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ Stage 6 geometry change
def stage6():
    out=ROOT/'stage6/deformation.json'
    if out.exists():raise FileExistsError('immutable')
    s3=json.loads((ROOT/'stage3/analysis.json').read_text())['results'];m4=json.loads((ROOT/'mech/mech.json').read_text())['part0_dev'];dc=CFG['stage6']
    rows={}
    def take(src,n,dn):
        d=src[n]['deformation'].get(dn,{})
        if d.get('unmeasured') or not d:return None
        return dict(err_hi=[c['hi'] for c in d['pooled']['abs_coarse']],corr=[c['corr'] for c in d['corr_slope']],slope=[c.get('slope') for c in d['corr_slope']],WL59=d['Q']['passed'])
    for dn in ('DEF71','CONFDEF72','DEF73'):rows[dn]={n:take(s3,n,dn) for n in Q78+A75+S79}
    rows['DEF_part0_te1']={n:take(m4,n,'DEF') for n in Q78+A75+AR+S79}
    flag={}
    for dn,r in rows.items():
        if any(r[n] is None for n in Q78+S79):flag[dn]='UNMEASURED';continue
        worst=lambda n:max(r[n]['err_hi']);wc=lambda n:min(r[n]['corr'])
        worse=all(worst(c)>=dc['error_factor']*max(worst(q) for q in Q78) or wc(c)<=min(wc(q) for q in Q78)-dc['corr_drop'] for c in S79)
        better=all(worst(c)<=max(worst(q) for q in Q78)/dc['error_factor'] and wc(c)>=max(wc(q) for q in Q78) for c in S79)
        flag[dn]='WORSE' if worse else 'RECOVERED' if better else 'MAINTAINED/OVERLAP'
    regression=flag.get(dc['primary'])=='WORSE'
    write_new(out,dict(rows=rows,flags=flag,deformation_regression=regression,rule=dc));print('STAGE6_DONE',json.dumps(flag),regression,flush=True)

if __name__=='__main__':
    {'stage3':stage3,'mech':mech,'stage5':stage5,'stage6':stage6}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
