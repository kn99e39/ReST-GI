"""WL73 evaluation (eval68 estimators unchanged): training-source fidelity (replicas 4-7), stratified fresh confirmation
(INT / BND (TBE, TBC, SM, SR) / UNS (SV) static on rest; te0 footprint static and te0 - rest deformation), development groups (labelled),
per-source rows with distances to the training population, physical / model flux profiles, WL59 full fields, passivity, labelled
fixed-A/O oracle, cost; matched S16 vs S16-R classification per architecture.
    python eval73.py stage1 | main"""
import sys,os,json,time
from pathlib import Path
import torch
from common73 import CFG,ROOT,R68,R71,R72,INP68,write_new,environment,dev
import family68 as FM
import family71 as FW
import eval68 as E
import eval72 as E72

HOME=Path.home();RADII=(.005,.01,.02,.04)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
FZ=CFG.get('frozen',{});POSE=CFG['stage0']['deformation_pose']['selected']

def geoms(RG):
    G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0','cf3')}
    G['cf2']=RG.device(load(R71/'inputs/geometry_cf2.pt'));G['te2']=RG.device(load(R72/'inputs/geometry_te2.pt'));G[POSE]=RG.device(load(ROOT/'inputs'/f'geometry_{POSE}.pt'))
    return G

def slot_ports():
    P={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    P.update({int(k):v['port'] for k,v in json.loads((R72/'ports.json').read_text())['slots'].items()})
    r=json.loads((ROOT/'ports.json').read_text())
    for part in ('new','confirmation'):P.update({int(k):v['port'] for k,v in r[part].items()})
    return P

NEW=sorted(int(s) for s in FZ.get('new_ports',{}));REMOVED=FZ.get('removed_slots',[]);KEPT=[s for s in range(16) if s not in REMOVED]
CONF=sorted(int(s) for s in FZ.get('confirmation_ports',{}))
SPECS=CFG['stage0']['confirmation_static']
def ctx_slots(*cs):return [s['slot'] for s in SPECS if s['context'] in cs and s['stratum'] in ('near','far')]
def strat_slots(st,*cs):return [s['slot'] for s in SPECS if s['context'] in cs and s['stratum']==st]
REM_NEAR=[s['slot'] for s in SPECS if s['stratum']=='near_removed']
FOOT=[s['slot'] for s in CFG['stage0']['footprint']['ports']]

def load_refs(G,P,dev_only=False):
    bm=FM.bin_map('cuda');R={}
    def ref(g,s,f,reps=None):
        b=load(f);S=b['S'] if reps is None else b['S'][reps];B=b['B16'] if reps is None else b['B16'][reps];i=P[s]
        if 'port' in b:assert b['port']==i,(g,s)
        return E.Ref((g,s),S,B,(1-G[g]['beta'][i]).float(),G[g]['parent'],bm,E.ball16(G[g],i,bm))
    ev=[4,5,6,7]
    for g in ('rest','pt0'):
        for s in range(16):
            f=INP68/'fit'/f'{g}_s{s}.pt' if s<4 else R68/'heldout/bundles'/f'{g}_s{s}.pt' if s<8 else R71/'refs/bundles'/f'{g}_s{s}.pt'
            R[(g,s)]=ref(g,s,f,ev)
        if not dev_only:
            for s in NEW:R[(g,s)]=ref(g,s,ROOT/'refs/bundles'/f'{g}_s{s}.pt',ev)
    for g in ('rest','cf2'):
        for s in range(20,28):R[(g,s)]=ref(g,s,R71/'refs/bundles'/f'{g}_s{s}.pt')
    for s in range(4,8):R[('cf3',s)]=ref('cf3',s,R68/'heldout/bundles'/f'cf3_s{s}.pt')
    for g in ('rest','te2'):
        for s in range(40,48):R[(g,s)]=ref(g,s,R72/'refs/bundles'/f'{g}_s{s}.pt')
    if not dev_only:
        for s in CONF:R[('rest',s)]=ref('rest',s,ROOT/'refs/bundles'/f'rest_s{s}.pt')
        for s in FOOT:R[(POSE,s)]=ref(POSE,s,ROOT/'refs/bundles'/f'{POSE}_s{s}.pt')
    D={}
    dl=[('DEF71','cf2',range(20,28)),('CONFDEF72','te2',range(40,48))]+([] if dev_only else [('DEF73',POSE,FOOT)])
    for name,g,slots in dl:
        for s in slots:
            a,b=R[(g,s)],R[('rest',s)];K=min(a.K,b.K);D[(name,s)]=E.Ref((name,s),a.S[:K]-b.S[:K],a.B16[:K]-b.B16[:K],a.d,a.parent,bm,a.ball16-b.ball16)
    return R,D

def groups(dev_only=False):
    G={'S16_rest':[('rest',s) for s in range(16)],'S16_pt0':[('pt0',s) for s in range(16)],'FIN71_rest':[('rest',s) for s in range(20,28)],
       'FIN71_cf2':[('cf2',s) for s in range(20,28)],'H3_cf3':[('cf3',s) for s in range(4,8)],'CONF72_rest':[('rest',s) for s in range(40,48)],'CONF72_te2':[('te2',s) for s in range(40,48)]}
    if dev_only:return G
    for g in ('rest','pt0'):
        G[f'TRAIN_KEEP_{g}']=[(g,s) for s in KEPT];G[f'TRAIN_REMOVED_{g}']=[(g,s) for s in REMOVED];G[f'TRAIN_NEW_{g}']=[(g,s) for s in NEW]
    r=lambda sl:[('rest',s) for s in sl]
    G.update({'BND':r(ctx_slots('TBE','TBC','SM','SR')),'BND_NEAR':r(strat_slots('near','TBE','TBC','SM','SR')),'BND_FAR':r(strat_slots('far','TBE','TBC','SM','SR')),
              'UNS':r(ctx_slots('SV')),'INT_DED':r(ctx_slots('INT')),'FP_rest':r(FOOT),'INT_ALL':r(ctx_slots('INT')+FOOT),f'FP_{POSE}':[(POSE,s) for s in FOOT],
              'SIDE':r(ctx_slots('SM','SR')),'TOPBOT_EDGE':r(ctx_slots('TBE','TBC')),'REM_NEAR':r(REM_NEAR)})
    return G

def candidate(name):
    """Historical (WL71 D1, WL72 A/B) through eval72.candidate; WL73 runs (D1R_/AR_ W+UV, BR_ RSK) from ROOT/runs."""
    import fit71,rsk72
    from runtime import Transport
    if not name.startswith(('D1R_','AR_','BR_')):return E72.candidate(name)
    if name.startswith('BR_'):
        a,o=fit71.angular();m=rsk72.RSK(a,o).cuda();m.load_state_dict(load(ROOT/'runs'/f'{name}.pt')['model']);m.eval()
        return m,(lambda g,ports:m.columns(g,ports)),'RSK'
    m=Transport().cuda();m.load_state_dict(load(ROOT/'runs'/f'{name}.pt')['model']);m.eval()
    return m,(lambda g,ports:FW.network_columns(m,g,ports)),'W+UV'

def ref_profile(r,i,x0):
    f=r.S.double().mean(0).sum((0,2,3));d=(x0-x0[i]).norm(dim=1).double().cpu();return [float(f[d<=R].sum()/f.sum()) for R in RADII]

def evaluate(name,G,P,R,D,qual,dq,lights,GR,full=True):
    m,cols,kind=candidate(name);row=dict(kind=kind,groups={},per_matrix={},deformation={},flux={})
    with torch.no_grad():
        hs={};cg={}
        for (g,s),r in R.items():
            T,A,O=cols(G[g],[P[s]]);h=FM.predict(T,(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0];hs[(g,s)]=h
            Sg,Bg=r.gpu();cg[(g,s)]=E.cross(r,Sg,Bg,h);del Sg,Bg
            row['per_matrix'][f'{g}_s{s}']={k:[c['hi'] for c in v] for k,v in E.pooled([r],[cg[(g,s)]],[E.STD[0],E.STD[1],E.STD[2],E.STD[5]]).items()}
            x0=G[g]['rest_xyz'];f=torch.einsum('na,rac,njrc->nj',(1-G[g]['beta'][[P[s]]]).float(),A,T)[0];rr=(x0-x0[P[s]]).norm(dim=1)
            row['flux'][f'{g}_s{s}']=[float(f[rr<=R_].sum()/f.sum()) for R_ in RADII]
        for gn,keys in GR.items():
            keys=[k for k in keys if qual[k]]
            if len(keys)<4:row['groups'][gn]=dict(unmeasured=True,matrices=len(keys));continue
            p=E.pooled([R[k] for k in keys],[cg[k] for k in keys],E.STD)
            row['groups'][gn]=dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys],corr_slope=E.corr_slope([R[k] for k in keys],[cg[k] for k in keys]))
        for dn,g in (('DEF71','cf2'),('CONFDEF72','te2'),('DEF73',POSE)):
            keys=[k for k in D if k[0]==dn and dq[k]]
            if dn=='DEF73' and not full:continue
            if len(keys)<4:row['deformation'][dn]=dict(unmeasured=True,matrices=len(keys));continue
            dcg=[]
            for k in keys:
                r=D[k];Sg,Bg=r.gpu();dcg.append(E.cross(r,Sg,Bg,hs[(g,k[1])]-hs[('rest',k[1])]));del Sg,Bg
            p=E.pooled([D[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D[k] for k in keys],dcg)
            row['deformation'][dn]=dict(pooled=p,corr_slope=cs,Q=Qdef(p,cs),matrices=[f's{k[1]}' for k in keys],
                per_matrix={f's{k[1]}':dict(err=[c['hi'] for c in E.pooled([D[k]],[c_],E.STD[2:3])['abs_coarse']],corr=[c['corr'] for c in E.corr_slope([D[k]],[c_])]) for k,c_ in zip(keys,dcg)})
        if not full:return row
        G3={g:G[g] for g in ('rest','pt0','cf3')};from r57_eval import physics
        row['passivity']=E.passivity(m,G3,lights)
        for g in ('cf2','te2',POSE):row['passivity'][g]=physics(m,G[g],lights[:2].cuda(),m.coefficients(G[g]))
        row['extended_fields']=E.extended(m,G3,lights)
        torch.cuda.synchronize();t0=time.perf_counter();n,co=E72.state_floats(m,kind,G['rest']);torch.cuda.synchronize();t1=time.perf_counter()
        for i in range(0,64,4):m.apply(G['rest'],lights[i:i+4].cuda(),co)
        torch.cuda.synchronize();t2=time.perf_counter()
        allp=sum(p.numel() for p in m.parameters());ao=m.in_logits.numel()+m.out_logits.numel()
        row['cost']=dict(learned_parameters_excl_fixed_AO=allp-ao,fixed_AO_logits=ao,runtime_state_floats=n,coefficient_build_seconds=t1-t0,apply_64_lights_seconds=t2-t1)
        src=(ROOT/'runs'/f'{name}.pt') if name.startswith(('D1R_','AR_','BR_')) else (R71/'runs'/f'{name}.pt') if name.startswith('D1_') else (R72/'runs'/f'{name}.pt')
        if '@' not in name and src.exists():
            rr_=load(src);row['run']={k:rr_.get(k) for k in ('J_initial','J_final','loss_final','L_sym_final','converged','last_decile_relative_change','seconds','peak_GPU_allocated','concurrent','trainable')}
            row['run']['history_J']=[(h['step'],h['J']) for h in rr_['history'] if h['step']%500==0]
    return row

def Qdef(p,cs):
    th=CFG['criteria']['deformation'];chk={c:dict(err=p['abs_coarse'][c]['hi']<=th['abs_coarse_hi'],corr=cs[c]['corr']>=th['corr_min']) for c in range(3)}
    return dict(per_channel=chk,passed=all(all(v.values()) for v in chk.values()))

def oracle(G,P,R,keys,iters=1500):
    """LABELLED ORACLE (eval72.oracle, unchanged arithmetic): admissible T for the fixed A/O fitted on the first replica, scored on the others."""
    import fit71
    if 'R73_DEV_ORACLE_ITERS' in os.environ:assert dev();iters=int(os.environ['R73_DEV_ORACLE_ITERS'])
    a,o=fit71.angular();A=torch.softmax(a,0).double().cuda();O=torch.softmax(o,1).double().cuda();bm=FM.bin_map('cuda');Hc=E.Hm().cuda();subs=[];cgs=[]
    proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
    for k in keys:
        r=R[k];K=r.K;fh,eh=[0],list(range(1,K));tg=FM.Target(r.S[fh].double().mean(0)[None].cuda(),r.d[None].cuda(),r.parent,bm)
        def f(T,tg=tg):
            v=sum((FM.predict(T,tg.d,A,O,sp,tg.parent,bm)-tg.S[sp]).square().sum((1,2,3))/tg.den[sp] for sp in E72.SP)/3
            zc=FM.predict(T,tg.d,A,O,'coarse',tg.parent,bm)/tg.d[:,:,None,None,None]
            return (v+(torch.einsum('ab,nbqoc->naqoc',Hc,zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']).sum()
        x,_=FM.fista(f,torch.full((1,r.S.shape[2],8,3),.5/r.S.shape[2],dtype=torch.float64,device='cuda'),proj,iters)
        h=FM.predict(x,tg.d,A,O,'native')[0].float();sub=E.Ref(r.key,r.S[eh],r.B16[eh],r.d,r.parent,bm,r.ball16);Sg,Bg=sub.gpu()
        subs.append(sub);cgs.append(E.cross(sub,Sg,Bg,h));del Sg,Bg,tg;torch.cuda.empty_cache()
    return E.pooled(subs,cgs,[('con_coarse','con','coarse'),('abs_coarse','abs','coarse'),('abs_fine16','abs','fine16')])

ARCH={'D1':(['D1_5901','D1_5902'],['D1R_5901','D1R_5902']),'A':(['A_5901','A_5902'],['AR_5901','AR_5902']),'B':(['B_7201','B_7202'],['BR_7201','BR_7202'])}
def decide(res):
    """Predeclared per-architecture classification of the matched S16 -> S16-R intervention (protocol 'decision'). Comparisons are
    three-valued: YES only if every seed clears the bound, NO only if the seeds clearly fail it, otherwise OVERLAP (seed variance)."""
    dc=CFG['decision'];W=lambda n,g,m:max(c['hi'] for c in res[n]['groups'][g]['pooled'][m]);M=('con_coarse','abs_coarse');f,dg=dc['improve_factor'],dc['degrade_factor']
    meas=lambda g:all(not res[n]['groups'][g].get('unmeasured') for v in ARCH.values() for n in v[0]+v[1])
    prim=('BND_FAR','BND_NEAR','INT_ALL','REM_NEAR','TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0')
    out={'measured':{g:meas(g) for g in prim+('BND','UNS','INT_DED','FP_rest')}}
    if not all(out['measured'][g] for g in prim):out['verdict']='INCONCLUSIVE (primary group unmeasured)';return out
    def better(g,old,new):
        r={m:dict(new=[W(n,g,m) for n in new],old=[W(n,g,m) for n in old]) for m in M}
        if all(max(r[m]['new'])<=f*min(r[m]['old']) for m in M):v='YES'
        elif any(min(r[m]['new'])>f*max(r[m]['old']) for m in M):v='NO'
        else:v='OVERLAP'
        return v,r
    def worse(g,old,new):
        r={m:dict(new=[W(n,g,m) for n in new],old=[W(n,g,m) for n in old]) for m in M}
        if any(min(r[m]['new'])>=dg*max(r[m]['old']) for m in M):v='YES'
        elif all(max(r[m]['new'])<dg*min(r[m]['old']) for m in M):v='NO'
        else:v='OVERLAP'
        return v,r
    cls={}
    for X,(old,new) in ARCH.items():
        valid=all(all(v['passed'] for v in res[n]['passivity'].values()) for n in old+new)
        q_fit=all(res[n]['groups'][f'TRAIN_NEW_{g}']['Q']['passed'] for n in new for g in ('rest','pt0'))
        rel_fit=all(W(n,f'TRAIN_NEW_{g}',m)<=dg*W(n,f'TRAIN_KEEP_{g}',m) for n in new for g in ('rest','pt0') for m in M)
        fit_ok=q_fit or rel_fit
        no_deficit=all(res[n]['groups']['BND_FAR']['Q']['passed'] for n in old)
        far,rf=better('BND_FAR',old,new);near,rn=better('BND_NEAR',old,new);acc=all(res[n]['groups']['BND_FAR']['Q']['passed'] for n in new)
        dI,ri=worse('INT_ALL',old,new);dR,rr=worse('REM_NEAR',old,new);deg='YES' in (dI,dR);degq='OVERLAP' in (dI,dR)
        if not valid:c=['INCONCLUSIVE (implementation / passivity gate failed)']
        elif not fit_ok:c=['INSUFFICIENT FIT']
        elif no_deficit:c=['NO SUPPORT DEFICIT (S16 models already pass the WL59 gates on BND_FAR)']
        elif far=='OVERLAP':c=['INCONCLUSIVE (seed variance on BND_FAR)']
        elif far=='YES':c=['TRADE-OFF / REDISTRIBUTION'] if deg else ['SUPPORT-SENSITIVE GENERALIZATION'] if acc else ['SUPPORT-SENSITIVE GENERALIZATION (partial)','SUPPORT INSUFFICIENT (accuracy)']
        elif near=='YES':c=['SUPPORT-SENSITIVE (local only, BND_NEAR)','SUPPORT INSUFFICIENT (transfer to BND_FAR)']+(['TRADE-OFF / REDISTRIBUTION (regression)'] if deg else [])
        else:c=['SUPPORT INSUFFICIENT']+(['TRADE-OFF / REDISTRIBUTION (regression)'] if deg else [])
        if degq and not deg:c=c+['possible regression (seed overlap)']
        cls[X]=dict(classes=c,valid=valid,fit_ok=fit_ok,fit_basis='Q' if q_fit else 'relative' if rel_fit else 'none',no_support_deficit=no_deficit,
            improves_BND_FAR=far,improves_BND_NEAR=near,accurate_BND_FAR=acc,degrades_INT_ALL=dI,degrades_REM_NEAR=dR,
            values=dict(BND_FAR=rf,BND_NEAR=rn,INT_ALL=ri,REM_NEAR=rr,UNS={m:dict(new=[W(n,'UNS',m) for n in new],old=[W(n,'UNS',m) for n in old]) for m in M} if out['measured']['UNS'] else None))
    out['per_architecture']=cls
    out['same_budget_ranking_BND_FAR']={f'{x}>{y}':better('BND_FAR',ARCH[y][1],ARCH[x][1])[0] for x in ARCH for y in ARCH if x!=y}
    return out

def stage1():
    """Reproduction gate: the evaluation path reproduces WL72's archived group bounds for the historical models on the archived references."""
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'stage1/eval_identity.json'
    if out.exists():raise FileExistsError('immutable')
    G=geoms(RG);P={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    P.update({int(k):v['port'] for k,v in json.loads((R72/'ports.json').read_text())['slots'].items()})
    R,D=load_refs(G,P,True);GR=groups(True);arc=json.loads((R72/'eval/analysis.json').read_text())['results'];rows={};worst=0.
    qual={k:min(E72.sigSE(r))>=CFG['precision']['abs_min'] for k,r in R.items()};dq={k:min(E72.sigSE(r))>=CFG['precision']['deformation_min'] for k,r in D.items()}
    for n in ('D1_5901','D1_5902','A_5901','A_5902','B_7201','B_7202'):
        r=evaluate(n,G,P,R,D,qual,dq,None,GR,full=False);d=0.
        for g in GR:
            ga=g if not g.startswith('CONF72') else g.replace('CONF72','CONF')
            if r['groups'][g].get('matrices')!=[m.replace('CONF72','CONF') for m in arc[n]['groups'][ga].get('matrices',[])]:
                mm=dict(model=n,group=g,wl73=r['groups'][g].get('matrices'),wl72=arc[n]['groups'][ga].get('matrices'))
                raise RuntimeError(f'STOP: qualified matrix set differs from WL72 {mm}')
            if r['deformation']['DEF71'].get('matrices')!=arc[n]['deformation']['DEF71'].get('matrices'):raise RuntimeError('STOP: DEF71 matrix set differs from WL72')
            for mtr,v in r['groups'][g]['pooled'].items():
                for c in range(3):a=arc[n]['groups'][g if not g.startswith('CONF72') else g.replace('CONF72','CONF')]['pooled'][mtr][c]['hi'];d=max(d,abs(v[c]['hi']-a)/max(abs(a),1e-12))
        a_=arc[n]['deformation']['DEF71']['pooled'];b_=r['deformation']['DEF71']['pooled']
        d=max([d]+[abs(b_[m][c]['hi']-a_[m][c]['hi'])/max(abs(a_[m][c]['hi']),1e-12) for m in a_ for c in range(3)]);rows[n]=d;worst=max(worst,d);print('S1',n,d,flush=True)
    ok=worst<=CFG['stage1']['eval_rel_tol'];write_new(out,dict(max_rel_diff=rows,worst=worst,tolerance=CFG['stage1']['eval_rel_tol'],passed=ok,groups=list(GR),
        note='same qualification rule as WL72 (abs signal/SE >= 10, deformation >= 3); qualified matrix lists asserted equal to WL72 per group and for DEF71 (WL72 needed >= 1 there, WL73 >= 4; WL72 had 4/8); CONFDEF72 UNMEASURED in both'))
    if not ok:raise RuntimeError('STOP: evaluation does not reproduce WL72')

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'eval/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=geoms(RG);P=slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=load_refs(G,P);GR=groups();print('REFS',round(time.perf_counter()-t0),flush=True)
    prec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=CFG['precision']['abs_min'] for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=CFG['precision']['deformation_min'] for k in D}
    prof={f'{g}_s{s}':ref_profile(r,P[s],G[g]['rest_xyz']) for (g,s),r in R.items() if s>=50 or s in (0,2,5,14)}
    hist=['D1_5901','D1_5902','A_5901','A_5902','B_7201','B_7202'];new=[n for v in ARCH.values() for n in v[1]]
    names=hist+new+([] if dev() else [f'{n}@{c}' for n in new for c in CFG['optimizer']['checkpoints']])   # dev smoke: finals only
    res={}
    for n in names:
        res[n]=evaluate(n,G,P,R,D,qual,dq,lights,GR);torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    dec=decide({n:res[n] for n in hist+new})
    orc={gn:oracle(G,P,R,[k for k in GR[gn] if qual[k]]) for gn in ('BND_NEAR','BND_FAR','UNS','INT_DED','REM_NEAR','TRAIN_NEW_rest')}
    write_new(out,json.loads(json.dumps(dict(results=res,decision=dec,oracle_fixed_AO=orc,reference_flux_profile=dict(radii=RADII,values=prof),
        precision=dict(abs=prec,deformation=dprec,qualified={f'{a}_s{b}':v for (a,b),v in qual.items()},deformation_qualified={f'{a}_s{b}':v for (a,b),v in dq.items()}),
        groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('EVAL_DONE',json.dumps({k:v.get('classes') for k,v in dec.get('per_architecture',{}).items()}) or dec.get('verdict'),round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':{'stage1':stage1,'main':main}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
