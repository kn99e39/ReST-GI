"""WL74 Stage 4 (conditional) evaluation: learned A with the G1 basis (A74_*) vs the frozen WL73 controls (AR_*) on the fresh Stage 4
confirmation population and on the WL73 groups (development), with the eval73 evaluator unchanged (eval68 estimators); DEF4 (pose - rest),
passivity on the Stage 4 pose, cost; labelled oracle (G0 and the chosen G1 basis) on the Stage 4 groups for the spatial-gap attribution.
    python eval74.py"""
import sys,os,json,time
import torch
from common74 import CFG,ROOT,R73,INP68,write_new,environment,dev
import family68 as FM
import family71 as FW
import eval68 as E
import eval72 as E72
import eval73 as E73
import ato74 as AT

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
ST=CFG['stage4'];POSE=ST['pose']['selected']
def sl(kind,st=None):return [s['slot'] for s in ST['static'] if s['context'] in kind and (st is None or s['stratum']==st)]
FP=[s['slot'] for s in ST['footprint']]

_cand=E73.candidate
def candidate(name):
    if name.startswith('A74_'):
        from runtime import Transport
        m=Transport().cuda();m.load_state_dict(load(ROOT/'runs'/f'{name}.pt')['model']);m.eval()
        return m,(lambda g,ports:FW.network_columns(m,g,ports)),'W+UV'
    return _cand(name)
E73.candidate=candidate

def oracle(bid,G,P,refs,keys):
    """Labelled oracle with a frozen basis: T fitted on replica 0, scored on replicas 1-3 (as ato74.stage2)."""
    b=load(ROOT/'bases'/f'{bid}.pt');A,O=(x.detach().double().cuda() for x in AT.AO(b['in_logits'],b['out_logits']));M=A.shape[0];bm=FM.bin_map('cuda')
    iters=CFG['oracle']['stage2_iters'] if 'R74_DEV_ITERS' not in os.environ else int(os.environ['R74_DEV_ITERS']);hs={};subs={}
    for k in keys:
        g,s=k;r=refs[k];tg=AT.target(r.S[[0]].double().cuda(),(1-G[g]['beta'][[P[s]]]).double(),G[g]['parent'],bm)
        T,_=AT.fista_T(tg,A,O,torch.full((1,G[g]['P'],M,3),.5/G[g]['P'],dtype=torch.float64,device='cuda'),iters)
        hs[k]=AT.predict_m(T.float(),tg.d.float(),A.float(),O.float(),'native')[0].detach()
        subs[k]=E.Ref(k,r.S[1:],r.B16[1:],r.d,r.parent,r.bm,r.ball16);del tg,T;torch.cuda.empty_cache()
    with torch.no_grad():cg,per=AT.score(E,subs,keys,hs)
    return cg,per,subs

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'eval4/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=E73.geoms(RG);G[POSE]=RG.device(load(ROOT/'inputs'/f'geometry_{POSE}.pt'));P=E73.slot_ports()
    P4={int(k):v['port'] for k,v in json.loads((ROOT/'ports4.json').read_text())['ports'].items()};P.update(P4)
    R,D=E73.load_refs(G,P);bm=FM.bin_map('cuda')
    for g,slots in (('rest',sl('TBE TBC SM SR INT')+FP),(POSE,FP)):
        for s in slots:
            b=load(ROOT/'refs/bundles'/f'{g}_s{s}.pt');assert b['port']==P[s]
            R[(g,s)]=E.Ref((g,s),b['S'],b['B16'],(1-G[g]['beta'][P[s]]).float(),G[g]['parent'],bm,E.ball16(G[g],P[s],bm))
    D4={}
    for s in FP:
        a,b=R[(POSE,s)],R[('rest',s)];D4[('DEF4',s)]=E.Ref(('DEF4',s),a.S-b.S,a.B16-b.B16,a.d,a.parent,bm,a.ball16-b.ball16)
    GR={k:v for k,v in E73.groups().items() if k in CFG['stage4']['wl73_groups']}
    r=lambda s:[('rest',x) for x in s]
    GR.update({'BND4_NEAR':r(sl('TBE TBC SM SR','near')),'BND4_FAR':r(sl('TBE TBC SM SR','far')),'INT4':r(sl('INT')+FP),f'FP4_{POSE}':[(POSE,s) for s in FP]})
    prec={f'{k[0]}_s{k[1]}':E72.sigSE(x) for k,x in R.items()};dprec={f'{k[0]}_s{k[1]}':E72.sigSE(x) for k,x in {**D,**D4}.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=10 for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=3 for k in {**D,**D4}}
    names=['AR_5901','AR_5902','A74_5901','A74_5902']+([] if dev() else [f'A74_{s}@{c}' for s in ('5901','5902') for c in CFG['stage3']['optimizer']['checkpoints']])
    lights=load(INP68/'wl59/lights.pt')['test'];res={}
    for n in names:
        row=E73.evaluate(n,G,P,R,D,qual,dq,lights,GR,full=True);m,cols,_=candidate(n);keys=[k for k in D4 if dq[k]]
        if len(keys)>=4:
            dcg=[]
            for k in keys:
                T,A,O=cols(G[POSE],[P[k[1]]]);hp=FM.predict(T,(1-G[POSE]['beta'][[P[k[1]]]]).float(),A,O,'native')[0]
                T,A,O=cols(G['rest'],[P[k[1]]]);hr=FM.predict(T,(1-G['rest']['beta'][[P[k[1]]]]).float(),A,O,'native')[0]
                x=D4[k];Sg,Bg=x.gpu();dcg.append(E.cross(x,Sg,Bg,hp-hr));del Sg,Bg
            p=E.pooled([D4[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D4[k] for k in keys],dcg);row['deformation']['DEF4']=dict(pooled=p,corr_slope=cs,Q=E73.Qdef(p,cs))
        else:row['deformation']['DEF4']=dict(unmeasured=True,matrices=len(keys))
        from r57_eval import physics
        row['passivity'][POSE]=physics(m,G[POSE],lights[:2].cuda(),m.coefficients(G[POSE]))
        if n.startswith('A74_') and '@' not in n:
            x=load(ROOT/'runs'/f'{n}.pt');row['run']={k:x.get(k) for k in ('J_initial','J_final','loss_final','L_sym_final','converged','seconds','peak_GPU_allocated','trainable','basis')}
        res[n]=row;torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    dec=json.loads((ROOT/'decision.json').read_text());orc={}
    for bid in ('G0_r0',dec['chosen_basis'] or 'G1_r0'):
        keys=[k for gn in ('BND4_NEAR','BND4_FAR','INT4') for k in GR[gn] if qual[k]];cg,per,subs=oracle(bid,G,P,R,keys)
        orc[bid]=dict(per_matrix=per,groups={gn:AT.pooled(E,subs,cg,[k for k in GR[gn] if qual[k]]) for gn in ('BND4_NEAR','BND4_FAR','INT4')})
    write_new(out,json.loads(json.dumps(dict(results=res,decision=decide4(res),oracle=orc,groups={k:[f'{a}_s{b}' for a,b in v] for k,v in GR.items()},
        precision=dict(abs=prec,deformation=dprec),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('EVAL4_DONE',round(time.perf_counter()-t0),flush=True)

def decide4(res):
    """Predeclared (protocol stage4.decision): A74 (G1 basis) vs AR (WL68 basis), three-valued as WL73 (f = 0.8, g = 1.25)."""
    W=lambda n,g,m:max(c['hi'] for c in res[n]['groups'][g]['pooled'][m]);M=('con_coarse','abs_coarse');old=['AR_5901','AR_5902'];new=['A74_5901','A74_5902']
    meas={g:all(not res[n]['groups'][g].get('unmeasured') for n in old+new) for g in ('BND4_NEAR','BND4_FAR','INT4','TRAIN_KEEP_rest','TRAIN_NEW_rest')}
    if not all(meas.values()):return dict(measured=meas,classes=['INCONCLUSIVE (unmeasured group)'])
    def cmp(g,f,worse=False):
        r={m:dict(new=[W(n,g,m) for n in new],old=[W(n,g,m) for n in old]) for m in M}
        if not worse:
            if all(max(r[m]['new'])<=f*min(r[m]['old']) for m in M):return 'YES',r
            if any(min(r[m]['new'])>f*max(r[m]['old']) for m in M):return 'NO',r
        else:
            if any(min(r[m]['new'])>=f*max(r[m]['old']) for m in M):return 'YES',r
            if all(max(r[m]['new'])<f*min(r[m]['old']) for m in M):return 'NO',r
        return 'OVERLAP',r
    far,rf=cmp('BND4_FAR',.8);near,rn=cmp('BND4_NEAR',.8);dI,ri=cmp('INT4',1.25,True);dK,rk=cmp('TRAIN_KEEP_rest',1.25,True);reg='YES' in (dI,dK)
    valid=all(all(v['passed'] for v in res[n]['passivity'].values()) for n in old+new)
    if not valid:c=['INCONCLUSIVE (passivity)']
    elif far=='OVERLAP':c=['INCONCLUSIVE (seed variance on BND4_FAR)']
    elif far=='YES':c=['TRADE-OFF (benefit with interior regression)'] if reg else ['LEARNED-K BENEFIT (far boundary transfer)']
    elif near=='YES':c=['LOCAL BENEFIT ONLY (near, not far)']+(['with interior regression'] if reg else [])
    else:c=['NO LEARNED-K BENEFIT']+(['with interior regression'] if reg else [])
    return dict(measured=meas,classes=c,far=far,near=near,regression_INT4=dI,regression_TRAIN_KEEP=dK,values=dict(BND4_FAR=rf,BND4_NEAR=rn,INT4=ri,TRAIN_KEEP_rest=rk))

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
