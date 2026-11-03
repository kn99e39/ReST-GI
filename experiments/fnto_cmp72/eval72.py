"""WL72 evaluation (eval68 estimators unchanged): development groups (WL68/WL71, labelled) and the fresh confirmation groups
(rest/te2 slots 40-47, signed te2 - rest), per-source errors and 1 cm flux, WL59 full fields, passivity, labelled oracle, cost; decision."""
import sys,os,json,time
from pathlib import Path
import torch
from common72 import CFG,ROOT,R68,R71,INP68,write_new,environment
import family68 as FM
import family71 as FW
import eval68 as E

HOME=Path.home();SP=('native','fine16','coarse')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def geoms(RG):
    G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0','cf3')}
    G['cf2']=RG.device(load(R71/'inputs/geometry_cf2.pt'));G['te2']=RG.device(load(ROOT/'inputs/geometry_te2.pt'));return G

def slot_ports():
    P={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    P.update({int(k):v['port'] for k,v in json.loads((ROOT/'ports.json').read_text())['slots'].items()});return P

def load_refs(G,P):
    bm=FM.bin_map('cuda');R={}
    def ref(g,s,f,reps=None):
        b=load(f);S=b['S'] if reps is None else b['S'][reps];B=b['B16'] if reps is None else b['B16'][reps];i=P[s]
        return E.Ref((g,s),S,B,(1-G[g]['beta'][i]).float(),G[g]['parent'],bm,E.ball16(G[g],i,bm))
    ev=[4,5,6,7]
    for g in ('rest','pt0'):
        for s in range(16):
            f=INP68/'fit'/f'{g}_s{s}.pt' if s<4 else R68/'heldout/bundles'/f'{g}_s{s}.pt' if s<8 else R71/'refs/bundles'/f'{g}_s{s}.pt'
            R[(g,s)]=ref(g,s,f,ev)
    for g in ('rest','cf2'):
        for s in range(20,28):R[(g,s)]=ref(g,s,R71/'refs/bundles'/f'{g}_s{s}.pt')
    for s in range(4,8):R[('cf3',s)]=ref('cf3',s,R68/'heldout/bundles'/f'cf3_s{s}.pt')
    for g in ('rest','te2'):
        for s in range(40,48):R[(g,s)]=ref(g,s,ROOT/'refs/bundles'/f'{g}_s{s}.pt')
    D={}
    for name,g,slots in (('DEF71','cf2',range(20,28)),('CONFDEF','te2',range(40,48))):
        for s in slots:
            a,b=R[(g,s)],R[('rest',s)];K=min(a.K,b.K);D[(name,s)]=E.Ref((name,s),a.S[:K]-b.S[:K],a.B16[:K]-b.B16[:K],a.d,a.parent,bm,a.ball16-b.ball16)
    return R,D

def sigSE(r):
    h=torch.zeros(16,r.S.shape[2],64,3,device='cuda');Sg,Bg=r.gpu();cg=E.cross(r,Sg,Bg,h);del Sg,Bg
    return [c['signal_over_SE'] for c in E.pooled([r],[cg],[('abs_coarse','abs','coarse')])['abs_coarse']]

GROUPS={'S16_rest':[('rest',s) for s in range(16)],'S16_pt0':[('pt0',s) for s in range(16)],'FIN71_rest':[('rest',s) for s in range(20,28)],
        'FIN71_cf2':[('cf2',s) for s in range(20,28)],'H3_cf3':[('cf3',s) for s in range(4,8)],'CONF_rest':[('rest',s) for s in range(40,48)],'CONF_te2':[('te2',s) for s in range(40,48)]}

def candidate(name):
    """-> (model, columns(g, ports), kind)."""
    import fit71,rsk72
    from runtime import Transport
    if name=='init_P5902':m=fit71.model_from('P_5902').eval()
    elif name.startswith(('A_','D1_','R71A')):
        m=Transport().cuda();src={'R71A_2000':HOME/'rest_drift_smoke/out/ckpt_R_02000.pt'}.get(name) or (R71/'runs'/f'{name}.pt' if name.startswith('D1_') else ROOT/'runs'/f'{name}.pt')
        m.load_state_dict(load(src)['model']);m.eval()
    else:
        a,o=fit71.angular();m=rsk72.RSK(a,o).cuda();m.load_state_dict(load(ROOT/'runs'/f'{name}.pt')['model']);m.eval()
        return m,(lambda g,ports:m.columns(g,ports)),'RSK'
    return m,(lambda g,ports:FW.network_columns(m,g,ports)),'W+UV'

def Qdef(p,cs):
    th=CFG['criteria']['deformation'];chk={c:dict(err=p['abs_coarse'][c]['hi']<=th['abs_coarse_hi'],corr=cs[c]['corr']>=th['corr_min']) for c in range(3)}
    return dict(per_channel=chk,passed=all(all(v.values()) for v in chk.values()))

def state_floats(m,kind,g):
    with torch.no_grad():co=m.coefficients(g)
    return int(sum(v.numel() for k,v in co.items() if k in ('W','U','V','WL','WF'))),co

def evaluate(name,G,P,R,D,qual,dq,lights):
    m,cols,kind=candidate(name);row=dict(kind=kind,groups={},per_matrix={},deformation={},flux={})
    with torch.no_grad():
        hs={};cg={}
        for (g,s),r in R.items():
            T,A,O=cols(G[g],[P[s]]);h=FM.predict(T,(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0];hs[(g,s)]=h
            Sg,Bg=r.gpu();cg[(g,s)]=E.cross(r,Sg,Bg,h);del Sg,Bg
            row['per_matrix'][f'{g}_s{s}']={k:[c['hi'] for c in v] for k,v in E.pooled([r],[cg[(g,s)]],[E.STD[0],E.STD[1],E.STD[2],E.STD[5]]).items()}
            x0=G[g]['rest_xyz'];f=torch.einsum('na,rac,njrc->nj',(1-G[g]['beta'][[P[s]]]).float(),A,T)[0];rr=(x0-x0[P[s]]).norm(dim=1)
            row['flux'][f'{g}_s{s}']=float(f[rr<=.01].sum()/f.sum())
        for gn,keys in GROUPS.items():
            keys=[k for k in keys if qual[k]]
            if len(keys)<4:row['groups'][gn]=dict(unmeasured=True,matrices=len(keys));continue
            p=E.pooled([R[k] for k in keys],[cg[k] for k in keys],E.STD)
            row['groups'][gn]=dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys],corr_slope=E.corr_slope([R[k] for k in keys],[cg[k] for k in keys]))
        for dn,g in (('DEF71','cf2'),('CONFDEF','te2')):
            keys=[k for k in D if k[0]==dn and dq[k]]
            if len(keys)<(4 if dn=='CONFDEF' else 1):row['deformation'][dn]=dict(unmeasured=True,matrices=len(keys));continue
            dcg=[]
            for k in keys:
                r=D[k];Sg,Bg=r.gpu();dcg.append(E.cross(r,Sg,Bg,hs[(g,k[1])]-hs[('rest',k[1])]));del Sg,Bg
            p=E.pooled([D[k] for k in keys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([D[k] for k in keys],dcg)
            row['deformation'][dn]=dict(pooled=p,corr_slope=cs,Q=Qdef(p,cs),matrices=[f's{k[1]}' for k in keys],
                per_matrix={f's{k[1]}':dict(err=[c['hi'] for c in E.pooled([D[k]],[c_],E.STD[2:3])['abs_coarse']],corr=[c['corr'] for c in E.corr_slope([D[k]],[c_])]) for k,c_ in zip(keys,dcg)})
        G3={g:G[g] for g in ('rest','pt0','cf3')};from r57_eval import physics
        row['passivity']=E.passivity(m,G3,lights)
        for g in ('cf2','te2'):row['passivity'][g]=physics(m,G[g],lights[:2].cuda(),m.coefficients(G[g]))
        row['extended_fields']=E.extended(m,G3,lights)
        torch.cuda.synchronize();t0=time.perf_counter();n,co=state_floats(m,kind,G['rest']);torch.cuda.synchronize();t1=time.perf_counter()
        for i in range(0,64,4):m.apply(G['rest'],lights[i:i+4].cuda(),co)
        torch.cuda.synchronize();t2=time.perf_counter()
        row['cost']=dict(parameters=sum(p.numel() for k,p in m.named_parameters() if k not in ('in_logits','out_logits') or kind=='W+UV'),
                         runtime_state_floats=n,coefficient_build_seconds=t1-t0,apply_64_lights_seconds=t2-t1)
        for k in ('J_final','loss_final','L_sym_final','converged','seconds','peak_GPU_allocated','concurrent','trainable'):
            if name.startswith(('A_','B_')) and '@' not in name:row[k]=load(ROOT/'runs'/f'{name}.pt').get(k)
    return row

def oracle(G,P,R,keys,iters=1500):
    import fit71
    if 'R72_DEV_ORACLE_ITERS' in os.environ:assert ROOT!=HOME/'rest_cmp72';iters=int(os.environ['R72_DEV_ORACLE_ITERS'])
    a,o=fit71.angular();A=torch.softmax(a,0).double().cuda();O=torch.softmax(o,1).double().cuda();bm=FM.bin_map('cuda');Hc=E.Hm().cuda();subs=[];cgs=[]
    proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
    for k in keys:
        r=R[k];K=r.K;fh,eh=[0],list(range(1,K));tg=FM.Target(r.S[fh].double().mean(0)[None].cuda(),r.d[None].cuda(),r.parent,bm)
        def f(T,tg=tg):
            v=sum((FM.predict(T,tg.d,A,O,sp,tg.parent,bm)-tg.S[sp]).square().sum((1,2,3))/tg.den[sp] for sp in SP)/3
            zc=FM.predict(T,tg.d,A,O,'coarse',tg.parent,bm)/tg.d[:,:,None,None,None]
            return (v+(torch.einsum('ab,nbqoc->naqoc',Hc,zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']).sum()
        x,_=FM.fista(f,torch.full((1,r.S.shape[2],8,3),.5/r.S.shape[2],dtype=torch.float64,device='cuda'),proj,iters)
        h=FM.predict(x,tg.d,A,O,'native')[0].float();sub=E.Ref(r.key,r.S[eh],r.B16[eh],r.d,r.parent,bm,r.ball16);Sg,Bg=sub.gpu()
        subs.append(sub);cgs.append(E.cross(sub,Sg,Bg,h));del Sg,Bg,tg;torch.cuda.empty_cache()
    return E.pooled(subs,cgs,[('con_coarse','con','coarse'),('abs_coarse','abs','coarse'),('abs_fine16','abs','fine16')])

def decide(res):
    w=lambda r,g,m:max(c['hi'] for c in r['groups'][g]['pooled'][m])
    meas=all(not res[n]['groups'][g].get('unmeasured') for n in res for g in ('CONF_rest','CONF_te2')) and all(not res[n]['deformation']['CONFDEF'].get('unmeasured') for n in res)
    if not meas:return dict(verdict='INCONCLUSIVE',reason='confirmation groups unmeasured')
    c=lambda n:max(w(res[n],g,'con_coarse') for g in ('CONF_rest','CONF_te2'));a=lambda n:max(w(res[n],g,'abs_coarse') for g in ('CONF_rest','CONF_te2'))
    e=lambda n:max(x['hi'] for x in res[n]['deformation']['CONFDEF']['pooled']['abs_coarse'])
    cD=min(c(n) for n in ('D1_5901','D1_5902'));aD=min(a(n) for n in ('D1_5901','D1_5902'))
    arms={'A':['A_5901','A_5902'],'B':['B_7201','B_7202']}
    valid={k:all(all(v['passed'] for v in res[n]['passivity'].values()) and res[n].get('J_final') is not None for n in ns) for k,ns in arms.items()}
    imp={k:all(c(n)<=.8*cD and a(n)<=.8*aD for n in ns) for k,ns in arms.items()}
    def sup(x,y):X,Y=arms[x],arms[y];return max(c(n) for n in X)<=.8*min(c(n) for n in Y) and max(a(n) for n in X)<=.8*min(a(n) for n in Y) and max(e(n) for n in X)<=1.2*min(e(n) for n in Y)
    S={'A':sup('A','B'),'B':sup('B','A')}
    if not all(valid.values()):v='INCOMPARABLE'
    elif imp['A'] and imp['B']:v='A SUPERIOR WITHIN TESTED SCOPE' if S['A'] else 'B SUPERIOR WITHIN TESTED SCOPE' if S['B'] else 'BOTH IMPROVE WITHOUT CLEAR WINNER'
    elif imp['A'] or imp['B']:
        k='A' if imp['A'] else 'B';v=f'{k} SUPERIOR WITHIN TESTED SCOPE' if S[k] else 'INCONCLUSIVE'
    else:v='NEITHER IMPROVES'
    return dict(verdict=v,valid=valid,improves=imp,superior=S,c={n:c(n) for n in res if n in sum(arms.values(),['D1_5901','D1_5902'])},
                a={n:a(n) for n in res if n in sum(arms.values(),['D1_5901','D1_5902'])},e={n:e(n) for n in res if n in sum(arms.values(),['D1_5901','D1_5902'])},cD=cD,aD=aD)

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False);out=ROOT/'eval/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();G=geoms(RG);P=slot_ports();lights=load(INP68/'wl59/lights.pt')['test'];R,D=load_refs(G,P);print('REFS',round(time.perf_counter()-t0),flush=True)
    prec={f'{k[0]}_s{k[1]}':sigSE(r) for k,r in R.items()};dprec={f'{k[0]}_s{k[1]}':sigSE(r) for k,r in D.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=10 for k in R};dq={k:min(dprec[f'{k[0]}_s{k[1]}'])>=3 for k in D}
    names=['D1_5901','D1_5902','init_P5902','R71A_2000']+[f'{x}{s}' for x in ('A_5901','A_5902','B_7201','B_7202') for s in ('@2000','@6000','')]
    res={}
    for n in names:
        res[n]=evaluate(n,G,P,R,D,qual,dq,lights);torch.cuda.empty_cache();print('EVAL',n,round(time.perf_counter()-t0),flush=True)
    finals={n:res[n] for n in ('D1_5901','D1_5902','A_5901','A_5902','B_7201','B_7202')};dec=decide(finals)
    conf=[k for k in R if k[1]>=40 and qual[k]];orc={g:oracle(G,P,R,[k for k in conf if k[0]==g]) for g in ('rest','te2')}
    write_new(out,json.loads(json.dumps(dict(results=res,decision=dec,oracle_fixed_AO=orc,precision=dict(abs=prec,deformation=dprec,qualified={f'{a}_s{b}':v for (a,b),v in qual.items()},
        deformation_qualified={f'{a}_s{b}':v for (a,b),v in dq.items()}),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()),default=float)))
    print('EVAL_DONE',dec['verdict'],round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
