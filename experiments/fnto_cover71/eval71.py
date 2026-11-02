"""WL71 evaluation: unbiased cross-replica statistics (eval68 estimators unchanged) on the fit populations (evaluation replicas 4-7),
the final held-out groups (FIN_rest, FIN_cf2, DEF_cf2), labelled development groups (cf3), noise-free teacher generalization,
the labelled angular-transfer oracle, passivity and WL59 extended fields; predeclared decision. Trains nothing but the oracle."""
import sys,os,json,time
from pathlib import Path
import numpy as np
import torch
from common71 import CFG,ROOT,INP68,R68,TRAIN_GEOMS,write_new,save_new,environment
import family68 as FM
import family71 as FW
import eval68 as E

SP=('native','fine16','coarse')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def geoms(RG):
    G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0','cf3')};G['cf2']=RG.device(load(ROOT/'inputs/geometry_cf2.pt'));return G

def slot_ports():
    P={int(k):v['port'] for k,v in json.loads((ROOT/'ports.json').read_text())['slots'].items()}
    wl68=json.loads((R68/'heldout/ports.json').read_text())['ports'];fit=json.loads((INP68/'wl66_ports.json').read_text())['ports']
    for s in fit+wl68:assert P[s['slot']]==s['port']
    return P

def ref(G,g,s,P,S,B16,key=None):
    bm=FM.bin_map('cuda');i=P[s];return E.Ref(key or (g,s),S,B16,(1-G[g]['beta'][i]).float(),G[g]['parent'],bm,E.ball16(G[g],i,bm))

def load_refs(G,P):
    refs={};ev=[4,5,6,7]
    for g in TRAIN_GEOMS:
        for s in range(16):
            f=INP68/'fit'/f'{g}_s{s}.pt' if s<4 else R68/'heldout/bundles'/f'{g}_s{s}.pt' if s<8 else ROOT/'refs/bundles'/f'{g}_s{s}.pt'
            b=load(f);refs[(g,s)]=ref(G,g,s,P,b['S'][ev],b['B16'][ev]);del b
    for g in ('rest','cf2'):
        for s in CFG['references']['final']['slots']:
            b=load(ROOT/'refs/bundles'/f'{g}_s{s}.pt');refs[(g,s)]=ref(G,g,s,P,b['S'],b['B16']);del b
    for s in range(8):
        b=load(R68/'heldout/bundles'/f'cf3_s{s}.pt');refs[('cf3',s)]=ref(G,'cf3',s,P,b['S'],b['B16']);del b
    return refs

def delta_refs(refs):
    bm=FM.bin_map('cuda');out={}
    for s in CFG['references']['final']['slots']:
        a,b=refs[('cf2',s)],refs[('rest',s)];K=min(a.K,b.K)
        out[('DEF',s)]=E.Ref(('DEF',s),a.S[:K]-b.S[:K],a.B16[:K]-b.B16[:K],a.d,a.parent,bm,a.ball16-b.ball16)
    return out

def signal_over_SE(r):
    """Coarse collided signal / jackknife SE per channel (candidate-independent)."""
    h=torch.zeros(16,r.S.shape[2],64,3,device='cuda');Sg,Bg=r.gpu();cg=E.cross(r,Sg,Bg,h);del Sg,Bg
    return [c['signal_over_SE'] for c in E.pooled([r],[cg],[('abs_coarse','abs','coarse')])['abs_coarse']]

# ------------------------------------------------------------------ candidates
def net(spec):
    from runtime import Transport
    m=Transport().cuda()
    p={'P_5901':INP68/'wl59/P_5901.pt','P_5902':INP68/'wl59/P_5902.pt'}.get(spec) or (R68/'runs'/f"{spec[5:]}.pt" if spec.startswith('wl68_') else ROOT/'runs'/f'{spec}.pt')
    m.load_state_dict(load(p)['model']);return m.eval()
def candidates():
    c={n:('P_5901' if n=='frozen_5901' else 'P_5902' if n=='frozen_5902' else n) for n in CFG['candidates_evaluated']};return c

def predict(m,G,g,ports,states):
    if g not in states:states[g]=m.state(G[g])
    T,A,O=FW.network_columns(m,G[g],ports,states[g]);return FM.predict(T,(1-G[g]['beta'][ports]).float(),A,O,'native')

def groups(qual,dqual):
    q=lambda k:qual.get(k,True)
    gr={'S4_rest':[('rest',s) for s in range(4)],'S4_pt0':[('pt0',s) for s in range(4)],
        'S16_rest':[('rest',s) for s in range(16)],'S16_pt0':[('pt0',s) for s in range(16)],
        'S16not4_rest':[('rest',s) for s in range(4,16)],'S16not4_pt0':[('pt0',s) for s in range(4,16)],
        'FIN_rest':[('rest',s) for s in range(20,28)],'FIN_cf2':[('cf2',s) for s in range(20,28)],
        'DEV_cf3_fitports':[('cf3',s) for s in range(4)],'DEV_cf3_wl68ports':[('cf3',s) for s in range(4,8)]}
    return {n:[k for k in v if q(k)] for n,v in gr.items()}

def Qdef(p,cs):
    th=CFG['criteria']['deformation']
    chk={c:dict(err=p['abs_coarse'][c]['hi']<=th['abs_coarse_hi'],corr=cs[c]['corr']>=th['corr_min']) for c in range(3)}
    return dict(per_channel=chk,passed=all(all(v.values()) for v in chk.values()))

def evaluate(name,spec,G,P,refs,drefs,gr,dkeys,lights):
    m=net(spec);states={};cg={};hs={};row=dict(groups={},per_matrix={})
    for key,r in refs.items():
        h=predict(m,G,key[0],[P[key[1]]],states)[0];Sg,Bg=r.gpu();cg[key]=E.cross(r,Sg,Bg,h);del Sg,Bg
        if key[1]>=20:hs[key]=h
        row['per_matrix'][f'{key[0]}_s{key[1]}']={k:[c['hi'] for c in v] for k,v in E.pooled([r],[cg[key]],E.STD[:3]+E.STD[5:6]).items()}
    for gname,keys in gr.items():
        if not keys:continue
        p=E.pooled([refs[k] for k in keys],[cg[k] for k in keys],E.STD)
        row['groups'][gname]=dict(pooled=p,Q=E.Q(p),matrices=[f'{k[0]}_s{k[1]}' for k in keys],corr_slope=E.corr_slope([refs[k] for k in keys],[cg[k] for k in keys]))
    dcg=[]
    for s in dkeys:
        dr=drefs[('DEF',s)];Sg,Bg=dr.gpu();dcg.append(E.cross(dr,Sg,Bg,hs[('cf2',s)]-hs[('rest',s)]));del Sg,Bg
    if dkeys:
        p=E.pooled([drefs[('DEF',s)] for s in dkeys],dcg,E.STD[:3]+E.STD[6:]);cs=E.corr_slope([drefs[('DEF',s)] for s in dkeys],dcg)
        row['groups']['DEF_cf2']=dict(pooled=p,corr_slope=cs,Q=Qdef(p,cs),matrices=[f'DEF_s{s}' for s in dkeys])
        row['deformation_per_matrix']={f'DEF_s{s}':dict(pooled={k:[c['hi'] for c in v] for k,v in E.pooled([drefs[('DEF',s)]],[c],E.STD[2:3]).items()},corr_slope=E.corr_slope([drefs[('DEF',s)]],[c])) for s,c in zip(dkeys,dcg)}
    if spec not in ('P_5901','P_5902'):
        r=load(ROOT/'runs'/f'{spec}.pt') if not spec.startswith('wl68_') else load(R68/'runs'/f'{spec[5:]}.pt')
        row.update(J_final=r['J_final'],converged=r['converged'],seconds=r['seconds'],peak_GPU_allocated=r['peak_GPU_allocated'],concurrent=r.get('concurrent'))
    G3={g:G[g] for g in ('rest','pt0','cf3')};from r57_eval import physics
    row['passivity']=E.passivity(m,G3,lights);row['extended_fields']=E.extended(m,G3,lights)
    with torch.no_grad():row['passivity']['cf2']=physics(m,G['cf2'],lights[:2].cuda(),m.coefficients(G['cf2']))
    return row,m

# ------------------------------------------------------------------ teacher generalization (noise-free)
def teacher_eval(G,P):
    import fit71
    out={};rng=np.random.default_rng(71064);excl=set(P[s] for s in list(range(16))+list(range(20,28)))
    R64=sorted(rng.choice([i for i in range(G['rest']['P']) if i not in excl],64,replace=False).tolist())
    sets={'train_S4':[P[s] for s in range(4)],'train_S16':[P[s] for s in range(16)],'FIN':[P[s] for s in range(20,28)],'R64':R64}
    bm=FM.bin_map('cuda')
    def resp(m,g,ports,st):
        T,A,O=FW.network_columns(m,G[g],ports,st);return FM.predict(T,(1-G[g]['beta'][ports]).float(),A,O,'native').double()
    def errs(m,t,g,ports,sm,stt):
        """Relative L2 over all listed sources/bins (accumulated per chunk) and the coarse responses for the deformation."""
        num={k:0. for k in ('abs_native','abs_fine16','abs_coarse','con_coarse')};den=dict(num);cy=[];ct=[]
        for c0 in range(0,len(ports),4):
            pp=ports[c0:c0+4];y=resp(m,g,pp,sm);tt=resp(t,g,pp,stt);d=(1-G[g]['beta'][pp]).double()
            for sp in SP:
                a=FM.aggregate(y,sp,G[g]['parent'],bm);b=FM.aggregate(tt,sp,G[g]['parent'],bm)
                num[f'abs_{sp}']=num[f'abs_{sp}']+(a-b).square().sum((0,1,2,3));den[f'abs_{sp}']=den[f'abs_{sp}']+b.square().sum((0,1,2,3))
                if sp=='coarse':
                    Hd=lambda z:torch.einsum('ab,nbqoc->naqoc',E.Hm().cuda(),z/d[:,:,None,None,None])
                    num['con_coarse']=num['con_coarse']+(Hd(a)-Hd(b)).square().sum((0,1,2,3));den['con_coarse']=den['con_coarse']+Hd(b).square().sum((0,1,2,3))
                    cy.append(a.cpu());ct.append(b.cpu())
            del y,tt
        return {k:(num[k]/den[k]).sqrt().tolist() for k in num},torch.cat(cy),torch.cat(ct)
    for rid,v in CFG['runs'].items():
        if v['target']=='phys':continue
        t=fit71.teacher(v['target']);res={}
        for stage in ('initial','final'):
            m=fit71.model_from(v['init']).eval() if stage=='initial' else net(rid);per={}
            with torch.no_grad():
                for sname,ports in sets.items():
                    geos=('rest','pt0') if sname.startswith('train') else ('rest','cf2')
                    ys={};ts={}
                    for g in geos:
                        per[f'{sname}/{g}'],ys[g],ts[g]=errs(m,t,g,ports,m.state(G[g]),t.state(G[g]))
                    if not sname.startswith('train'):
                        dy=ys['cf2']-ys['rest'];dt=ts['cf2']-ts['rest']
                        e=[];co=[]
                        for c in range(3):
                            x=dt[...,c].flatten();y=dy[...,c].flatten();e.append(float((x-y).norm()/x.norm()));xc=x-x.mean();yc=y-y.mean();co.append(float(xc@yc/(xc.norm()*yc.norm())))
                        per[f'{sname}/DEF']=dict(abs_coarse=e,corr=co)
            res[stage]=per;torch.cuda.empty_cache()
        th=CFG['criteria']['teacher'];fin=res['final'];ini=res['initial']
        def ok(keys):return all(fin[k]['con_coarse'][c]<=th['con_coarse'] and fin[k]['abs_coarse'][c]<=th['abs_coarse'] and fin[k]['abs_fine16'][c]<=th['abs_fine16'] for k in keys for c in range(3))
        def okd(k):return all(fin[k]['abs_coarse'][c]<=th['deformation_abs_coarse'] and fin[k]['corr'][c]>=th['deformation_corr'] for c in range(3))
        tr=[f"train_{v['pop']}/{g}" for g in ('rest','pt0')]
        informative={k:any(ini[k]['con_coarse'][c]>=th['informative_con_coarse'] or ini[k]['abs_coarse'][c]>=th['informative_abs_coarse'] for c in range(3)) for k in ini if 'DEF' not in k}
        run=load(ROOT/'runs'/f'{rid}.pt')
        out[rid]=dict(errors=res,training_recovery=ok(tr),generalization_FIN=ok(['FIN/rest','FIN/cf2']),generalization_R64=ok(['R64/rest','R64/cf2']),
            deformation_FIN=okd('FIN/DEF'),deformation_R64=okd('R64/DEF'),informative=informative,converged=run['converged'],J_final=run['J_final'],seconds=run['seconds'])
        out[rid]['generalizes']=out[rid]['generalization_FIN'] and out[rid]['generalization_R64'] and out[rid]['deformation_FIN'] and out[rid]['deformation_R64']
    return out,R64

# ------------------------------------------------------------------ oracle
def oracle(A,O,G,refs,keys,iters):
    """LABELLED ORACLE: admissible spatial T for fixed A/O fitted on final replica 0, scored on replicas 1-3 (eval68 rule)."""
    bm=FM.bin_map('cuda');A=A.double().cuda();O=O.double().cuda();Hc=E.Hm().cuda();subs=[];cgs=[]
    proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
    for k in keys:
        r=refs[k];K=r.K;nf=K//2 if K>=6 else 1;fh,eh=list(range(nf)),list(range(nf,K))
        tg=FM.Target(r.S[fh].double().mean(0)[None].cuda(),r.d[None].cuda(),r.parent,bm)
        def f(T,tg=tg):
            v=sum((FM.predict(T,tg.d,A,O,sp,tg.parent,bm)-tg.S[sp]).square().sum((1,2,3))/tg.den[sp] for sp in SP)/3
            zc=FM.predict(T,tg.d,A,O,'coarse',tg.parent,bm)/tg.d[:,:,None,None,None]
            return (v+(torch.einsum('ab,nbqoc->naqoc',Hc,zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']).sum()
        x,_=FM.fista(f,torch.full((1,r.S.shape[2],8,3),.5/r.S.shape[2],dtype=torch.float64,device='cuda'),proj,iters)
        h=FM.predict(x,tg.d,A,O,'native')[0].float();sub=E.Ref(r.key,r.S[eh],r.B16[eh],r.d,r.parent,bm,r.ball16);Sg,Bg=sub.gpu()
        subs.append(sub);cgs.append(E.cross(sub,Sg,Bg,h));del Sg,Bg,tg;torch.cuda.empty_cache()
    return E.pooled(subs,cgs,[('con_coarse','con','coarse'),('abs_coarse','abs','coarse'),('abs_fine16','abs','fine16')])

# ------------------------------------------------------------------ decision
def decide(res,teacher,qual_ok):
    d=CFG['decision'];prim=min(('D1_5901','D1_5902'),key=lambda r:res[r]['J_final']);r=res[prim];gq=lambda n:r['groups'].get(n,{}).get('Q',{}).get('passed',False)
    S_fit=gq('S16_rest') and gq('S16_pt0');S_cross=gq('FIN_rest');S_state=gq('FIN_cf2');S_def=gq('DEF_cf2')
    C=res['C_'+prim.split('_')[1]]
    def worst(x,g,m):return max(c['hi'] for c in x['groups'][g]['pooled'][m])
    improves=all(worst(r,g,m)<=.8*worst(C,g,m) for g in ('FIN_rest','FIN_cf2') for m in ('con_coarse','abs_coarse'))
    tr=all(teacher[t]['training_recovery'] for t in ('TA_S16','TB_S16'));adequate=bool(r['converged']) and tr and qual_ok
    gen=all(teacher[t]['generalizes'] for t in ('TA_S16','TB_S16'))
    outcome='A' if S_fit and S_cross and S_state and S_def else 'B' if improves else 'C' if adequate else 'D'
    label=outcome+('/teacher-generalizes' if gen else '/teacher-fails') if outcome=='C' else outcome
    return dict(primary=prim,S_fit=S_fit,S_cross=S_cross,S_state=S_state,S_def=S_def,improves=improves,adequate=adequate,primary_converged=r['converged'],
                teacher_training_recovery=tr,teacher_generalizes=gen,references_qualified=qual_ok,outcome=outcome,label=label,
                statement=CFG['decision']['family_statement'])

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False)
    out=ROOT/'eval/analysis.json'
    if out.exists():raise FileExistsError('analysis is immutable')
    t0=time.perf_counter();G=geoms(RG);P=slot_ports();lights=load(INP68/'wl59/lights.pt')['test']
    refs=load_refs(G,P);drefs=delta_refs(refs);print('REFS',round(time.perf_counter()-t0),flush=True)
    th=CFG['precision'];prec={f'{k[0]}_s{k[1]}':signal_over_SE(r) for k,r in refs.items()};dprec={f'DEF_s{k[1]}':signal_over_SE(r) for k,r in drefs.items()}
    qual={k:min(prec[f'{k[0]}_s{k[1]}'])>=th['abs_matrix_signal_over_SE'] for k in refs};dkeys=[k[1] for k in drefs if min(dprec[f'DEF_s{k[1]}'])>=th['deformation_matrix_signal_over_SE']]
    gr=groups(qual,None);qual_ok=len(gr['FIN_rest'])>=4 and len(gr['FIN_cf2'])>=4 and len(dkeys)>=4
    print('PRECISION',qual_ok,len(dkeys),round(time.perf_counter()-t0),flush=True)
    res={};models={}
    for name,spec in candidates().items():
        res[name],m=evaluate(name,spec,G,P,refs,drefs,gr,dkeys,lights);del m;torch.cuda.empty_cache();print('EVAL',name,round(time.perf_counter()-t0),flush=True)
    teacher,R64=teacher_eval(G,P);print('TEACHER',round(time.perf_counter()-t0),flush=True)
    dec=decide(res,teacher,qual_ok)
    import fit71
    A,O=[torch.softmax(x.float(),d) for x,d in zip(fit71.angular(),(0,1))]
    orc={g:oracle(A.cpu(),O.cpu(),G,refs,gr[g],int(os.environ.get('R71_DEV_ORACLE_ITERS',CFG['oracle'].get('iters',1500)))) for g in ('FIN_rest','FIN_cf2')}
    print('ORACLE',round(time.perf_counter()-t0),flush=True)
    write_new(out,dict(results=res,teacher=teacher,R64_ports=R64,decision=dec,oracle_fixed_AO=orc,
        precision=dict(abs_signal_over_SE=prec,deformation_signal_over_SE=dprec,qualified={f'{k[0]}_s{k[1]}':v for k,v in qual.items()},deformation_qualified=dkeys,final_qualified=qual_ok),
        groups={k:[f'{a}_s{b}' for a,b in v] for k,v in gr.items()},seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()))
    print('EVAL_DONE',dec['label'],round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
