"""WL75 Stage 4 analysis (descriptive; nothing is trained, selected or corrected here). Uses the Stage 2 / Stage 3 evaluation records and
geometry-only labels; the orientation part evaluates the frozen final models on exact slab-symmetry images (no reference needed).
A  learned vs source-specific oracle: per matrix (replicas 1-3 for both) AR vs G0 and A75 vs G2 (and A75 vs G0), per metric.
B  coarse-cell sub-position: each source's index inside its 1 cm coarse cell (rest material coordinates) -> center / edge / corner;
   per-matrix native / fine16 / coarse errors stratified by it (descriptive; small counts).
C  orientation: face and context labels; label-free symmetry invariants of the learned operators under the 15 exact non-identity slab
   operations (D4 x z-flip): I1 angle-integrated spatial field (exact under any symmetry), I2 sorted per-incoming-bin totals and
   I3 native contrast norm (exact when the operation permutes bins; checked by the sorted ballistic budget rows).
D  per-source error vs distance to the supervised sources (actual and their exact images).
E  checkpoint drift (2,000 / 6,000 / 10,000 / final) on training and Stage 4 groups (no selection).
    python analysis75.py"""
import sys,os,json,time
import numpy as np
import torch
from common75 import CFG,ROOT,R74,R73,INP68,write_new,environment,dev
import family68 as FM
import eval73 as E73
import ports73 as P73
import sym72 as SY
import rank75 as RK
import eval75 as EV

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
WM=lambda per,m:max(c['hi'] for c in per[m])

def subcell(g,i):
    x=g['rest_xyz'].double().numpy()*1e3;par=g['parent'].numpy();mem=np.where(par==par[i])[0];X=x[mem];idx=[];ext=0
    for ax in range(3):
        u=np.unique(np.round(X[:,ax],4))
        if len(u)<2:continue
        k=int(np.searchsorted(u,np.round(x[i,ax],4)));idx.append(dict(axis=ax,index=k,n=len(u)));ext+=k in (0,len(u)-1)
    return dict(cell=int(par[i]),members=len(mem),indices=idx,label={0:'center',1:'edge'}.get(ext,'corner'),offset_mm=[round(float(v),4) for v in x[i]-X.mean(0)])

def invariants(m,g,ports,perms,ops):
    """Per source i: relative discrepancy of I1/I2/I3 between i and each exact image perm[i] (model only)."""
    out={};beta=g['beta']
    with torch.no_grad():
        for i in ports:
            js=[int(p[i]) for p in perms];T,A,O=RK.columns(m,g,[i]+js) if isinstance(m,RK.TransportM) else __import__('family71').network_columns(m,g,[i]+js)
            d=(1-beta[[i]+js]).float();K=FM.predict(T,d,A,O,'native')   # (n,16,P,64,3)
            F=K.sum((1,3));e=K.sum((2,3)).sort(1).values;Kc=K-K.mean(1,keepdim=True);cn=Kc.square().sum((1,2,3)).sqrt()
            rows=[]
            for k,(op,p) in enumerate(zip(ops,perms)):
                pt=torch.as_tensor(p,device=F.device)
                I1=float((F[k+1][pt]-F[0]).norm()/F[0].norm());I2=float((e[k+1]-e[0]).norm()/e[0].norm());I3=float(((cn[k+1]-cn[0]).abs()/cn[0]).max())
                bp=float((beta[js[k]].sort().values-beta[i].sort().values).abs().max())
                rows.append(dict(op=f'{op[0]}{"+zflip" if op[1] else ""}',image=js[k],I1=I1,I2=I2,I3=I3,beta_multiset_maxdiff=bp,bins_permute=bp<=1e-6))
            out[str(i)]=rows;del K,F,Kc;torch.cuda.empty_cache()
    return out

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'analysis/analysis.json'
    if out.exists():raise FileExistsError('immutable')
    t0=time.perf_counter();s2=json.loads((ROOT/'stage2/analysis.json').read_text());p3=ROOT/'stage3/analysis.json';s3=json.loads(p3.read_text()) if p3.exists() else None
    gcpu=load(INP68/'wl59/geometry_rest.pt');sl=P73.Slab(gcpu);ops=[o for o in SY.ops(True) if o!=('id',False)];perms=[p for o,p in zip(SY.ops(True),sl.perms) if o!=('id',False)]
    import fit73
    S16R=[p for _,p in fit73.populations()['S16R']];trainR=sl.images(S16R);basis=CFG['basis']['selected']
    P4={int(k):v for k,v in json.loads((ROOT/'ports4.json').read_text())['ports'].items()} if (ROOT/'ports4.json').exists() else {}
    def labels(i):return dict(port=i,face=int(sl.face[i]),context=str(sl.ctx[i]),edge_mm=float(sl.e[i]),d_actual_mm=float(sl.dist([i],S16R)[0]),d_images_mm=float(sl.dist([i],trainR)[0]),subcell=subcell(gcpu,i))
    res=dict(labels=dict(train={str(s):labels(p) for s,p in fit73.populations()['S16R']},stage4={str(s):labels(v['port']) for s,v in P4.items()}))
    # A + B + D on Stage 4 (independent) matrices
    if s3:
        R3=s3['results'];O3=s3['oracle'];rows=[]
        for key in O3['G0_r0']['per_matrix']:
            g,s=key.split('_s');s=int(s);row=dict(matrix=key,**{k:v for k,v in res['labels']['stage4'][str(s)].items() if k!='port'},subcell_label=res['labels']['stage4'][str(s)]['subcell']['label'])
            for m in ('abs_native','abs_fine16','abs_coarse','con_native','con_fine16','con_coarse'):
                row[m]={n:WM(R3[n]['per_matrix_r123'][key],m) for n in EV.OLD+EV.NEW}
                row[m].update({b:WM(O3[b]['per_matrix'][key],m) for b in O3})
                ar=sum(row[m][n] for n in EV.OLD)/2;a7=sum(row[m][n] for n in EV.NEW)/2
                row[m]['gap_AR_minus_G0']=ar-row[m]['G0_r0'];row[m]['gap_A75_minus_G2']=a7-row[m][basis];row[m]['A75_over_AR']=a7/ar if ar else None
            rows.append(row)
        res['per_source_stage4']=rows
        def strat(key):
            out={}
            for lab in sorted({r[key] for r in rows}):
                rr=[r for r in rows if r[key]==lab];out[str(lab)]={m:dict(n=len(rr),AR=float(np.mean([np.mean([r[m][n] for n in EV.OLD]) for r in rr])),A75=float(np.mean([np.mean([r[m][n] for n in EV.NEW]) for r in rr])),
                    G0=float(np.mean([r[m]['G0_r0'] for r in rr])),G2=float(np.mean([r[m][basis] for r in rr]))) for m in ('abs_native','abs_fine16','abs_coarse','con_native','con_fine16','con_coarse')}
            return out
        res['by_subcell']=strat('subcell_label');res['by_face']=strat('face');res['by_context']=strat('context')
        res['distance']=[dict(matrix=r['matrix'],d_actual_mm=r['d_actual_mm'],d_images_mm=r['d_images_mm'],con_coarse=r['con_coarse'],abs_coarse=r['abs_coarse']) for r in rows]
        res['oracle_vs_learned_groups']={gn:{**{n:{m:WM(R3[n]['groups'][gn]['pooled'],m) for m in ('abs_coarse','abs_fine16','abs_native','con_coarse','con_fine16','con_native')} for n in EV.OLD+EV.NEW},
            **{b:{m:WM(O3[b]['groups'][gn]['pooled'],m) for m in ('abs_coarse','abs_fine16','abs_native','con_coarse','con_fine16','con_native')} for b in O3 if not O3[b]['groups'][gn].get('unmeasured')}}
            for gn in ('BND4_NEAR','BND4_FAR','INT4','ALL4_rest') if not R3['A75_5901']['groups'][gn].get('unmeasured')}
    # B on training sources (held-out replicas 4-7, Stage 2)
    R2=s2['results'];tr=[]
    for key,pm in R2['A75_5901']['per_matrix'].items():
        g,s=key.split('_s');s=int(s)
        if g not in ('rest','pt0') or str(s) not in res['labels']['train']:continue
        tr.append(dict(matrix=key,subcell=res['labels']['train'][str(s)]['subcell']['label'],context=res['labels']['train'][str(s)]['context'],
            **{m:{n:max(R2[n]['per_matrix'][key][m]) for n in EV.OLD+EV.NEW} for m in ('abs_native','abs_fine16','abs_coarse','con_coarse')}))
    res['per_source_train']=tr
    # C orientation invariants (finals), Stage 4 rest sources and S16-R training sources
    G=RG.device(gcpu);ori={}
    for n in EV.OLD+EV.NEW:
        m=EV.candidate(n)[0];ori[n]=dict(stage4=invariants(m,G,[v['port'] for v in P4.values()],perms,ops) if P4 else {},
                                         train=invariants(m,G,S16R,perms,ops));del m;torch.cuda.empty_cache();print('ORIENT',n,round(time.perf_counter()-t0),flush=True)
    summ={}
    for n,v in ori.items():
        for part,d in v.items():
            rr=[r for rows in d.values() for r in rows]
            if not rr:continue
            perm=[r for r in rr if r['bins_permute']]
            summ[f'{n}:{part}']=dict(n=len(rr),I1_median=float(np.median([r['I1'] for r in rr])),I1_max=float(max(r['I1'] for r in rr)),
                bins_permute=len(perm),I2_median=float(np.median([r['I2'] for r in perm])) if perm else None,I3_median=float(np.median([r['I3'] for r in perm])) if perm else None,
                I3_max=float(max(r['I3'] for r in perm)) if perm else None,
                by_op={o:dict(I1=float(np.median([r['I1'] for r in rr if r['op']==o])),I3=float(np.median([r['I3'] for r in rr if r['op']==o]))) for o in sorted({r['op'] for r in rr})})
    res['orientation']=dict(summary=summ,per_source=ori)
    # E drift
    dr={}
    for n in EV.NEW:
        for c in CFG['optimizer']['checkpoints']+['final']:
            k=n if c=='final' else f'{n}@{c}'
            if k in R2:dr[f'{k}:train']={g:{m:WM(R2[k]['groups'][g]['pooled'],m) for m in ('con_coarse','abs_coarse','abs_fine16')} for g in ('TRAIN_NEW_rest','TRAIN_KEEP_rest','BND_NEAR','BND_FAR','INT_ALL')}
            if s3 and k in s3['results']:dr[f'{k}:stage4']={g:{m:WM(s3['results'][k]['groups'][g]['pooled'],m) for m in ('con_coarse','abs_coarse','abs_fine16')} for g in ('BND4_NEAR','BND4_FAR','INT4') if not s3['results'][k]['groups'][g].get('unmeasured')}
    if s3:
        for n in EV.OLD:
            for c in CFG['optimizer']['checkpoints']+['final']:
                k=n if c=='final' else f'{n}@{c}'
                if k in s3['results']:dr[f'{k}:stage4']={g:{m:WM(s3['results'][k]['groups'][g]['pooled'],m) for m in ('con_coarse','abs_coarse','abs_fine16')} for g in ('BND4_NEAR','BND4_FAR','INT4') if not s3['results'][k]['groups'][g].get('unmeasured')}
    res['drift']=dr;res['seconds']=time.perf_counter()-t0
    write_new(out,json.loads(json.dumps(res,default=float)));print('ANALYSIS_DONE',round(time.perf_counter()-t0),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
