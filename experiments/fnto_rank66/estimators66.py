"""Pure WL66 estimator algebra (no IO). Every statistic is a function of replica cross-products, so per-matrix and
pooled statistics share one code path and delete-one-replica jackknife applies to all of them.

Notation (one matrix = geometry x port x channel x output space): Z_k (16 x Q) replica-k collided response divided by
the model budget d; C_k = H Z_k with H = I - 11^T/16. M[k,l] = Z_k Z_l^T. U-statistic means run over ordered pairs k != l.
"""
import itertools
import torch

NB=16
def H(n=NB):return torch.eye(n,dtype=torch.float64)-torch.full((n,n),1./n,dtype=torch.float64)

def ust(M,S):
    """Unbiased cross-Gram: mean over ordered pairs k!=l in S of M[k,l], symmetrised."""
    G=sum(M[k,l] for k in S for l in S if k!=l)/(len(S)*(len(S)-1));return .5*(G+G.transpose(-1,-2))
def ust_scalar(P,S):return sum(float(P[k,l]) for k in S for l in S if k!=l)/(len(S)*(len(S)-1))
def mean_over(X,S):return sum(X[k] for k in S)/len(S)
def meangram(M,S):G=sum(M[k,l] for k in S for l in S)/len(S)**2;return .5*(G+G.transpose(-1,-2))
def tail(G,r):
    """Sum of eigenvalues beyond the r largest = min over rank-r projectors of tr((I-P)G)."""
    ev=torch.linalg.eigvalsh(G);return float(ev[:max(len(ev)-r,0)].sum())
def upper(M,S,r):
    """Held-out residual of a subspace fitted on the other replica half: E[value | V] >= true tail (high-biased)."""
    S=sorted(S);h=len(S)//2;halves=(S[:h],S[h:]);vals=[]
    for fit,ev in (halves,halves[::-1]):
        if len(fit)<1 or len(ev)<2:continue
        V=torch.linalg.eigh(meangram(M,fit))[1][:,-r:];G=ust(M,ev);vals.append(float(torch.trace(G)-torch.trace(V.T@G@V)))
    return sum(vals)/len(vals)
def basis(X,tol=1e-10,cap=None):
    """Orthonormal basis of range(X) (n x m), rank by relative singular-value tolerance (optionally capped)."""
    U,s,_=torch.linalg.svd(X,full_matrices=False);r=int((s>tol*s[0]).sum());return U[:,:r if cap is None else min(r,cap)]

def frozen_bases(A,O,space_bin_map=None):
    """A (8,16,3), O (8,64,3) frozen checkpoint. Returns per channel: incoming projector Pi onto range(A^T), orthonormal TA
    of H range(A^T), and Otilde (8, n_out) aggregated outgoing lobes."""
    out=[]
    for c in range(3):
        At=A[:,:,c].T.double();Pi=At@torch.linalg.pinv(At)
        # 1 in range(A^T) (column-stochastic A) => dim H range(A^T) <= 7; the float32 softmax leaves a ~1e-7 spurious 8th direction.
        TA=basis(H()@At,tol=1e-5,cap=7)
        O_=O[:,:,c].double()
        if space_bin_map is not None:O_=torch.zeros(O_.shape[0],int(space_bin_map.max())+1,dtype=torch.float64).index_add_(1,space_bin_map,O_)
        out.append(dict(At=At,Pi=Pi,TA=TA,Ot=O_,one_in_range=float((torch.ones(NB,dtype=torch.float64)-Pi@torch.ones(NB,dtype=torch.float64)).norm()/4)))
    return out

def block_projector(vectors):
    """Orthogonal projector onto span of the given column vectors."""
    B=basis(vectors);return B@B.T

def summary(Z,d,Bp,R,Zhat,bhat,cells,n_out,bases,second=None,secondB=None,N=None):
    """Per-matrix sufficient statistics. Z,Bp,R: (K,16,Q) normalised collided, raw zero-collision, raw collided.
    Zhat: dict ck -> (16,Q) model scatter / d. bhat: (16,Q) model ball. bases: dict ck -> channel basis dict."""
    K=Z.shape[0];Hm=H().to(Z.device);D=torch.diag(d).to(Z.device)
    C=torch.einsum('ab,kbq->kaq',Hm,Z)
    s=dict(K=K,d=d.cpu())
    s['MZ']=torch.einsum('kaq,lbq->klab',Z,Z).cpu();s['MC']=torch.einsum('kaq,lbq->klab',C,C).cpu()
    s['MR']=torch.einsum('kaq,lbq->klab',R,R).cpu();s['MB']=torch.einsum('kaq,lbq->klab',Bp,Bp).cpu();s['MRB']=torch.einsum('kaq,lbq->klab',R,Bp).cpu()
    def outgoing(X):Y=X.reshape(K,NB,cells,n_out).permute(0,3,1,2).reshape(K,n_out,-1);return torch.einsum('koq,lpq->klop',Y,Y).cpu()
    s['OGC']=outgoing(C);s['OGZ']=outgoing(Z)
    if second is not None:
        v=((N*second-R.square())/(N-1)).clamp(min=0)/d[None,:,None].square().to(Z.device)
        s['noise_packet_rows']=v.sum(2).cpu()
        s['noise_packet_C']=torch.einsum('ab,kbq->k',Hm.square(),v).cpu()
        s['noise_sample_rows']=Z.var(0,unbiased=True).sum(1).cpu();s['noise_sample_C']=C.var(0,unbiased=True).sum().cpu()
        vb=((N*secondB-Bp.square())/(N-1)).clamp(min=0);s['noise_packet_B']=vb.sum((1,2)).cpu()
    s['ck']={}
    bb=bhat@bhat.T
    for ck,zh in Zhat.items():
        bs=bases[ck];zc=Hm@zh;e=dict(hh=(zh@zh.T).cpu(),hZ=torch.einsum('aq,kbq->kab',zh,Z).cpu(),
            hB=torch.einsum('aq,kbq->kab',zh,Bp).cpu(),hb=(zh@bhat.T).cpu(),bR=torch.einsum('aq,kbq->kab',bhat,R).cpu(),
            bB=torch.einsum('aq,kbq->kab',bhat,Bp).cpu(),bb=bb.cpu())
        Ot=bs['Ot'].to(Z.device);At=bs['At'].to(Z.device)
        # Frozen A and O, free signed spatial coefficients: per destination cell, span{a_r otilde_r^T}.
        Pabs=block_projector(torch.einsum('ar,ro->aor',At,Ot).reshape(NB*n_out,-1))
        Pcon=block_projector(torch.einsum('ar,ro->aor',Hm@At,Ot).reshape(NB*n_out,-1))
        Po=block_projector(Ot.T)
        def resid_blocks(X,P):
            Y=X.reshape(K,NB,cells,n_out).permute(0,2,1,3).reshape(K,cells,NB*n_out);Y=Y-Y@P;return Y.reshape(K,-1)
        def resid_out(X,P):Y=X.reshape(K,NB,cells,n_out);return (Y-Y@P).reshape(K,-1)
        for name,X,P,f in (('PAOz',Z,Pabs,resid_blocks),('PAOc',C,Pcon,resid_blocks),('POz',Z,Po,resid_out),('POc',C,Po,resid_out)):
            Y=f(X,P);e[name]=(Y@Y.T).cpu()
        # Exact model-span sanity on the actual prediction (must be ~0): AO block span of Zhat.
        Yh=zh.reshape(NB,cells,n_out).permute(1,0,2).reshape(cells,-1);e['model_AO_span_residual']=float(((Yh-Yh@Pabs).norm()/Yh.norm()).item())
        e['model_A_span_residual']=float(((zh-bs['Pi'].to(Z.device)@zh).norm()/zh.norm()).item())
        s['ck'][ck]=e
    s['D']=D.cpu()
    return s

def add(summaries):
    """Pool matrices by summing every additive statistic (Z-normalised Grams and raw Grams were formed per matrix)."""
    out=dict(K=summaries[0]['K'],ck={})
    keys=[k for k in summaries[0] if k not in ('K','ck','d','D') and isinstance(summaries[0][k],torch.Tensor)]
    for k in keys:out[k]=sum(s[k] for s in summaries)
    out['raw']=[(s['D'],s['MZ'],{ck:(e['hh'],e['hZ'],e['hB'],e['hb']) for ck,e in s['ck'].items()}) for s in summaries]
    for ck in summaries[0]['ck']:
        out['ck'][ck]={k:sum(s['ck'][ck][k] for s in summaries) for k in summaries[0]['ck'][ck] if isinstance(summaries[0]['ck'][ck][k],torch.Tensor)}
    return out

def stats(s,S,bases,channel,pairs=True):
    """All statistics on replica subset S. s: pooled-form summary (from add()); bases: ck -> per-channel basis list."""
    Hm=H();GZ=ust(s['MZ'],S);GC=ust(s['MC'],S);out={}
    out['S_Z']=float(torch.trace(GZ));out['S_C']=float(torch.trace(GC))
    ev=torch.linalg.eigvalsh(GC).flip(0)
    for m in range(NB):out[f'eig{m}']=float(ev[m])
    out['L7']=tail(GC,7);out['U7']=upper(s['MC'],S,7)
    out['L7_rel']=out['L7']/out['S_C'];out['U7_rel']=out['U7']/out['S_C'];out['L7_absrel']=out['L7']/out['S_Z'];out['U7_absrel']=out['U7']/out['S_Z']
    GR=ust(s['MR'],S);out['S_raw']=float(torch.trace(GR));out['L8_raw']=tail(GR,8);out['U8_raw']=upper(s['MR'],S,8)
    out['L8_raw_rel']=out['L8_raw']/out['S_raw'];out['U8_raw_rel']=out['U8_raw']/out['S_raw']
    no=s['OGC'].shape[-1]
    if no>8:
        for nm in ('OGC','OGZ'):
            G=ust(s[nm],S);out[f'{nm}_L8']=tail(G,8);out[f'{nm}_U8']=upper(s[nm],S,8)
        out['OGC_L8_rel']=out['OGC_L8']/out['S_C'];out['OGC_U8_rel']=out['OGC_U8']/out['S_C']
    GB=ust(s['MB'],S);out['S_B']=float(torch.trace(GB));out['S_T']=float(torch.trace(GR)+2*torch.trace(ust(s['MRB'],S))+torch.trace(GB))
    for ck,e in s['ck'].items():
        b=bases[ck][channel];TA=b['TA'];Pi=b['Pi'];p=ck+'/'
        out[p+'F_A']=float(torch.trace(GC)-torch.trace(TA.T@GC@TA));out[p+'F_A_absZ']=float(torch.trace(GZ)-torch.trace(Pi@GZ@Pi.T))
        mh=mean_over(e['hZ'],S)
        out[p+'E_Mz']=float(torch.trace(e['hh'])-2*torch.trace(mh)+torch.trace(GZ))
        out[p+'E_Mc']=float(torch.trace(Hm@e['hh']@Hm)-2*torch.trace(Hm@mh@Hm)+torch.trace(GC))
        for nm in ('PAOz','PAOc','POz','POc'):out[p+nm]=ust_scalar(e[nm],S)
        # Raw-space (un-normalised) scatter, zero-collision and total errors, summed per matrix with its own D.
        sc=bl=tt=0.
        for D,MZ,hs in s['raw']:
            hh,hZ,hB,hb=hs[ck];GRm=D@ust(MZ,S)@D;mz=D@mean_over(hZ,S)@D
            sc+=float(torch.trace(D@hh@D)-2*torch.trace(mz)+torch.trace(GRm))
        bb=e['bb'];bl=float(torch.trace(bb)-2*torch.trace(mean_over(e['bB'],S))+torch.trace(GB))
        for D,MZ,hs in s['raw']:
            hh,hZ,hB,hb=hs[ck];tt+=float(torch.trace(D@hh@D)+2*torch.trace(D@hb)-2*(torch.trace(D@mean_over(hZ,S)@D)+torch.trace(D@mean_over(hB,S))))
        tt+=float(torch.trace(bb)-2*(torch.trace(mean_over(e['bR'],S))+torch.trace(mean_over(e['bB'],S))))+out['S_T']
        out[p+'E_raw_scatter']=sc;out[p+'E_raw_ball']=bl;out[p+'E_raw_total']=tt
        for nm in ('F_A','E_Mc','PAOc','POc'):out[p+nm+'_rel']=out[p+nm]/out['S_C']
        for nm in ('F_A_absZ','E_Mz','PAOz','POz'):out[p+nm+'_rel']=out[p+nm]/out['S_Z']
        out[p+'E_raw_scatter_rel']=sc/out['S_raw'];out[p+'E_raw_ball_rel']=bl/max(out['S_B'],1e-300);out[p+'E_raw_total_rel']=tt/out['S_T']
        out[p+'FA_over_EM']=out[p+'F_A']/out[p+'E_Mc'];out[p+'FAO_over_EM']=out[p+'PAOc']/out[p+'E_Mc']
        if pairs:
            R=torch.eye(NB,dtype=torch.float64)-Pi;MM=e['hh']-mh-mh.T+GZ
            for a,bq in itertools.combinations(range(NB),2):
                u=torch.zeros(NB,dtype=torch.float64);u[a]=1;u[bq]=-1;ph=float(u@GZ@u)
                out[p+f'pair{a}_{bq}_phys']=ph;out[p+f'pair{a}_{bq}_FA']=float(u@R@GZ@R.T@u);out[p+f'pair{a}_{bq}_EM']=float(u@MM@u)
    return out

def jackknife(fn,K,t):
    """Delete-one-replica jackknife of every statistic returned by fn(subset)."""
    full=fn(list(range(K)));loo=[fn([j for j in range(K) if j!=k]) for k in range(K)];res={}
    for name,v in full.items():
        xs=torch.tensor([l[name] for l in loo],dtype=torch.float64);se=float(((K-1)/K*(xs-xs.mean()).square().sum()).sqrt())
        res[name]=dict(est=v,se=se,lo=v-t*se,hi=v+t*se)
    return res
