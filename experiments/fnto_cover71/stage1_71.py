"""WL71 Stage 1 (diagnostic, development evidence only): spatial-map identifiability of the WL68 fits.
Reads WL68 inputs/runs/bundles read-only; writes ROOT/stage1/diag.json. No fit, no new reference, no held-out population of WL71."""
import sys,json,math
import numpy as np
import torch
from common71 import ROOT,R68,INP68,write_new,environment
import family71 as FW

RADII=(0.005,0.01,0.02,0.04)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    RG=environment();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0','cf3')}
    fitp=[s['port'] for s in sorted(json.loads((INP68/'wl66_ports.json').read_text())['ports'],key=lambda s:s['slot'])]
    hop=[s['port'] for s in json.loads((R68/'heldout/ports.json').read_text())['ports']]
    xyz0=G['rest']['rest_xyz'].cpu().numpy();face=np.rint(G['rest']['node'][:,15].cpu().numpy()*5).astype(int);P=len(face)
    out=dict(fit_ports=fitp,wl68_new_ports=hop,note='development evidence (WL66/WL68 populations, WL68 fits)')
    # 1. features: which local columns vary, and non-position identity of ports
    names=['x0','y0','z0','x','y','z','dx','dy','dz','nx','ny','nz','area_ratio','thick','normspread','face']
    feat={}
    for g,gg in G.items():
        X=gg['node'][:,:16].cpu().numpy();np_=np.round(X[:,6:],6)
        top=face==5;uniq=len({tuple(r) for r in np_[top]})
        feat[g]=dict(std_by_column={n:float(X[:,k].std()) for k,n in enumerate(names)},
                     distinct_nonposition_vectors_top=uniq,top_ports=int(top.sum()),
                     ports_sharing_nonposition_features_with_fit_port={str(i):int((np.abs(np_-np_[i]).max(1)<1e-6).sum()) for i in fitp+hop})
    out['features']=feat
    # 2. coverage: rest-xy and standardized-feature distance to nearest fit source (rest+pt0 rows)
    Xall=torch.cat([G[g]['node'][:,:16] for g in ('rest','pt0')]).cpu().numpy();mu,sd=Xall.mean(0),Xall.std(0)+1e-9
    Z={g:(G[g]['node'][:,:16].cpu().numpy()-mu)/sd for g in G}
    fitZ=np.concatenate([Z[g][fitp] for g in ('rest','pt0')])
    def cov(i,g):
        same=[j for j in fitp if face[j]==face[i]]
        dxy=min(float(np.hypot(*(xyz0[i,:2]-xyz0[j,:2]))) for j in same) if same else None
        return dict(rest_xy_to_nearest_same_face_fit=dxy,feature_distance_to_nearest_fit_row=float(np.linalg.norm(fitZ-Z[g][i],axis=1).min()))
    out['coverage']={f'{g}_{i}':cov(i,g) for g in G for i in fitp+hop}
    allfd={g:np.linalg.norm(Z[g][:,None,:]-fitZ[None],axis=2).min(1) for g in G}
    out['coverage_all_ports']={g:dict(feature_distance_quantiles=np.quantile(allfd[g],[.1,.5,.9,.99]).tolist()) for g in G}
    # 3. receiver-neighbourhood overlap of each new port with the fit sources (same geometry)
    ov={}
    for g in ('rest','pt0','cf3'):
        rec=lambda ps:set(G[g]['dst'][FW.edges(G[g],ps)].flatten().tolist())
        fr=rec(fitp);ov[g]={str(i):len(rec([i])&fr)/128 for i in hop}
    out['receiver_overlap_with_fit_sources']=ov
    # 4-5. model diagnosis: state rank, change of T relative to the init checkpoint vs distance to fit sources
    from runtime import Transport
    def model(p):
        m=Transport().cuda();m.load_state_dict(load(p)['model']);return m.eval()
    M={'P_5901':model(INP68/'wl59/P_5901.pt'),'P_5902':model(INP68/'wl59/P_5902.pt')}
    for r in ('net_all_5901','net_all_5902','net_all_scratch6801','net_net_5902','net_AO_5902'):M[r]=model(R68/'runs'/f'{r}.pt')
    init={'net_all_5901':'P_5901','net_all_5902':'P_5902','net_net_5902':'P_5902','net_AO_5902':'P_5902','net_all_scratch6801':None}
    dist=np.array([min(math.hypot(*(xyz0[i,:2]-xyz0[j,:2])) if face[j]==face[i] else 9. for j in fitp) for i in range(P)])
    bins=[(0,.01),(.01,.02),(.02,.04),(.04,9.)]
    st={}
    for g in ('rest','pt0','cf3'):
        st[g]={}
        for name,m in M.items():
            s=m.state(G[g]);z=s['z'];sv=torch.linalg.svdvals(z-z.mean(0));sf=torch.linalg.svdvals(z[fitp]-z[fitp].mean(0))
            st[g][name]=dict(z_singular_all=sv[:12].tolist(),z_singular_fit_sources=sf.tolist())
    out['state_rank']=st
    chg={}
    for g in ('rest','cf3'):
        gg=G[g];chg[g]={}
        for name,ini in init.items():
            if ini is None:continue
            rel=np.zeros(P);dmass=np.zeros(P);loc_m=np.zeros(P);loc_i=np.zeros(P)
            sm,si=M[name].state(gg),M[ini].state(gg);x0=gg['rest_xyz']
            for c0 in range(0,P,64):
                ps=list(range(c0,min(P,c0+64)))
                Tm,_,_=FW.network_columns(M[name],gg,ps,sm);Ti,_,_=FW.network_columns(M[ini],gg,ps,si)
                rel[ps]=((Tm-Ti).flatten(1).norm(dim=1)/Ti.flatten(1).norm(dim=1)).cpu().numpy()
                dmass[ps]=(Tm.sum(1)-Ti.sum(1)).abs().mean((1,2)).cpu().numpy()
                d=(x0[None,:,:]-x0[ps][:,None,:]).norm(dim=2)<=.01
                loc_m[ps]=((Tm.sum((2,3))*d).sum(1)/Tm.sum((1,2,3))).cpu().numpy();loc_i[ps]=((Ti.sum((2,3))*d).sum(1)/Ti.sum((1,2,3))).cpu().numpy()
            chg[g][name]=dict(init=ini,
                rel_T_change_by_distance_to_fit={f'{a}-{b}':dict(median=float(np.median(rel[(dist>=a)&(dist<b)])),n=int(((dist>=a)&(dist<b)).sum())) for a,b in bins},
                rel_T_change_at_fit_ports=rel[fitp].tolist(),rel_T_change_at_wl68_new_ports=rel[hop].tolist(),
                mass_change_by_distance={f'{a}-{b}':float(np.median(dmass[(dist>=a)&(dist<b)])) for a,b in bins},
                T_mass_within_1cm_fit_model={'fit_ports':loc_m[fitp].tolist(),'new_ports':loc_m[hop].tolist(),'median_all':float(np.median(loc_m))},
                T_mass_within_1cm_init={'fit_ports':loc_i[fitp].tolist(),'new_ports':loc_i[hop].tolist(),'median_all':float(np.median(loc_i))})
    out['T_change']=chg
    # 6. physical locality (development references): fraction of collided flux within radii of the source
    def prof(S,i,x0):
        f=S.double().sum((0,2))  # S (16,P,64) channel-summed collided flux -> per receiver
        r=(x0-x0[i]).norm(dim=1).double();return [float(f[r<=R].sum()/f.sum()) for R in RADII]
    phys={}
    x0c=G['rest']['rest_xyz'].cpu()
    for g in ('rest','pt0'):
        for s,i in enumerate(fitp):
            S=load(INP68/'fit'/f'{g}_s{s}.pt')['S'][:4].mean(0);phys[f'{g}_s{s}_port{i}']=prof(S.sum(-1),i,x0c)
    for g,slots in (('rest',[4,5,6,7]),('cf3',[4,5,6,7])):
        for s in slots:
            S=load(R68/'heldout/bundles'/f'{g}_s{s}.pt')['S'].mean(0);i=hop[s-4];phys[f'{g}_s{s}_port{i}']=prof(S.sum(-1),i,x0c)
    out['physical_flux_fraction_within_radii']=dict(radii=RADII,values=phys)
    pred={}
    for name in ('P_5902','net_all_5902','net_all_5901','net_all_scratch6801'):
        for g in ('rest','cf3'):
            T,A,O=FW.network_columns(M[name],G[g],fitp+hop);d=(1-G[g]['beta'][fitp+hop]).float()
            f=torch.einsum('na,rac,njrc->nj',d,A,T).cpu()
            for k,i in enumerate(fitp+hop):
                r=(x0c-x0c[i]).norm(dim=1);pred[f'{name}_{g}_port{i}']=[float(f[k][r<=R].sum()/f[k].sum()) for R in RADII]
    out['model_flux_fraction_within_radii']=pred
    write_new(ROOT/'stage1/diag.json',out);print('STAGE1_DONE',flush=True)

if __name__=='__main__':main();sys.stdout.flush();import os;os._exit(0)
