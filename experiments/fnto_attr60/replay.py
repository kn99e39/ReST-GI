"""Finite frozen-checkpoint CPU replay; exact original algebra, no fitting/cache updates."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from analyze_stored import load,sha,agg,relative,ballistic

def coefficients(ck,g):
    lin=lambda z,n:F.linear(z,ck[n+'.weight'],ck.get(n+'.bias'))
    z=F.silu(lin(g['node'][:,:16],'encoder'))
    for k in range(2):z=z+.25*lin(F.silu(lin(torch.cat([z,z[g['graph']].mean(1)],1),f'messages.{k}.0')),f'messages.{k}.2')
    s=lin(z,'src_layer');d=lin(z,'dst_layer');raw=[]
    for ix in torch.arange(len(g['src'])).split(65536):
        h=s[g['src'][ix]]+d[g['dst'][ix]]+lin(g['pair'][ix],'pair_layer')
        v=lin(F.silu(lin(F.silu(h),'edge.1')),'edge.3')
        raw.append(F.softplus(v).view(-1,8,3)*g['prior'][ix,None])
    raw=torch.cat(raw);head=lin(F.silu(lin(z,'node_head.0')),'node_head.2');p=g['P']
    U=head[:,:384].view(p,16,8,3).softmax(0);V=F.softplus(head[:,384:768].view(p,16,8,3));mass=head[:,768:].view(p,8,3).sigmoid()
    norm=torch.zeros(p,8,3).index_add_(0,g['src'],raw)+V.sum(1);scale=mass/(norm+1e-12)
    return dict(W=raw*scale[g['src']],U=U,V=V*scale[:,None],A=ck['in_logits'].softmax(0),O=ck['out_logits'].softmax(1))

def action(g,x,co):
    q=torch.einsum('biac,rac->birc',x*(1-g['beta'])[None,:,:,None],co['A']);y=torch.zeros(len(x),g['P'],8,3)
    for ix in torch.arange(len(g['src'])).split(65536):y.index_add_(1,g['dst'][ix],q[:,g['src'][ix]]*co['W'][ix][None])
    latent=torch.einsum('birc,ikrc->bkrc',q,co['V']);y+=torch.einsum('jkrc,bkrc->bjrc',co['U'],latent)
    return torch.einsum('bjrc,roc->bjoc',y,co['O'])+ballistic(g,x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();root=a.root;torch.set_num_threads(4);torch.set_grad_enabled(False);t0=time.perf_counter()
    out=dict(source_commit=a.source_commit,script_sha256=sha(__file__),device='CPU',inputs={},seeds={},sampling='Fixed diagnostic subset, not full training risk; no new quality adjudication')
    def record(p):out['inputs'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    lp=root/'data/lights.pt';record(lp);lights=load(lp)
    # Choose nearest training dent to primary by geometry parameters only: tr11; no target/error selection.
    for seed in (5901,5902):
        cp=root/'checkpoints'/f'P_{seed}.pt';record(cp);ck=load(cp)['model'];rows={};pred={};target={}
        for pose in ('pt0','rest','tr11','va1'):
            gp=root/'data/geometry'/f'{pose}.pt';record(gp);g=load(gp);co=coefficients(ck,g)
            if pose=='pt0':
                xx=lights['test'][[8,9,62,63]];yy=action(g,xx,co);apath=root/'out'/f'actions_{seed}.pt';obj=load(apath);stored=obj['predictions']['pt0']['F'][[8,9,62,63]]
                er=relative(yy.double(),stored.double());assert er<2e-5,'CPU replay does not reproduce immutable primary actions'
                rows['replay_verification_relative_L2']=er;del obj;continue
            tp=root/'data/training_targets'/f'{pose}_r0.pt';record(tp);tt=load(tp)
            splits=('train','validation')
            for split in splits:
                ids=[8,9,94,95] if split=='train' else [8,9,30,31]
                xx=lights[split][ids];yy=agg(action(g,xx,co),g['parent']);rr=agg(tt[split][ids],g['parent'])
                pred[pose,split]=yy;target[pose,split]=rr
                rows[f'{pose}_{split}']=dict(fields=ids,absolute_relative_L2=[[(float((yy[j,:,:,c]-rr[j,:,:,c]).norm()/rr[j,:,:,c].norm())) for c in range(3)] for j in range(4)],directional_pair_error=[relative(yy[2,:,:,c]-yy[3,:,:,c],rr[2,:,:,c]-rr[3,:,:,c]) for c in range(3)])
        for pose,split in [('tr11','train'),('tr11','validation'),('va1','train'),('va1','validation')]:
            yd=pred[pose,split][:2]-pred['rest',split][:2];td=target[pose,split][:2]-target['rest',split][:2]
            rows[f'{pose}_{split}']['focused_delta_error_unqualified']=[relative(yd[:,:,:,c],td[:,:,:,c]) for c in range(3)]
        out['seeds'][str(seed)]=rows
    out['seconds']=time.perf_counter()-t0;print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':main()
