"""CPU, nonzero-weight computational-dependency controls (no geometry oracle)."""
import sys,os
import torch
from common import *
from runtime import Transport,full,selective,apply

def main():
    torch.set_num_threads(1);torch.manual_seed(591102)
    n,k=64,8
    graph=(torch.arange(n)[:,None]+torch.arange(k)[None,:])%n
    src=torch.arange(n).repeat_interleave(k);dst=graph.flatten()
    columns=torch.arange(n*16);rows=(columns//16)*64+(columns%16)*4
    B=torch.sparse_coo_tensor(torch.stack([rows,columns]),torch.full((n*16,),.1),(n*64,n*16)).coalesce()
    g=dict(P=n,node=torch.randn(n,140),graph=graph,src=src,dst=dst,parent=torch.arange(n)//16,
           pair=torch.randn(n*k,6),prior=torch.rand(n*k,3),B=B,beta=torch.full((n,16),.1),
           signature='s0',cache_contract='cpu-dependency-control')
    m=Transport().eval()
    with torch.no_grad():
        m.edge[-1].weight.normal_(0,.02);m.node_head[-1].weight.normal_(0,.02)
        c=full(m,g);x=torch.rand(3,n,16,3);records=[]
        for step in range(12):
            new=dict(g,node=g['node'].clone(),pair=g['pair'].clone(),prior=g['prior'].clone(),signature=f's{step+1}')
            if step==10:new['node'][:,0]+=.02
            elif step!=11:new['node'][(step*7)%n,0]+=.02
            # Independently changed metric with possibly unchanged endpoint latent.
            new['pair'][(step*37)%(n*k),0]+=.03
            new['prior'][(step*31)%(n*k),1]*=.9
            raw=c['raw'].clone();head=c['head'].clone();f=full(m,new);c,audit=selective(m,c,new);mask=audit.pop('masks')
            y,z=apply(m,c,x),apply(m,f,x)
            error=float((y-z).abs().sum()/z.abs().sum())
            retained=bool(torch.equal(raw[~mask['edges']],c['raw'][~mask['edges']]) and torch.equal(head[~mask['nodes']],c['head'][~mask['nodes']]))
            assert error<=1e-5 and retained
            records.append(dict(step=step,error=error,retained_bitwise=retained,audit=audit));g=new
    dump(dict(passed=True,device='CPU',nonzero_decoder_weights=True,edits=records,
              purpose='Exercise all learned dependency paths; complements real-geometry initial-weight GPU tests'),OUT/'nonzero_dependency_tests.json')
    print('12 sequential nonzero-weight CPU F/S dependency tests passed',flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
