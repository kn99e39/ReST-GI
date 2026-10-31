"""WL65 top-up extension contract: fixed v2 protocol, packet population, count-weighted moments."""
import json
from pathlib import Path
from common65 import HERE,CFG as CFG1,sha,write_new,save_new,reference
CFG=json.loads((HERE/'protocol/topup_ab_v2.json').read_text())
TOP=CFG['topup']
HOME=Path.home();PHASE_A=HOME/'rest_clean65';EXT=Path(__import__('os').environ.get('C65X_ROOT',HOME/'rest_clean65_ext'));ORIG=HOME/'rest_persist_v1'
assert TOP['poses']==CFG1['poses'] and set(TOP['fields'])<=set(CFG1['fields'])
assert TOP['N_per_packet']==CFG1['N_per_half']==TOP['action_chunk']==CFG1['action_chunk']
assert len(TOP['poses'])*len(TOP['fields'])*2*TOP['packets_per_replica_group']==TOP['total_packets']
assert TOP['total_packets']*TOP['N_per_packet']==TOP['total_photons']

def phaseA_seed(pi,field,rep,half):return 650000000+pi*100000+field*100+rep*10+half
def topup_seed(pi,field,rep,k):return 660000000+pi*100000+field*100+rep*10+k

def packets(pose,field,root_a=PHASE_A,root_t=EXT):
    """Fixed packet list in accumulation order: (source, replica group, path, N, seed)."""
    pi=TOP['poses'].index(pose);out=[]
    for rep in range(2):
        for half in range(2):out.append(('phaseA',rep,root_a/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt',CFG1['N_per_half'],phaseA_seed(pi,field,rep,half)))
    if field in TOP['fields']:
        for rep in range(2):
            for k in range(TOP['packets_per_replica_group']):out.append(('topup',rep,root_t/'halves'/f'{pose}_f{field}_r{rep}_k{k}.pt',TOP['N_per_packet'],topup_seed(pi,field,rep,k)))
    return out

def packet_variance(first,second,N):return (second-first.square()/N)*(N/(N-1))

def override(orig,response,first=80):
    """B training-target dict: only train[first:first+len(response)] replaced; every other tensor/key copied."""
    new={k:v.clone() for k,v in orig.items()};new['train'][first:first+len(response)]=response
    return new

class Combined:
    """Count-weighted mean, variance of the mean and replica-group means of independent packets."""
    def __init__(self,counts,groups):
        self.counts=list(counts);self.groups=list(groups);total=sum(self.counts)
        self.w=[n/total for n in self.counts];self.mean=self.var=None;self.group={0:None,1:None}
        self.gw={g:sum(n for n,h in zip(self.counts,self.groups) if h==g) for g in (0,1)};self.i=0
    def add(self,first,second):
        i=self.i;w=self.w[i];n=self.counts[i];g=self.groups[i];self.i+=1
        m=first*w;v=packet_variance(first,second,n)*(w*w);gm=first*(n/self.gw[g])
        self.mean=m if self.mean is None else self.mean+m
        self.var=v if self.var is None else self.var+v
        self.group[g]=gm if self.group[g] is None else self.group[g]+gm
    def result(self):
        assert self.i==len(self.counts)
        import torch
        return dict(mean=self.mean,var=self.var,rep=torch.stack([self.group[0],self.group[1]]))
