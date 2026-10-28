"""Actual unique tensor storage accounting; aliases/views are counted once.
CPU and CUDA separate. Sparse indices/values included, no device-to-host copies.
"""
import torch

def storages(x,out=None):
    if out is None:out={}
    if isinstance(x,torch.Tensor):
        if x.layout==torch.sparse_coo:
            storages(x._indices(),out);storages(x._values(),out)
        elif x.layout==torch.sparse_csr:
            for a in (x.crow_indices(),x.col_indices(),x.values()):storages(a,out)
        elif x.numel():
            s=x.untyped_storage();key=(str(x.device),s.data_ptr());out[key]=s.nbytes()
    elif isinstance(x,dict):
        for a in x.values():storages(a,out)
    elif isinstance(x,(list,tuple)):
        for a in x:storages(a,out)
    return out
def bytes_of(*xs,device='cuda'):
    acc={}
    for x in xs:storages(x,acc)
    return sum(n for (d,p),n in acc.items() if d.startswith(device))
def cache_record(g,st,co=None,parameters=()):
    b=bytes_of(g);s=bytes_of(g,st);c=bytes_of(g,st,co)
    return dict(geometry_bytes=b,compact_incremental_bytes=s-b,coefficient_incremental_bytes=c-s,
        compact_total_bytes=s,full_total_bytes=c,parameters_bytes=bytes_of(list(parameters)),
        all_retained_bytes=bytes_of(g,st,co,list(parameters)),storage_identity='unique (device, untyped_storage.data_ptr), full allocated storage.nbytes; aliases/views once')
