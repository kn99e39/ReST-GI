"""Lossless CPU tuple packing, pinned staging and asynchronous CUDA prefetch.

No model/precision/batch changes. Five float32 vectors plus an exact 0/1 mask
form one 64-byte row. The GPU receives one contiguous transfer per batch.
"""
import queue, threading
import torch
FIELDS=('xi','wi','xo','wo','throughput')

def unpack(rows):
    out={key:rows[:,3*i:3*i+3].contiguous() for i,key in enumerate(FIELDS)}
    out['mask']=rows[:,15].to(torch.bool).contiguous()
    return out

class Slot:
    def __init__(self,batch_size):
        self.host=torch.empty((batch_size,16),dtype=torch.float32,pin_memory=True)
        self.event=None

class CUDAPrefetch:
    def __init__(self,packed,indices,batch_size,device='cuda',slots=3,max_batches=None):
        if packed.device.type!='cpu' or not packed.is_contiguous(): raise ValueError('Contiguous CPU tuple storage required')
        if indices.device.type!='cpu' or indices.dtype!=torch.int64: raise ValueError('CPU int64 permutation required')
        if len(indices)%batch_size: raise ValueError('Whole batches required')
        self.packed=packed; self.indices=indices; self.batch_size=batch_size; self.device=torch.device(device)
        self.count=len(indices)//batch_size if max_batches is None else max_batches
        self.stop=threading.Event(); self.free=queue.Queue(slots); self.ready=queue.Queue(slots)
        self.stream=torch.cuda.Stream(device=self.device)
        for _ in range(slots): self.free.put(Slot(batch_size))
        self.worker=threading.Thread(target=self._produce,daemon=True,name='ck21b_cpu_gather')
        self.worker.start(); self.pending=None; self.started=False

    def _put(self,target,item):
        while not self.stop.is_set():
            try: target.put(item,timeout=.1); return True
            except queue.Full: pass
        return False

    def _produce(self):
        try:
            with torch.no_grad():
                for step in range(self.count):
                    while not self.stop.is_set():
                        try: slot=self.free.get(timeout=.1); break
                        except queue.Empty: pass
                    else: return
                    if slot.event is not None: slot.event.synchronize()
                    first=(step%(len(self.indices)//self.batch_size))*self.batch_size
                    torch.index_select(self.packed,0,self.indices[first:first+self.batch_size],out=slot.host)
                    if not self._put(self.ready,slot): return
                self._put(self.ready,None)
        except BaseException as error:
            self._put(self.ready,error)

    def _preload(self):
        item=self.ready.get()
        if item is None: return None
        if isinstance(item,BaseException): raise item
        with torch.no_grad(),torch.cuda.stream(self.stream):
            transferred=item.host.to(self.device,non_blocking=True)
            batch=unpack(transferred)
            event=torch.cuda.Event(); event.record(self.stream)
        item.event=event; self._put(self.free,item)
        return batch,event

    def __iter__(self): return self
    def __next__(self):
        if not self.started: self.pending=self._preload(); self.started=True
        if self.pending is None:
            self.close(); raise StopIteration
        batch,event=self.pending
        current=torch.cuda.current_stream(self.device); current.wait_event(event)
        for value in batch.values(): value.record_stream(current)
        self.pending=self._preload()
        return batch

    def close(self):
        self.stop.set()
        if self.worker.is_alive(): self.worker.join(timeout=2)

def packed_dataset_class(native):
    class PackedDataset(native):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            count=self.N*self.N*self.H*self.H
            self.packed=torch.empty((count,16),dtype=torch.float32)
            for i,key in enumerate(FIELDS): setattr(self,key,self.packed[:,3*i:3*i+3])
            self.mask=self.packed[:,15]
            self.inds=None
            torch.cuda.empty_cache()
        def __getitem__(self,index):
            raise RuntimeError('Use the pinned-prefetch loader, not synchronous item fetching')
    return PackedDataset

class PrefetchLoader(torch.utils.data.DataLoader):
    def __init__(self,dataset):
        super().__init__(dataset,batch_size=None,shuffle=False,num_workers=0)
    def __iter__(self):
        iterator=CUDAPrefetch(self.dataset.packed,self.dataset.inds,self.dataset.batch_size,self.dataset.device)
        try:
            yield from iterator
        finally: iterator.close()
