"""Runtime dependencies injected; no framework imports or global patches.
Narrow qualified-shape proposal. Unsupported inputs fail BEFORE allocation/enqueue.
No retry/fallback after launch attempt. All outputs are fresh dense BF16 storage.
"""
def check_copy(x, offset, torch):
    if not (x.dtype == torch.bfloat16 and x.device.type == 'cuda'
            and tuple(x.shape) in ((32,32),(1696,32)) and tuple(x.stride()) == (64,1)
            and x.storage_offset() == offset and not x.requires_grad
            and not x.is_conj() and not x.is_neg()):
        raise ValueError('unqualified GDN copy input')
    end = (x.storage_offset() + (x.shape[0]-1)*64 + 32)*2
    if end > x.untyped_storage().nbytes(): raise ValueError('input extent')

class OwnedCopies:
    def __init__(self, torch, ct, kernels):
        self.t,self.ct,self.k=torch,ct,kernels
        self.failed=False
        self.attempts={'zero':0,'b':0,'a':0}; self.enqueues=dict(self.attempts)
    def _healthy(self):
        if self.failed: raise RuntimeError('owned-copy failure latched')
    def _launch(self,key,grid,kernel,args,device):
        self.attempts[key]+=1
        try:
            with self.t.cuda.device(device):
                self.ct.launch(self.t.cuda.current_stream(device),grid,kernel,args)
            self.enqueues[key]+=1
        except BaseException:
            self.failed=True
            raise
    def zeros(self, shape, *, dtype, device):
        self._healthy()
        if tuple(shape) not in ((32,32,128),(1696,32,128)) or dtype != self.t.bfloat16 or device.type!='cuda':
            raise ValueError('unqualified GDN zeros')
        out=self.t.empty(shape,dtype=dtype,device=device)
        self._launch('zero',(out.numel()//1024,1,1),self.k.zero_bits,(out.view(self.t.uint16).view(-1),),device)
        return out
    def _copy(self,x,offset,key):
        self._healthy(); check_copy(x,offset,self.t)
        out=self.t.empty(tuple(x.shape),dtype=x.dtype,device=x.device)
        s,o=x.untyped_storage(),out.untyped_storage()
        if not (s.data_ptr()+s.nbytes()<=o.data_ptr() or o.data_ptr()+o.nbytes()<=s.data_ptr()):
            raise ValueError('fresh output backing overlaps input')
        # Same-itemsize dtype view, not numeric cast: preserves every BF16 bit.
        self._launch(key,(x.shape[0]//32,1,1),self.k.copy_bits,(x.view(self.t.uint16),out.view(self.t.uint16)),x.device)
        return out
    def contiguous_b(self,x): return self._copy(x,0,'b')
    def contiguous_a(self,x): return self._copy(x,32,'a')
