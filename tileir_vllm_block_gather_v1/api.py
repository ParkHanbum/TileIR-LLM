"""Public-source block-table gather companion. No registration/runtime patch."""
def validate(src, counts, indices, out, padded, *, require_cuda=True):
    import torch
    if not isinstance(src,tuple) or not isinstance(out,tuple) or not src or len(src)!=len(out):
        raise ValueError('nonempty equal-length tuples required')
    if type(padded) is not int or padded<0: raise ValueError('padded must be nonnegative int')
    tensors=(*src,counts,indices,*out)
    for t in tensors:
        if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad:
            raise ValueError('strided integer tensors required')
        if not t.is_contiguous(): raise ValueError('contiguous tensors required')
        if require_cuda and t.device.type!='cuda': raise ValueError('CUDA-visible tensor required')
        if t.device!=indices.device: raise ValueError('same device required')
    if indices.ndim!=1 or indices.dtype not in (torch.int32,torch.int64): raise ValueError('indices int32/int64 vector required')
    if counts.ndim!=2 or counts.dtype!=torch.int32 or counts.shape[0]!=len(src): raise ValueError('counts int32 [groups,requests] required')
    if indices.numel()>padded: raise ValueError('padded smaller than actual requests')
    for s,o in zip(src,out):
        if s.ndim!=2 or o.ndim!=2 or s.dtype!=torch.int32 or o.dtype!=torch.int32: raise ValueError('tables int32 matrices required')
        if s.shape[0]!=counts.shape[1] or s.shape[1]<=0 or o.shape[1]!=s.shape[1] or o.shape[0]<padded: raise ValueError('table capacity mismatch')
    # Entire backing intervals, including distinct Storage wrappers, may not overlap.
    def overlap(a,b):
        a,b=a.untyped_storage(),b.untyped_storage()
        return a.data_ptr()<b.data_ptr()+b.nbytes() and b.data_ptr()<a.data_ptr()+a.nbytes()
    for i,o in enumerate(out):
        if any(overlap(o,t) for t in (*src,counts,indices,*out[:i])): raise ValueError('overlapping backing storage')
    return len(src),indices.numel()

def validate_values_cpu(src,counts,indices,out,padded):
    """Qualification only. Never silently copy CUDA data or sync in public API."""
    validate(src,counts,indices,out,padded,require_cuda=False)
    if any(t.device.type!='cpu' for t in (*src,counts,indices,*out)): raise ValueError('CPU snapshots only')
    ids=indices.tolist()
    if any(i<0 or i>=counts.shape[1] for i in ids): raise ValueError('request index out of range; -1 not supported by source gather')
    for g,s in enumerate(src):
        if any(int(counts[g,i])<0 or int(counts[g,i])>s.shape[1] for i in ids): raise ValueError('invalid live block count')

def reference_cpu(src,counts,indices,out,padded):
    validate_values_cpu(src,counts,indices,out,padded)
    for g,(s,o) in enumerate(zip(src,out)):
        for b,r in enumerate(indices.tolist()):
            n=int(counts[g,r]); o[b,:n].copy_(s[r,:n])
        o[len(indices):padded].zero_()
    return tuple(o[:padded] for o in out)

def _enqueue(src,counts,indices,out,padded,actual):
    import torch
    import cuda.tile as ct
    from .kernels import gather_group
    with torch.cuda.device(indices.device):
        stream=torch.cuda.current_stream(indices.device)
        for g,(s,o) in enumerate(zip(src,out)):
            ct.launch(stream,(padded,(s.shape[1]+255)//256,1),gather_group,(s,counts,indices,o,g,actual,256))

def gather_block_tables(src,counts,indices,out,padded):
    """Mutate provided tables, returning prefix views retaining their storage.

    Preconditions: each live req index in [0,R), counts[g,req] in [0,width_g].
    Metadata/alias validated before any launch; GPU values not synchronously read.
    Actual-row tail unchanged; padded rows zero; rows >= padded untouched.
    One group per launch, not ABI/launch-equivalent to vLLM pointer-array fusion.
    CUDA-visible UVA views are metadata-accepted but NOT qualified yet.
    """
    _,actual=validate(src,counts,indices,out,padded)
    if padded: _enqueue(src,counts,indices,out,padded,actual)
    return tuple(o[:padded] for o in out)
