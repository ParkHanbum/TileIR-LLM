"""Experimental public-source scalar cumsum. No runtime adoption or raw equality claim."""
def chunk_local_cumsum(g, chunk_size=64, reverse=False, cu_seqlens=None,
                       chunk_indices=None, head_first=False, output_dtype=None):
    from .contract import validate_descriptor,validate_metadata
    import torch
    validate_descriptor(g.shape,g.stride(),str(g.dtype),g.storage_offset(),chunk_size,reverse,head_first)
    assert g.is_cuda and not g.requires_grad
    assert output_dtype is None or output_dtype==torch.float32
    for x in (cu_seqlens,chunk_indices):
        assert isinstance(x,torch.Tensor) and x.device==g.device
        assert x.dtype in (torch.int32,torch.int64) and x.is_contiguous() and x.storage_offset()==0
    assert tuple(cu_seqlens.shape)==(33,) and tuple(chunk_indices.shape)==(32,2)
    # Initial qualification only: host validation synchronizes. Not a performance implementation.
    validate_metadata(cu_seqlens.tolist(),chunk_indices.tolist())
    import cuda.tile as ct
    from .kernels import scalar_cumsum
    out=torch.empty_like(g)
    ct.launch(torch.cuda.current_stream(g.device),(32,32),scalar_cumsum,(g,out,cu_seqlens,chunk_indices))
    return out
