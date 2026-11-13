"""Pure CPU narrow-contract checks; observed metadata still pending."""
def validate_metadata(cu_seqlens, chunk_indices):
    assert type(cu_seqlens) is list and cu_seqlens == list(range(0,1697,53))
    assert type(chunk_indices) is list and chunk_indices == [[i,0] for i in range(32)]
    return 32

def validate_descriptor(shape, stride, dtype, offset, chunk_size, reverse, head_first):
    assert tuple(shape)==(1,1696,32)
    assert tuple(stride)==(54272,32,1)
    assert dtype=='torch.float32' and offset==0
    assert type(chunk_size) is int and chunk_size==64
    assert reverse is False and head_first is False
