def gather_descriptor(shape,stride,dtype,offset,idx_shape,idx_stride,idx_dtype,idx_offset):
 assert tuple(shape)in ((32,),(1696,))and tuple(stride)==(1,)and dtype in ('torch.int32','torch.int64')and offset==0
 assert tuple(idx_shape)==(32,)and tuple(idx_stride)==(1,)and idx_dtype=='torch.int64'and idx_offset==0
 return shape[0]
def validate_indices(indices,n):
 assert len(indices)==32 and all(type(i)is int and -n<=i<n for i in indices),'Index outside [-N,N)'
 return [i+n if i<0 else i for i in indices]
def ones_descriptor(shape,dtype,n):
 assert tuple(shape)==(32,)and dtype=='torch.int32'and type(n)is int and n==32
 return n
