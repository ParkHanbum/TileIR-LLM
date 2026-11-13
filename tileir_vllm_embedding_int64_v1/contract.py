"""Observed fixed descriptors, not a general torch.Tensor.long replacement."""
def validate(shape,stride,dtype,offset,device_type,layout,requires_grad,is_neg,is_conj):
 assert tuple(shape)in ((32,),(1696,)), 'Only observed embedding token counts32/1696'
 assert tuple(stride)==(1,)and offset==0, 'Only contiguous offset-zero token IDs'
 assert dtype=='torch.int32'and device_type=='cuda'and layout=='torch.strided', 'CUDA strided int32 required'
 assert not requires_grad and not is_neg and not is_conj, 'Ordinary integer view required; flagged views unqualified'
 return shape[0]
def signed_reference(values):
 out=[]
 for v in values:
  assert type(v)is int and -2147483648<=v<=2147483647
  out.append(v) # Sign-preserving exact integer widening, not unsigned reinterpretation.
 return out
