def validate_finite_descriptor(shape,stride,dtype,offset,requires_grad,is_conj,is_neg):
 assert dtype in ('torch.bfloat16','torch.float32') and not requires_grad and offset==0
 n=1;expected=[]
 for size in reversed(shape):
  assert type(size)is int and size>0
  expected.append(n);n*=size
 assert tuple(stride)==tuple(reversed(expected)) and 1<=n<=8388608
 return n
def checked_status(value):
 assert type(value)is int and value>=0
 assert value==0,'NaN/Inf outside finite-only contract'
