import math
MAX=1<<24
def descriptor(shape,stride,dtype,offset,storage_bytes,constant,neg=False,conj=False):
 if type(constant) is not int or constant not in (0,1):raise ValueError('constant must be int0/1')
 if dtype!='torch.float32' or neg or conj:raise ValueError('only ordinary real FP32')
 if len(shape)!=len(stride) or any(type(x)is not int or x<0 for x in (*shape,*stride)) or type(offset)is not int or offset<0:raise ValueError('descriptor')
 n=math.prod(shape)
 if n>MAX:raise ValueError('element cap')
 if n==0:return dict(logical=0,physical=0,kind='empty')
 if (offset+1+sum((d-1)*s for d,s in zip(shape,stride)))*4>storage_bytes:raise ValueError('storage extent')
 if all(s==0 for s in stride):return dict(logical=n,physical=1,kind='broadcast')
 expected=1
 for d,s in zip(reversed(shape),reversed(stride)):
  if d>1 and s!=expected:raise ValueError('noncontiguous')
  expected*=d
 return dict(logical=n,physical=n,kind='contiguous')
def reference(values,constant):return all(x==constant for x in values)
