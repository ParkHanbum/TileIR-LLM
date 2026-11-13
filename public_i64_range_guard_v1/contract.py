MIN=-(1<<63)
MAX=(1<<63)-1
def bounds(lower,upper):
 if type(lower)is not int or type(upper)is not int or not MIN<=lower<=MAX or not MIN<=upper<=MAX:raise ValueError('Signed int64 scalar bounds required')
def descriptor(shape,stride,dtype,offset,storage_bytes,lower,upper,neg=False,conj=False):
 bounds(lower,upper)
 if dtype!='torch.int64' or neg or conj or type(offset)is not int or offset<0 or storage_bytes%8:raise ValueError('Unsupported int64 view')
 if len(shape)!=len(stride) or any(type(d)is not int or d<0 for d in shape):raise ValueError('Invalid dimensions')
 n=1
 for d in shape:n*=d
 if n>1024:raise ValueError('Bounded predicate supports at most1024 elements')
 if any(type(s)is not int or s<0 for s in stride):raise ValueError('Unsupported strides')
 expected=1
 if n:
  for d,s in zip(reversed(shape),reversed(stride)):
   if d>1 and s!=expected:raise ValueError('Contiguous view required')
   expected*=d
 if (offset+n)*8>storage_bytes:raise ValueError('View exceeds backing')
 return n
def reference(values,lower,upper):
 bounds(lower,upper)
 return all(lower<=v<upper for v in values)
