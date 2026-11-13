"""Draft bounded metadata contract; not yet authorized for model replacement."""
SIZES={'torch.float32':4,'torch.bfloat16':2}
def vector(shape,stride,dtype,offset,nbytes,neg=False,conj=False):
 if dtype not in SIZES or len(shape)!=1 or len(stride)!=1:raise ValueError('Typed 1D vector required')
 n=shape[0];s=SIZES[dtype]
 if type(n)is not int or not 0<=n<=1024 or tuple(stride)!=(1,) or offset<0 or neg or conj:raise ValueError('Unsupported vector layout')
 if (offset+n)*s>nbytes:raise ValueError('Backing too small')
 return n,s
