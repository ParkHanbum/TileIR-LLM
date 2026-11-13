"""Metadata and bit-level references only; no CUDA imports."""
MODES={'has_nan':0,'has_posinf':1,'rows_have_finite':2}
def matrix(shape,stride,dtype,offset,storage_bytes,neg=False,conj=False):
 if dtype not in ('torch.float32','torch.bfloat16') or neg or conj:raise ValueError('Unsupported floating view')
 if len(shape)!=2 or len(stride)!=2 or any(type(v)is not int or v<0 for v in shape):raise ValueError('Bounded matrix required')
 if any(type(s)is not int or s<0 for s in stride):raise ValueError('Unsupported strides')
 rows,cols=shape
 if rows>32 or cols>248320 or type(offset)is not int or offset<0:raise ValueError('Unsupported extent/offset')
 if rows*cols and ((cols>1 and stride[1]!=1)or(rows>1 and stride[0]!=cols)):raise ValueError('Contiguous matrix required')
 size=4 if dtype=='torch.float32'else 2
 if storage_bytes%size or (offset+rows*cols)*size>storage_bytes:raise ValueError('Backing bounds')
 return rows,cols,size
def masks(dtype):
 if dtype=='torch.float32':return 0x7fffffff,0x7f800000
 if dtype=='torch.bfloat16':return 0x7fff,0x7f80
 raise ValueError('Unsupported dtype')
def reference(words,mode,dtype):
 if mode not in MODES:raise ValueError('Unknown predicate')
 absolute,inf=masks(dtype)
 if mode=='has_nan':return any((w&absolute)>inf for row in words for w in row)
 if mode=='has_posinf':return any(w==inf for row in words for w in row)
 return all(any((w&absolute)<inf for w in row)for row in words)
