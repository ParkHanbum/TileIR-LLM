"""Separate large-GDN design. No change to qualified <=2^23 primitive."""
MAX_ELEMENTS=16777216
def descriptor(shape,dtype,offset,contiguous):
 assert dtype in ('torch.bfloat16','torch.float32') and offset==0 and contiguous
 n=1
 for s in shape:assert type(s)is int and s>=0;n*=s
 assert n<=MAX_ELEMENTS
 return n
def status_boolean(value,n):
 assert type(value)is int and 0<=value<=n
 return value==0
def scalar_reference(values):
 # Python comparison oracle only, not actual tensor/BF16 execution evidence.
 return all(v==v and -float('inf')<v<float('inf') for v in values)
