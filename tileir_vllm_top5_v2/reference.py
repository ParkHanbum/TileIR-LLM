import struct,math
def key(v):
 assert math.isfinite(v)
 bits=struct.unpack('<I',struct.pack('<f',v))[0]>>16
 return bits^(65535 if bits&32768 else 32768)
def network(values,indices):
 n=len(values);k=list(values)+[0.]*(32-n);i=list(indices)+[0]*(32-n);v=[True]*n+[False]*(32-n)
 def stage(s,size=None):
  for tid in range(16):
   a=2*tid-(tid&(s-1));b=a+s;direction=False if size is None else bool(tid&(size//2))
   swap=(k[a]>k[b] and v[a]) or not v[b]
   if swap==direction:
    for t in [k,i,v]:t[a],t[b]=t[b],t[a]
 for size in [2,4,8,16]:
  stride=size//2
  while stride:stage(stride,size);stride//=2
 for stride in [16,8,4,2,1]:stage(stride)
 return i[:n]
def reference(row):
 assert len(row)>=5 and all(math.isfinite(x)for x in row)
 keys=[key(x)for x in row];threshold=sorted(keys,reverse=True)[4]
 selected=[i for i,k in enumerate(keys)if k>threshold];selected += [i for i,k in enumerate(keys)if k==threshold][:5-len(selected)]
 return network([row[i]for i in selected],selected)
