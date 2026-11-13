"""Bounded host chunk hashes: NOT exact untouched-byte comparison."""
import hashlib
CHUNK=8*1024**2
def selected_intervals(indices,pitch,inner):
 return sorted((r*pitch*4,(r*pitch+inner)*4) for r in indices)
def complements(start,end,selected):
 cursor=start
 for lo,hi in selected:
  if hi<=cursor:continue
  if lo>=end:break
  if lo>cursor:yield cursor,min(lo,end)
  cursor=max(cursor,min(hi,end))
 if cursor<end:yield cursor,end

def owner_hashes(owner,normalized):
 import torch
 n,h,k,v=owner.shape;inner=h*k*v;length=(n-1)*owner.stride(0)+inner
 assert length*4<=1024**3
 flat=owner.as_strided((length,),(1,),owner.storage_offset()).view(torch.uint8)
 selected=selected_intervals(normalized,owner.stride(0),inner);allhash=hashlib.sha256();untouched=hashlib.sha256()
 for lo in range(0,flat.numel(),CHUNK):
  hi=min(flat.numel(),lo+CHUNK);cpu=flat[lo:hi].cpu();b=memoryview(cpu.numpy()).cast('B');allhash.update(b)
  for a,z in complements(lo,hi,selected):untouched.update(b[a-lo:z-lo])
 return dict(all_sha256=allhash.hexdigest(),unselected_padding_sha256=untouched.hexdigest(),bytes=length*4,chunk_bytes=CHUNK,outside_view_verified=False)

def same(a,b):
 import torch
 assert a.shape==b.shape and a.dtype==b.dtype and a.is_contiguous() and b.is_contiguous()
 x=a.view(torch.uint8).flatten();y=b.view(torch.uint8).flatten()
 return all(torch.equal(x[i:i+CHUNK].cpu(),y[i:i+CHUNK].cpu()) for i in range(0,x.numel(),CHUNK))

def tensor_hash(t):
 import torch
 assert t.is_contiguous() and t.numel()*t.element_size()<=64*1024**2
 x=t.view(torch.uint8).flatten();h=hashlib.sha256()
 for i in range(0,x.numel(),CHUNK):h.update(memoryview(x[i:i+CHUNK].cpu().numpy()).cast('B'))
 return h.hexdigest()
