"""Public staged list semantics, conservative no-race specialization. CPU only."""
def validate(shape,stride,dtype,indices,starts,contents,ends):
 assert dtype in ('torch.int32','torch.int64','torch.float32')
 assert len(shape)in(1,2)and all(type(x)is int and x>0 for x in shape)
 assert tuple(stride)==((1,)if len(shape)==1 else(shape[1],1))
 assert all(isinstance(x,(list,tuple))for x in (indices,starts,contents,ends))
 assert len(indices)==len(starts)==len(ends)<=65536 and len(contents)<=8388608
 assert len(contents)==(ends[-1]if ends else 0)
 width=shape[1]if len(shape)==2 else 1;prior=0;intervals=[]
 for row,start,end in zip(indices,starts,ends):
  assert all(type(x)is int for x in (row,start,end))
  assert 0<=row<shape[0] and 0<=start<=width and prior<=end<=len(contents)
  n=end-prior;assert n<=width-start and end<=2147483647
  lo=row*stride[0]+start;hi=lo+n
  assert all(n==0 or a==b or hi<=a or b<=lo for a,b in intervals),'Overlapping native parallel writes unsupported'
  intervals.append((lo,hi));prior=end
 for x in contents:
  if dtype=='torch.float32':assert type(x)in(int,float)
  else:assert type(x)is int and -(2**(31 if dtype=='torch.int32'else 63))<=x<2**(31 if dtype=='torch.int32'else 63)
 return tuple(intervals)
def snapshot(owner):
 t=owner.gpu
 data=tuple(tuple(getattr(owner,n))for n in ('_staged_write_indices','_staged_write_starts','_staged_write_contents','_staged_write_cu_lens'))
 validate(tuple(t.shape),tuple(t.stride()),str(t.dtype),*data)
 return data
