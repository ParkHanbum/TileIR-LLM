"""CPU interval complement; actual inner [H,V,K] is contiguous."""
def untouched_ranges(metadata,slots):
 shape=metadata['shape'];stride=metadata['stride'];offset=metadata['storage_offset']
 assert len(shape)==len(stride)==4 and stride[1:]==[shape[2]*shape[3],shape[3],1]
 size=shape[1]*shape[2]*shape[3];assert stride[0]>=size
 assert len(slots)==len(set(slots)) and all(0<=s<shape[0]for s in slots)
 total=offset+1+sum((n-1)*s for n,s in zip(shape,stride));cursor=0;result=[]
 for slot in sorted(slots):
  lo=offset+slot*stride[0];hi=lo+size
  if lo>cursor:result.append((cursor,lo))
  cursor=hi
 if cursor<total:result.append((cursor,total))
 return result
