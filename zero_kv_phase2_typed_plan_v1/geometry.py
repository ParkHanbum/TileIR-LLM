"""CPU-only offset geometry. Returned plans remain process-local, never persisted."""
CHUNK=64*1024**2
class ContractError(RuntimeError):pass
def require(x):
 if not x:raise ContractError('invalid_owned_geometry')
def merge(size,intervals):
 require(type(size)is int and 0<size<=32*1024**3 and size%4==0)
 out=[]
 for lo,hi in sorted(intervals):
  require(type(lo)is int and type(hi)is int and 0<=lo<hi<=size and lo%4==hi%4==0)
  if out and lo<=out[-1][1]:out[-1]=(out[-1][0],max(hi,out[-1][1]))
  else:out.append((lo,hi))
 return out
def partition(size,intervals):
 target=merge(size,intervals);cursor=0;result=[]
 def append(kind,lo,hi):
  while lo<hi:
   end=min(hi,lo+CHUNK);result.append((kind,lo,end));lo=end
 for lo,hi in target:
  append('outside',cursor,lo);append('target',lo,hi);cursor=hi
 append('outside',cursor,size)
 return result
def from_registry(registry,block_ids):
 registry.check();require(type(block_ids)is list and all(type(i)is int for i in block_ids))
 plans=[]
 for allocation in registry.allocations.values():
  spans=[]
  for segment in registry.segments.values():
   if segment.allocation is not allocation:continue
   require(all(0<=i<segment.blocks for i in block_ids))
   spans.extend((segment.offset+i*segment.stride,segment.offset+i*segment.stride+segment.page)for i in block_ids)
  plans.append((allocation,partition(allocation.size,spans)))
 return plans
