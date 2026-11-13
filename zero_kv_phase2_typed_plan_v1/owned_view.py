"""Draft public typed-storage view constructor; no integer pointer cast."""
from .geometry import require,CHUNK
def chunk_view(torch,allocation,lo,hi):
 allocation.check();require(0<=lo<hi<=allocation.size and lo%4==hi%4==0 and hi-lo<=CHUNK)
 # Storage object originates only in retained typed Tensor, never an address table.
 out=torch.empty(0,dtype=torch.int32,device=allocation.owner.device)
 out.set_(allocation.storage,lo//4,((hi-lo)//4,),(1,))
 require(out.untyped_storage().data_ptr()==allocation.base and out.untyped_storage().nbytes()==allocation.size)
 require(out.data_ptr()==allocation.base+lo and out.storage_offset()==lo//4)
 return out
