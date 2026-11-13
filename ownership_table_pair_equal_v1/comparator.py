from .api import equal

def comparator(torch,pair_equal=equal):
 # Same expected tensor creation/order and short circuit as original comparator.
 # data_ptr values are integer payload only; never cast to pointer or dereferenced.
 def compare(ptrs,strides,tensors):
  expected_ptrs=torch.tensor([t.data_ptr()for t in tensors],dtype=torch.uint64,device=ptrs.device)
  expected_strides=torch.tensor([t.stride(0)for t in tensors],dtype=torch.int64,device=strides.device)
  return bool(pair_equal(ptrs,expected_ptrs) and pair_equal(strides,expected_strides))
 return compare
