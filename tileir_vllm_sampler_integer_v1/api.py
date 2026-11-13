"""Fixed-workload integer APIs; CPU-prepared and uncompiled."""
def gather_integer_rows(x,indices):
 import torch
 from .contract import gather_descriptor,validate_indices
 assert isinstance(x,torch.Tensor)and isinstance(indices,torch.Tensor)and x.device.type=='cuda'and indices.device==x.device
 assert not x.requires_grad and not indices.requires_grad and not x.is_neg()and not x.is_conj()and not indices.is_neg()and not indices.is_conj()
 n=gather_descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),tuple(indices.shape),tuple(indices.stride()),str(indices.dtype),indices.storage_offset())
 # Scope-preserving safety guard; host sync is explicit, not a zero-native claim.
 validate_indices(indices.cpu().tolist(),n)
 import cuda.tile as ct
 from .kernels import gather_rows
 out=torch.empty((32,),device=x.device,dtype=x.dtype)
 assert out.untyped_storage().data_ptr()not in (x.untyped_storage().data_ptr(),indices.untyped_storage().data_ptr())
 ct.launch(torch.cuda.current_stream(x.device),(1,),gather_rows,(x,indices,out,n))
 return out
def new_sampled_ones(seq_lens,num_reqs):
 import torch
 from .contract import ones_descriptor
 assert isinstance(seq_lens,torch.Tensor)and seq_lens.device.type=='cuda'
 ones_descriptor(tuple(seq_lens.shape),str(seq_lens.dtype),num_reqs)
 import cuda.tile as ct
 from .kernels import ones
 out=torch.empty((32,),dtype=seq_lens.dtype,device=seq_lens.device)
 assert out.untyped_storage().data_ptr()!=seq_lens.untyped_storage().data_ptr()
 ct.launch(torch.cuda.current_stream(seq_lens.device),(1,),ones,(out,))
 return out
