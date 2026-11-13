"""Public Mamba accepted-token tensor scatter. No surrounding state commit/align port."""
def validate_indices(indices,capacity):
 assert len(indices)==32 and type(capacity)is int and 1<=capacity<=4096
 selected=[i for i in indices if i>=0]
 assert all(type(i)is int for i in indices) and all(i<capacity for i in selected)
 assert len(set(selected))==len(selected),'Duplicate destinations unsupported (native write race)'
def scatter_num_accepted_(indices,num_sampled,accepted):
 import torch
 assert indices.device.type=='cuda' and indices.device==num_sampled.device==accepted.device
 for t,dtype in ((indices,torch.int64),(num_sampled,torch.int32),(accepted,torch.int32)):
  assert t.dtype==dtype and t.ndim==1 and t.stride()==(1,) and not t.requires_grad
 assert indices.shape==(32,) and num_sampled.shape==(32,) and 1<=accepted.numel()<=4096
 assert len({t.untyped_storage().data_ptr()for t in (indices,num_sampled,accepted)})==3,'No alias permitted'
 validate_indices(indices.detach().cpu().tolist(),accepted.numel())
 import cuda.tile as ct
 from .kernel import scatter_accepted
 ct.launch(torch.cuda.current_stream(accepted.device),(1,),scatter_accepted,(indices,num_sampled,accepted))
 return None
