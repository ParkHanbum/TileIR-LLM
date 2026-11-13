from .contract import layout,index_values,raw_interval,raw_disjoint
def common(owner,indices,scatter=False):
 import torch
 assert owner.is_cuda and owner.dtype==torch.float32 and not owner.requires_grad
 raw_disjoint(owner,indices)
 inner=layout(tuple(owner.shape),tuple(owner.stride()),owner.storage_offset(),owner.element_size(),owner.untyped_storage().nbytes())
 assert indices.is_cuda and indices.device==owner.device and indices.dtype in (torch.int32,torch.int64) and indices.ndim==1 and 1<=indices.numel()<=32 and indices.stride()[0]>=0
 assert indices.numel()*indices.element_size()<=256 and indices.untyped_storage().nbytes()<=16*1024**2
 index_values(indices.detach().cpu().tolist(),owner.shape[0],scatter)

 return inner
def gather_state(owner,indices):
 import torch,cuda.tile as ct
 from .kernels import gather_kernel
 inner=common(owner,indices);out=torch.empty((indices.numel(),*owner.shape[1:]),dtype=owner.dtype,device=owner.device)
 ct.launch(torch.cuda.current_stream(owner.device),(indices.numel(),(inner+1023)//1024),gather_kernel,(owner,indices,out));return out
def mask_initial_state_(initial,has_initial):
 import torch,cuda.tile as ct
 from .kernels import mask_kernel
 raw_disjoint(initial,has_initial)
 assert initial.is_cuda and initial.dtype==torch.float32 and not initial.requires_grad and initial.is_contiguous() and initial.storage_offset()==0
 inner=layout(tuple(initial.shape),tuple(initial.stride()),0,4,initial.untyped_storage().nbytes());assert initial.shape[0]<=32 and initial.untyped_storage().nbytes()==initial.numel()*4<=64*1024**2
 assert has_initial.is_cuda and has_initial.device==initial.device and has_initial.dtype==torch.bool and tuple(has_initial.shape)==(initial.shape[0],) and has_initial.stride()[0]>=0 and has_initial.untyped_storage().nbytes()<=16*1024**2

 ct.launch(torch.cuda.current_stream(initial.device),(initial.shape[0],(inner+1023)//1024),mask_kernel,(initial,has_initial));return initial
def scatter_state_(owner,indices,last):
 import torch,cuda.tile as ct
 from .kernels import scatter_kernel
 raw_disjoint(owner,indices,last)
 inner=common(owner,indices,True)
 assert last.is_cuda and last.device==owner.device and last.dtype==owner.dtype and not last.requires_grad and tuple(last.shape)==(indices.numel(),*owner.shape[1:]) and last.is_contiguous() and last.storage_offset()==0 and last.untyped_storage().nbytes()==last.numel()*4<=64*1024**2

 ct.launch(torch.cuda.current_stream(owner.device),(indices.numel(),(inner+1023)//1024),scatter_kernel,(owner,indices,last));return owner
