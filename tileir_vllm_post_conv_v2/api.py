"""Narrow actual prefill API. Fresh tuple; no cache or input mutation."""
def validate(conv_output,a,b,A_log,dt_bias,num_k_heads,head_k_dim,head_v_dim,apply_l2norm=True,output_g_exp=False,*,require_cuda=True):
 import torch
 assert (num_k_heads,head_k_dim,head_v_dim)==(16,128,128) and all(type(v) is int for v in (num_k_heads,head_k_dim,head_v_dim))
 assert apply_l2norm is True and output_g_exp is False
 tensors=(conv_output,a,b,A_log,dt_bias)
 expected=(((1696,8192),torch.bfloat16),((1696,32),torch.bfloat16),((1696,32),torch.bfloat16),((32,),torch.float32),((32,),torch.bfloat16))
 for t,(shape,dtype) in zip(tensors,expected):
  assert isinstance(t,torch.Tensor) and tuple(t.shape)==shape and t.dtype==dtype
  assert t.is_contiguous() and t.storage_offset()==0 and not t.requires_grad
  assert t.device==conv_output.device and (not require_cuda or t.is_cuda)
 return 1696

def fused_post_conv_prep(conv_output,a,b,A_log,dt_bias,num_k_heads,head_k_dim,head_v_dim,apply_l2norm=True,output_g_exp=False):
 L=validate(conv_output,a,b,A_log,dt_bias,num_k_heads,head_k_dim,head_v_dim,apply_l2norm,output_g_exp)
 import torch,cuda.tile as ct
 from .kernels import qk_kernel,vg_kernel
 device=conv_output.device
 q=torch.empty((L,16,128),dtype=torch.bfloat16,device=device);k=torch.empty_like(q)
 v=torch.empty((L,32,128),dtype=torch.bfloat16,device=device)
 g=torch.empty((L,32),dtype=torch.float32,device=device);beta=torch.empty_like(g)
 stream=torch.cuda.current_stream(device)
 ct.launch(stream,(L,16,2),qk_kernel,(conv_output,q,k))
 ct.launch(stream,(L,32),vg_kernel,(conv_output,a,b,A_log,dt_bias,v,g,beta))
 return q,k,v,g,beta
