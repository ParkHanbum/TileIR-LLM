"""Nonquantized BF16 KV insertion, captured strided shared-cache layout only."""
def validate(key,value,key_cache,value_cache,slot_mapping,kv_cache_dtype,k_scale,v_scale,*,require_cuda=True):
 import torch
 assert kv_cache_dtype in ('auto','bfloat16')
 tensors=(key,value,key_cache,value_cache,slot_mapping,k_scale,v_scale)
 assert all(isinstance(t,torch.Tensor) and not t.requires_grad and t.device==key.device and (not require_cuda or t.is_cuda) for t in tensors)
 assert key.dtype==value.dtype==key_cache.dtype==value_cache.dtype==torch.bfloat16
 assert key.ndim==value.ndim==3 and tuple(key.shape[1:])==tuple(value.shape[1:])==(4,256)
 assert key.stride()[1:]==value.stride()[1:]==(256,1) and key.stride(0)>=1024 and value.stride(0)>=1024
 assert key_cache.ndim==value_cache.ndim==4 and key_cache.shape==value_cache.shape and tuple(key_cache.shape[2:])==(4,256)
 b,bs,_,_=key_cache.shape;assert b>0 and bs>0
 assert key_cache.stride()==value_cache.stride()==(bs*2048,2048,512,1)
 assert key_cache.untyped_storage().data_ptr()==value_cache.untyped_storage().data_ptr() and value_cache.storage_offset()==key_cache.storage_offset()+256
 assert slot_mapping.ndim==1 and slot_mapping.dtype==torch.int64 and slot_mapping.stride()==(1,)
 n=slot_mapping.numel();assert n in (32,1696) and min(key.shape[0],value.shape[0])>=n
 for t in (k_scale,v_scale):assert t.dtype==torch.float32 and t.ndim==0
 # Inference-only checked values; no unsupported duplicate-write race.
 slots=slot_mapping.detach().cpu().tolist();assert all(0<=s<b*bs for s in slots) and len(set(slots))==n
 cs=key_cache.untyped_storage();lo=cs.data_ptr();hi=lo+cs.nbytes()
 for t in (key,value,slot_mapping,k_scale,v_scale):
  st=t.untyped_storage();assert not(lo<st.data_ptr()+st.nbytes() and st.data_ptr()<hi)
 return n

def reshape_and_cache_flash(key,value,key_cache,value_cache,slot_mapping,kv_cache_dtype,k_scale,v_scale):
 n=validate(key,value,key_cache,value_cache,slot_mapping,kv_cache_dtype,k_scale,v_scale)
 import torch,cuda.tile as ct
 from .kernels import scatter
 ct.launch(torch.cuda.current_stream(key.device),(n,4),scatter,(key,value,key_cache,value_cache,slot_mapping,key_cache.shape[1]))
 return None
