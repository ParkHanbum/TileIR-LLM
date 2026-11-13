"""CPU-prepared int32 to int64 copy for TP1 embedding inputs; uncompiled."""
def embedding_ids_long(x):
 import torch
 from .contract import validate
 assert isinstance(x,torch.Tensor)
 n=validate(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.device.type,str(x.layout),x.requires_grad,x.is_neg(),x.is_conj())
 import cuda.tile as ct
 from .kernel import widen
 out=torch.empty((n,),dtype=torch.int64,device=x.device)
 assert out.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
 ct.launch(torch.cuda.current_stream(x.device),((n+255)//256,),widen,(x,out))
 return out
