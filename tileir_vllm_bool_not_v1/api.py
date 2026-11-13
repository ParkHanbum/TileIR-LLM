"""Fresh bool32 logical inverse; no out/inplace/fallback/global patch."""
def validate_contract(shape,stride,dtype,offset,owner_bytes,is_conj=False,is_neg=False):
 assert tuple(shape)==(32,) and tuple(stride)==(1,) and dtype=='torch.bool'
 assert offset==0 and owner_bytes==32 and not is_conj and not is_neg
def bitwise_not_bool32(mask):
 import torch,cuda.tile as ct
 validate_contract(mask.shape,mask.stride(),str(mask.dtype),mask.storage_offset(),mask.untyped_storage().nbytes(),mask.is_conj(),mask.is_neg())
 assert mask.device.type=='cuda' and not mask.requires_grad
 from .kernel import invert_bool
 out=torch.empty((32,),dtype=torch.bool,device=mask.device)
 ct.launch(torch.cuda.current_stream(mask.device),(1,),invert_bool,(mask,out))
 return out
