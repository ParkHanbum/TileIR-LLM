from .contract import SHAPES,STRIDES,validate_values
from tileir_gdn_parent_state_v2.contract import raw_disjoint
def validate(tensors):
 import torch
 assert len(tensors)==7
 for i,t in enumerate(tensors):
  assert t.is_cuda and t.device==tensors[0].device and tuple(t.shape)==SHAPES[i] and tuple(t.stride())==STRIDES[i]
  assert t.dtype==(torch.int64 if i==2 else torch.int32) and not t.requires_grad and not t.is_conj() and not t.is_neg()
  assert t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()
 raw_disjoint(*tensors)
 validate_values(*[tensors[i].detach().cpu().tolist() for i in (2,3,5,6)])
def prepare_prefill_inputs(input_ids,next_prefill_tokens,idx_mapping,query_start_loc,all_token_ids,prefill_len,num_computed_tokens):
 import torch,cuda.tile as ct
 from .kernel import prepare_prefill_tokens
 ts=(input_ids,next_prefill_tokens,idx_mapping,query_start_loc,all_token_ids,prefill_len,num_computed_tokens);validate(ts)
 ct.launch(torch.cuda.current_stream(input_ids.device),(32,),prepare_prefill_tokens,(*ts,1))
 return None
