"""Narrow observed-layout functional API, checked live values, no fallback.
CUDA imports are lazy. First compile and all numerical checks are pending.
"""
import json
from pathlib import Path
from . import guards
LAYOUTS=json.loads(Path(__file__).with_name('OBSERVED-LAYOUTS.json').read_text())
def validate_layout(label,values):
 expected=LAYOUTS[label]
 if set(values)!=set(expected):raise ValueError('tensor argument set')
 spans=[]
 for name,t in values.items():
  d=expected[name]
  actual=dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),requires_grad=bool(t.requires_grad),is_conj=t.is_conj(),is_neg=t.is_neg())
  if actual!=d:raise ValueError('unqualified layout '+name)
  st=t.untyped_storage();a=st.data_ptr();z=st.nbytes()
  if z:
   for _,b,w in spans:
    if max(a,b)<min(a+z,b+w):raise ValueError('overlapping arguments')
   spans.append((name,a,z))
def host(t):return t.detach().cpu().tolist()
def launch(fn,args,grid,device):
 import torch
 import cuda.tile as ct
 with torch.cuda.device(device):ct.launch(torch.cuda.current_stream(device),grid,fn,args)
def prepare_rope_positions(positions,prefill_positions,prefill_delta,idx_mapping,query_start_loc,prefill_lens,num_computed_tokens):
 vals=dict(idx_mapping=idx_mapping,query_start_loc=query_start_loc,prefill_lens=prefill_lens,num_computed_tokens=num_computed_tokens)
 vals.update({'self.positions':positions,'self.prefill_positions.gpu':prefill_positions,'self.prefill_delta.gpu':prefill_delta})
 validate_layout('prepare_rope_positions',vals)
 guards.rope(*map(host,(idx_mapping,query_start_loc,prefill_lens,num_computed_tokens,prefill_delta)))
 from .kernels import rope
 launch(rope,(positions,prefill_positions,prefill_delta,idx_mapping,query_start_loc,prefill_lens,num_computed_tokens),(32,3,1),positions.device)
 return None

def combine_sampled_and_draft_tokens(input_ids,idx_mapping,last_sampled_tokens,query_start_loc,seq_lens,prefill_len,draft_tokens,cu_num_logits,num_logits,num_new_sampled_tokens=1):
 vals={k:v for k,v in locals().items() if k not in ('num_logits','num_new_sampled_tokens')}
 validate_layout('combine_sampled_and_draft_tokens',vals)
 guards.combine(host(idx_mapping),host(query_start_loc),host(seq_lens),host(prefill_len),host(cu_num_logits),[x[0] for x in host(last_sampled_tokens)],num_logits,num_new_sampled_tokens)
 import torch
 out=torch.empty((32,),dtype=torch.int64,device=input_ids.device)
 from .kernels import combine
 launch(combine,(input_ids,idx_mapping,last_sampled_tokens,query_start_loc,seq_lens,prefill_len,cu_num_logits,out),(32,1,1),input_ids.device)
 return out

def apply_logit_bias(logits,expanded_idx_mapping,pos,num_allowed_token_ids,allowed_token_ids,num_logit_bias,logit_bias_token_ids,logit_bias,min_lens,num_stop_token_ids,restore_when_all_masked,stop_token_ids,check_all_masked_rows=False):
 vals={k:v for k,v in locals().items() if k!='check_all_masked_rows'}
 validate_layout('bias',vals)
 guards.bias(*map(host,(expanded_idx_mapping,pos,num_allowed_token_ids,num_logit_bias,min_lens,num_stop_token_ids,stop_token_ids,restore_when_all_masked)),check_all_masked_rows)
 from .kernels import bias
 launch(bias,(logits,expanded_idx_mapping,pos,min_lens,num_stop_token_ids,stop_token_ids),(32,1,1),logits.device)
 return None
