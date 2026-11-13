"""Lean subclass: original API safety/value/ownership/address guards unchanged."""
from collections import Counter
from residual_four_lean26_runtime_v1.adapter import Replacement as Qualified,api,LABELS
class Replacement(Qualified):
 def save(self):pass # No per-boundary JSON proof I/O; final summary remains in RESULT.
 def summary(self):
  return dict(rows=self.rows,calls=dict(Counter(x['label']for x in self.rows)),restored=self.restored,failed=self.failed,complete=not self.failed and len(self.rows)==64 and all(x['passed']for x in self.rows),raw_rechecked=False,cpu_oracle_calls=0,scope='Lean operational contracts; prior actual raw proof separate')
 def call(self,label,original,bound):
  import torch
  a=bound.arguments;row=dict(label=label,native_calls=0,candidate_calls=0,passed=False);self.rows.append(row)
  try:
   if label=='prepare_rope_positions':
    owner=a['self'];assert owner.num_dims==3 and owner.max_model_len==128;self.owners.refs.append(owner)
    vals={k:v for k,v in a.items()if k!='self'};vals.update({'self.positions':owner.positions,'self.prefill_positions.gpu':owner.prefill_positions.gpu,'self.prefill_delta.gpu':owner.prefill_delta.gpu})
    required=('self.prefill_positions.gpu','self.prefill_delta.gpu')
   else:vals={k:v for k,v in a.items()if isinstance(v,torch.Tensor)};required=()
   if label=='post_update':
    from tileir_vllm_post_update_v3.contract import NAMES,validate_metadata,validate_values
    args=tuple(a[n]for n in NAMES);validate_metadata(args);validate_values(torch,args)
   else:
    api.validate_layout(label,vals);api.validate_values(label,vals,a)
    if label=='combine_sampled_and_draft_tokens':required=('prefill_len',)
    if label=='bias':required=('num_allowed_token_ids','num_logit_bias','min_lens','num_stop_token_ids','restore_when_all_masked')
   row['ownership']=self.owners.retain(vals,required);row['preflight_passed']=True
   signatures={k:(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))for k,t in vals.items()}
   row['candidate_calls']=1
   if label=='prepare_rope_positions':
    ret=api.prepare_rope_positions(owner.positions,owner.prefill_positions.gpu,owner.prefill_delta.gpu,**{k:v for k,v in a.items()if k!='self'})
   elif label=='combine_sampled_and_draft_tokens':ret=api.combine_sampled_and_draft_tokens(**a)
   elif label=='bias':ret=api.apply_logit_bias(**a)
   else:
    import cuda.tile as ct
    from tileir_vllm_post_update_v3.kernels import post_update_kernel
    flat=tuple(t.view(-1)for t in args)
    with torch.cuda.device(args[0].device):ct.launch(torch.cuda.current_stream(args[0].device),(32,1,1),post_update_kernel,flat+(1,128,248320,True,True))
    ret=None
   assert signatures=={k:(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))for k,t in vals.items()},'Original tensor/storage identity changed'
   row['original_addresses_preserved']=True
   if label!='combine_sampled_and_draft_tokens':row['return_contract']=ret is None
   else:
    row['return_contract']=tuple(ret.shape)==(32,)and ret.dtype==torch.int64 and ret.device==vals['input_ids'].device and ret.stride()==(1,)and ret.storage_offset()==0 and all(ret.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in vals.values())
   row['passed']=row['return_contract']
   assert row['passed'],'Lean return metadata mismatch'
   return ret
  except BaseException as e:row['error']=repr(e);raise

