"""Four original-address replacements, no native helper oracle or fallback."""
import functools,hashlib,inspect,json
from pathlib import Path
from collections import Counter
from residual_rope_combine_bias_port_v4.shadow import Shadow
from residual_rope_combine_bias_port_v4 import api
from .oracle import expected
CURRENT=None
LABELS=('prepare_rope_positions','combine_sampled_and_draft_tokens','bias','post_update')
class Replacement(Shadow):
 def __init__(self,output,owners):super().__init__(output);self.owners=owners;self.controller=None;self.owned_checks=[]
 def install(self):
  global CURRENT
  assert CURRENT is None
  CURRENT=self
  try:
   super().install()
   from vllm.v1.worker.gpu import model_runner,input_batch
   from tileir_vllm_post_update_v3.contract import NAMES
   original=model_runner.post_update;assert original is input_batch.post_update and tuple(inspect.signature(original).parameters)==NAMES
   new=self.wrap(original,'post_update');self.bindings.append((model_runner,'post_update',original,new));model_runner.post_update=new
  except BaseException:self.close();raise
 def latch(self,label,error):
  self.failed=True
  if self.controller is not None:self.controller.latch('residual_four/'+label,error)
 def guard(self,label):
  assert not self.failed and self.controller is not None and self.controller.failed is None and self.controller.active,'Shared first-failure/active guard'
  assert sum(x['label']==label for x in self.rows)<16,'Per-helper call cap'
  for namespace,name,expected in self.owned_checks:assert namespace[name] is expected,'Owned helper binding changed'
 def wrap(self,original,label):
  sig=inspect.signature(original)
  @functools.wraps(original)
  def wrapped(*args,**kwargs):
   if self.failed or (self.controller is not None and self.controller.failed is not None):raise RuntimeError('Shared failure latched')
   if not self.active:return original(*args,**kwargs)
   try:
    self.guard(label);bound=sig.bind(*args,**kwargs);bound.apply_defaults();return self.call(label,original,bound)
   except BaseException as e:self.latch(label,e);raise
  return wrapped
 def bind_controller(self,controller):
  self.controller=controller
  # Follow only Python function closure/wrapped links of the installed runner binding.
  part=next(p for p in controller.parts if p.key=='runner_padding')
  funcs=[b[-1] for b in part.bindings];seen=set();copies=[]
  while funcs:
   fn=funcs.pop()
   if not inspect.isfunction(fn) or id(fn) in seen:continue
   seen.add(id(fn))
   if fn.__code__.co_filename.endswith('/runner_padding_runtime_integration_v1/prepare_inputs.copied.py.txt'):copies.append(fn)
   wrapped=getattr(fn,'__wrapped__',None)
   if wrapped is not None:funcs.append(wrapped)
   funcs.extend(v for v in inspect.getclosurevars(fn).nonlocals.values()if inspect.isfunction(v))
  assert len(copies)==1,'Exact owned prepare_inputs copy required'
  expected=next(new for obj,name,old,new in self.bindings if name=='combine_sampled_and_draft_tokens')
  assert copies[0].__globals__['combine_sampled_and_draft_tokens'] is expected
  self.owned_checks=[(copies[0].__globals__,'combine_sampled_and_draft_tokens',expected)]
  for obj,name,old,new in self.bindings:self.owned_checks.append((vars(obj),name,new))
  from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
  bias=next(new for obj,name,old,new in self.bindings if name=='apply_logit_bias')
  assert LogitBiasState.apply.__globals__['apply_logit_bias'] is bias
  self.owned_checks.append((LogitBiasState.apply.__globals__,'apply_logit_bias',bias))
  return dict(shared_latch=True,owned_copy_identity=True,per_helper_cap=16)
 def close(self):
  global CURRENT
  result=super().close()
  if CURRENT is self:CURRENT=None
  return result
 def summary(self):
  return dict(rows=self.rows,calls=dict(Counter(x['label']for x in self.rows)),restored=self.restored,failed=self.failed,complete=not self.failed and len(self.rows)==64 and all(x['passed']for x in self.rows),scope='Original address CPU-oracle correctness, native helpers0, no performance/ABI claim')
 def call(self,label,original,bound):
  import torch
  a=bound.arguments;row=dict(label=label,native_calls=0,candidate_calls=0,passed=False);self.rows.append(row);self.save()
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
   def raw(t):return t.detach().view(torch.uint8).cpu().contiguous().numpy().tobytes()
   before={k:raw(t)for k,t in vals.items()};outputs,return_expected=expected(label,before,a)
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
   row['output_raw']=all(raw(vals[k])==v for k,v in outputs.items());row['readonly_raw']=all(raw(t)==before[k]for k,t in vals.items()if k not in outputs)
   if return_expected is None:row['return_contract']=ret is None
   else:
    row['return_contract']=tuple(ret.shape)==(32,)and ret.dtype==torch.int64 and ret.device==vals['input_ids'].device and ret.stride()==(1,)and ret.storage_offset()==0 and all(ret.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in vals.values())and raw(ret)==return_expected
   row['output_bytes']=sum(map(len,outputs.values()));row['readonly_bytes']=sum(len(v)for k,v in before.items()if k not in outputs)
   row['passed']=row['output_raw']and row['readonly_raw']and row['return_contract']
   assert row['passed'],'Original-address CPU oracle mismatch'
   return ret
  except BaseException as e:row['error']=repr(e);raise
  finally:self.save()
 def finish(self):
  assert not self.failed and Counter(x['label']for x in self.rows)==Counter({x:16 for x in LABELS})
  assert all(x['passed']and x['native_calls']==0 and x['candidate_calls']==1 for x in self.rows)
  self.save();return self.summary()

def bind_lean_controller(model):
 from lean_public_composition_v2.adapter import ATTR
 assert CURRENT is not None
 return CURRENT.bind_controller(getattr(model,ATTR))
