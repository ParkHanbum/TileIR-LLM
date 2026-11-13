"""48 selected native calls, same-preimage candidate shadows, never replacement.
No blanket finite check: bias is permitted to create -Inf. Full raw bytes decide.
"""
import functools,hashlib,importlib,inspect,json
from pathlib import Path
from collections import Counter
from . import api
R=Path(__file__).resolve().parent
SPECS=json.loads((R.parent/'residual_rope_combine_bias_contract_v1/HELPER-SPECS.json').read_text())
class Shadow:
 def __init__(self,output):self.output=Path(output);self.bindings=[];self.rows=[];self.active=False;self.restored=False;self.failed=False
 def save(self):
  self.output.mkdir(exist_ok=True)
  (self.output/'SHADOW.json').write_text(json.dumps(self.summary(),indent=2)+'\n')
 def summary(self):return dict(rows=self.rows,calls=Counter(x['label'] for x in self.rows),restored=self.restored,failed=self.failed,complete=not self.failed and len(self.rows)==48 and all(x.get('passed') for x in self.rows))
 def install(self):
  try:
   for spec in SPECS:
    p=Path(spec['source_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['source_sha256']
    obj=importlib.import_module(spec['module'])
    for bit in spec['owner'].split('.'):
     if bit:obj=getattr(obj,bit)
    name=spec['attribute'];old=getattr(obj,name);assert list(inspect.signature(old).parameters)==spec['parameter_names']
    new=self.wrap(old,spec['label']);self.bindings.append((obj,name,old,new));setattr(obj,name,new)
  except BaseException:self.close();raise
 def wrap(self,original,label):
  sig=inspect.signature(original)
  @functools.wraps(original)
  def wrapped(*args,**kwargs):
   if not self.active:return original(*args,**kwargs)
   if self.failed:raise RuntimeError('Failed helper shadow cannot resume')
   try:
    bound=sig.bind(*args,**kwargs);bound.apply_defaults()
    return self.call(label,original,bound)
   except BaseException:self.failed=True;raise
  return wrapped
 def call(self,label,original,bound):
  import torch
  a=bound.arguments
  if label=='prepare_rope_positions':
   owner=a['self'];vals={k:v for k,v in a.items() if k!='self'}
   vals.update({'self.positions':owner.positions,'self.prefill_positions.gpu':owner.prefill_positions.gpu,'self.prefill_delta.gpu':owner.prefill_delta.gpu})
   assert owner.num_dims==3 and owner.max_model_len==128,'Actual owner scalar contract'
   target='self.positions'
  else:vals={k:v for k,v in a.items() if isinstance(v,torch.Tensor)};target='input_ids' if label=='combine_sampled_and_draft_tokens' else 'logits'
  try:
   api.validate_layout(label,vals)
   api.validate_values(label,vals,a) # All public value/scope checks precede native or candidate.
  except BaseException as e:
   self.rows.append(dict(label=label,native_calls=0,candidate_calls=0,passed=False,stage='preflight',error=repr(e)));self.save();raise
  def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
  before={k:hashlib.sha256(raw(t)).hexdigest() for k,t in vals.items()}
  pre_target=vals[target].detach().cpu().clone()
  clones={k:t.clone(memory_format=torch.preserve_format) for k,t in vals.items()}
  assert all(raw(t)==raw(clones[k]) for k,t in vals.items()),'Preimage copy mismatch'
  row=dict(label=label,native_calls=0,candidate_calls=0,passed=False,preimage_exact=True,preflight_passed=True);self.rows.append(row);self.save()
  try:
   row['native_calls']=1;result=original(*bound.args,**bound.kwargs)
   if label=='prepare_rope_positions':
    fn=api.prepare_rope_positions
    candidate_args=dict(positions=clones['self.positions'],prefill_positions=clones['self.prefill_positions.gpu'],prefill_delta=clones['self.prefill_delta.gpu'],**{k:clones[k] for k in a if k!='self'})
   else:
    fn=api.combine_sampled_and_draft_tokens if label=='combine_sampled_and_draft_tokens' else api.apply_logit_bias
    candidate_args={k:clones[k] if k in clones else v for k,v in a.items()}
   row['candidate_calls']=1;candidate=fn(**candidate_args)
   row['native_readonly_preserved']=all(hashlib.sha256(raw(t)).hexdigest()==before[k] for k,t in vals.items() if k!=target)
   row['candidate_readonly_preserved']=all(hashlib.sha256(raw(t)).hexdigest()==before[k] for k,t in clones.items() if k!=target)
   row['mutated_buffer_raw_equal']=raw(vals[target])==raw(clones[target])
   row['input_layout_preserved']=True
   api.validate_layout(label,vals);api.validate_layout(label,clones)
   if label=='combine_sampled_and_draft_tokens':
    row['return_raw_equal']=raw(result)==raw(candidate)
    row['fresh_output']=all(result.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for t in vals.values()) and all(candidate.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for t in clones.values())
    row['return_layout']=tuple(result.shape)==tuple(candidate.shape)==(32,) and result.dtype==candidate.dtype==torch.int64 and result.stride()==candidate.stride()==(1,) and result.storage_offset()==candidate.storage_offset()==0
   else:row['return_raw_equal']=row['fresh_output']=row['return_layout']=result is None and candidate is None
   row['passed']=all(row[k] for k in ('native_readonly_preserved','candidate_readonly_preserved','mutated_buffer_raw_equal','input_layout_preserved','return_raw_equal','fresh_output','return_layout'))
   if not row['passed']:
    bank=self.output/('mismatch'+str(len(self.rows)));bank.mkdir()
    torch.save(vals[target].detach().cpu(),bank/'native.pt');torch.save(clones[target].detach().cpu(),bank/'candidate.pt')
    raise AssertionError('Residual helper raw/contract HOLD')
   row['original_return_preserved']=True;return result
  except BaseException as e:
   row['error']=repr(e)
   bank=self.output/('failed_preimage'+str(len(self.rows)));bank.mkdir(exist_ok=True)
   torch.save(pre_target,bank/'target_before.pt')
   for k,t in vals.items():
    if k!=target:torch.save(t.detach().cpu(),bank/(k+'.pt'))
   row['preimage_path']=str(bank);raise
  finally:self.save()
 def finish(self):
  assert not self.failed
  assert Counter(x['label'] for x in self.rows)==Counter({x['label']:16 for x in SPECS})
  assert all(x['passed'] and x['native_calls']==x['candidate_calls']==1 and x['original_return_preserved'] for x in self.rows)
  self.save();return self.summary()
 def close(self):
  self.active=False;errors=[]
  for obj,name,old,new in reversed(self.bindings):
   if getattr(obj,name)is new:setattr(obj,name,old)
   else:errors.append(name)
  self.bindings=[];self.restored=not errors
  if errors:raise RuntimeError('Foreign helper bindings: '+str(errors))
