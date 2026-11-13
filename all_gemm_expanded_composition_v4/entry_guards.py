"""Fail-closed process-local guards on each actual installed entry and numeric method.
Each binding guard is added immediately after its owning component installation.
Nested later components capture that guarded binding. Unwind per component in reverse.
"""
import functools,inspect,types
METHODS=('apply','apply_gate','enter','cat','dispatch','invoke','gather','construct','fill_padding','copy_logits','scatter','observe')

def endpoints(key,obj):
 if key in ('norm','attention_gate','gated_norm'):
  return [(m,'forward',new) for m,had,prior,new in obj.saved]
 if key=='all_gemm':return [(q,'apply',new) for q,old,new in obj.saved]
 if key=='silu_mul':return [(m,'op',new) for m,had,prior,new in obj.bindings]
 if key=='embedding':
  m,had,prior,new=obj.binding;return [(m,'embedding',new)]
 if key=='gdn_decode_core':
  m,name,old,new=obj.binding;return [(m,name,new)]
 if key=='gdn_prefill_core':return [(m,'_forward_method',new) for m,had,prior,new in obj.bindings]
 if key in ('attention_kv','attention_core','gdn_parent_state'):
  return [(m,name,new) for m,name,had,prior,new in obj.bindings]
 if key in ('runner_padding','hidden_gather','sampler_fp32'):
  return [(obj.owner,dict(runner_padding='prepare_inputs',hidden_gather='sample',sampler_fp32='apply_sampling_params')[key],obj.wrapper)]
 if key=='accepted_scatter':return [(obj.owner_class,'postprocess_state',obj.wrapper)]
 return [(m,name,new) for m,name,old,new in obj.bindings]

class EntryGuards:
 def __init__(self,controller):self.controller=controller;self.records=[];self.map={}
 def check(self):
  c=self.controller;assert c.failed is None,('Composition first failure latched',c.failed)
  # In-flight rows may intentionally have passed=False. Only explicit component failure
  # markers are safe during nested entry; final row checks remain in finish/save guards.
  for key,module in c.installed:
   obj=getattr(c.model,module.ATTR,None)
   if getattr(obj,'failed',None) or getattr(obj,'first_mismatch',None) or getattr(obj,'first_failure',None):
    err=RuntimeError('Existing component failure '+key);c.latch(key,err);raise err
 def add(self,key,obj,name,expected,kind):
  current=getattr(obj,name)
  same_method=(kind=='component_method' and getattr(expected,'__self__',None) is obj and getattr(current,'__self__',None) is obj and getattr(current,'__func__',None) is getattr(expected,'__func__',None))
  assert current is expected or same_method,(key,name,'Unexpected installed endpoint')
  had=name in obj.__dict__;prior=obj.__dict__.get(name)
  @functools.wraps(expected)
  def guarded(*args,**kwargs):
   self.check()
   try:return expected(*args,**kwargs)
   except BaseException as error:self.controller.latch(key,error);raise
  guarded._composition_entry_key=key
  setattr(obj,name,guarded);self.records.append((key,obj,name,had,prior,guarded))
  self.map.setdefault(key,[]).append(dict(kind=kind,attribute=name))
 def install(self,key,obj):
  # A same-named instance field is not a callable boundary. Preserve observers,
  # counters, None and even callable data exactly. Validate method identities
  # before adding any guards; static/class methods are outside this contract.
  methods=[]
  for name in METHODS:
   descriptor=inspect.getattr_static(type(obj),name,None)
   if not isinstance(descriptor,types.FunctionType):continue
   value=getattr(obj,name)
   assert isinstance(value,types.MethodType) and value.__self__ is obj and value.__func__ is descriptor,(key,name,'Unexpected component method identity')
   methods.append((name,value))
  actual=endpoints(key,obj);assert actual,(key,'No actual entry guarded')
  for target,name,new in actual:self.add(key,target,name,new,'installed_binding')
  for name,value in methods:self.add(key,obj,name,value,'component_method')
 def restore(self,key):
  errors=[];remaining=[]
  for row in reversed(self.records):
   k,obj,name,had,prior,new=row
   if k!=key:remaining.append(row);continue
   try:
    assert getattr(obj,name) is new,(key,name,'Entry guard restore conflict')
    if had:setattr(obj,name,prior)
    else:delattr(obj,name)
   except BaseException as error:errors.append(repr(error))
  self.records=list(reversed(remaining));assert not errors,errors
 def receipt(self):return {k:list(v) for k,v in self.map.items()}
