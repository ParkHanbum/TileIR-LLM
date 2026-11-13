"""Parent endpoint wrapper around exact already-installed qualified core guard.
Only wrapper composition/receipts differ from standalone no-oracle v2. Its observe,
invoke and four numeric APIs are inherited unchanged. No operator/check disabled.
"""
from pathlib import Path
import json
from gdn_parent_state_no_oracle_v2 import adapter as base
from gdn_prefill_core_runtime_integration_v3 import adapter as core_module
from .parent_contract import parent_gate,verify_qualified_parent
ATTR='_gdn_parent_composed_v3';ROOT=Path(__file__).resolve().parent
class Adapter(base.Adapter):
 def __init__(self,output):
  super().__init__(output);self.core_adapter=None;self.core_links=[];self.expected_core_bindings=[]
 def install(self,model):
  core_adapter=getattr(model,core_module.ATTR);self.core_adapter=core_adapter
  named=dict(model.named_modules());expected={}
  for path in core_adapter.modules:
   assert path.endswith('.linear_attn.chunk_gated_delta_rule');core=named[path]
   matches=[r for r in core_adapter.bindings if r[0] is core];assert len(matches)==1
   own=matches[0][3];current=core._forward_method
   assert getattr(current,'_composition_entry_key',None)=='gdn_prefill_core' and current.__wrapped__ is own
   assert current.__name__=='forward_native' and getattr(own,'__wrapped__',None) is not None
   parent=path.rsplit('.',1)[0];expected[id(core)]=(parent,path,current)
  assert len(expected)==24
  result=super().install(model)
  for obj,key,had,old,new in self.bindings:
   if key!='_forward_method':continue
   parent,path,current=expected[id(obj)]
   assert had and old is current and new.__wrapped__ is current and obj._forward_method is new
   self.expected_core_bindings.append((obj,current));self.core_links.append(dict(parent=parent,core_path=path,exact_installed_guard=True,captured_by_parent=True))
  assert len(self.core_links)==24
  result.update(core_native=False,core_delegated=True,core_links=self.core_links,scope='Parent endpoint wraps exact existing core shadow/replacement guard; not native core')
  return result
 def core_phase(self):
  a=self.core_adapter
  if a is None:return dict(rows=[],native=0,tile=0)
  return dict(rows=[dict(path=r['path'],passed=r['passed'],phase=r['phase']) for r in a.rows if 'passed' in r],native=a.native_gate_calls,tile=a.tileir_gate_calls)
 def save(self):
  c=self.core_phase()
  r=dict(mode=self.mode,rows=self.rows,failed=self.failed,native_only=False,parent_targets_native=self.mode=='shadow',owner_payload_bytes=0,metadata_only=False,core_native=False,core_delegated=True,native_invert_retained=self.mode=='shadow',target_oracles=0,supported_calls=len(self.rows),logical_suboperation_calls=4*len(self.rows),parent_api_calls=4*len(self.rows),supported_vendor_fallback_calls=0,shadow_pass=self.baseline_pass,core_links=self.core_links,core_phase_rows=c['rows'],core_native_calls=c['native'],core_tileir_calls=c['tile'])
  (self.out/((self.mode or 'install')+'-PROGRESS.json')).write_text(json.dumps(r,indent=2));return r
 def start(self,mode):
  assert self.core_adapter.mode==mode,'Core phase must start before parent'
  super().start(mode)
  return dict(mode=mode,native_only=False,parent_targets_native=mode=='shadow',core_delegated=True,target_oracles=0)
 def finish(self):
  self.guard()
  try:
   assert self.active is None and self.core_adapter.mode==self.mode and not self.core_adapter.failed
   assert self.core_adapter.shadow_pass,'Core phase numerical gate must finish first'
   r=self.save();assert parent_gate(r),'Parent/core composition count/order/route HOLD'
   if self.mode=='shadow':self.baseline_pass=True
   return self.save()
  except BaseException as error:self.fail(error);raise
 def close(self):
  super().close()
  assert all(obj._forward_method is prior for obj,prior in self.expected_core_bindings),'Parent must restore guarded core before core teardown'
  return dict(restored=True,core_guard_restored=True)

def install(model,output):
 verify_qualified_parent(ROOT) # Pending receipt rejects before torch import/binding mutation.
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 try:return a.close()
 finally:delattr(model,ATTR)
