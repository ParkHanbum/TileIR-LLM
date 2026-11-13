"""Actual entry schema wiring, cross-component latches and nested restoration."""
import sys,types,tempfile,pathlib,unittest
from unittest.mock import patch
from .entry_guards import EntryGuards,endpoints,METHODS
from .adapter import PARTS,ORDER
from . import runner_padding_bridge as bridge
NS=types.SimpleNamespace
class Controller:
 def __init__(self):self.model=NS();self.installed=[];self.failed=None
 def latch(self,key,error):
  if self.failed is None:self.failed=dict(component=key,error=repr(error))
 def attach(self,key,obj):
  mod=NS(ATTR='_test_'+key);setattr(self.model,mod.ATTR,obj);self.installed.append((key,mod))

class Fixture:
 def __init__(self,body):self.body=body;self.rows=[];self.failed=False;self.first_mismatch=None
for _name in METHODS:
 def _method(self,*args,**kwargs):return self.body(*args,**kwargs)
 setattr(Fixture,_name,_method)

def fixture(key,body):
 owner=NS();name='entry';obj=Fixture(body)
 if key in ('norm','attention_gate','gated_norm'):name='forward';obj.saved=[(owner,False,None,body)]
 elif key=='all_gemm':name='apply';obj.saved=[(owner,None,body)]
 elif key=='silu_mul':name='op';obj.bindings=[(owner,False,None,body)]
 elif key=='embedding':name='embedding';obj.binding=(owner,False,None,body)
 elif key=='gdn_decode_core':obj.binding=(owner,name,None,body)
 elif key=='gdn_prefill_core':name='_forward_method';obj.bindings=[(owner,False,None,body)]
 elif key in ('attention_kv','attention_core','gdn_parent_state'):obj.bindings=[(owner,name,False,None,body)]
 elif key in ('runner_padding','hidden_gather','sampler_fp32'):
  name=dict(runner_padding='prepare_inputs',hidden_gather='sample',sampler_fp32='apply_sampling_params')[key];obj.owner=owner;obj.wrapper=body
 elif key=='accepted_scatter':name='postprocess_state';obj.owner_class=owner;obj.wrapper=body
 else:obj.bindings=[(owner,name,None,body)]
 setattr(owner,name,body)
 return obj,owner,name
class EntryTests(unittest.TestCase):
 def test_every_component_failure_blocks_every_other_actual_entry_and_method(self):
  # Every25 source component fails its installed binding. All other24*12 entries
  # are then attempted; their bodies must never run. No synthetic GPU APIs used.
  for failed_key in ORDER:
   c=Controller();g=EntryGuards(c);rows={};calls=[]
   for key in ORDER:
    def body(*a,_key=key,**kw):
     calls.append(_key)
     if _key==failed_key:raise ValueError('first-source-failure')
    obj,owner,name=fixture(key,body);c.attach(key,obj);g.install(key,obj);rows[key]=(obj,owner,name,body)
   with self.assertRaisesRegex(ValueError,'first-source-failure'):getattr(rows[failed_key][1],rows[failed_key][2])()
   first=dict(c.failed)
   for key,(obj,owner,name,body) in rows.items():
    for entry in [getattr(owner,name),*[getattr(obj,m) for m in METHODS]]:
     with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):entry()
   self.assertEqual(calls,[failed_key]);self.assertEqual(c.failed,first)
   for key in reversed(ORDER):g.restore(key)
   self.assertFalse(g.records)
   for obj,owner,name,body in rows.values():self.assertIs(getattr(owner,name),body)
 def test_explicit_failure_marker_blocks_other_without_prior_throw(self):
  for marker in ('failed','first_mismatch','first_failure'):
   c=Controller();g=EntryGuards(c);calls=[];a,ao,an=fixture('attention_core',lambda:calls.append('a'));b,bo,bn=fixture('hidden_gather',lambda:calls.append('b'))
   c.attach('attention_core',a);g.install('attention_core',a);c.attach('hidden_gather',b);g.install('hidden_gather',b);setattr(a,marker,{'error':1})
   with self.assertRaisesRegex(RuntimeError,'Existing component failure'):getattr(bo,bn)()
   self.assertFalse(calls);self.assertEqual(c.failed['component'],'attention_core');g.restore('hidden_gather');g.restore('attention_core')
 def test_inflight_false_row_is_not_a_failure(self):
  c=Controller();g=EntryGuards(c);calls=[];a,ao,an=fixture('attention_core',lambda:calls.append('a'));a.rows=[dict(passed=False)];b,bo,bn=fixture('hidden_gather',lambda:calls.append('b'))
  c.attach('attention_core',a);g.install('attention_core',a);c.attach('hidden_gather',b);g.install('hidden_gather',b);getattr(bo,bn)();self.assertEqual(calls,['b']);self.assertIsNone(c.failed)
  g.restore('hidden_gather');g.restore('attention_core')
 def test_nested_guard_identity_unwinds_per_component(self):
  c=Controller();g=EntryGuards(c);owner=NS();calls=[]
  def native():calls.append('native')
  owner.method=native
  def inner():calls.append('inner');return native()
  a=NS(bindings=[(owner,'method',native,inner)]);owner.method=inner;c.attach('pos_seq',a);g.install('pos_seq',a);inner_guard=owner.method
  def outer():calls.append('outer');return inner_guard()
  b=NS(bindings=[(owner,'method',inner_guard,outer)]);owner.method=outer;c.attach('sample_counts',b);g.install('sample_counts',b)
  owner.method();self.assertEqual(calls,['outer','inner','native'])
  g.restore('sample_counts');self.assertIs(owner.method,outer);owner.method=inner_guard
  # Later component snapshot still points at the exact guarded earlier binding.
  self.assertIs(owner.method,inner_guard);g.restore('pos_seq');self.assertIs(owner.method,inner);owner.method=native;self.assertIs(owner.method,native)
 def test_guard_restore_conflict_is_not_silent(self):
  c=Controller();g=EntryGuards(c);a,o,n=fixture('hidden_gather',lambda:None);c.attach('hidden_gather',a);g.install('hidden_gather',a);setattr(o,n,lambda:None)
  with self.assertRaisesRegex(AssertionError,'restore conflict'):g.restore('hidden_gather')
 def test_actual_attention_binding_latches_before_native_entry(self):
  from attention_scoped_runtime_v5 import adapter as attention
  mod=types.ModuleType('_cpu_attention_live');calls=[]
  def forward(self,attn_metadata=None,output=None):calls.append('native');return output
  forward.__module__=mod.__name__;mod.flash_attn_varlen_func=lambda **kw:kw['out']
  Impl=type('FlashAttentionImpl',(),dict(forward=forward));owners=[(n,NS(impl=Impl())) for n in attention.MODULES];model=NS(named_modules=lambda:owners)
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{mod.__name__:mod}),patch.object(attention,'identity',lambda x:{}):
   a=attention.Adapter(pathlib.Path(d)/'a');a.install(model);c=Controller();c.attach('attention_core',a);g=EntryGuards(c);g.install('attention_core',a);c.latch('another',ValueError('failed'))
   for n,o in owners:
    with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):o.impl.forward()
   with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):mod.flash_attn_varlen_func()
   self.assertFalse(calls);g.restore('attention_core');a.close()

class BridgeTests(unittest.TestCase):
 def exercise(self,fail=False):
  pending=NS(pending=None);events=[];runner=NS(pcp_manager=None);batch=object();answer=object()
  def invoke(copied,r,s,b,desc,loras):
   self.assertIs(pending.pending['runner'],r);self.assertEqual(pending.pending['batch'],{'observed':True});return copied(r,s,b,desc,loras)
  def copied(*args):
   events.extend(['construct1','construct2','fill1','fill2']);pending.pending['calls']+=1;events.append('posseq')
   if fail:raise ValueError('copied-body-failure')
   return answer
  with patch.object(bridge.pos,'state_metadata',lambda b:{'observed':True}):
   if fail:
    with self.assertRaisesRegex(ValueError,'copied-body-failure'):bridge.combined_invoke(pending,NS(invoke=invoke),copied,runner,None,batch,None,0)
   else:self.assertIs(bridge.combined_invoke(pending,NS(invoke=invoke),copied,runner,None,batch,None,0),answer)
  self.assertIsNone(pending.pending);self.assertEqual(events,['construct1','construct2','fill1','fill2','posseq'])
 def test_context_sideeffects_once(self):self.exercise()
 def test_context_restored_on_failure(self):self.exercise(True)
 def test_missing_and_duplicate_helper_count_rejected(self):
  for count in (0,2):
   pending=NS(pending=None)
   def invoke(*args):pending.pending['calls']=count
   with patch.object(bridge.pos,'state_metadata',lambda b:{}):
    with self.assertRaises(AssertionError):bridge.combined_invoke(pending,NS(invoke=invoke),None,NS(pcp_manager=None),None,None,None,0)
   self.assertIsNone(pending.pending)
 def test_two_binding_native_source_chain_install_restore(self):
  from .test_extensions import fake_modules
  from pos_seq_runtime_integration_v1 import adapter as pos
  # Use the actual pos_seq install_classes and bridge install paths; only public
  # source identity and metadata are substituted because this is a CPU mock.
  mod=types.ModuleType('vllm.v1.worker.gpu.model_runner')
  def native_prepare(*args):return None
  def native_helper(*args):return None
  Runner=type('GPUModelRunner',(),dict(prepare_inputs=native_prepare));mod.GPUModelRunner=Runner;mod.prepare_pos_seq_lens=native_helper
  manifest=pathlib.Path(pos.__file__).parent.parent/'pos_seq_runtime_capture_v2/posSeqCapture02/bank/MANIFEST.json';import json
  source=json.loads(manifest.read_text())['samples'][0]['source'];model=NS()
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,fake_modules(mod.__name__,mod)),patch.object(pos,'identity',lambda f:source['prepare'] if f is native_prepare else source['helper']):
   a=pos.Adapter(pathlib.Path(d)/'pos');a.install_classes(Runner,mod);setattr(model,pos.ATTR,a)
   c=Controller();c.model=model;c.installed=[('pos_seq',pos)];g=EntryGuards(c);g.install('pos_seq',a);old=Runner.prepare_inputs
   with patch('inspect.getsource',lambda f:(bridge.padding.R/'prepare_inputs.original.py.txt').read_text()):bridge.install(model,str(pathlib.Path(d)/'pad'))
   pad=getattr(model,bridge.ATTR);c.installed.append(('runner_padding',bridge));g.install('runner_padding',pad)
   c.latch('other',ValueError('fail'))
   with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):Runner.prepare_inputs()
   g.restore('runner_padding');bridge.close(model);self.assertIs(Runner.prepare_inputs,old)
   g.restore('pos_seq');pos.close(model);self.assertIs(Runner.prepare_inputs,native_prepare);self.assertIs(mod.prepare_pos_seq_lens,native_helper)
if __name__=='__main__':unittest.main()
