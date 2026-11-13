"""No tensor framework imports: binding, restoration, provenance and gate tests."""
import ast,copy,json,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
from . import adapter as a,result_gate as g,gates
from . import test_cpu as prior_tests
P=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path(__file__).resolve().parent

def fake_modules(name,leaf):
 result={};parts=name.split('.')
 for i in range(1,len(parts)+1):
  key='.'.join(parts[:i]);result[key]=leaf if i==len(parts) else types.ModuleType(key)
  if i>1:setattr(result['.'.join(parts[:i-1])],parts[i-1],result[key])
 return result
class ExtensionGates(unittest.TestCase):
 @classmethod
 def setUpClass(cls):prior_tests.GateTests.setUpClass();cls.example=prior_tests.GateTests.example
 def test_all_six_actual_exit_prerequisites(self):
  r=gates.qualification(R);self.assertEqual(len(r['evidence']),8);self.assertTrue(r['parent_actual']['complete'])
  self.assertTrue(all(x['actual_exit']==0 for x in r['evidence'].values()));self.assertNotIn('all_gemm_sampler_four_gdn_composition_v1',json.dumps(r))
 def test_every_added_row_failure_rejected(self):
  for key in ('sampler_four','gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter'):
   with self.subTest(key=key):
    r=copy.deepcopy(self.example);r['shadow_request']['selected_dispatch']['components'][key]['rows'][0]['passed']=False
    self.assertFalse(g.audit(r)['passed'])
 def test_attention_native_exception_rejected(self):
  r=copy.deepcopy(self.example);r['replacement_request']['selected_dispatch']['components']['attention_core']['native_gate_calls']=1;self.assertFalse(g.audit(r)['passed'])
 def test_gdn_nonzero_initial_state_rejected(self):
  r=copy.deepcopy(self.example);r['shadow_request']['selected_dispatch']['components']['gdn_prefill_core']['rows'][0]['initial_state_all_zero']=False;self.assertFalse(g.audit(r)['passed'])
 def test_padding_empty_fill_must_remain_counted(self):
  r=copy.deepcopy(self.example);r['replacement_request']['selected_dispatch']['components']['runner_padding']['fill_rows']=[x for x in r['replacement_request']['selected_dispatch']['components']['runner_padding']['fill_rows'] if x['length']];self.assertFalse(g.audit(r)['passed'])
 def test_each_restore_missing_rejected(self):
  for key in a.ORDER:
   r=copy.deepcopy(self.example);del r['adapter_restore'][0]['components'][key];self.assertFalse(g.audit(r)['passed'])
 def test_counts_and_unique_model_attrs(self):
  self.assertEqual(len(a.PARTS),25);self.assertEqual(sum(n for k,m,n in a.PARTS),5888)
  self.assertEqual(5888+3*16+3*16+3*24,6056)
  self.assertEqual(len({m.ATTR for k,m,n in a.PARTS}),25)
 def test_optional_unqualified_apis_absent(self):
  text=(R/'adapter.py').read_text()
  self.assertIn('parent_bridge as parent_state',text);self.assertNotIn('gdn_parent_state_runtime_integration_v2',text)
 def test_runtime_unchanged(self):
  def options(path):
   tree=ast.parse(path.read_text());return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='options' for t in n.targets)]
  self.assertEqual(options(R/'run.py'),options(P/'all_gemm_sixteen_sampler_composition_v2/run.py'))

class BindingTests(unittest.TestCase):
 def test_attention_forward_distinct_from_kv_update_and_restores(self):
  from attention_scoped_runtime_v5 import adapter as attention
  module=types.ModuleType('_cpu_mock_flash')
  def forward(self,attn_metadata=None,output=None):return output
  forward.__module__=module.__name__
  def helper(**kwargs):return kwargs['out']
  module.flash_attn_varlen_func=helper
  kv=object();module.reshape_and_cache_flash=kv
  Impl=type('FlashAttentionImpl',(),{'forward':forward});owners=[(n,types.SimpleNamespace(impl=Impl())) for n in sorted(attention.MODULES)]
  for n,o in owners:o.impl.do_kv_cache_update=object()
  before={n:o.impl.do_kv_cache_update for n,o in owners};model=types.SimpleNamespace(named_modules=lambda:owners)
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{module.__name__:module}),patch.object(attention,'identity',lambda x:{}):
   x=attention.Adapter(pathlib.Path(d)/'a');x.install(model)
   self.assertEqual(len(x.bindings),9);self.assertIsNot(module.flash_attn_varlen_func,helper)
   for n,o in owners:self.assertIs(o.impl.do_kv_cache_update,before[n]);self.assertIn('forward',o.impl.__dict__)
   x.close()
   for n,o in owners:self.assertIs(o.impl.do_kv_cache_update,before[n]);self.assertNotIn('forward',o.impl.__dict__)
   self.assertIs(module.flash_attn_varlen_func,helper);self.assertIs(module.reshape_and_cache_flash,kv)
 def test_gdn_prefill_binding_does_not_replace_parent_or_decode(self):
  from gdn_prefill_core_runtime_integration_v3 import adapter as gdn
  fc=types.ModuleType('vllm.forward_context');fc.get_forward_context=lambda:None
  def forward_native(*args,**kwargs):raise AssertionError('CPU must not execute numeric path')
  named={}
  for i in range(32):
   if i%4==3:continue
   prefix=f'language_model.model.layers.{i}.linear_attn';named[prefix]=types.SimpleNamespace(prefix=prefix)
   named[prefix+'.chunk_gated_delta_rule']=types.SimpleNamespace(gdn_prefill_backend='triton',_forward_method=forward_native)
  model=types.SimpleNamespace(named_modules=lambda:list(named.items()))
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,fake_modules('vllm.forward_context',fc)),patch.object(gdn,'identity',lambda x:{}):
   x=gdn.Adapter(pathlib.Path(d)/'g');x.install(model);self.assertEqual(len(x.bindings),24)
   for k,v in named.items():
    if k.endswith('chunk_gated_delta_rule'):self.assertIsNot(v._forward_method,forward_native)
    else:self.assertEqual(set(v.__dict__),{'prefix'})
   x.close()
   for k,v in named.items():
    if k.endswith('chunk_gated_delta_rule'):self.assertIs(v._forward_method,forward_native)
 def test_runner_two_class_bindings_and_reverse_restore(self):
  from runner_padding_runtime_integration_v1 import adapter as padding
  from runner_hidden_gather_runtime_integration_v2 import adapter as gather
  mod=types.ModuleType('vllm.v1.worker.gpu.model_runner')
  def prepare_inputs(*args,**kwargs):return None
  def sample(*args,**kwargs):return None
  Runner=type('GPUModelRunner',(),dict(prepare_inputs=prepare_inputs,sample=sample));mod.GPUModelRunner=Runner
  originals={prepare_inputs:(padding.R/'prepare_inputs.original.py.txt').read_text(),sample:(gather.R/'sample.original.py.txt').read_text()}
  model=types.SimpleNamespace()
  with patch.dict(sys.modules,fake_modules(mod.__name__,mod)),patch('inspect.getsource',lambda fn:originals[fn]):
   padding.install(model,'/tmp/not-written-padding');pw=Runner.prepare_inputs
   gather.install(model,'/tmp/not-written-gather');gw=Runner.sample
   self.assertIs(Runner.prepare_inputs,pw);self.assertIsNot(gw,sample)
   gather.close(model);self.assertIs(Runner.sample,sample);self.assertIs(Runner.prepare_inputs,pw)
   padding.close(model);self.assertIs(Runner.prepare_inputs,prepare_inputs);self.assertEqual(model.__dict__,{})
 def test_partial_new_install_cleanup(self):
  helper=prior_tests.ControllerTests();helper.setUp();self.addCleanup(helper.patch.stop)
  for key in a.ORDER[17:]:
   events=[];model=types.SimpleNamespace()
   with tempfile.TemporaryDirectory() as d:
    c=a.Controller(model,d,helper.fake(events,fail_install=key))
    with self.assertRaises(RuntimeError):c.install()
    c.close();self.assertEqual([k for t,k in events if t=='close'],list(reversed(a.ORDER[:a.ORDER.index(key)+1])));self.assertEqual(model.__dict__,{})
if __name__=='__main__':unittest.main()
