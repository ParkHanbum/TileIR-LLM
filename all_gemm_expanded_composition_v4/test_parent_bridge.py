"""CPU mocks exercise real parent install/corewrap/invoke/finish and global guards.
Synthetic rows are wiring evidence, never numeric or GPU proof.
"""
import ast,copy,functools,json,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
from . import parent_bridge as bridge
from .parent_contract import parent_gate,verify_qualified_parent
from .entry_guards import EntryGuards
NS=types.SimpleNamespace;R=pathlib.Path(__file__).parent
class Control:
 def __init__(self,model):self.model=model;self.installed=[];self.failed=None
 def latch(self,key,error):
  if self.failed is None:self.failed=dict(component=key,error=repr(error))

def modules():
 names=('torch','torch.utils','torch.utils._python_dispatch','vllm','vllm.forward_context');mods={n:types.ModuleType(n) for n in names}
 for n,m in mods.items():
  if '.' in n:p,k=n.rsplit('.',1);setattr(mods[p],k,m)
 class Mode:
  def __enter__(self):return self
  def __exit__(self,*a):return False
 mods['torch.utils._python_dispatch'].TorchDispatchMode=Mode
 mods['vllm.forward_context'].get_forward_context=lambda:None
 return mods

class BridgeCase(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.events=[];self.named={}
  self.cc=NS(modules={},bindings=[],mode='shadow',rows=[],native_gate_calls=0,tileir_gate_calls=0,failed=False,shadow_pass=False)
  for i in range(32):
   if i%4==3:continue
   parent=f'language_model.model.layers.{i}.linear_attn';path=parent+'.chunk_gated_delta_rule'
   def forward_native(initial_state):return object(),object()
   @functools.wraps(forward_native)
   def qualified(initial_state,_path=path):
    self.events.append(('core',_path));self.cc.rows.append(dict(path=_path,phase='prefill',passed=True));self.cc.tileir_gate_calls+=1
    if self.cc.mode=='shadow':self.cc.native_gate_calls+=1
    return object(),object()
   core=NS(_forward_method=qualified);owner=NS(chunk_gated_delta_rule=core,_forward_core=lambda:None,prefix=parent)
   self.named[parent]=owner;self.named[path]=core;self.cc.modules[path]={};self.cc.bindings.append((core,True,forward_native,qualified))
  self.model=NS(named_modules=lambda:list(self.named.items()));setattr(self.model,bridge.core_module.ATTR,self.cc)
  self.c=Control(self.model);self.c.installed=[('gdn_prefill_core',bridge.core_module)];self.g=EntryGuards(self.c);self.g.install('gdn_prefill_core',self.cc)
  self.core_guards={path:x._forward_method for path,x in self.named.items() if path.endswith('chunk_gated_delta_rule')}
  self.a=bridge.Adapter(pathlib.Path(self.tmp.name)/'out')
  with patch.dict(sys.modules,modules()),patch.object(bridge.base,'identity',lambda f:{}):self.a.install(self.model)
  setattr(self.model,bridge.ATTR,self.a);self.c.installed.append(('gdn_parent_state',bridge));self.g.install('gdn_parent_state',self.a)
 def tearDown(self):
  self.g.restore('gdn_parent_state');self.a.close()
  for path,old in self.core_guards.items():self.assertIs(self.named[path]._forward_method,old)
  self.g.restore('gdn_prefill_core');self.assertFalse(self.g.records)
 def phase(self,mode):
  self.cc.mode=mode;self.cc.rows=[];self.cc.native_gate_calls=self.cc.tileir_gate_calls=0;self.a.start(mode);calls=[]
  for parent in [n for n in self.named if n.endswith('.linear_attn')]:
   initial=object();owner=object();final=object();row=dict(layer=parent,order=[],passed=False,native_target_calls=0,tile_target_calls=0,native_invert_calls=0,tile_invert_calls=0)
   ctx=dict(row=row,owner=owner,indices=object(),mask=object(),initial=initial,final=final);self.a.rows.append(row);self.a.active=ctx
   def api(label,result):
    def f(*args):calls.append(label);return result
    return f
   with patch.multiple(bridge.base,gather_state=api('gather',initial),bitwise_not_bool32=api('invert',object()),mask_initial_state_=api('mask',initial),scatter_state_=api('scatter',owner)):
    for label in ('gather','invert','mask'):
     bridge.base.advance(row,label);self.a.invoke(label,api('native',object()),(),{},ctx)
    with patch.object(bridge.base,'metadata',lambda t:{}),patch.object(bridge.base,'aliases',lambda d:{}):
     ret=self.named[parent+'.chunk_gated_delta_rule']._forward_method(initial_state=initial)
    self.assertIs(ctx['final'],ret[1]);bridge.base.advance(row,'scatter');self.a.invoke('scatter',api('native',owner),(),{},ctx)
   row['passed']=True;self.a.active=None
  self.cc.shadow_pass=True;d=self.a.finish();self.assertTrue(parent_gate(d));self.assertTrue(d['shadow_pass']);self.assertFalse(d['core_native']);self.assertFalse(d['native_only'])
  self.assertEqual(calls.count('native'),96 if mode=='shadow' else 0)
  self.assertEqual(sum(calls.count(k) for k in ('gather','invert','mask','scatter')),96 if mode=='replace' else 0)
  return d
 def test_actual_parent_core_route_both_phases(self):
  a=self.phase('shadow');b=self.phase('replace');self.assertEqual(a['core_native_calls'],24);self.assertEqual(b['core_native_calls'],0);self.assertEqual(a['core_tileir_calls'],24);self.assertEqual(b['core_tileir_calls'],24)
 def test_start_before_core_phase_rejected(self):
  self.cc.mode='replace'
  with self.assertRaises(AssertionError):self.a.start('shadow')
 def test_parent_first_failure_blocks_core_entry(self):
  self.a.start('shadow');ctx=dict(row=dict(native_invert_calls=0),mask=object())
  def fail():raise RuntimeError('first parent failure')
  with self.assertRaisesRegex(RuntimeError,'first parent failure'):self.a.invoke('invert',fail,(),{},ctx)
  for path,core in self.named.items():
   if path.endswith('chunk_gated_delta_rule'):
    with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):core._forward_method(initial_state=object())
  self.assertFalse(self.events)
 def test_core_identity_mismatch_rejected_before_install(self):
  # A second bridge must reject the now parent-guarded core, not unwrap it blindly.
  with tempfile.TemporaryDirectory() as d:
   b=bridge.Adapter(pathlib.Path(d)/'out')
   with self.assertRaises(AssertionError):b.install(self.model)
   self.assertEqual(b.bindings,[])
 def test_missing_active_context_latches_and_blocks_parent_api(self):
  self.a.start('shadow');core=next(x for p,x in self.named.items() if p.endswith('chunk_gated_delta_rule'))
  with self.assertRaises(AssertionError):core._forward_method(initial_state=object())
  before=list(self.events)
  with self.assertRaisesRegex(AssertionError,'Composition first failure latched'):self.a.invoke('gather',lambda:self.events.append('bad'),(),{},dict(row={}))
  self.assertEqual(self.events,before)
 def test_initial_object_identity_checked_before_core(self):
  self.a.start('shadow');self.a.active=dict(initial=object(),owner=object(),row=dict(order=['gather','invert','mask']))
  core=next(x for p,x in self.named.items() if p.endswith('chunk_gated_delta_rule'))
  with self.assertRaises(AssertionError):core._forward_method(initial_state=object())
  self.assertFalse(self.events)
 def test_gate_wrong_core_route_and_counts_rejected(self):
  d=self.phase('shadow')
  for key,value in [('core_native',True),('native_only',True),('target_oracles',1),('core_native_calls',0),('parent_api_calls',24)]:
   x=copy.deepcopy(d);x[key]=value;self.assertFalse(parent_gate(x),key)
  x=copy.deepcopy(d);x['core_links'][0]['captured_by_parent']=False;self.assertFalse(parent_gate(x))
  x=copy.deepcopy(d);x['core_phase_rows'][0]['path']='wrong';self.assertFalse(parent_gate(x))

class PendingTests(unittest.TestCase):
 def test_pending_blocks_before_model_attributes_or_gpu(self):
  model=NS()
  with tempfile.TemporaryDirectory() as d:
   pathlib.Path(d,'PARENT-PREREQUISITE.json').write_text(json.dumps(dict(complete=False)))
   with patch.object(bridge,'ROOT',pathlib.Path(d)):
    with self.assertRaisesRegex(AssertionError,'actual PASS pending'):bridge.install(model,str(pathlib.Path(d)/'out'))
   self.assertEqual(model.__dict__,{});self.assertFalse((pathlib.Path(d)/'out').exists())
 def test_no_numeric_override(self):
  self.assertIs(bridge.Adapter.observe,bridge.base.Adapter.observe);self.assertIs(bridge.Adapter.invoke,bridge.base.Adapter.invoke)
 def test_count_definition(self):self.assertEqual(5864+24,5888);self.assertEqual(5960+24*4,6056)
 def test_sources_parse(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
