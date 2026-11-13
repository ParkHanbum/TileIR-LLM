import unittest
from types import SimpleNamespace as N
from .bridge import Bus
class Tests(unittest.TestCase):
 def test_failure_blocks_every_other_domain(self):
  for origin in ('scalar','metadata','staged','integer','zero','four','base'):
   b=Bus();calls=[];entries={k:b.entry(k,lambda k=k:calls.append(k))for k in ('scalar','metadata','staged','integer','zero','four','base')}
   b.latch(origin,ValueError('private'))
   for entry in entries.values():
    with self.assertRaises(RuntimeError):entry()
   self.assertEqual(calls,[]);b.latch('second',KeyError());self.assertEqual(b.failure['component'],origin)
 def test_scalar_outer_owner_check_propagates(self):
  b=Bus();called=[]
  def bad(*a):raise ValueError('owner mismatch before candidate')
  s=N(check_global=lambda:None,call=lambda:None,finish=lambda:None,start=lambda:None,binding=N(replacement=bad))
  b.bind_scalar(s)
  with self.assertRaises(ValueError):s.binding.replacement()
  self.assertEqual(b.failure['component'],'scalar/owned_add_request')
  with self.assertRaises(RuntimeError):b.entry('other',lambda:called.append(1))()
  self.assertEqual(called,[])
 def test_metadata_validation_before_invoke_propagates(self):
  b=Bus()
  def bad(*a):raise ValueError('input before invoke')
  s=N(subq=bad,subs=bad,gt=bad,full=bad,invoke=lambda:None,start=lambda:None,finish=lambda:None,guard=lambda fn:fn)
  b.bind_metadata(s);copied_entry=s.guard(lambda:s.gt(None))
  with self.assertRaises(ValueError):copied_entry()
  self.assertIsNotNone(b.failed)
 def test_partial_install_reverse_best_effort(self):
  b=Bus();events=[];b.install('first',lambda:None,lambda:events.append('first'))
  def bad():raise ValueError()
  with self.assertRaises(ValueError):b.install('second',bad,lambda:events.append('second'))
  self.assertTrue(b.close()['restored']);self.assertEqual(events,['second','first'])
 def test_foreign_base_failure_blocks(self):
  c=N(failed=None,active=True);b=Bus(lambda:c);c.failed={'component':'base'}
  with self.assertRaises(RuntimeError):b.guard()
 def test_cleanup_failure_retains(self):
  b=Bus();events=[]
  def bad():raise ValueError()
  b.install('first',lambda:None,lambda:events.append('first'));b.install('second',lambda:None,bad)
  with self.assertRaises(RuntimeError):b.close()
  self.assertEqual(events,['first'])
  with self.assertRaises(RuntimeError):b.require_release(synchronized=True,shutdown=True)

class ExtensionTests(unittest.TestCase):
 def test_actual_extension_request_entry_blocks_late_or_failed(self):
  from .extension import Extension
  x=Extension();calls=[];f=x.request_entry('metadata/owned',lambda:calls.append(1))
  with self.assertRaises(RuntimeError):f()
  self.assertEqual(calls,[]);self.assertIsNotNone(x.bus.failed)
 def test_actual_extension_lifecycle_once(self):
  from .extension import Extension
  x=Extension();x.phase='warmup';x.reference_observed={};events=[]
  x.pair=N(start=lambda mode:events.append('pair'),finish=lambda:{})
  x.scalar=N(start=lambda:events.append('scalar'),finish=lambda:{})
  x.metadata=N()
  x.metadata_joint=N(start=lambda:events.append('metadata'),finish=lambda:dict(metadata={},batch_cast={}))
  x.staged=N(safe_receipt=lambda:{})
  x.post_guard=N(start=lambda:None,finish=lambda:{})
  x.start_new();self.assertEqual(events,['pair','scalar','metadata'])
  with self.assertRaises(RuntimeError):x.start_new()
  x.finish_new();self.assertEqual(x.phase,'finished')
  with self.assertRaises(RuntimeError):x.request_guard()

class PublicControllerSourceTests(unittest.TestCase):
 def source_class(self,folder,namespace):
  import ast
  from pathlib import Path
  path=Path(__file__).resolve().parent.parent/folder/'controller.py'
  tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  exec(compile(tree,str(path),'exec'),namespace);return namespace['Controller']
 def test_real_scalar_outer_entry_latches_before_other_domains(self):
  from .extension import Extension
  class Binding:
   def __init__(self,cls,call):self.replacement=lambda actual,index,request:call(actual.num_accepted_tokens_gpu,index)
   def install(self):pass
   def close(self):pass
  class Owner:
   num_accepted_tokens_gpu=object()
   def add_request(self,index,request):pass
  Scalar=self.source_class('runner_scalar32_lean_v1',dict(Binding=Binding,owner_signature=lambda x:id(x),validate=lambda *x:None,fill_request_slot=lambda *x:None))
  e=Extension();e.phase='request';e.request_started=True;o=Owner();c=Scalar(o);e.bus.bind_scalar(c);c.check_global=e.request_guard;c.binding.replacement=e.request_entry('scalar/entry',c.binding.replacement);c.start()
  with self.assertRaises(ValueError):c.binding.replacement(Owner(),0,None)
  count=[]
  with self.assertRaises(RuntimeError):e.bus.entry('staged',lambda:count.append(1))()
  self.assertEqual(count,[])
 def test_real_metadata_guard_validate_failure_and_after_finish(self):
  from .extension import Extension
  def invalid(*a,**k):raise ValueError('before invoke')
  Metadata=self.source_class('runtime_metadata_five_lean_v1',dict(api=N(validate=invalid),Bindings=None))
  e=Extension();e.phase='request';e.request_started=True;c=Metadata();e.bus.bind_metadata(c);old=c.guard;c.guard=lambda fn:e.request_entry('metadata/entry',old(fn));c.start('lean');entry=c.guard(lambda:c.gt(None))
  with self.assertRaises(ValueError):entry()
  self.assertIsNotNone(e.bus.failed)
  e2=Extension();e2.phase='finished';e2.request_started=True;e2.request_finished=True;calls=[]
  with self.assertRaises(RuntimeError):e2.request_entry('metadata/entry',lambda:calls.append(1))()
  self.assertEqual(calls,[])

class RealIntegerActivationTests(unittest.TestCase):
 def classes(self):
  import ast
  from pathlib import Path
  p=Path(__file__).resolve().parent.parent
  def load(folder,file,name,ns):
   tree=ast.parse((p/folder/file).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name];exec(compile(tree,str(p/folder/file),'exec'),ns);return ns[name]
  BaseCast=load('embedding_int64_runtime_v2','cast.py','Cast',{})
  BaseInteger=load('sampler_integer_runtime_v1','integer.py','Integer',{})
  Cast=load('integer_pair_runtime_v1','adapter.py','Cast',dict(cast=N(Cast=BaseCast)))
  Integer=load('integer_pair_runtime_v1','adapter.py','Integer',dict(integer=N(Integer=BaseInteger)))
  return Cast,Integer
 def state(self,cls):
  s=cls.__new__(cls);s.active=False;s.failed=None;s.shadow_pass=False;s.rows=[];return s
 def fill(self,c,i):
  c.rows=[dict(n=1696 if x==0 else 32)for x in range(16)]
  i.rows=[dict(site=('positions','input_ids','ones')[x%3])for x in range(48)]
 def test_actual_inherited_start_requires_real_completed_native_phase(self):
  Cast,Integer=self.classes();c,i=[self.state(x)for x in (Cast,Integer)]
  for s in (c,i):
   with self.assertRaises(AssertionError):s.start('replace')
   self.assertFalse(s.shadow_pass);s.start('shadow')
  self.fill(c,i);cr=c.finish();ir=i.finish()
  self.assertEqual((cr['native_calls'],ir['native_calls']),(16,48));self.assertTrue(c.shadow_pass and i.shadow_pass)
  for s in(c,i):s.start('replace');self.assertTrue(s.active)
  self.fill(c,i);self.assertEqual((c.finish()['tile_calls'],i.finish()['tile_calls']),(16,48))
 def test_incomplete_native_finish_does_not_unlock(self):
  Cast,Integer=self.classes()
  for cls in(Cast,Integer):
   s=self.state(cls);s.start('shadow')
   with self.assertRaises(AssertionError):s.finish()
   self.assertFalse(s.shadow_pass)
 def test_extension_never_assigns_shadow_pass(self):
  import ast
  from pathlib import Path
  root=Path(__file__).parent
  for file in('extension.py','adapter.py','run.py'):
   t=ast.parse((root/file).read_text())
   self.assertFalse(any(isinstance(n,ast.Attribute)and isinstance(n.ctx,ast.Store)and n.attr=='shadow_pass'for n in ast.walk(t)))

class ProductionCloseSerializationTests(unittest.TestCase):
 def invoke_close(self,folder):
  import ast,json,copy
  from pathlib import Path
  from integer_pair_bridge_v1.bridge import Bridge
  root=Path(__file__).resolve().parent.parent
  source=root/folder/'adapter.py';tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  native=dict(restored=True,close_order=['b','a'],components={'nested':{'restored':True}})
  state=lambda:N(failed=None,guard=lambda:None,close=lambda:dict(restored=True))
  pair=Bridge(state(),state(),lambda:None);pair.bind_guards()
  ns=dict(base=N(ATTR='model',close=lambda m:native),deepcopy=copy.deepcopy)
  exec(compile(tree,str(source),'exec'),ns);ns['EXT']=N(pair=pair)
  result=ns['close'](object());return result,native,pair
 def test_old_v2_cycle_reproduced(self):
  import json
  result,native,pair=self.invoke_close('whole_lean_extensions_runtime_v2')
  self.assertTrue(pair.restored)
  with self.assertRaisesRegex(ValueError,'Circular reference'):json.dumps({'adapter_restore':[result]})
 def test_actual_v3_close_receipt_independent_and_serializable(self):
  import json
  result,native,pair=self.invoke_close('whole_lean_extensions_runtime_v5')
  self.assertTrue(pair.restored);self.assertNotIn('integer_pair_restore',native)
  decoded=json.loads(json.dumps({'adapter_restore':[result]}))
  self.assertTrue(decoded['adapter_restore'][0]['integer_pair_restore']['restored'])
  self.assertIsNot(result['components'],native['components'])
  self.assertIsNot(result['components'],result['integer_pair_restore']['components']['lean']['components'])
  result['components']['nested']['restored']=False
  self.assertTrue(native['components']['nested']['restored']);self.assertTrue(result['integer_pair_restore']['components']['lean']['components']['nested']['restored'])

class PredicateLifecycleTests(unittest.TestCase):
 def make(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'predicate_controller.py').read_text())
  events=[]
  class Checks:
   def __init__(self,c):self.c=c
   def start(self,*args):events.append('check_start')
   def finish(self):
    assert self.c.active;events.append('check_finish');return {}
  class Base:
   def __init__(self,model,specs):self.active=False;self.failed=None;self.parts=[N(key='gdn_prefill_core',body='finite')]
   def install(self):events.append('base_install');return {}
   def start(self):self.active=True;events.append('base_start')
   def finish(self):self.active=False;events.append('base_finish');return {}
   def latch(self,k,e):self.failed=k
   def close(self):events.append('base_close');return {}
  class Binding:
   def __init__(self,part,checks):self.part=part
   def install(self):events.append('eq_install')
   def close(self):events.append('eq_close')
  ns=dict(Base=Base,SPECS=[('attention_core',object,128),('greedy_sample',object,16),('sampler_four',object,16)],Finite=Checks,Equal=Checks,Range=Checks,Sampling=Checks,Temperature=Checks,TemperatureGreedy=object,annotate_sampling_receipt=lambda s,t:s,sampling_proof=N(Greedy=object,Sampler=object),replace_specs=lambda x:x,integer_specs=lambda x:x,LifetimeController=Base,GDNBodyBinding=Binding,Attention=object)
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Controller')
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'<actual-controller>','exec'),ns)
  return ns['Controller'](None),events
 def test_install_finish_restore_order(self):
  c,e=self.make();c.install();c.start();c.finish();c.close()
  self.assertEqual(e,['base_install','eq_install','check_start','check_start','check_start','check_start','check_start','base_start','check_finish','check_finish','check_finish','check_finish','check_finish','base_finish','eq_close','base_close'])
 def test_predicate_failure_latches(self):
  c,e=self.make()
  def bad():raise ValueError('no user data')
  c.eq408.start=bad
  with self.assertRaises(ValueError):c.start()
  self.assertEqual(c.failed,'predicate/start');self.assertNotIn('base_start',e)
 def test_both_new_domains_guarded_by_actual_entry(self):
  from .extension import Extension
  for name in ('batch_cast','finite504','eq408'):
   x=Extension();x.phase='request';x.request_started=True;calls=[]
   x.bus.latch(name,ValueError())
   with self.assertRaises(RuntimeError):x.request_entry('metadata_batch/entry',lambda:calls.append(1))()
   self.assertEqual(calls,[])

class ExplicitRootTests(unittest.TestCase):
 def test_actual_audit_passes_current_root_to_inherited_gate(self):
  import ast
  from pathlib import Path
  root=Path(__file__).parent
  tree=ast.parse((root/'gates.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit')
  seen=[]
  def base(r,p):seen.append(p);return {'passed':False}
  ns=dict(Path=Path,__file__=str(root/'gates.py'),base_audit=base)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-audit>','exec'),ns)
  ns['audit']({});ns['audit']({},Path('/explicit/source/root'))
  self.assertEqual(seen,[root.resolve(),Path('/explicit/source/root')])
 def test_launcher_explicit_root_and_source_only(self):
  import ast
  from pathlib import Path
  root=Path(__file__).parent
  tree=ast.parse((root/'launch.py').read_text())
  calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='audit']
  self.assertEqual(len(calls),1)
  self.assertEqual(ast.dump(calls[0].keywords[0].value),"Name(id='R', ctx=Load())")
  self.assertEqual(calls[0].keywords[0].arg,'root')
  self.assertFalse((root/'cache_v1').exists());self.assertFalse((root/'CACHE-PRESEED-01.json').exists())

class PendingV5Tests(unittest.TestCase):
 def test_actual_qualification_requires_added_proofs_first(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'gates.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='qualification')
  self.assertIsInstance(fn.body[0],ast.ImportFrom)
  self.assertEqual(fn.body[0].module,'new_proofs')
  self.assertEqual(ast.unparse(fn.body[1]),'added = verify_new()')
 def test_post_binding_does_not_repeat_owner_cap(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'extension.py').read_text())
  method=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='install_post_guard')
  text=ast.unparse(method)
  self.assertNotIn('owner.guard',text)
  self.assertIn('binding.callback = self.request_entry',text)
  self.assertIn("self.bus.install('post_guard'",text)

class LifetimeCompositionTests(PredicateLifecycleTests):
 def test_actual_safe_close_failure_keeps_owners_and_cannot_retry(self):
  import ast
  from pathlib import Path
  c,events=self.make()
  source=Path(__file__).parent.parent/'gdn_integer_finite_runtime_qualification_v3/adapter.py'
  tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Controller')
  close=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='close')
  ns={};exec(compile(ast.Module(body=[close],type_ignores=[]),str(source),'exec'),ns)
  c.__class__.close.__globals__['LifetimeController']=N(close=ns['close'])
  calls=[]
  def bad():calls.append('bad');raise ValueError('restore')
  badpart=N(key='failed',close=bad,bindings=[])
  good=N(key='good',close=lambda:dict(restored=True),bindings=[])
  c.parts=[good,badpart]
  with self.assertRaises(RuntimeError):c.close()
  self.assertTrue(c._permanent_close_hold);self.assertFalse(c._restoration_proven)
  self.assertEqual(c._lifetime_parts,[good,badpart]);self.assertEqual(c.parts,[badpart])
  with self.assertRaises(RuntimeError):c.close()
  self.assertEqual(calls,['bad'])

class TypedGateTests(unittest.TestCase):
 def receipt(self):
  return {'components':{'gdn_prefill_core':{'gdn_finite':dict(typed_integer_checks=72,typed_integer_site_counts=dict.fromkeys(('cu_seqlens','chunk_indices','chunk_offsets'),24),integer_gpu_kernels=0,integer_stream_synchronizations=72,integer_native_predicates=0,float_checks=360)}}}
 def test_actual_typed_contract_strict(self):
  from .integer_gate import typed_contract
  typed_contract(self.receipt())
  for key,value in [('retained_integer_checks',72),('integer_stream_synchronizations',0),('integer_native_predicates',72),('float_checks',359),('typed_integer_site_counts',{})]:
   d=self.receipt();d['components']['gdn_prefill_core']['gdn_finite'][key]=value
   with self.assertRaises(AssertionError):typed_contract(d)
