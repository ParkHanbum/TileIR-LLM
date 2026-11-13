import unittest,ast
from pathlib import Path
from .registry import Registry,Hold,CAP
class Storage:
 def __init__(self,base,size):self.base=base;self.size=size
 def data_ptr(self):return self.base
 def nbytes(self):return self.size
class View:
 requires_grad=False
 def __init__(self,s,shape,stride,offset=0,el=4):self.s=s;self.shape=shape;self.st=stride;self.off=offset;self.el=el;self.device='cuda:0'
 def untyped_storage(self):return self.s
 def data_ptr(self):return self.s.base+self.off*self.el
 def storage_offset(self):return self.off
 def stride(self):return self.st
 def element_size(self):return self.el
 def is_neg(self):return False
 def is_conj(self):return False
class Tests(unittest.TestCase):
 def tables(self,r):
  ss=list(r.segments.values());return [[s.address for s in ss],[s.stride//4 for s in ss],[s.page//4 for s in ss]]
 def test_overlay_address_only(self):
  s=Storage(4096,400);r=Registry('cuda:0');r.add_view(View(s,(4,2),(20,1)),4);r.add_view(View(s,(4,3),(20,1)),4)
  self.assertEqual(len(r.segments),1);d=r.compare(*self.tables(r),[1,3,1]);self.assertEqual(d['selected_union_bytes'],24)
 def test_multi_owner_and_virtual(self):
  r=Registry('cuda:0');r.add_view(View(Storage(4096,1024),(8,2),(16,1)),4);r.add_view(View(Storage(8192,1024),(4,2),(16,1)),4)
  self.assertEqual(len(r.segments),3);self.assertEqual(r.compare(*self.tables(r),[1])['selected_union_bytes'],24)
 def test_outer_dim_segments(self):
  r=Registry('cuda:0');r.add_view(View(Storage(4096,1024),(4,2,2),(4,64,1)),4);self.assertEqual(len(r.segments),2)
 def test_no_unowned_address(self):
  r=Registry('cuda:0');r.add_view(View(Storage(4096,1024),(4,2),(16,1)),4);ts=self.tables(r);ts[0][0]+=4
  with self.assertRaises(Hold):r.compare(*ts,[1])
 def test_stale_and_bounds(self):
  s=Storage(4096,1024);r=Registry('cuda:0');r.add_view(View(s,(4,2),(16,1)),4)
  with self.assertRaises(Hold):r.compare(*self.tables(r),[4])
  s.base+=4096
  with self.assertRaises(Hold):r.check()
 def test_partial_allocation_overlap(self):
  r=Registry('cuda:0');r.add_view(View(Storage(4096,1024),(4,2),(16,1)),4)
  with self.assertRaises(Hold):r.add_view(View(Storage(4608,1024),(4,2),(16,1)),4)
 def test_hypothetical_budget_not_actual(self):
  page=2162688;s=Storage(4096,8*310*page);r=Registry('cuda:0')
  for i in range(8):r.add_view(View(s,(310,4,528,512),(1081344,512,2048,1),i*310*1081344,2),310)
  d=r.compare(*self.tables(r),list(range(32)));self.assertEqual(d['selected_union_bytes'],528*1024**2);self.assertFalse(d['within_budget'])
 def test_source_no_framework_import_or_pointer_construction(self):
  for p in Path(__file__).parent.glob('*.py'):
   ast.parse(p.read_text())
  s=(Path(__file__).parent/'registry.py').read_text()+(Path(__file__).parent/'diagnostic.py').read_text()
  for bad in ('import torch','import cuda','import vllm','from_address','ctypes','set_(','torch.tensor'):self.assertNotIn(bad,s)

class Table(View):
 def __init__(self,base,dtype,values):super().__init__(Storage(base,64),(8,),(1,),el=8);self.dtype=dtype;self.values=values
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return list(self.values)
class BoundaryTests(unittest.TestCase):
 def diagnostic(self):
  from .diagnostic import Diagnostic
  from types import SimpleNamespace
  d=Diagnostic();r=Registry('cuda:0')
  for i in range(8):r.add_view(View(Storage(4096+i*4096,1024),(32,2),(4,1)),32)
  ss=list(r.segments.values());tables=(Table(100000,'torch.uint64',[s.address for s in ss]),Table(100128,'torch.int64',[s.stride//4 for s in ss]),Table(100256,'torch.int64',[s.page//4 for s in ss]))
  d.registry=r;d.zeroer=SimpleNamespace(_meta=(*tables,1,2,8));return d
 def test_original_exactonce_and_no_second_entry(self):
  d=self.diagnostic();calls=[];ids=list(range(32))
  self.assertIsNone(d.zero(lambda z,b:calls.append((z,b)),d.zeroer,ids));self.assertEqual(len(calls),1);self.assertIs(calls[0][1],ids)
  with self.assertRaises(Hold):d.zero(lambda *x:calls.append(x),d.zeroer,ids)
  self.assertEqual(len(calls),1)
 def test_bad_address_latch_native_zero(self):
  d=self.diagnostic();d.zeroer._meta[0].values[0]+=4;calls=[]
  for _ in range(2):
   with self.assertRaises(Hold):d.zero(lambda *x:calls.append(x),d.zeroer,list(range(32)))
  self.assertEqual(calls,[]);self.assertNotIn('4096',str(d.failed))
 def test_budget_hold_before_original(self):
  from unittest.mock import patch
  d=self.diagnostic();calls=[]
  with patch.object(d.registry,'compare',return_value={'within_budget':False,'selected_union_bytes':CAP+1}):
   with self.assertRaises(Hold):d.zero(lambda *x:calls.append(x),d.zeroer,list(range(32)))
  self.assertEqual(calls,[]);self.assertEqual(d.failed,'selected_union_exceeds_256MiB')
 def test_release_requires_sync(self):
  d=self.diagnostic()
  with self.assertRaises(Hold):d.close()
  d.close(after_synchronize=True);self.assertIsNone(d.registry)

class MaliciousTests(BoundaryTests):
 def test_malicious_native_hold_redacted(self):
  d=self.diagnostic()
  def bad(*x):raise Hold('address 0xdeadbeef secret')
  with self.assertRaises(Hold)as cm:d.zero(bad,d.zeroer,list(range(32)))
  self.assertEqual(str(cm.exception),'native_or_metadata_failure');self.assertNotIn('deadbeef',d.failed);self.assertEqual(d.native_zero_calls,1);self.assertEqual(d.native_zero_returned,0)
 def test_setup_attempt_return_counts(self):
  from .diagnostic import Diagnostic
  d=Diagnostic()
  def bad(*x):raise Hold('0x12345')
  with self.assertRaises(Hold):d.setup(bad,object(),object,object)
  self.assertEqual((d.setup_attempted,d.setup_returned),(1,0));self.assertEqual(d.failed,'setup_failed')
 def test_safe_code_never_calls_str(self):
  from .diagnostic import safe_code
  class Bad(Hold):
   def __str__(self):raise AssertionError('mustnotcall')
  self.assertEqual(safe_code(Bad('address'),'generic'),'generic')
class WrapperTests(unittest.TestCase):
 def modules(self):
  from types import SimpleNamespace
  class Zero:
   def zero_block_ids(self,ids):return None
   def warmup(self,n):
    if n>0:self.zero_block_ids([0])
  class Runner:
   def _init_kv_zero_meta(self):return None
  return SimpleNamespace(GPUModelRunner=Runner),SimpleNamespace(KVBlockZeroer=Zero,AttentionSpec=object,torch=SimpleNamespace(Tensor=object))
 def test_warmup_separate_restore(self):
  from .wrapper import Hooks
  mr,u=self.modules();old=u.KVBlockZeroer.zero_block_ids;h=Hooks().install(mr,u,validator=lambda *x:None)
  z=u.KVBlockZeroer();z.warmup(1);self.assertEqual((h.warmup_attempted,h.warmup_returned,h.diag.native_zero_calls),(1,1,0))
  h.close();self.assertIs(u.KVBlockZeroer.zero_block_ids,old);h.release(after_synchronize=True)
 def test_unclassified_initial_zero_latches(self):
  from .wrapper import Hooks
  mr,u=self.modules();h=Hooks().install(mr,u,validator=lambda *x:None)
  with self.assertRaises(Hold):u.KVBlockZeroer().zero_block_ids([0])
  self.assertEqual(h.warmup_attempted,0);self.assertIsNotNone(h.failed);h.close()
 def test_native_binding_ast_pins(self):
  import json,hashlib
  specs=json.loads((Path(__file__).parent/'SOURCE-BINDINGS.json').read_text())
  for s in specs:
   t=ast.parse(Path(s['source_path']).read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef)and x.name==s['owner']);f=next(x for x in c.body if isinstance(x,ast.FunctionDef)and x.name==s['attribute']);fragment=''.join(Path(s['source_path']).read_text().splitlines(True)[f.lineno-1:f.end_lineno]);import textwrap;node=ast.parse(textwrap.dedent(fragment)).body[0];self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),s['binding_ast_sha256'])

class RuntimeTests(unittest.TestCase):
 def test_endpoint_failure_retains_values(self):
  from .runtime_contract import record_endpoints
  import copy
  r={};saved=[]
  def bad(*args):raise AssertionError('mismatch')
  with self.assertRaises(AssertionError):record_endpoints(r,[[1]],{'actual':2},lambda:saved.append(copy.deepcopy(r)),bad,bad,None,None)
  self.assertEqual(saved,[{'output_token_ids':[[1]],'public_logprobs':{'actual':2}}])
 def test_runtime_guards(self):
  from .runtime_contract import runtime_guard
  from types import SimpleNamespace as S
  class InprocClient:pass
  class Policy:
   def values(self):return dict(bf16=True,split_k=True)
   def select(self,x):assert x=='native';assert self.values()==dict(bf16=True,split_k=True)
  cfg=S(model_config=S(enforce_eager=True),compilation_config=S(mode=0,cudagraph_mode=S(name='NONE')),scheduler_config=S(async_scheduling=True))
  e=S(engine_core=InprocClient(),vllm_config=cfg);runtime_guard(e,Policy())
  for obj,key,bad in [(cfg.model_config,'enforce_eager',False),(cfg.compilation_config,'mode',1),(cfg.compilation_config.cudagraph_mode,'name','FULL'),(cfg.scheduler_config,'async_scheduling',False)]:
   before=getattr(obj,key);setattr(obj,key,bad)
   with self.assertRaises(AssertionError):runtime_guard(e,Policy())
   setattr(obj,key,before)

class BindingRegression(unittest.TestCase):
 def test_closed_binding_codes_survive_install(self):
  from .wrapper import Hooks
  from .diagnostic import safe_code
  for code in ('source_changed','binding_identity','binding_signature','binding_source','constructor_alias'):
   def fail(*args):raise Hold(code)
   h=Hooks()
   with self.assertRaises(Hold) as cm:h.install(None,None,validator=fail)
   self.assertEqual(h.failed,code);self.assertEqual(safe_code(cm.exception,'generic'),code)
  self.assertEqual(safe_code(Hold('binding_source 0xdeadbeef'),'generic'),'generic')
 def test_whole_file_and_dedent_docstring_regression(self):
  import json,textwrap,hashlib
  spec=json.loads((Path(__file__).parent/'SOURCE-BINDINGS.json').read_text())[0]
  src=Path(spec['source_path']).read_text();c=next(x for x in ast.parse(src).body if isinstance(x,ast.ClassDef)and x.name==spec['owner']);f=next(x for x in c.body if isinstance(x,ast.FunctionDef)and x.name==spec['attribute'])
  fragment=''.join(src.splitlines(True)[f.lineno-1:f.end_lineno]);g=ast.parse(textwrap.dedent(fragment)).body[0]
  self.assertNotEqual(ast.dump(f,include_attributes=False),ast.dump(g,include_attributes=False))
  self.assertEqual(hashlib.sha256(ast.dump(g,include_attributes=False).encode()).hexdigest(),spec['binding_ast_sha256'])

class SourceOnlyPackaging(unittest.TestCase):
 def test_no_runtime_receipt_or_cache_before_preseed(self):
  root=Path(__file__).parent
  self.assertFalse((root/'cache_v1').exists())
  self.assertEqual(list(root.glob('CACHE-PRESEED*')),[])
  self.assertEqual(list(root.glob('*-launch.json')),[])
  self.assertEqual(list(root.glob('*-recovery.json')),[])
 def test_pins_contain_no_inherited_cache_receipt(self):
  import json
  root=Path(__file__).parent;p=root/'PINS.json'
  self.assertTrue(p.is_file())
  pins=json.loads(p.read_text())
  self.assertFalse(any(Path(x).name.startswith('CACHE-PRESEED') for x in pins))
  self.assertFalse(any('/zero_kv_registry_phase1_v3/'in x for x in pins))
