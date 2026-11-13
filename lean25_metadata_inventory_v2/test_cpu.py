import unittest,sys,ast,json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .observer import Ledger,create_mode
from .helpers import wrap_helper,helper_observers
R=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_original_return_alias_and_once(self):
  x=torch.arange(4);ledger=Ledger();calls=[]
  def original(a):calls.append(1);a.add_(1);return a
  w=wrap_helper(torch,ledger,original,dict(label='x'))
  out=w(x);self.assertIs(out,x);self.assertEqual(calls,[1]);self.assertEqual(ledger.rows[0]['returns'],1)
  rows=json.loads(next(iter(ledger.rows[0]['return_metadata'])))['aliases'];self.assertTrue(rows[0]['same_object']);self.assertTrue(rows[0]['storage_overlap'])
 def test_nested_mode_forwarding(self):
  ledger=Ledger();seen=[]
  class Inner(TorchDispatchMode):
   def __torch_dispatch__(self,func,types,args=(),kwargs=None):seen.append(str(func));return func(*args,**(kwargs or {}))
  x=torch.ones(4)
  with create_mode(ledger):
   with Inner():out=x.add_(2)
  self.assertIs(out,x);self.assertTrue(torch.equal(x,torch.full((4,),3.)));self.assertEqual(len(seen),1);self.assertEqual(ledger.calls,1);self.assertEqual(ledger.rows[0]['returns'],1)
 def test_exception_latch(self):
  ledger=Ledger();calls=[]
  def bad(a):calls.append(1);raise ValueError('original')
  w=wrap_helper(torch,ledger,bad,dict(label='bad'))
  with self.assertRaisesRegex(ValueError,'original'):w(torch.ones(1))
  with self.assertRaises(AssertionError):w(torch.ones(1))
  self.assertEqual(calls,[1]);self.assertEqual(ledger.rows[0]['errors'],1)
 def test_cap_before_original(self):
  ledger=Ledger(max_calls=0);calls=[];w=wrap_helper(torch,ledger,lambda a:calls.append(a),dict(label='cap'))
  with self.assertRaises(RuntimeError):w(torch.ones(1))
  self.assertEqual(calls,[])
 def test_wrapper_restore_and_partial_failure(self):
  import types,hashlib
  module=types.ModuleType('fake_public_helpers');fn=lambda x:x;module.fn=fn;ledger=Ledger()
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);f=p/'source.txt';f.write_text('source');spec=dict(label='f',module='fake_public_helpers',owner='',attribute='fn',source_path=str(f),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),parameter_names=['x'])
   (p/'HELPER-SPECS.json').write_text(json.dumps(dict(complete=True,helpers=[spec])))
   with patch.dict(sys.modules,{'fake_public_helpers':module}):
    with self.assertRaisesRegex(ValueError,'body'):
     with helper_observers(ledger,p):self.assertIsNot(module.fn,fn);raise ValueError('body')
    self.assertIs(module.fn,fn)
    bad=dict(spec,attribute='absent');(p/'HELPER-SPECS.json').write_text(json.dumps(dict(complete=True,helpers=[spec,bad])))
    with self.assertRaises(AttributeError):
     with helper_observers(ledger,p):pass
    self.assertIs(module.fn,fn)
 def test_no_value_io(self):
  for name in ['observer.py','helpers.py']:
   s=(R/name).read_text();ast.parse(s)
   for banned in ['.cpu(','.numpy(','.item(','.tolist(','.clone(','torch.cuda.synchronize','repr(value)']:self.assertNotIn(banned,s)
 def test_no_cuda_initialized(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()

class CorrelationTests(unittest.TestCase):
 def test_nested_launch_and_async_unknown(self):
  from .correlate import analyze
  def event(cat,name,ts,dur,pid=1,tid=1,args=None):return dict(cat=cat,name=name,ts=ts,dur=dur,pid=pid,tid=tid,args=args or {})
  es=[event('user_annotation','metadata_request_execution::lean',0,100),event('user_annotation','python_helper_site::0',10,50),event('user_annotation','metadata_site::1',20,5),event('cuda_runtime','cudaLaunchKernel',21,1,args={'correlation':7}),event('kernel','k',200,5,pid=0,tid=7,args={'correlation':7}),event('kernel','uncovered',220,5,pid=0,tid=8,args={'correlation':8})]
  x=analyze(es);self.assertEqual(x['attributed'][0]['site'],'metadata_site::1');self.assertEqual(x['unknown'][0]['symbol'],'uncovered')
 def test_constructor_after_fields(self):
  from .helpers import wrap_helper
  ledger=Ledger()
  class C:
   def __init__(self):self.cpu=torch.zeros(3);self.view=self.cpu.view(3)
  fn=wrap_helper(torch,ledger,C.__init__,dict(label='ctor',self_attrs=['cpu','view']))
  obj=C.__new__(C);self.assertIsNone(fn(obj));after=json.loads(next(iter(ledger.rows[0]['return_metadata'])))
  self.assertEqual(after['inputs']['self.cpu']['shape'],[3]);self.assertTrue(any(x['storage_overlap']for x in after['aliases']))

class EntryTests(unittest.TestCase):
 def test_pending_before_runtime(self):
  import subprocess,os
  for entry in ('run.py','launch.py'):
   code="""import sys,runpy,importlib.abc,json
from pathlib import Path
old=Path.read_text
def read(self,*a,**kw):
 if self.name=='HELPER-SPECS.json':return json.dumps({'complete':False,'helpers':[]})
 return old(self,*a,**kw)
Path.read_text=read
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in ('torch','vllm','cuda'):raise RuntimeError('Unexpected runtime import')
sys.meta_path.insert(0,Block());sys.argv=[ENTRY,'--run-gpu']
try:runpy.run_path(ENTRY,run_name='__main__')
except AssertionError as e:assert 'Helper metadata contracts pending'in str(e),repr(e)
else:raise AssertionError('Pending unexpectedly released')
""".replace('ENTRY',repr(str(R/entry)))
   p=subprocess.run([sys.executable,'-B','-c',code],cwd=R.parent,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,timeout=60)
   self.assertEqual(p.returncode,0,p.stderr.decode())
 def test_source_fields_and_no_runtime_settings_change(self):
  for f in R.glob('*.py'):ast.parse(f.read_text())
  s=(R/'run.py').read_text()
  self.assertIn("record_shapes=False",s);self.assertIn("with_stack=False",s);self.assertNotIn('perf_counter',s)
  self.assertLess(s.index("'SETUP-SPECS.json'"),s.index('engine=LLMEngine.from_engine_args'))
  self.assertLess(s.index("native=request('native')"),s.index("lean=request('lean')"))

class ActualRequestPathTests(unittest.TestCase):
 def test_lean_start_uses_imported_partial(self):
  import functools,types
  tree=ast.parse((R/'run.py').read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='request')
  module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));calls=[]
  class StopAtStart(Exception):pass
  def start(*args,**kw):pass
  def apply(fn):
   self.assertIsInstance(fn,functools.partial);self.assertIs(fn.func,start);calls.append(fn.keywords);raise StopAtStart()
  env=dict(partial=functools.partial,engine=types.SimpleNamespace(has_unfinished_requests=lambda:False,apply_model=apply),policy=types.SimpleNamespace(select=lambda mode:(None,True)),adapter=types.SimpleNamespace(start=start))
  exec(compile(module,str(R/'run.py'),'exec'),env)
  with self.assertRaises(StopAtStart):env['request']('lean')
  self.assertEqual(calls,[dict(mode='lean')])
 def test_zero_typed_projection(self):
  from .helpers import projected_owners
  import types
  owner=types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context={'layer':types.SimpleNamespace(kv_cache=torch.zeros(2))}))
  tensors=[];d=projected_owners(torch,owner,dict(typed_owner_projection=dict(container='compilation_config.static_forward_context',per_value_attribute='kv_cache')),tensors,'kv')
  self.assertEqual(d['layer']['shape'],[2]);self.assertEqual(len(tensors),1)
