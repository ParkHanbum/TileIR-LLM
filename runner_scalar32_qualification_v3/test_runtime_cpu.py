import ast,json,subprocess,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
class Tests(unittest.TestCase):
 def test_namespace_only_diff(self):
  old=(P/'runner_scalar32_qualification_v1/api.py').read_text()
  self.assertEqual((R/'api.py').read_text(),old.replace('from kernel import fill_one','from runner_scalar32_qualification_v3.kernel import fill_one'))
  self.assertEqual((R/'kernel.py').read_bytes(),(P/'runner_scalar32_qualification_v1/kernel.py').read_bytes())
 def test_fresh_stop_before_framework(self):
  code='''import sys,importlib.abc
sys.path.insert(0,sys.argv[1])
class Stop(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name in ('torch','cuda'):raise RuntimeError('STOP_BEFORE_FRAMEWORK')
sys.meta_path.insert(0,Stop())
from runner_scalar32_qualification_v3.api import fill_request_slot
from runner_scalar32_qualification_v3.test_cpu import V
try:fill_request_slot(V(),0)
except RuntimeError as e:assert str(e)=='STOP_BEFORE_FRAMEWORK'
else:raise AssertionError()
assert 'torch' not in sys.modules and 'cuda' not in sys.modules
'''
  subprocess.run([sys.executable,'-B','-c',code,str(P)],check=True)
 def test_qualified_kernel_request_no_collision(self):
  code='''import sys,types,importlib.abc
sys.path.insert(0,sys.argv[1])
sys.modules['torch']=types.ModuleType('torch');cuda=types.ModuleType('cuda');cuda.__path__=[];sys.modules['cuda']=cuda
sys.modules['cuda.tile']=types.ModuleType('cuda.tile');sys.modules['kernel']=types.ModuleType('kernel')
class Stop(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='runner_scalar32_qualification_v3.kernel':raise RuntimeError('QUALIFIED_KERNEL')
sys.meta_path.insert(0,Stop())
from runner_scalar32_qualification_v3.api import fill_request_slot
from runner_scalar32_qualification_v3.test_cpu import V
try:fill_request_slot(V(),0)
except RuntimeError as e:assert str(e)=='QUALIFIED_KERNEL'
else:raise AssertionError()
'''
  subprocess.run([sys.executable,'-B','-c',code,str(P)],check=True)
 def test_source_only_no_runtime_artifacts(self):
  self.assertFalse((R/'cache_v1').exists());self.assertFalse((R/'CACHE-PRESEED-01.json').exists())
 def test_fail_audit(self):
  from runner_scalar32_qualification_v3.gates import audit
  self.assertFalse(audit({})['passed'])
 def test_timeout_and_no_profile(self):
  s=(R/'run.py').read_text();self.assertIn('signal.alarm(900)',s);self.assertNotIn('torch.profiler',s)
 def test_launcher_cache_before_admission(self):
  s=(R/'launch.py').read_text();self.assertLess(s.index('cache_preseed=preflight(R)'),s.index("rec['pre']=h.admit()"))
if __name__=='__main__':unittest.main()

class CallerTests(unittest.TestCase):
 def test_real_caller_dedent(self):
  import __future__
  from runner_scalar32_qualification_v3.caller_gate import verify
  d=json.loads((R/'CALLER.json').read_text());t=ast.parse(Path(d['path']).read_text())
  fn=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='add_requests')
  cl=ast.ClassDef(name='Runner',bases=[],keywords=[],body=[fn],decorator_list=[])
  tree=ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[]));ns={}
  exec(compile(tree,d['path'],'exec',flags=__import__('__future__').annotations.compiler_flag),ns)
  self.assertTrue(verify(ns['Runner']())['dynamic_model_state_call_source_verified'])
 def test_cleanup_failure_is_not_complete(self):
  from runner_scalar32_qualification_v3.cleanup import cleanup
  class Observer:
   def close(self):raise RuntimeError('expected')
  r={'candidate_checks_passed':True}
  with self.assertRaises(RuntimeError):cleanup(r,lambda:None,None,None,Observer(),lambda:None,lambda:None,False)
  self.assertFalse(r['complete']);self.assertFalse(r['observer_restored'])
