import unittest,ast,sys
from pathlib import Path
from . import guards,api
class Tests(unittest.TestCase):
 def test_no_gpu_import(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_source_ast(self):ast.parse(Path(__file__).with_name('kernels.py').read_text())
 def test_rope_prefill_decode(self):
  guards.rope(list(range(32)),[53*i for i in range(33)],[53]*32,[0]*32,[0]*32)
  guards.rope(list(range(32)),list(range(33)),[53]*32,[68]*32,[0]*32)
 def test_rope_invalid(self):
  with self.assertRaises(ValueError):guards.rope([32]*32,list(range(33)),[53]*32,[0]*32,[0]*32)
  with self.assertRaises(ValueError):guards.rope(list(range(32)),[53*i for i in range(33)],[200]*32,[100]*32,[0]*32)
 def test_combine(self):guards.combine(list(range(32)),list(range(33)),[54]*32,[53]*32,list(range(33)),[42]*32,32,1)
 def test_combine_invalid(self):
  for cu,new in [(list(range(33)),0),([0]*33,1)]:
   with self.assertRaises(ValueError):guards.combine(list(range(32)),list(range(33)),[54]*32,[53]*32,cu,[42]*32,32,new)
 def test_bias_noop_unread_invalid_ids(self):guards.bias(list(range(32)),[68]*32,[0]*32,[0]*32,[69]*32,[2]*32,[[-999]*128 for _ in range(32)],[0]*32,False)
 def test_bias_active(self):guards.bias(list(range(32)),[67]*32,[0]*32,[0]*32,[69]*32,[1]*32,[[42]+[-999]*127 for _ in range(32)],[0]*32,False)
 def test_bias_invalid(self):
  with self.assertRaises(ValueError):guards.bias(list(range(32)),[67]*32,[0]*32,[0]*32,[69]*32,[2]*32,[[42]*128 for _ in range(32)],[0]*32,False)
  with self.assertRaises(ValueError):guards.bias(list(range(32)),[67]*32,[1]*32,[0]*32,[69]*32,[1]*32,[[42]*128 for _ in range(32)],[0]*32,False)
if __name__=='__main__':unittest.main()
class ShadowCPU(unittest.TestCase):
 def test_exact_once_original_return(self):
  import tempfile
  from .shadow import Shadow
  with tempfile.TemporaryDirectory() as d:
   x=Shadow(d);x.active=True;calls=[];out=object()
   def native(a):calls.append(a);return out
   x.call=lambda label,fn,b:fn(*b.args,**b.kwargs)
   fn=x.wrap(native,'bias');self.assertIs(fn(7),out);self.assertEqual(calls,[7])
 def test_inactive_bypass(self):
  import tempfile
  from .shadow import Shadow
  with tempfile.TemporaryDirectory() as d:
   x=Shadow(d);calls=[];out=object()
   def native():calls.append(1);return out
   self.assertIs(x.wrap(native,'bias')(),out);self.assertEqual(calls,[1]);self.assertEqual(x.rows,[])
 def test_incomplete_finish(self):
  import tempfile
  from .shadow import Shadow
  with tempfile.TemporaryDirectory() as d:
   x=Shadow(d)
   with self.assertRaises(AssertionError):x.finish()
 def test_layout_reject_mock(self):
  with self.assertRaises(ValueError):api.validate_layout('bias',{})
class CleanupCPU(unittest.TestCase):
 def test_post_close_failure_does_not_skip_three(self):
  import types
  from .cleanup import close_helpers
  events=[]
  class Post:
   def __exit__(self,*a):events.append('post');raise ValueError('drift')
  three=types.SimpleNamespace(close=lambda:events.append('three'),summary=lambda:{},save=lambda:events.append('save'))
  r={};e=close_helpers(r,Post(),None,three)
  self.assertTrue(e);self.assertEqual(events,['post','three','save']);self.assertFalse(r['candidate_checks_passed'])
 def test_failed_shadow_cannot_retry_native(self):
  import tempfile
  from .shadow import Shadow
  with tempfile.TemporaryDirectory() as d:
   s=Shadow(d);s.active=True;s.failed=True;calls=[]
   with self.assertRaises(RuntimeError):s.wrap(lambda:calls.append(1),'bias')()
   self.assertEqual(calls,[])
