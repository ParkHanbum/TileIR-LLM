import unittest,types,sys,ast
from pathlib import Path
from unittest.mock import patch
from .checks import Checks
class Tensor:
 def __init__(self,n):self.n=n;self.device='cuda:0';self.shape=(32,);self.dtype='int32'
 def data_ptr(self):return self.n*1000
 def untyped_storage(self):return self
 def nbytes(self):return 128
 def storage_offset(self):return 0
 def stride(self):return (1,)
 def detach(self):return self
 def view(self,*_):return self
 def cpu(self):return self
 def numpy(self):return self
 def tobytes(self):return bytes([self.n])*128
class FakeGuard:
 def __init__(self):self.pending=[];self.stages=[]
 def validate(self,torch,args):self.stages=list(range(9))
class Tests(unittest.TestCase):
 def make(self):
  controller=types.SimpleNamespace(active=True,failed=None)
  owner=types.SimpleNamespace(failed=False,active=False,controller=controller)
  owner.latch=lambda label,error:setattr(controller,'failed',label)
  c=Checks(owner);c.binding.copied=object();owner.call=c.binding.copied
  torch=types.SimpleNamespace(uint8=object(),cuda=types.SimpleNamespace(current_stream=lambda *_:types.SimpleNamespace(cuda_stream=123)))
  contract=types.ModuleType('tileir_vllm_post_update_v3.contract');contract.validate_metadata=lambda _:None;contract.validate_values=lambda *_:None
  return c,owner,torch,contract,tuple(Tensor(i+1)for i in range(10))
 def test_shadow_replace_and_distinct_rows(self):
  c,o,t,contract,args=self.make()
  with patch.dict(sys.modules,{'tileir_vllm_post_update_v3.contract':contract}),patch('post_update_guard_runtime_v1.checks.Guard',FakeGuard):
   c.start('shadow');o.active=True
   for _ in range(16):c.validate(t,args)
   first=c.finish();o.active=False;c.start('replace');o.active=True
   for _ in range(16):c.validate(t,args)
   second=c.finish()
  self.assertEqual(first['mode'],'shadow');self.assertTrue(first['rows'][0]['native_calls']==1)
  self.assertEqual(second['rows'][0]['native_calls'],0);self.assertIsNot(first['rows'],second['rows'])
 def test_replace_without_shadow_rejected(self):
  c,*_=self.make()
  with self.assertRaises(AssertionError):c.start('replace')
 def test_native_mismatch_latches(self):
  c,o,t,contract,args=self.make();c.start('shadow');o.active=True
  def bad(*_):raise ValueError('computed integer overflow')
  contract.validate_values=bad
  with patch.dict(sys.modules,{'tileir_vllm_post_update_v3.contract':contract}),patch('post_update_guard_runtime_v1.checks.Guard',FakeGuard):
   with self.assertRaisesRegex(AssertionError,'Same-input'):c.validate(t,args)
  self.assertTrue(c.failed);self.assertIsNotNone(o.controller.failed)
 def test_cap(self):
  c,o,t,contract,args=self.make();c.start('shadow');o.active=True;c.rows=[{}]*16
  with self.assertRaises(AssertionError):c.validate(t,args)
  self.assertTrue(c.failed)
 def test_binding_tamper(self):
  c,o,t,contract,args=self.make();c.start('shadow');o.active=True;o.call=object()
  with self.assertRaisesRegex(AssertionError,'Owned'):c.validate(t,args)
  self.assertTrue(c.failed)
 def test_no_double_per_helper_cap(self):
  # Outer four wrapper already checks its <16 cap before adding current row.
  source=(Path(__file__).parent/'checks.py').read_text();self.assertNotIn("owner.guard(",source)
 def test_cleanup_still_attempts_under_restore_failure(self):
  from .cleanup import finish
  events=[]
  def fail():events.append('binding');raise RuntimeError('restore')
  c=types.SimpleNamespace(binding=types.SimpleNamespace(close=fail))
  def close(*a,**kw):events.append(('four',kw['propagating']))
  r={}
  with patch('post_update_guard_runtime_v1.cleanup.close_and_finish',close):
   with self.assertRaises(RuntimeError):finish(r,lambda:None,c,None,None,None,propagating=False)
  self.assertEqual(events,['binding',('four',True)]);self.assertFalse(r['complete'])
 def test_sources_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
