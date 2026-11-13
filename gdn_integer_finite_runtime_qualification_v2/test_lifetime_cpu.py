import unittest
from types import SimpleNamespace
from unittest.mock import patch
from . import adapter
class Part:
 def __init__(self,key,fail=0):self.key=key;self.fail=fail;self.calls=0;self.owner=object()
 def close(self):
  self.calls+=1
  if self.calls<=self.fail:raise RuntimeError('injected close failure')
  return {'restored':True}
class Tests(unittest.TestCase):
 def state(self,parts):
  s=object.__new__(adapter.Controller);s.parts=parts;s.active=True;return s
 def test_install_failure_keeps_owner_and_retry(self):
  a=Part('a');b=Part('b',1);state=self.state([a,b]);model=SimpleNamespace()
  def fail_install():
   state.close() # real close implementation, fails after partial restoration
  state.install=fail_install
  with patch.object(adapter,'Controller',return_value=state):
   with self.assertRaises(RuntimeError):adapter.install(model)
  self.assertIs(getattr(model,adapter.ATTR),state);self.assertEqual(state.parts,[b]);self.assertEqual(state._lifetime_parts,[a,b])
  with self.assertRaises(AssertionError):adapter.release(model)
  receipt=adapter.close(model);self.assertTrue(receipt['restored']);self.assertEqual(a.calls,1);self.assertEqual(b.calls,2)
  self.assertEqual(state._lifetime_parts,[a,b]);adapter.release(model);self.assertFalse(hasattr(model,adapter.ATTR))
 def test_repeated_close_failure_never_release(self):
  a=Part('a',3);s=self.state([a]);m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  for _ in range(2):
   with self.assertRaises(RuntimeError):adapter.close(m)
   with self.assertRaises(AssertionError):adapter.release(m)
   self.assertIs(getattr(m,adapter.ATTR),s);self.assertEqual(s.parts,[a])
 def test_success_reverse_and_retained(self):
  a=Part('a');b=Part('b');s=self.state([a,b]);r=s.close()
  self.assertEqual(r['close_order'],['b','a']);self.assertEqual(s._lifetime_parts,[a,b]);self.assertTrue(s._restoration_proven)
 def test_missing_receipt_retains_part(self):
  a=Part('a');a.close=lambda:None;s=self.state([a])
  with self.assertRaises(RuntimeError):s.close()
  self.assertEqual(s.parts,[a]);self.assertFalse(s._restoration_proven)

class CleanupTests(unittest.TestCase):
 def test_failed_sync_blocks_release(self):
  from .cleanup import final_cleanup
  a=Part('a');s=object.__new__(adapter.Controller);s.parts=[a];s.active=True;m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  engine=SimpleNamespace(apply_model=lambda fn:[fn(m)],engine_core=SimpleNamespace(shutdown=lambda:None));result={}
  def fail_sync():raise RuntimeError('CPU sync exception')
  final_cleanup(result,lambda:None,None,engine,adapter,True,fail_sync,lambda:None,propagating=True)
  self.assertIs(getattr(m,adapter.ATTR),s);self.assertFalse(result.get('adapter_released',False));self.assertFalse(result['complete'])
 def test_later_cleanup_retries_same_owner(self):
  from .cleanup import final_cleanup
  a=Part('a',1);s=object.__new__(adapter.Controller);s.parts=[a];s.active=True;m=SimpleNamespace();setattr(m,adapter.ATTR,s)
  with self.assertRaises(RuntimeError):s.close()
  seen=[]
  def apply(fn):seen.append(getattr(m,adapter.ATTR));return [fn(m)]
  engine=SimpleNamespace(apply_model=apply,engine_core=SimpleNamespace(shutdown=lambda:None));result={}
  final_cleanup(result,lambda:None,None,engine,adapter,True,lambda:None,lambda:None,propagating=True)
  self.assertTrue(all(x is s for x in seen));self.assertEqual(a.calls,2);self.assertTrue(result['adapter_released']);self.assertFalse(hasattr(m,adapter.ATTR));self.assertFalse(result['complete'])
