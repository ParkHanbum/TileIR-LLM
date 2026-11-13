import unittest,types
from unittest.mock import patch
from .adapter import Checks
class Guard:
 def __init__(self):self.pending=[];self.failed=False;self.stages=[]
 def validate(self,*args):self.stages=list(range(9))
class Tests(unittest.TestCase):
 def make(self):
  c=types.SimpleNamespace(active=True,failed=None)
  o=types.SimpleNamespace(active=True,failed=False,controller=c)
  o.latch=lambda *args:setattr(c,'failed',args[0])
  x=Checks(o);x.binding.copied=object();o.call=x.binding.copied;x.start();return x,o
 def test_complete_and_next_request(self):
  x,o=self.make()
  with patch('post_update_guard_lean_v1.adapter.Guard',Guard):
   for _ in range(16):x.validate(None,())
   self.assertEqual(x.finish()['stages'],144);x.start();self.assertEqual(x.calls,0)
 def test_short_request_rejected(self):
  x,_=self.make()
  with self.assertRaises(RuntimeError):x.finish()
 def test_foreign_latch(self):
  x,o=self.make();o.call=object()
  with self.assertRaises(RuntimeError):x.validate(None,())
  self.assertTrue(x.failed);self.assertIsNotNone(o.controller.failed)
 def test_error_preserved_pending_retained(self):
  class Bad(Guard):
   def validate(self,*a):raise ValueError('state index bounds')
  x,o=self.make()
  with patch('post_update_guard_lean_v1.adapter.Guard',Bad):
   with self.assertRaisesRegex(ValueError,'state index bounds'):x.validate(None,())
  self.assertEqual(len(x.pending),1);self.assertTrue(x.failed)
 def test_call_cap(self):
  x,o=self.make();x.calls=16
  with self.assertRaises(RuntimeError):x.validate(None,())
if __name__=='__main__':unittest.main()
