import unittest
from types import SimpleNamespace
from .controller import Controller
from batch_ptr_int_cast_v1.test_cpu import Test as Fixture
class Tests(unittest.TestCase):
 def setup_case(self):
  src,dst,t,s,a,c=Fixture().fixture();q=Controller(t);q.start();return q,src,dst,t,s,a,c
 def test_lifetime_and_alias(self):
  q,src,dst,t,s,a,c=self.setup_case();self.assertIs(q.call(dst,src),dst);self.assertEqual(len(t.pending),1)
  r=q.finish();self.assertFalse(t.pending);self.assertEqual(r['native_private_oracle'],0);q.start();q.close()
 def test_early_reset_rejects(self):
  q,src,dst,t,s,a,c=self.setup_case();q.call(dst,src)
  with self.assertRaises(RuntimeError):q.start()
  self.assertEqual(len(t.pending),1);q.close()
 def test_second_call_latches(self):
  q,src,dst,t,s,a,c=self.setup_case();q.call(dst,src)
  with self.assertRaises(RuntimeError):q.call(dst,src)
  self.assertEqual(len(c),1);q.close()
 def test_unpinned_no_launch(self):
  q,src,dst,t,s,a,c=self.setup_case();src.pinned=False
  with self.assertRaises(ValueError):q.call(dst,src)
  self.assertFalse(c);self.assertTrue(q.failed)
 def test_restore_failure_still_drains(self):
  q,src,dst,t,s,a,c=self.setup_case();q.call(dst,src)
  def fail():raise RuntimeError()
  q.binding=SimpleNamespace(close=fail)
  with self.assertRaises(RuntimeError):q.close()
  self.assertFalse(t.pending);self.assertTrue(q.failed)
 def test_sync_failure_retains(self):
  q,src,dst,t,s,a,c=self.setup_case();q.call(dst,src)
  def fail():raise RuntimeError()
  s.synchronize=fail
  with self.assertRaises(RuntimeError):q.finish()
  self.assertEqual(len(t.pending),1);self.assertTrue(q.failed)
