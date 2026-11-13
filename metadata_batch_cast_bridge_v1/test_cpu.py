import ast,unittest
from types import SimpleNamespace
from .bridge import Bridge,conv_tree
from runtime_metadata_five_v1.binding import EXPECTED
class Tests(unittest.TestCase):
 def fixture(self):
  m=SimpleNamespace(failed=False);c=SimpleNamespace(failed=False,transport=SimpleNamespace(failed=False));return Bridge(m,c)
 def test_actual_ast_single_owned_conv(self):
  tree=conv_tree(EXPECTED['conv']['source']);calls=[n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
  self.assertEqual(calls.count('_meta_full'),2);self.assertEqual(calls.count('_batch_ptr_cast'),1)
 def test_missing_copy_rejects(self):
  s=EXPECTED['conv']['source'].replace('batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)','pass')
  with self.assertRaises(ValueError):conv_tree(s)
 def test_joint_failure_blocks_other_helper(self):
  b=self.fixture();calls=[]
  def bad():raise ValueError()
  with self.assertRaises(ValueError):b.guard(bad)()
  with self.assertRaises(RuntimeError):b.guard(lambda:calls.append(1))()
  self.assertFalse(calls);self.assertTrue(b.metadata.failed);self.assertTrue(b.cast.failed)
 def test_transport_failure_blocks_entry(self):
  b=self.fixture();b.cast.transport.failed=True
  with self.assertRaises(RuntimeError):b.guard(lambda:None)()
 def test_reverse_restore(self):
  b=self.fixture();old=lambda:1;new=lambda:2
  class A:pass
  A.f=new;b.installed=[(A,'f',old,new)];b.restore();self.assertIs(A.f,old)
 def test_restore_collision_drains(self):
  b=self.fixture();events=[]
  class A:f=1
  b.installed=[(A,'f',0,2)];b.cast.close=lambda:events.append('cast drain');b.metadata.close=lambda:events.append('meta close')
  with self.assertRaises(RuntimeError):b.close()
  self.assertEqual(events,['cast drain','meta close']);self.assertEqual(A.f,1)
