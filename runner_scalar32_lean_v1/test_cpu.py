import unittest,sys,ast
from pathlib import Path
from runner_scalar32_qualification_v3.test_cpu import owner,candidate
from runner_scalar32_lean_v1.controller import Controller
class Tests(unittest.TestCase):
 def test_full_census_restore(self):
  x=owner();fn=type(x).add_request;c=Controller(x,candidate);c.install();c.start()
  for i in reversed(range(32)):x.add_request(i,None)
  self.assertEqual(c.finish()['calls'],32);self.assertEqual(x.num_accepted_tokens_gpu.v,[1]*32);c.close();self.assertIs(type(x).add_request,fn)
 def test_duplicate_before_super(self):
  x=owner();c=Controller(x,candidate);c.install();c.start();x.add_request(0,None)
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(x.base_calls,1);c.close()
 def test_cap_before_super(self):
  x=owner();c=Controller(x,candidate);c.install();c.start()
  for i in range(32):x.add_request(i,None)
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(x.base_calls,32);c.close()
 def test_global_latch(self):
  x=owner()
  def fail():raise RuntimeError('global')
  c=Controller(x,candidate,fail);c.install();c.start()
  with self.assertRaises(RuntimeError):x.add_request(0,None)
  self.assertEqual(x.base_calls,0);c.close()
 def test_owner_swap(self):
  x=owner();c=Controller(x,candidate);c.install();c.start();x.num_accepted_tokens_gpu=x.num_accepted_tokens_gpu.clone()
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(x.base_calls,0);c.close()
 def test_candidate_failure_counts(self):
  x=owner()
  def fail(*a):raise RuntimeError('kernel')
  c=Controller(x,fail);c.install();c.start()
  with self.assertRaises(RuntimeError):x.add_request(0,None)
  with self.assertRaises(RuntimeError):x.add_request(1,None)
  self.assertEqual((c.attempts,c.returns,x.base_calls),(1,0,1));c.close()
 def test_inactive_and_restore(self):
  x=owner();c=Controller(x,candidate);c.install()
  with self.assertRaises(RuntimeError):x.add_request(0,None)
  self.assertEqual(x.base_calls,0);c.close()
 def test_restart_after_finish(self):
  x=owner();c=Controller(x,candidate);c.install()
  for _ in range(2):
   c.start()
   for i in range(32):x.add_request(i,None)
   c.finish()
  self.assertEqual(x.base_calls,64);c.close()
 def test_no_proof_operations(self):
  s=Path(__file__).with_name('controller.py').read_text();ast.parse(s)
  for forbidden in ['.clone(','.cpu(','.fill_(','.numpy(','.fsync(','.synchronize(']:self.assertNotIn(forbidden,s)
 def test_no_framework_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
