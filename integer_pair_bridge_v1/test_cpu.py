import unittest
from bridge import Bridge
class State:
 def __init__(self,n,log):self.n=n;self.log=log;self.failed=None;self.calls=0;self.active=False
 def guard(self):assert self.active and self.failed is None and self.calls<self.n
 def start(self,mode):self.active=True;self.calls=0
 def finish(self):assert self.calls==self.n;self.active=False;return {'calls':self.calls}
 def close(self):self.log.append(self.n);return {'restored':True}
class Tests(unittest.TestCase):
 def setup_pair(self):
  log=[];a=State(16,log);b=State(48,log);p=Bridge(a,b,lambda:None);p.bind_guards();p.start('shadow');return p,a,b,log
 def test_counts(self):
  p,a,b,_=self.setup_pair()
  for i in range(16):
   a.guard();a.calls+=1
   for j in range(3):b.guard();b.calls+=1
  self.assertEqual(p.finish(),{'embedding_cast':{'calls':16},'sampler_integer':{'calls':48}})
 def test_exhaustion_blocks_peer(self):
  p,a,b,_=self.setup_pair();a.calls=16
  with self.assertRaises(AssertionError):a.guard()
  with self.assertRaises(AssertionError):b.guard()
  self.assertEqual(b.calls,0)
 def test_close_failure_attempts_both(self):
  p,a,b,log=self.setup_pair()
  def fail():log.append('lean');raise RuntimeError('test')
  with self.assertRaises(RuntimeError):p.close(fail)
  self.assertEqual(log,['lean',48,16])
  with self.assertRaises(AssertionError):p.require_release()
 def test_reverse_success(self):
  p,a,b,log=self.setup_pair();p.close(lambda:dict(restored=True));p.require_release();self.assertEqual(log,[48,16])
if __name__=='__main__':unittest.main()
