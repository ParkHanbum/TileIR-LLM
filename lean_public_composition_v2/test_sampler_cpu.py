import unittest,ast
from pathlib import Path
from types import SimpleNamespace
from .sampler import SamplerFP32,SamplerFour
class C:
 active=True;failed=None
 def latch(self,k,e):self.failed=(k,str(e))
class TestSampler(unittest.TestCase):
 def test_invoke_return_and_mutation(self):
  c=C();a=SamplerFP32(c,'sampler_fp32',16);a.sampler_owner=None;a.current=None;owner=object();out=[]
  def copied(o):
   self.assertIs(o,owner);a.current=out;a.counts[('sampler',a.pending['phase'])]+=1;out.append('processor');return out
  for i in range(16):self.assertIs(a.invoke(copied,owner),out)
  self.assertEqual(a.finish()['calls'],16);self.assertIsNone(a.pending)
 def test_return_mismatch_latches(self):
  c=C();a=SamplerFP32(c,'cast',16);a.sampler_owner=None;a.current=None
  def bad(o):a.current=object();return object()
  with self.assertRaises(AssertionError):a.invoke(bad,object())
  self.assertIsNotNone(c.failed);self.assertIsNone(a.pending)
 def test_cat_once(self):
  a=SamplerFour(C(),'sampler_four',16);a.pending={};a.cat_calls=0;sentinel=object();a.cat_api=lambda x,y:sentinel
  self.assertIs(a.cat(None,None),sentinel)
  with self.assertRaises(AssertionError):a.cat(None,None)
 def test_no_payload_diagnostics(self):
  s=(Path(__file__).parent/'sampler.py').read_text();ast.parse(s)
  for banned in ['torch.cuda.synchronize','torch.save','detach().cpu','empty_like','bytecopy(']:self.assertNotIn(banned,s)
 def test_methods_not_data(self):
  a=SamplerFP32(C(),'cast',16);a.dispatch=17
  self.assertEqual(a.dispatch,17);self.assertTrue(callable(a.invoke))
if __name__=='__main__':unittest.main()
