"""Actual CPU torch tensors only. CUDA initialization is explicitly forbidden."""
import unittest
from unittest.mock import patch
import torch
from . import metrics
class Tests(unittest.TestCase):
 def test_raw_negative_zero(self):
  a=torch.tensor([[0.,1.,2.]],dtype=torch.bfloat16);b=a.clone();b[0,0]=-0.
  with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
   c=metrics.compare(a,b,torch);self.assertEqual(c['raw_mismatches'],1);self.assertEqual(c['strict_failures'],0);self.assertFalse(metrics.passing(c))
 def test_chunk_boundaries_exact_hash(self):
  a=torch.arange(30,dtype=torch.float32).reshape(5,6).to(torch.bfloat16)
  with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')),patch.object(metrics,'CHUNK_BYTES',8):
   c=metrics.compare(a,a.clone(),torch);self.assertTrue(metrics.passing(c));self.assertEqual(c['elements'],30);self.assertEqual(c['native_sha256'],metrics.digest(a,torch)['sha256'])
 def test_nonfinite_and_strict(self):
  a=torch.ones((2,8),dtype=torch.bfloat16);b=a.clone();b[0,0]=2.;b[1,7]=float('nan')
  with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')),patch.object(metrics,'CHUNK_BYTES',8):
   c=metrics.compare(a,b,torch);self.assertEqual(c['raw_mismatches'],2);self.assertEqual(c['strict_failures'],2);self.assertEqual(c['candidate_nonfinite'],1);self.assertEqual([x['index']for x in c['first_differences']],[[0,0],[1,7]])
if __name__=='__main__':unittest.main()
