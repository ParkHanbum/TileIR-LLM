import unittest
from unittest.mock import patch
import torch
from . import api
class Tests(unittest.TestCase):
 def fixture(self):return (torch.tensor([1,1,1],dtype=torch.int32),torch.tensor([2,5,6],dtype=torch.int32),torch.tensor([0,1,2,4],dtype=torch.int32),torch.tensor([0,1,1]),torch.tensor([3,5],dtype=torch.int32))
 def test_mutation_return_and_readonly(self):
  a=self.fixture();saved=[t.clone() for t in a[1:]];s,r=api.reference_cpu(*a)
  self.assertIs(s,a[0]);self.assertEqual(s.tolist(),[0,1,1]);self.assertEqual(r.tolist(),[0,0,1]);self.assertNotEqual(s.untyped_storage().data_ptr(),r.untyped_storage().data_ptr())
  self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[1:],saved)))
 def test_all32_equal_boundary(self):
  a=(torch.ones(32,dtype=torch.int32),torch.full((32,),53,dtype=torch.int32),torch.arange(33,dtype=torch.int32),torch.arange(32),torch.full((32,),53,dtype=torch.int32));s,r=api.reference_cpu(*a);self.assertTrue(bool((s==1).all()));self.assertTrue(bool((r==0).all()))
 def test_values(self):
  for kind in ('index','prefix','count'):
   a=self.fixture()
   if kind=='index':a[3][0]=-1
   if kind=='prefix':a[2][0]=1
   if kind=='count':a[0][0]=2
   with self.assertRaises(ValueError):api.reference_cpu(*a)
   self.assertEqual(a[0][1:].tolist(),[1,1])
 def test_overlap(self):
  a=list(self.fixture());buf=bytearray(64);a[0]=torch.frombuffer(buf,dtype=torch.int32,count=3);a[1]=torch.frombuffer(buf,dtype=torch.int32,count=3,offset=4)
  with self.assertRaises(ValueError):api.validate(*a,require_cuda=False)
 def test_invalid_zero_launch(self):
  with patch.object(api,'_enqueue') as launch:
   with self.assertRaises(ValueError):api.get_num_sampled_and_rejected(*self.fixture())
   launch.assert_not_called()
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
