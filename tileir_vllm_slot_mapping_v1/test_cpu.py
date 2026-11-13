import unittest
import torch
from .api import reference_cpu,validate,validate_values_cpu
class Tests(unittest.TestCase):
 def fixture(self):
  return ((torch.tensor([[3,7],[9,11]],dtype=torch.int32),)*2,torch.tensor([1,-1,0]),torch.tensor([0,2,3,4],dtype=torch.int32),torch.tensor([0,5,2,3]),torch.tensor([4,4],dtype=torch.int32),torch.tensor([True,False]),torch.full((2,9),99,dtype=torch.int64),6)
 def test_mapping_tail_disable_dummy_alias(self):
  args=self.fixture();out=args[-2];v=reference_cpu(*args)
  self.assertEqual(out[0].tolist(),[36,45,-1,15,-1,-1,-1,-1,-1]);self.assertEqual(out[1].tolist(),[-1]*9)
  self.assertEqual(v.shape,(2,6));self.assertEqual(v.stride(),(9,1));self.assertEqual(v.data_ptr(),out.data_ptr())
 def test_zero_tokens(self):
  a=list(self.fixture());a[1]=torch.empty(0,dtype=torch.int64);a[2]=torch.tensor([0],dtype=torch.int32);a[3]=torch.empty(0,dtype=torch.int64);a[-1]=0
  v=reference_cpu(*a);self.assertEqual(v.shape,(2,0));self.assertTrue(bool((a[-2]==-1).all()))
 def test_invalid_values(self):
  for which in ('prefix','block','product'):
   a=list(self.fixture())
   if which=='prefix':a[2][1]=-1
   if which=='block':a[3][1]=8
   if which=='product':a[0][0][1,1]=2147483647
   with self.assertRaises(ValueError):validate_values_cpu(*a)
   self.assertTrue(bool((a[-2]==99).all()))
 def test_cp_and_dtype(self):
  a=self.fixture()
  with self.assertRaises(NotImplementedError):validate(*a,cp_size=2,require_cuda=False)
  a=list(a);a[1]=a[1].int()
  with self.assertRaises(ValueError):validate(*a,require_cuda=False)
 def test_overlap(self):
  a=list(self.fixture());buf=bytearray(256);a[0]=(torch.frombuffer(buf,dtype=torch.int32,count=4).reshape(2,2),)*2;a[-2]=torch.frombuffer(buf,dtype=torch.int64,count=18,offset=8).reshape(2,9)
  with self.assertRaises(ValueError):validate(*a,require_cuda=False)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
