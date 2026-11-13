import unittest
from unittest.mock import patch
import torch
from . import api
class Tests(unittest.TestCase):
 def fixture(self,dtype=torch.int32):
  return (torch.arange(12,dtype=torch.int32).reshape(3,4),torch.arange(18,dtype=torch.int32).reshape(3,6)),torch.tensor([[2,0,4],[1,3,6]],dtype=torch.int32),torch.tensor([2,0],dtype=dtype),(torch.full((4,4),-9,dtype=torch.int32),torch.full((4,6),-9,dtype=torch.int32)),3
 def test_tail_pad_and_views(self):
  for dtype in (torch.int32,torch.int64):
   s,c,i,o,p=self.fixture(dtype); result=api.reference_cpu(s,c,i,o,p)
   self.assertEqual(o[0].tolist(),[[8,9,10,11],[0,1,-9,-9],[0,0,0,0],[-9]*4])
   self.assertEqual(o[1][1].tolist(),[0,-9,-9,-9,-9,-9]); self.assertEqual(result[0].stride(),(4,1));result[0][0,0]=123;self.assertEqual(o[0][0,0],123)
 def test_zero_count(self):
  s,c,i,o,p=self.fixture();i[0]=1;api.reference_cpu(s,c,i,o,p);self.assertEqual(o[0][0].tolist(),[-9]*4)
 def test_value_rejects_no_mutation(self):
  for bad in (-1,3):
   s,c,i,o,p=self.fixture();i[0]=bad
   with self.assertRaises(ValueError):api.reference_cpu(s,c,i,o,p)
   self.assertTrue(bool((o[0]==-9).all()))
  s,c,i,o,p=self.fixture();c[0,2]=5
  with self.assertRaises(ValueError):api.reference_cpu(s,c,i,o,p)
 def test_metadata_no_launch(self):
  s,c,i,o,p=self.fixture()
  with patch.object(api,'_enqueue') as launch:
   for pp in (True,-1,1):
    with self.assertRaises(ValueError):api.gather_block_tables(s,c,i,o,pp)
   launch.assert_not_called()
 def test_overlap_distinct_storage(self):
  s,c,i,o,p=self.fixture();buf=bytearray(128)
  a=torch.frombuffer(buf,dtype=torch.int32,count=12,offset=0).reshape(3,4)
  b=torch.frombuffer(buf,dtype=torch.int32,count=16,offset=4).reshape(4,4)
  with self.assertRaises(ValueError):api.validate((a,s[1]),c,i,(b,o[1]),p,require_cuda=False)
 def test_noncontiguous_and_crossoutput(self):
  s,c,i,o,p=self.fixture()
  with self.assertRaises(ValueError):api.validate((s[0].T,s[1]),c,i,o,p,require_cuda=False)
  with self.assertRaises(ValueError):api.validate((s[0],s[0]),c,i,(o[0],o[0]),p,require_cuda=False)
 def test_no_cuda(self): self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
