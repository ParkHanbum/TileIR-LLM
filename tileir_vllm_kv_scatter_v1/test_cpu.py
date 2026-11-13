import unittest,torch
from .api import validate
class Tests(unittest.TestCase):
 def values(self):
  base=torch.zeros(8*4*2048,dtype=torch.bfloat16)
  k=torch.as_strided(base,(8,4,4,256),(8192,2048,512,1),0);v=torch.as_strided(base,k.shape,k.stride(),256)
  return [torch.zeros(32,4,256,dtype=torch.bfloat16),torch.zeros(32,4,256,dtype=torch.bfloat16),k,v,torch.arange(32,dtype=torch.int64),'auto',torch.tensor(1.),torch.tensor(1.)]
 def test_valid(self):self.assertEqual(validate(*self.values(),require_cuda=False),32)
 def test_negative_duplicate(self):
  for x in (-1,0):
   v=self.values();v[4][-1]=x
   with self.assertRaises(AssertionError):validate(*v,require_cuda=False)
 def test_quantized_rejected(self):
  v=self.values();v[5]='fp8'
  with self.assertRaises(AssertionError):validate(*v,require_cuda=False)
 def test_separate_cache_rejected(self):
  v=self.values();v[3]=v[3].clone()
  with self.assertRaises(AssertionError):validate(*v,require_cuda=False)
 def test_offset_input_preserved(self):
  v=self.values();base=torch.zeros(32*10240,dtype=torch.bfloat16);v[1]=torch.as_strided(base,(32,4,256),(10240,256,1),9216);self.assertEqual(validate(*v,require_cuda=False),32)
 def test_no_cuda_init(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
