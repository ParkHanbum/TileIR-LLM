import ast,os,unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import torch
from tileir_vllm_gemma_norm_v3.api import validate,gemma_rms_norm
class Tests(unittest.TestCase):
 def values(self):return torch.empty((32,2560),dtype=torch.bfloat16),torch.zeros(2560,dtype=torch.bfloat16)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_valid_pair(self):
  x,w=self.values();self.assertEqual(validate(x,w,1e-6,x,require_cuda=False),(32,1e-6))
 def test_strided_offset_input(self):
  x=torch.empty((34,2600),dtype=torch.bfloat16)[1:33,:2560];w=torch.empty(2561,dtype=torch.float32)[1:];self.assertEqual(validate(x,w,1e-6,require_cuda=False)[0],32)
 def test_epsilon(self):
  x,w=self.values()
  for e in (True,0,-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):validate(x,w,e,require_cuda=False)
 def test_bad_member_zero_launch(self):
  x,w=self.values()
  with patch('tileir_vllm_gemma_norm_v3.api._enqueue') as enq:
   with self.assertRaises(ValueError):gemma_rms_norm(x,w,1e-6,torch.empty(3))
   enq.assert_not_called()
 def test_dtype_shape(self):
  x,w=self.values()
  for z in (x.float(),x[:3]):
   with self.assertRaises(NotImplementedError):validate(z,w,1e-6,require_cuda=False)
 def test_nonunit_inner_stride(self):
  x,w=self.values();z=torch.empty((32,5120),dtype=torch.bfloat16)[:,::2]
  with self.assertRaises(ValueError):validate(z,w,1e-6,require_cuda=False)
 def test_autograd(self):
  x,w=self.values();w.requires_grad_(True)
  with self.assertRaises(ValueError):validate(x,w,1e-6,require_cuda=False)
if __name__=='__main__':unittest.main()
