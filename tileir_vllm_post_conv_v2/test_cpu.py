import unittest,sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from tileir_vllm_post_conv_v2.api import validate,fused_post_conv_prep
from unittest.mock import patch
class Tests(unittest.TestCase):
 def values(self):return [torch.empty((1696,8192),dtype=torch.bfloat16),torch.empty((1696,32),dtype=torch.bfloat16),torch.empty((1696,32),dtype=torch.bfloat16),torch.empty((32,),dtype=torch.float32),torch.empty((32,),dtype=torch.bfloat16)]
 def test_actual_metadata(self):self.assertEqual(validate(*self.values(),16,128,128,require_cuda=False),1696)
 def test_invalid_options(self):
  v=self.values()
  for args in [(True,128,128,True,False),(16,128,128,False,False),(16,128,128,True,True)]:
   with self.assertRaises(AssertionError):validate(*v,*args,require_cuda=False)
 def test_invalid_tensor_contract(self):
  v=self.values()
  for i,replacement in [(0,v[0][:32]),(1,v[1].float()),(3,v[3].bfloat16()),(4,torch.empty(33,dtype=torch.bfloat16)[1:])]:
   copy=list(v);copy[i]=replacement
   with self.assertRaises(AssertionError):validate(*copy,16,128,128,require_cuda=False)
 def test_bad_metadata_before_backend_import(self):
  with patch.dict(sys.modules,{'cuda.tile':None}):
   with self.assertRaises(AssertionError):fused_post_conv_prep(*self.values(),16,128,128)
 def test_public_tree_no_rsqrt_sum(self):
  s=Path(__file__).with_name('kernels.py').read_text();ast.parse(s)
  self.assertNotIn('ct.rsqrt',s);self.assertNotIn('ct.sum',s);self.assertIn('range(1,8)',s);self.assertIn('rounding_mode=RM.FULL',s)
 def test_cuda_not_initialized(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
