import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from gdn_post_conv_runtime_integration_v2.adapter import candidate_views,INPUTS
from tileir_vllm_post_conv_v2.api import validate
class Tests(unittest.TestCase):
 def values(self):return dict(conv_output=torch.empty((1696,8192),dtype=torch.bfloat16),a=torch.empty((1696,32),dtype=torch.bfloat16),b=torch.empty((1696,32),dtype=torch.bfloat16),A_log=torch.nn.Parameter(torch.ones(32,dtype=torch.float32)),dt_bias=torch.nn.Parameter(torch.ones(32,dtype=torch.bfloat16)),num_k_heads=16,head_k_dim=128,head_v_dim=128,apply_l2norm=True,output_g_exp=False)
 def test_actual_parameters_pass_candidate_only(self):
  v=self.values();origA=v['A_log'];origD=v['dt_bias']
  with self.assertRaises(AssertionError):validate(**v,require_cuda=False)
  args,record=candidate_views(v);self.assertEqual(validate(**args,require_cuda=False),1696)
  self.assertIs(v['A_log'],origA);self.assertIs(v['dt_bias'],origD);self.assertTrue(origA.requires_grad and origD.requires_grad)
  for k in INPUTS:self.assertTrue(record[k]['same_storage_layout_dtype']);self.assertFalse(args[k].requires_grad)
 def test_alias_and_mutation_visible_but_no_original_write(self):
  v=self.values();v['a'].fill_(2);args,_=candidate_views(v)
  self.assertEqual(args['a'].data_ptr(),v['a'].data_ptr());self.assertTrue(torch.equal(args['a'],v['a']))
  self.assertEqual(v['a'][0,0].item(),2)
 def test_noncuda_no_init(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
