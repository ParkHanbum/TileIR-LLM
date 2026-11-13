import ast,tempfile,unittest,types
from pathlib import Path
from unittest.mock import patch
import torch
from gated_norm_runtime_integration_v1.adapter import Adapter
from gated_norm_runtime_integration_v1.precision import PrecisionPolicy
class RMSNormGated:
 def __init__(self):
  self.weight=torch.nn.Parameter(torch.ones(128,dtype=torch.bfloat16));self.eps=1e-6;self.group_size=None;self.norm_before_gate=True;self.bias=None;self.activation='silu';self._forward_method=self.forward_cuda;self.calls=0
 def forward_cuda(self,x,z):self.calls+=1;self.last=x.clone();return self.last
 def forward(self,x,z):return self._forward_method(x,z)
class Model:
 def __init__(self):self.mods=[(f'layers.{i}.linear_attn.norm',RMSNormGated()) for i in range(32) if i%4!=3]
 def named_modules(self):return iter(self.mods)
class Tests(unittest.TestCase):
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
 def test_identity_detach_and_conditional(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a');m=Model();a.install(m);v=m.mods[0][1];x=torch.zeros((32,32,128),dtype=torch.bfloat16);z=x.clone()
   def candidate(xx,zz,ww,eps):
    self.assertFalse(ww.requires_grad);self.assertEqual(ww.data_ptr(),v.weight.data_ptr());return xx.clone()
   with patch('tileir_vllm_gated_norm_full_div_v1.api.validate'),patch('tileir_vllm_gated_norm_full_div_v1.gated_rms_norm',side_effect=candidate):
    self.assertIs(v.forward(x,z),v.last);self.assertEqual(v.calls,1);self.assertTrue(a.rows[0]['passed'])
    with self.assertRaises(AssertionError):a.start('replace')
    a.shadow_pass=True;a.start('replace');v.forward(x,z);self.assertEqual(v.calls,1)
   a.close();self.assertTrue(all('forward' not in v.__dict__ for _,v in m.mods))
 def test_counts(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a');m=Model();a.install(m);a.rows=[dict(path=n,shape=[sz,32,128],passed=True) for n,_ in m.mods for sz in [1696]+[32]*15];self.assertTrue(a.finish()['shadow_pass']);a.rows.pop()
   with self.assertRaises(AssertionError):a.finish()
   a.close()
 def test_readonly_precision(self):
  mat=types.SimpleNamespace(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=False);t=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=mat)));p=PrecisionPolicy(t);self.assertEqual(p.select('F'),('unchanged_native',True));p.restore();self.assertFalse(mat.allow_bf16_reduced_precision_reduction_split_k)
if __name__=='__main__':unittest.main()
