import os,ast,tempfile,types,unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import torch
from norm_runtime_integration_v4.adapter import Adapter
class GemmaRMSNorm:
 def __init__(self,n=2560):self.weight=torch.zeros(n,dtype=torch.bfloat16);self.variance_epsilon=1e-6;self.last=None
 def forward(self,x,residual=None):
  self.last=x.clone() if residual is None else (x.clone(),residual.clone());return self.last
class Model:
 def __init__(self):self.mods=[(f'layers.{i//2}.'+('input_layernorm' if i%2==0 else 'post_attention_layernorm'),GemmaRMSNorm()) for i in range(64)]+[('norm',GemmaRMSNorm()),('q_norm',GemmaRMSNorm(256))]
 def named_modules(self):return iter(self.mods)
class FakeDispatch:
 def enter(self,key):pass
 def leave(self,key):return [dict(op='rms_norm',provider='native',inplace=False,supported=True)]
 def close(self):pass
class Tests(unittest.TestCase):
 def setUp(self):
  p=patch('norm_runtime_integration_v4.adapter.new_dispatch_observer',return_value=FakeDispatch());p.start();self.addCleanup(p.stop)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_native_identity_and_conditional_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'adapter');m=Model();a.install(m);mod=m.mods[0][1];x=torch.zeros((32,2560),dtype=torch.bfloat16)
   with patch('tileir_vllm_gemma_norm_v3.validate'),patch('tileir_vllm_gemma_norm_v3.gemma_rms_norm',side_effect=lambda x,w,e,r:x.clone()):
    y=mod.forward(x);self.assertIs(y,mod.last);self.assertTrue(a.rows[0]['passed'])
    with self.assertRaises(AssertionError):a.start('replace')
    a.shadow_pass=True;a.start('replace');native_last=mod.last;y=mod.forward(x);self.assertIs(mod.last,native_last);self.assertIsNot(y,native_last)
   a.close();self.assertTrue(all('forward' not in v.__dict__ for _,v in m.mods))
 def test_parameter_detach_preserves_storage(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'adapter');m=Model();mod=m.mods[0][1];mod.weight=torch.nn.Parameter(mod.weight);a.install(m);x=torch.zeros((32,2560),dtype=torch.bfloat16)
   def candidate(xx,ww,e,r):
    self.assertFalse(ww.requires_grad);self.assertEqual(ww.data_ptr(),mod.weight.data_ptr());self.assertEqual(xx.data_ptr(),x.data_ptr());return xx.clone()
   with patch('tileir_vllm_gemma_norm_v3.validate'),patch('tileir_vllm_gemma_norm_v3.gemma_rms_norm',side_effect=candidate):mod.forward(x)
   self.assertTrue(a.rows[0]['detach_checks']['weight']['original']['requires_grad']);self.assertFalse(a.rows[0]['detach_checks']['weight']['candidate']['requires_grad']);self.assertTrue(a.rows[0]['detach_checks']['weight']['same_storage_view']);a.close()
 def test_all_counts_required(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'adapter');m=Model();a.install(m)
   a.rows=[dict(path=n,shape=[size,2560],passed=True) for n,v in m.mods if len(v.weight)==2560 for size in [1696]+[32]*15]
   self.assertTrue(a.finish()['shadow_pass']);a.rows.pop()
   with self.assertRaises(AssertionError):a.finish()
   a.close()
 def test_mismatch_snapshot(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'adapter');m=Model();a.install(m);x=torch.zeros((32,2560),dtype=torch.bfloat16)
   with patch('tileir_vllm_gemma_norm_v3.validate'),patch('tileir_vllm_gemma_norm_v3.gemma_rms_norm',side_effect=lambda x,w,e,r:x+1):m.mods[0][1].forward(x)
   self.assertFalse(a.rows[0]['passed']);self.assertEqual(set(a.first_mismatch['artifacts']),{'x','weight','nativeY0','tileY0'});a.close()
if __name__=='__main__':unittest.main()
