import tempfile,unittest,types,sys
from pathlib import Path
from unittest.mock import patch
import torch
import adapter
class Tests(unittest.TestCase):
 def test_shadow_return_and_replace_count(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');x=torch.ones((32,18432),dtype=torch.bfloat16);out=torch.empty((32,9216),dtype=x.dtype);calls=[]
   def op(o,x):calls.append(1);o.copy_(x[:,:9216])
   def tile(o,x):o.copy_(x[:,:9216])
   with patch('tileir_vllm_silu_mul_v2.api.validate'),patch('tileir_vllm_silu_mul_v2.api.silu_and_mul',side_effect=tile):
    self.assertIsNone(a.apply('layer',out,x,op));self.assertEqual(len(calls),1)
    a.shadow_pass=True;a.start('replace');self.assertIsNone(a.apply('layer',out,x,op));self.assertEqual(len(calls),1);self.assertEqual(a.native_gate_calls,0)
 def test_mismatch_stops(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');x=torch.ones((32,18432),dtype=torch.bfloat16);out=torch.empty((32,9216),dtype=x.dtype)
   def native(o,x):o.zero_()
   def tile(o,x):o.fill_(1)
   with patch('tileir_vllm_silu_mul_v2.api.validate'),patch('tileir_vllm_silu_mul_v2.api.silu_and_mul',side_effect=tile):
    with self.assertRaises(AssertionError):a.apply('layer',out,x,native)
   self.assertIsNotNone(a.first_mismatch)
   with self.assertRaises(AssertionError):a.start('replace')
 def test_census(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');a.modules=[str(i) for i in range(32)]
   a.rows=[dict(path=n,shape=[m,18432],passed=True) for n in a.modules for m,k in [(1696,1),(32,15)] for _ in range(k)];a.native_gate_calls=a.tileir_gate_calls=512
   self.assertTrue(a.finish()['shadow_pass']);a.rows.pop()
   with self.assertRaises(AssertionError):a.finish()
 def test_install_restore_cuda_dispatch(self):
  original=lambda *a:None
  class SiluAndMul:
   def __init__(self):self.op=original;self._forward_method=self.forward_cuda
   def forward_cuda(self,x):pass
  mods=[(f'model.layers.{i}.mlp.act_fn',SiluAndMul()) for i in range(32)];model=types.SimpleNamespace(named_modules=lambda:mods)
  fake=types.ModuleType('vllm.model_executor.layers.activation');fake.SiluAndMul=SiluAndMul
  with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'vllm.model_executor.layers.activation':fake}),patch.object(torch.ops._C,'silu_and_mul',original,create=True):
   a=adapter.Adapter(Path(tmp)/'a');a.install(model);self.assertEqual(len(a.modules),32);a.close();self.assertTrue(all(m.op is original for n,m in mods))
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
