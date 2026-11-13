import tempfile,unittest,types
from pathlib import Path
from qk_rope_runtime_integration_v1.adapter import Adapter
from qk_rope_runtime_integration_v1.result_gate import audit
class T(unittest.TestCase):
 def test_gate_incomplete(self):self.assertFalse(audit({})['passed'])
 def test_replace_requires_shadow(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a')
   with self.assertRaises(AssertionError):a.start('replace')
 def test_incomplete_count(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a');a.paths=['x']
   with self.assertRaises(AssertionError):a.finish()
 def test_binding_restore(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a');o=types.SimpleNamespace(f=1);a.bindings=[(o,'f',0,1)];self.assertTrue(a.close()['restored']);self.assertEqual(o.f,0)
 def test_binding_conflict_not_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a');o=types.SimpleNamespace(f=2);a.bindings=[(o,'f',0,1)]
   with self.assertRaises(AssertionError):a.close()
   self.assertEqual(o.f,2)
if __name__=='__main__':unittest.main()
class BindingTests(unittest.TestCase):
 def test_original_once_tuple_and_v_preserved(self):
  from unittest.mock import Mock
  class Tensor:
   shape=(32,10240)
   def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:7)
   def stride(self):return (10240,1)
   def storage_offset(self):return 0
  class V(Tensor):
   shape=(32,1024)
   def storage_offset(self):return 9216
  calls=[];q,k,g=object(),object(),object();original_tuple=(q,k,g)
  def native(q_gate,k,q_weight,k_weight,cos_sin_cache,positions):calls.append(1);return original_tuple
  module=types.SimpleNamespace(fused_qk_rmsnorm_rope_gate=native)
  class Owner:
   use_fused_qk_norm_rope_gate=True
   q_norm=types.SimpleNamespace(weight=object());k_norm=types.SimpleNamespace(weight=object());rotary_emb=types.SimpleNamespace(cos_sin_cache=object())
   def _project_qkv_gate(self,qkv,positions):
    z=module.fused_qk_rmsnorm_rope_gate(qkv,qkv,self.q_norm.weight,self.k_norm.weight,self.rotary_emb.cos_sin_cache,positions)
    self.returned=(z[0],z[1],V(),z[2]);return self.returned
  original=Owner._project_qkv_gate;o=Owner()
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'a')
   def apply(ctx,args,fn):return fn(),dict(passed=True)
   a.apply=apply;a.install_bindings([('layers.3.self_attn',o)],Owner,module)
   out=o._project_qkv_gate(Tensor(),object());self.assertIs(out,o.returned);self.assertEqual(calls,[1]);self.assertTrue(a.rows[0]['v_contract']);self.assertIsNone(a.pending)
   a.close();self.assertIs(Owner._project_qkv_gate,original);self.assertIs(module.fused_qk_rmsnorm_rope_gate,native)
