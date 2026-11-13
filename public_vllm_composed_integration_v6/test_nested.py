import ast,types,tempfile,unittest
from pathlib import Path
from qk_rope_runtime_integration_v1.adapter import Adapter as QK
from attention_gate_runtime_integration_v1 import adapter as attention
class T:
 shape=(32,10240)
 def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:123)
 def storage_offset(self):return 0
class V(T):
 shape=(32,1024)
 def stride(self):return (10240,1)
 def storage_offset(self):return 9216
class Tests(unittest.TestCase):
 def test_actual_forward_and_project_wrappers(self):
  events=[]
  def helper(q_gate,k,q_weight,k_weight,cos_sin_cache,positions):events.append('native_helper');return ('q','k','gate')
  module=types.SimpleNamespace(fused_qk_rmsnorm_rope_gate=helper)
  class Owner:
   use_fused_qk_norm_rope_gate=True
   q_norm=types.SimpleNamespace(weight=object());k_norm=types.SimpleNamespace(weight=object());rotary_emb=types.SimpleNamespace(cos_sin_cache=object())
   def _project_qkv_gate(self,qkv,positions):
    q,k,g=module.fused_qk_rmsnorm_rope_gate(qkv,qkv,self.q_norm.weight,self.k_norm.weight,self.rotary_emb.cos_sin_cache,positions)
    return q,k,V(),g
   def forward(self,*args):return 'original_forward'
   def qkv_proj(self,x):events.append('qkv');return T(),None
   def attn(self,q,k,v):events.append('attention');return 'attn'
   def o_proj(self,x):events.append('out');return x,None
  owner=Owner();original_project=Owner._project_qkv_gate;original_forward=owner.forward.__func__
  # Compile actual nested attention forward, preserving dynamic project lookup.
  tree=ast.parse(Path(attention.__file__).read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
  def apply_gate(name,x,g):events.append('gate');return 'result'
  ns={'self':types.SimpleNamespace(apply_gate=apply_gate),'name':'layers.3.self_attn'}
  ast.fix_missing_locations(tree);exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_attention_forward','exec'),ns)
  owner.forward=types.MethodType(ns['forward'],owner)
  with tempfile.TemporaryDirectory() as d:
   q=QK(str(Path(d)/'q'))
   def apply(ctx,a,native):events.append('qk_wrapper');return native(),{'passed':True}
   q.apply=apply;q.install_bindings([('layers.3.self_attn',owner)],Owner,module)
   self.assertEqual(owner.forward(object(),object()),'result')
   self.assertEqual(events,['qkv','qk_wrapper','native_helper','attention','gate','out'])
   self.assertEqual(len(q.rows),1);self.assertTrue(q.rows[0]['v_contract']);self.assertTrue(q.rows[0]['project_tuple_preserved'])
   q.close();self.assertIs(Owner._project_qkv_gate,original_project);self.assertIs(module.fused_qk_rmsnorm_rope_gate,helper)
   del owner.forward;self.assertIs(owner.forward.__func__,original_forward)
if __name__=='__main__':unittest.main()
