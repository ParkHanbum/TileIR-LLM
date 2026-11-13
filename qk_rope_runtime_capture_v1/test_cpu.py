import unittest,tempfile,types,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
import observer as o
class Tests(unittest.TestCase):
 def fixture(self,out):
  module=types.SimpleNamespace();calls=[]
  def helper(q_gate,k,q_weight,k_weight,cos_sin_cache,positions,eps,num_q_heads,num_kv_heads,head_dim,rotary_dim,mrope_section=None,norm_beta=0.0):
   calls.append(1);a=q_gate.reshape(-1,num_q_heads,head_dim*2)
   return a[:,:,:head_dim].reshape(-1,num_q_heads*head_dim).clone(),k.clone(),a[:,:,head_dim:].reshape(-1,num_q_heads*head_dim).clone()
  module.fused_qk_rmsnorm_rope_gate=helper
  class Model:
   def __init__(self):
    self.use_fused_qk_norm_rope_gate=True;self.q_norm=types.SimpleNamespace(weight=torch.ones(256));self.k_norm=types.SimpleNamespace(weight=torch.ones(256));self.rotary_emb=types.SimpleNamespace(cos_sin_cache=torch.zeros(80,64))
   def _project_qkv_gate(self,qkv,positions):
    qg,k,v=qkv.split((8192,1024,1024),dim=-1)
    if self.use_fused_qk_norm_rope_gate:q,k,g=module.fused_qk_rmsnorm_rope_gate(qg,k,self.q_norm.weight,self.k_norm.weight,self.rotary_emb.cos_sin_cache,positions,1e-6,16,4,256,64,norm_beta=1.0)
    else:q,g=qg[:,:4096],qg[:,4096:]
    self.lastret=(q,k,v,g);return self.lastret
  modules=[(f'language_model.model.layers.{i}.self_attn',Model()) for i in range(3,32,4)]
  obs=o.Observer(out);obs.tensor=lambda t,label:dict(storage_byte_sha256='same');old=Model._project_qkv_gate;obs.install_bindings(modules,Model,module)
  return obs,modules,Model,module,helper,old,calls
 def test_owner_helper_exact_once_return_restore(self):
  with tempfile.TemporaryDirectory() as td:
   obs,mods,cls,module,fn,old,calls=self.fixture(Path(td)/'bank');x=torch.zeros((32,10240),dtype=torch.bfloat16);p=torch.arange(32)
   for _,m in mods:self.assertIs(m._project_qkv_gate(x,p),m.lastret)
   self.assertEqual(len(calls),8);self.assertEqual(sum(obs.seen.values()),8);self.assertEqual(len(obs.samples),1);self.assertTrue(obs.samples['decode']['gate_copy_raw_equal']);self.assertTrue(obs.samples['decode']['v_aliases']['qkv|v']['same_storage']);obs.close();self.assertIs(cls._project_qkv_gate,old);self.assertIs(module.fused_qk_rmsnorm_rope_gate,fn)
 def test_false_branch_holds_and_clears_pending(self):
  with tempfile.TemporaryDirectory() as td:
   obs,mods,cls,module,fn,old,calls=self.fixture(Path(td)/'bank');m=mods[0][1];m.use_fused_qk_norm_rope_gate=False
   with self.assertRaises(AssertionError):m._project_qkv_gate(torch.zeros((32,10240),dtype=torch.bfloat16),torch.arange(32))
   self.assertIsNone(obs.pending);self.assertTrue(obs.errors);self.assertEqual(calls,[]);obs.close()
 def test_storage_dedup_and_mutation(self):
  with tempfile.TemporaryDirectory() as td:
   obs=o.Observer(Path(td)/'bank');x=torch.arange(32,dtype=torch.int32);view=x[4:12]
   with patch.object(torch.cuda,'current_stream',return_value=types.SimpleNamespace(cuda_stream=0)):
    a=obs.tensor(view,'a');b=obs.tensor(view,'b');self.assertEqual(a['path'],b['path']);self.assertEqual(a['storage_path'],b['storage_path']);x[0]=99;c=obs.tensor(view,'c');self.assertEqual(a['path'],c['path']);self.assertNotEqual(a['storage_byte_sha256'],c['storage_byte_sha256'])
 def test_incomplete_gate_rejects(self):
  from result_gate import audit
  self.assertFalse(audit({'complete':False})['passed'])
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
