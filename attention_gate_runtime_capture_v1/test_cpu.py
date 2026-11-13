import ast,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import observer as o
from tileir_vllm_attention_gate_v1.api import validate_metadata
class Tensor:
 shape=(32,4096)
class Hook:
 def remove(self):pass
class Sub:
 def forward(self,*a):pass
 def register_forward_hook(self,f,**kw):self.post=f;return Hook()
 def register_forward_pre_hook(self,f,**kw):self.pre=f;return Hook()
class Model:
 attn_output_gate=True
 def __init__(self):self.attn=Sub();self.o_proj=Sub();self.calls=0;self.ret=(None,None,None,Tensor())
 def forward(self,*a):pass
 def _project_qkv_gate(self,*a,**kw):self.calls+=1;return self.ret
class Tests(unittest.TestCase):
 def test_identity_restore_association(self):
  obs=o.Observer.__new__(o.Observer)
  for k in ('pending','samples','weights','modules','live','seen'):setattr(obs,k,{})
  obs.handles=[];obs.save=lambda:None;obs.tensor=lambda t,l:dict(logical_sha256='same')
  m=Model();model=types.SimpleNamespace(named_modules=lambda:[('language_model.model.layers.3.self_attn',m)])
  with patch.object(o,'aliases',return_value={}):
   obs.install(model);self.assertIs(m._project_qkv_gate('q','p'),m.ret);self.assertEqual(m.calls,1)
   self.assertIsNone(m.attn.post(m.attn,(),{},Tensor()));self.assertIsNone(m.o_proj.pre(m.o_proj,(Tensor(),),{}))
   self.assertEqual(obs.samples['decode']['step'],0);obs.close()
  self.assertNotIn('_project_qkv_gate',m.__dict__)
 def test_exception_restore(self):
  m=Model();obs=o.Observer.__new__(o.Observer)
  for k in ('pending','samples','weights','modules','live','seen'):setattr(obs,k,{})
  obs.handles=[];obs.save=lambda:None
  model=types.SimpleNamespace(named_modules=lambda:[('layers.3.self_attn',m)])
  obs.install(model)
  try:
   with self.assertRaises(AssertionError):m.o_proj.pre(m.o_proj,(Tensor(),),{})
  finally:obs.close()
  self.assertNotIn('_project_qkv_gate',m.__dict__)
 def test_layout(self):
  m=dict(shape=(32,4096),stride=(12288,1),offset=8192,dtype='torch.bfloat16');self.assertEqual(validate_metadata(m,m),32)
  with self.assertRaises(ValueError):validate_metadata(dict(m,stride=(2048,1)),m)
 def test_cpu_ast(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
  for d in (Path(__file__).parent,Path(__file__).parent.parent/'tileir_vllm_attention_gate_v1'):
   for f in d.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
