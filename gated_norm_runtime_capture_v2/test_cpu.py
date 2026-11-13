import ast,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import observer as o
class T:
 def __init__(self,shape,value='same'):self.shape=shape;self.value=value
class H:
 def remove(self):self.removed=True
class RMSNormGated:
 def __init__(self):self.weight=T((128,));self.bias=None;self.group_size=None;self.norm_before_gate=True;self.activation='silu';self.eps=1e-6;self._forward_method=self.forward_cuda
 def forward(self,*args):pass
 def forward_cuda(self,*args):pass
 def forward_native(self,*args):pass
 def register_forward_pre_hook(self,f,with_kwargs):self.pre=f;return H()
 def register_forward_hook(self,f,with_kwargs,always_call):self.post=f;return H()
class Tests(unittest.TestCase):
 def test_original_descriptor(self):
  class Parameter:
   requires_grad=True
   is_leaf=True
  t=Parameter();d=o.original_tensor_descriptor(t)
  self.assertIs(d['requires_grad'],True);self.assertIs(d['is_leaf'],True)
  self.assertTrue(d['python_type'].endswith('.Parameter'))
 def test_tensor_original_before_clone(self):
  class Parameter:
   requires_grad=True
   is_leaf=True
  t=Parameter();obs=o.Observer.__new__(o.Observer)
  with patch.object(o.Base,'tensor',return_value={'metadata':{'shape':[128]}}) as b:
   r=obs.tensor(t,'W');b.assert_called_once_with(t,'W')
  self.assertTrue(r['metadata']['requires_grad']);self.assertTrue(r['metadata']['is_leaf'])
 def test_two_shapes(self):
  obs=o.Observer.__new__(o.Observer)
  for k in ('pending','samples','weights','modules','live','seen'):setattr(obs,k,{})
  obs.handles=[];obs.save=lambda:None;obs.digest=lambda t:t.value;obs.tensor=lambda t,l:dict(logical_sha256=t.value)
  m=RMSNormGated();model=types.SimpleNamespace(named_modules=lambda:[('model.layers.0.linear_attn.norm',m)])
  with patch.object(o,'aliases',return_value={}):
   obs.install(model)
   for n in (1696,32):
    x=T((n,32,128));z=T(x.shape);y=T(x.shape)
    self.assertIsNone(m.pre(m,(x,z),{}));self.assertIsNone(m.post(m,(x,z),{},y))
   obs.verify();obs.close()
  self.assertEqual(len(obs.samples),2);self.assertEqual(len(obs.weights),1)
 def test_ast_no_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
