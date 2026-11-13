import ast,sys,unittest,types
from unittest.mock import patch
from pathlib import Path
from .api import validate_metadata,validate
class Tests(unittest.TestCase):
 def args(self,t=32):
  x=dict(shape=(t,32,128),stride=(4096,128,1),offset=0,dtype='torch.bfloat16');z=dict(x,stride=(12288,128,1),offset=8192);w=dict(shape=(128,),stride=(1,),offset=0,dtype='torch.float32');return x,z,w
 def test_both_shapes_padded_z(self):
  for n in (32,1696):self.assertEqual(validate_metadata(*self.args(n),1e-6),(n,1e-6))
 def test_no_gemma_offset(self):
  s=(Path(__file__).parent/'kernels.py').read_text();self.assertNotIn('+1.0',s);self.assertIn('weighted=normalized*w',s)
 def test_bad_shape(self):
  x,z,w=self.args();x['shape']=(1,32,128)
  with self.assertRaises(NotImplementedError):validate_metadata(x,z,w,1e-6)
 def test_overlap(self):
  x,z,w=self.args();z['stride']=(128,128,1)
  with self.assertRaises(ValueError):validate_metadata(x,z,w,1e-6)
 def test_bad_epsilon(self):
  for e in (True,0,-1,float('inf'),float('nan')):
   with self.assertRaises(ValueError):validate_metadata(*self.args(),e)
 def test_parameter_subclass_nongrad(self):
  class Tensor:
   def __init__(self,spec):self.shape=spec['shape'];self.dtype=spec['dtype'];self.layout='strided';self.requires_grad=False;self.device=types.SimpleNamespace(type='cuda');self.spec=spec
   def is_conj(self):return False
   def is_neg(self):return False
   def stride(self):return self.spec['stride']
   def storage_offset(self):return self.spec['offset']
  class Parameter(Tensor):pass
  x,z,w=self.args();x=Tensor(x);z=Tensor(z);w=Parameter(w);z.device=w.device=x.device
  fake=types.SimpleNamespace(Tensor=Tensor,strided='strided')
  with patch.dict(sys.modules,{'torch':fake}):
   self.assertEqual(validate(x,z,w),(32,1e-6));w.requires_grad=True
   with self.assertRaises(ValueError):validate(x,z,w)
 def test_ast_no_gpu_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
