import unittest,types,sys,ast
from pathlib import Path
from unittest.mock import patch
from .adapter import Checks,LIMITS,SPECS,Post,KV,Attention
class Tensor:
 shape=(32,);device=types.SimpleNamespace(type='cuda');dtype='torch.bfloat16'
 def __init__(self):self.bytes=b'unchanged';self.storage=types.SimpleNamespace(nbytes=lambda:64)
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def data_ptr(self):return 42
 def is_neg(self):return False
 def is_conj(self):return False
 def untyped_storage(self):return self.storage
class Snap:
 def __init__(self,x):self.x=x
 def set_(self,*a):return self
 def cpu(self):return self.x.bytes
class T(unittest.TestCase):
 def make(self):
  c=types.SimpleNamespace(active=True,failed=None)
  def latch(k,e):c.failed=repr(e)
  c.latch=latch;return c,Checks(c)
 def context(self,x,native=True,tile=True):
  fake=types.SimpleNamespace(uint8='u8',empty=lambda *a,**k:Snap(x),equal=lambda a,b:a==b,isfinite=lambda x:types.SimpleNamespace(all=lambda:native),cuda=types.SimpleNamespace(current_stream=lambda device:types.SimpleNamespace(cuda_stream=7)))
  return patch.dict(sys.modules,{'torch':fake,'tileir_validation_primitives_v3.api':types.SimpleNamespace(all_finite=lambda x:tile),'tileir_kv_strided_finite_v1.api':types.SimpleNamespace(all_finite_value=lambda x:tile)})
 def test504_and_conditional(self):
  c,s=self.make();x=Tensor()
  with self.assertRaises(AssertionError):s.start('replace')
  with self.context(x):
   for mode in ('shadow','replace'):
    s.start(mode)
    for site,n in LIMITS.items():
     for _ in range(n):self.assertTrue(s.finite(site,x))
    self.assertEqual(s.finish()['calls'],504)
 def test_cap_latch(self):
  c,s=self.make();x=Tensor();s.start('shadow')
  with self.context(x):
   for _ in range(24):s.finite('post.a',x)
   with self.assertRaises(AssertionError):s.finite('post.a',x)
   with self.assertRaises(AssertionError):s.finite('kv.key',x)
  self.assertIsNotNone(c.failed);self.assertEqual(len(s.rows),24)
 def test_false_returns_to_original_assert(self):
  c,s=self.make();s.start('shadow');x=Tensor()
  with self.context(x,False,False):self.assertIs(s.finite('kv.key',x),False)
  self.assertIsNone(c.failed)
 def test_difference_latches(self):
  c,s=self.make();s.start('shadow');x=Tensor()
  with self.context(x,True,False):
   with self.assertRaises(AssertionError):s.finite('kv.key',x)
  self.assertIsNotNone(c.failed)
 def test_bindings_and_source(self):
  mapping={k:c for k,c,n in SPECS};self.assertIs(mapping['post_conv'],Post);self.assertIs(mapping['attention_kv'],KV);self.assertIs(mapping['attention_core'],Attention)
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  for name in ('torch','vllm','cuda.tile'):self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
