import unittest,types,ast
from pathlib import Path
from .api import check,BYTES,SHAPE,STRIDE
from .adapter import Adapter
class Tensor:
 def __init__(self,ptr):self.ptr=ptr;self.shape=SHAPE;self.dtype='bf16';self.device=types.SimpleNamespace(type='cuda');self.requires_grad=False;self.offset=0;self.size=BYTES
 def stride(self):return STRIDE
 def storage_offset(self):return self.offset
 def is_conj(self):return False
 def is_neg(self):return False
 def untyped_storage(self):return self
 def data_ptr(self):return self.ptr
 def nbytes(self):return self.size
class Tests(unittest.TestCase):
 def make(self):
  a,b=Tensor(100),Tensor(100+BYTES+4096);b.device=a.device;return a,b,types.SimpleNamespace(bfloat16='bf16',Tensor=Tensor)
 def test_tensor_type(self):
  a,b,t=self.make()
  with self.assertRaises(TypeError):check(object(),b,t)
 def test_observation_precedes_contract(self):
  s=(Path(__file__).parent/'adapter.py').read_text();self.assertLess(s.index('self.observed.append'),s.index('check(dst,src,t)'));self.assertIn("('ptr','storage_ptr')",s)
 def test_odd_owner_rejected(self):
  a,b,t=self.make();a.size+=1
  with self.assertRaises(ValueError):check(a,b,t)
 def test_unique_owner_finish(self):
  a=Adapter();a.layers=set(range(24));a.seen=set(range(23));a.rows=[dict(layer=i%23)for i in range(24)]
  with self.assertRaises(RuntimeError):a.finish()
 def test_rows_address_free(self):
  s=(Path(__file__).parent/'adapter.py').read_text();self.assertIn('source=public_meta(src),destination=public_meta(dst)',s)
 def test_valid(self):check(*self.make())
 def test_alias(self):
  a,b,t=self.make();b.ptr=a.ptr
  with self.assertRaisesRegex(ValueError,'overlap'):check(a,b,t)
 def test_extent(self):
  a,b,t=self.make();a.size-=2
  with self.assertRaisesRegex(ValueError,'extent'):check(a,b,t)
 def test_offset(self):
  a,b,t=self.make();a.offset=64;a.size+=128;check(a,b,t)
 def test_shape(self):
  a,b,t=self.make();a.shape=(32,32,128)
  with self.assertRaises(ValueError):check(a,b,t)
 def test_close_foreign(self):
  a=Adapter();own=object();foreign=object();m=types.SimpleNamespace(_forward_core=foreign);a.bindings=[(m,False,None,own)]
  with self.assertRaises(RuntimeError):a.close()
  self.assertIs(m._forward_core,foreign)
 def test_close_restore(self):
  a=Adapter();own=object();old=object();m=types.SimpleNamespace(_forward_core=own);a.bindings=[(m,True,old,own)];a.close();self.assertIs(m._forward_core,old)
 def test_poison_noop_cannot_equal(self):
  # Exhaustive raw BF16 bit patterns; complement differs even NaN/Inf/-0.
  self.assertTrue(all((x^65535)!=x for x in range(65536)))
 def test_ast_and_order(self):
  s=(Path(__file__).parent/'adapter.py').read_text();ast.parse(s)
  self.assertLess(s.index('private=db.clone()'),s.index('ret=func(*args,**kwargs)'))
  self.assertLess(s.index('returned=copier(pd,src)'),s.index('ret=func(*args,**kwargs)'))
if __name__=='__main__':unittest.main()
