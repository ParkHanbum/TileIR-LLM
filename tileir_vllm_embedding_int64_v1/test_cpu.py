import unittest,ast,sys,types
from pathlib import Path
from unittest.mock import patch
from .contract import validate,signed_reference
from .api import embedding_ids_long
R=Path(__file__).parent;D=R.parent/'integer_boundary_ports_design_v1'
class T(unittest.TestCase):
 def test_signed_extremes(self):self.assertEqual(signed_reference([-2**31,-1,0,1,2**31-1]),[-2**31,-1,0,1,2**31-1])
 def test_descriptors(self):
  for n in (32,1696):self.assertEqual(validate((n,),(1,),'torch.int32',0,'cuda','torch.strided',False,False,False),n)
  for shape,stride,dtype,off,neg in [((0,),(1,),'torch.int32',0,False),((32,),(2,),'torch.int32',0,False),((32,),(1,),'torch.int64',0,False),((32,),(1,),'torch.int32',1,False),((32,),(1,),'torch.int32',0,True)]:
   with self.assertRaises(AssertionError):validate(shape,stride,dtype,off,'cuda','torch.strided',False,neg,False)
 def test_actual_api_invalid_before_allocation_import_launch(self):
  class Tensor:
   shape=(32,);dtype='torch.int64';device=types.SimpleNamespace(type='cuda');layout='torch.strided';requires_grad=False
   def stride(self):return (1,)
   def storage_offset(self):return 0
   def is_neg(self):return False
   def is_conj(self):return False
  writes=[]
  with patch.dict(sys.modules,{'torch':types.SimpleNamespace(Tensor=Tensor,empty=lambda *a,**k:writes.append(1))}):
   with self.assertRaises(AssertionError):embedding_ids_long(Tensor())
  self.assertEqual(writes,[])
 def test_owned_tp1_delegates_current_embedding_once(self):
  ns={};exec((D/'embedding_tp1_body.py.txt').read_text(),ns);events=[];input_=object();casted=object();output=object()
  owner=types.SimpleNamespace(tp_size=1,quant_method=types.SimpleNamespace(embedding=lambda own,x:events.append(('embedding',own,x))or output))
  def cast(x):events.append(('cast',x));return casted
  self.assertIs(ns['tp1_forward'](owner,input_,cast),output);self.assertEqual(events,[('cast',input_),('embedding',owner,casted)])
  owner.tp_size=2;events.clear()
  with self.assertRaises(AssertionError):ns['tp1_forward'](owner,input_,cast)
  self.assertEqual(events,[])
 def test_tail_and_ast(self):
  self.assertEqual(7*256-1696,96);self.assertEqual(256-32,224)
  for p in R.glob('*.py'):ast.parse(p.read_text())
  for mod in ('torch','vllm','cuda.tile'):self.assertNotIn(mod,sys.modules)
if __name__=='__main__':unittest.main()
