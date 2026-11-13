import unittest,ast
from pathlib import Path
from types import SimpleNamespace
from .api import validate
from .comparator import comparator
class T:
 def __init__(self,n=4,dtype='torch.uint64'):self.shape=(n,);self.dtype=dtype;self.device=SimpleNamespace(type='cuda');self.layout='torch.strided';self.requires_grad=False
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def untyped_storage(self):return SimpleNamespace(nbytes=lambda:self.shape[0]*8)
 def is_neg(self):return False
 def is_conj(self):return False
class Tests(unittest.TestCase):
 def test_types_lengths(self):
  for dtype in ('torch.uint64','torch.int64'):
   for n in (1,2,3,4):self.assertEqual(validate(T(n,dtype),T(n,dtype)),n)
 def test_reject_before_compute(self):
  for t in (T(0),T(5),T(2,'torch.float64')):
   with self.assertRaises(ValueError):validate(t,t)
 def test_creation_order_shortcircuit(self):
  events=[]
  def tensor(vals,**kw):events.append(('create',kw['dtype']));return vals
  torch=SimpleNamespace(tensor=tensor,uint64='u64',int64='i64');owners=[SimpleNamespace(data_ptr=lambda:2**64-1,stride=lambda i:128)]
  ptr=SimpleNamespace(device='cuda');stride=SimpleNamespace(device='cuda')
  def equal(a,b):events.append(('compare',a is ptr));return False
  self.assertFalse(comparator(torch,equal)(ptr,stride,owners));self.assertEqual(events,[('create','u64'),('create','i64'),('compare',True)])
 def test_both_equal_calls(self):
  events=[];torch=SimpleNamespace(tensor=lambda x,**k:x,uint64='u',int64='i');t=SimpleNamespace(device='cuda');o=SimpleNamespace(data_ptr=lambda:0,stride=lambda i:0)
  self.assertTrue(comparator(torch,lambda a,b:events.append(b)or True)(t,t,[o]));self.assertEqual(events,[[0],[0]])
 def test_kernel_no_pointer_conversion(self):
  source=Path(__file__).with_name('kernel.py').read_text();tree=ast.parse(source)
  self.assertNotIn('data_ptr',source);self.assertNotIn('bitcast',source)
  calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
  self.assertEqual(calls.count('gather'),2);self.assertEqual(calls.count('scatter'),1)
 def test_kernel_model_extrema(self):
  for values in ([0,2**63,2**64-1,1],[-2**63,2**63-1,0,-1]):
   for n in (1,2,3,4):
    a=values[:n];self.assertEqual(sum(x!=y for x,y in zip(a,a)),0)
    for lane in range(n):
     b=a[:];b[lane]=1 if b[lane]!=1 else 0;self.assertEqual(sum(x!=y for x,y in zip(a,b)),1)

class TypeTests(unittest.TestCase):
 def test_real_entry_rejects_duck_before_tile_import(self):
  import sys
  from unittest.mock import patch
  from .api import equal
  class OwnedTensor:pass
  with patch.dict(sys.modules,{'torch':SimpleNamespace(Tensor=OwnedTensor)}):
   with self.assertRaises(TypeError):equal(T(),T())
  self.assertNotIn('cuda.tile',sys.modules)
class BindingTests(unittest.TestCase):
 def test_actual_origin_and_restore(self):
  from unittest.mock import patch
  from staged_ownership_runtime_v1.observer import comparator
  from .binding import Binding
  torch=SimpleNamespace();old=comparator(torch);observer=SimpleNamespace(compare=old)
  with patch('ownership_table_pair_equal_v1.binding.request_comparator',return_value=(lambda *a:True,lambda:{})):
   b=Binding(observer,torch,lambda:'request');b.install();self.assertTrue(observer.compare(None,None,None));self.assertTrue(b.finish()['actual_callback_origin']['code_identity_verified']);b.close();self.assertIs(observer.compare,old)
 def test_wrong_origin_rejected(self):
  from .binding import Binding
  with self.assertRaises(ValueError):Binding(SimpleNamespace(compare=lambda *a:True),SimpleNamespace(),lambda:'request')
 def test_foreign_restore_not_overwritten(self):
  from unittest.mock import patch
  from staged_ownership_runtime_v1.observer import comparator
  from .binding import Binding
  torch=SimpleNamespace();observer=SimpleNamespace(compare=comparator(torch))
  with patch('ownership_table_pair_equal_v1.binding.request_comparator',return_value=(lambda *a:True,lambda:{})):
   b=Binding(observer,torch,lambda:'request');b.install();foreign=lambda:False;observer.compare=foreign
   with self.assertRaises(RuntimeError):b.close()
   self.assertIs(observer.compare,foreign);self.assertTrue(b.failed)
class DiagnosticTests(unittest.TestCase):
 def test_unsupported_metadata_saved_before_rejection(self):
  from .qualifier import Qualifier
  saved=[];q=Qualifier(None,record=lambda value:saved.append(value));a=T(0);b=T(0)
  with self.assertRaises(ValueError):q.pair(a,b)
  self.assertTrue(q.failed);self.assertEqual(saved[0]['observed'][0]['a']['shape'],[0])
  self.assertEqual(saved[0]['observed'][0]['a']['storage_bytes'],0)
  self.assertNotIn('ptr',saved[0]['observed'][0]['a'])
