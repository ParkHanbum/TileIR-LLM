import unittest,types,sys
from pathlib import Path
from tileir_staged_typed_v2 import api
from . import gates
class Tests(unittest.TestCase):
 def test_empty_api_no_framework_import(self):
  t=types.SimpleNamespace(shape=(2,),stride=lambda:(1,),dtype='torch.int32',device=types.SimpleNamespace(type='cuda'),requires_grad=False,storage_offset=lambda:0,numel=lambda:2,element_size=lambda:4,untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:8))
  self.assertIsNone(api.apply(t,[],[],[],[]));self.assertNotIn('torch',sys.modules)
 def test_unsupported_guard_before_native(self):
  from staged_ownership_runtime_v1.test_observer import Tests as Mocks
  from .shadow import OwnershipObserver
  S,F,T,p,o=Mocks().make();o.__class__=OwnershipObserver;o.shadow_rows=[]
  with o:
   a=S();a._staged_write_indices=[2];a._staged_write_starts=[0];a._staged_write_contents=[1];a._staged_write_cu_lens=[1];p[0]='request'
   with self.assertRaises(RuntimeError):a.apply_write()
   self.assertEqual(a.n,0);self.assertEqual(o.shadow_rows[0]['native_returns'],0)
   with self.assertRaises(RuntimeError):a.apply_write()
   self.assertEqual(a.n,0)
 def test_actual_prerequisite(self):self.assertTrue(gates.qualification(Path(__file__).parent)['source_verified'])
