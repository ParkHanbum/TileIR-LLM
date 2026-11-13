import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import torch
import tileir_vllm_hidden_gather_v1 as api
from runner_hidden_gather_runtime_integration_v2.adapter import Adapter
class Tests(unittest.TestCase):
 def test_negative_oob_and_alias_before_native_or_tile(self):
  for kind in ('negative','oob','alias'):
   with self.subTest(kind=kind):
    a=Adapter('/tmp/unused-hidden-guard');a.model=object();r=NS(model=a.model,batch_sharder=None);a.start('shadow')
    x=torch.zeros(1696,2560,dtype=torch.bfloat16);idx=torch.full((32,),-1 if kind=='negative' else 1696 if kind=='oob' else 0,dtype=torch.int64)
    real=torch.Tensor.untyped_storage
    def storage(t):
     if kind=='alias':return NS(data_ptr=lambda:1000,nbytes=lambda:256 if t.dtype==torch.int64 else 1696*5120)
     return real(t)
    with patch.object(torch.Tensor,'__getitem__',side_effect=AssertionError('Native indexing must not run')) as native,patch.object(api,'gather_hidden_rows',side_effect=AssertionError('Tile must not run')) as tile,patch.object(torch.Tensor,'untyped_storage',storage):
     with self.assertRaises(ValueError):a.invoke(lambda owner,h,i:a.gather(h,i),r,x,idx)
     self.assertEqual(native.call_count,0);self.assertEqual(tile.call_count,0)
    self.assertTrue(a.failed);self.assertFalse(a.pending);self.assertEqual(a.rows,[])
if __name__=='__main__':unittest.main()
