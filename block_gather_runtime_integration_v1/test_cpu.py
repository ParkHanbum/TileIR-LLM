import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
import adapter
from tileir_vllm_block_gather_v1.api import reference_cpu
class UvaBuffer:
 pass
class Tests(unittest.TestCase):
 def fixture(self):
  c=torch.ones((4,32),dtype=torch.int32);active=UvaBuffer();active.cpu=c;active._uva=c
  src=[torch.arange(32*n,dtype=torch.int32).reshape(32,n) for n in (1,1,1,8)]
  owner=types.SimpleNamespace(block_tables=[types.SimpleNamespace(gpu=t) for t in src],input_block_tables=[torch.full_like(t,-7) for t in src],num_blocks=types.SimpleNamespace(gpu=c,pool=types.SimpleNamespace(_curr=0,_uva_bufs=[active])))
  idx=torch.arange(31,-1,-1,dtype=torch.int64);return owner,idx,src,c
 def test_full_shadow_replace_exact_once(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');owner,idx,src,c=self.fixture();calls=[];last=[]
   def native():
    calls.append(1);ret=reference_cpu(tuple(src),c,idx,tuple(owner.input_block_tables),32);last[:]=[ret];return ret
   with patch.object(torch.Tensor,'is_pinned',return_value=True),patch('torch.cuda.synchronize'),patch('tileir_vllm_block_gather_v1.api.gather_block_tables',side_effect=reference_cpu):
    a.start('shadow')
    for i in range(16):self.assertIs(a.apply(owner,idx,32,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode')),last[0])
    self.assertTrue(a.finish()['shadow_pass']);self.assertEqual(len(calls),16)
    a.start('replace')
    for i in range(16):a.apply(owner,idx,32,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode'))
    self.assertEqual(a.finish()['native_gate_calls'],0);self.assertEqual(len(calls),16)
 def test_replace_requires_shadow(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a')
   with self.assertRaises(AssertionError):a.start('replace')
 def test_mismatch_blocks(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');owner,idx,src,c=self.fixture()
   def native():return reference_cpu(tuple(src),c,idx,tuple(owner.input_block_tables),32)
   def wrong(s,c,i,o,p):o[0].zero_();return tuple(x[:p] for x in o)
   with patch.object(torch.Tensor,'is_pinned',return_value=True),patch('torch.cuda.synchronize'),patch('tileir_vllm_block_gather_v1.api.gather_block_tables',side_effect=wrong):
    with self.assertRaises(AssertionError):a.apply(owner,idx,32,native,dict(num_reqs=32,phase='prefill'))
   self.assertIsNotNone(a.first_mismatch)
   with self.assertRaises(AssertionError):a.start('replace')
 def test_class_boundary_restore(self):
  class Blocks:
   def gather_block_tables(self,idx,num_reqs_padded,out=None,out_ptrs=None):return self.marker
  class Runner:
   pcp_manager=None
   def prepare_attn(self,b):return self.block_tables.gather_block_tables(b.idx_mapping,b.num_reqs_after_padding)
  original_r=Runner.prepare_attn;original_b=Blocks.gather_block_tables
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');block=Blocks();block.marker=object();r=Runner();r.block_tables=block
   batch=types.SimpleNamespace(idx_mapping=object(),num_reqs_after_padding=32)
   a.apply=lambda owner,idx,padded,native,meta:native()
   with patch.object(adapter,'phase_metadata',return_value={'num_reqs_after_padding':32}):
    a.install_classes(Runner,Blocks);self.assertIs(r.prepare_attn(batch),block.marker);self.assertIsNone(a.pending);a.close()
   self.assertIs(Runner.prepare_attn,original_r);self.assertIs(Blocks.gather_block_tables,original_b)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
