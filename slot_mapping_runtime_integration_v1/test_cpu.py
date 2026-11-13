import tempfile,unittest,types
from pathlib import Path
from unittest.mock import patch
import torch
import adapter
from tileir_vllm_slot_mapping_v1.api import reference_cpu
class Tests(unittest.TestCase):
 def fixture(self):
  src=[torch.arange(32*n,dtype=torch.int32).reshape(32,n) for n in (1,1,1,8)]
  o=types.SimpleNamespace(cp_size=1,block_tables=[types.SimpleNamespace(gpu=t) for t in src],slot_mappings=torch.full((4,2048),99,dtype=torch.int64),block_sizes_tensor=torch.tensor([128,128,128,528],dtype=torch.int32),slot_mapping_enabled=torch.ones(4,dtype=torch.bool),dcp_sharded=torch.ones(4,dtype=torch.bool),block_table_ptrs=torch.tensor([1,2,3,4]),block_table_strides=torch.tensor([1,1,1,8]))
  return o,torch.arange(32,dtype=torch.int64),torch.arange(33,dtype=torch.int32),torch.zeros(32,dtype=torch.int64)
 def test_all16_shadow_conditional_replace(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');o,idx,q,pos=self.fixture();calls=[];last=[]
   def native():
    calls.append(1);ret=reference_cpu(tuple(x.gpu for x in o.block_tables),idx,q,pos,o.block_sizes_tensor,o.slot_mapping_enabled,o.slot_mappings,32);last[:]=[ret];return ret
   def tile(*args,cp_size=1):return reference_cpu(*args)
   with patch('torch.cuda.synchronize'),patch('tileir_vllm_slot_mapping_v1.api.compute_slot_mappings',side_effect=tile):
    a.start('shadow')
    for i in range(16):self.assertIs(a.apply(o,idx,q,pos,32,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode')),last[0])
    self.assertTrue(a.finish()['shadow_pass']);a.start('replace')
    for i in range(16):a.apply(o,idx,q,pos,32,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode'))
    self.assertEqual(a.finish()['native_gate_calls'],0);self.assertEqual(len(calls),16)
 def test_mismatch_blocks_replacement(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');o,idx,q,pos=self.fixture()
   def native():return reference_cpu(tuple(x.gpu for x in o.block_tables),idx,q,pos,o.block_sizes_tensor,o.slot_mapping_enabled,o.slot_mappings,32)
   def wrong(*args,**kw):args[-2].zero_();return args[-2][:,:args[-1]]
   with patch('torch.cuda.synchronize'),patch('tileir_vllm_slot_mapping_v1.api.compute_slot_mappings',side_effect=wrong):
    with self.assertRaises(AssertionError):a.apply(o,idx,q,pos,32,native,dict(num_reqs=32,phase='prefill'))
   self.assertIsNotNone(a.first_mismatch)
   with self.assertRaises(AssertionError):a.start('replace')
 def test_boundary_restore(self):
  class B:
   def compute_slot_mappings(self,*a,**k):return self.marker
  class R:
   pcp_manager=None
   def prepare_attn(self,b):return self.block_tables.compute_slot_mappings(b.idx_mapping,b.query_start_loc,b.positions,b.num_tokens_after_padding)
  br=R.prepare_attn;bs=B.compute_slot_mappings
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');o=B();o.marker=object();r=R();r.block_tables=o;b=types.SimpleNamespace(idx_mapping=object(),query_start_loc=object(),positions=object(),num_tokens_after_padding=32)
   a.apply=lambda owner,idx,q,p,padded,native,batch:native()
   with patch.object(adapter,'phase_metadata',return_value={}):
    a.install_classes(R,B);self.assertIs(r.prepare_attn(b),o.marker);self.assertIsNone(a.pending);a.close()
   self.assertIs(R.prepare_attn,br);self.assertIs(B.compute_slot_mappings,bs)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
