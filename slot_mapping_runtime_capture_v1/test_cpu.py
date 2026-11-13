import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
import observer as o
class Tests(unittest.TestCase):
 def test_exact_once_phase_return_restore(self):
  class B:
   num_kv_cache_groups=2;cp_size=1;cp_rank=0;cp_interleave=1;block_sizes=[4,8];_slot_mapping_enabled=[True,False]
   def __init__(self):self.slot_mappings=torch.zeros((2,9),dtype=torch.int64);self.calls=0;self.last=None
   def compute_slot_mappings(self,i,q,p,num_tokens_padded,out=None):self.calls+=1;self.slot_mappings.fill_(-1);self.last=self.slot_mappings[:,:num_tokens_padded];return self.last
  class R:
   pcp_manager=None
   def prepare_attn(self,b):return self.block_tables.compute_slot_mappings(b.idx_mapping,b.query_start_loc,b.positions,b.num_tokens_after_padding)
  oldR=R.prepare_attn;oldB=B.compute_slot_mappings
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'bank');obs.snapshot=lambda *a:{'indices':{'storage_byte_sha256':'same'},'out':{'storage_byte_sha256':'mutable'}};obs.tensor=lambda *a:{}
   obs.install_classes(R,B);r=R();r.block_tables=B();batch=types.SimpleNamespace(idx_mapping=object(),query_start_loc=object(),positions=object(),num_tokens_after_padding=6)
   for phase in ('prefill','decode','decode'):
    with patch.object(o,'phase_metadata',return_value={'phase':phase}):self.assertIs(r.prepare_attn(batch),r.block_tables.last)
   self.assertEqual(r.block_tables.calls,3);self.assertEqual(set(obs.samples),{'prefill','decode'});obs.close();self.assertIs(R.prepare_attn,oldR);self.assertIs(B.compute_slot_mappings,oldB)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
