import unittest,tempfile,types,json
from pathlib import Path
import numpy as np
import torch
import observer as o
class Tests(unittest.TestCase):
 def batch(self,flag):
  return types.SimpleNamespace(num_reqs=2,num_reqs_after_padding=3,num_tokens=2,num_tokens_after_padding=3,req_ids=['a','b'],is_prefilling_np=np.array([flag,flag]),has_prefill=flag,num_scheduled_tokens=np.array([1,1]),num_computed_tokens_np=np.array([0,0] if flag else [53,53]),prefill_len_np=np.array([53,53]),num_computed_prefill_tokens_np=np.array([0,0] if flag else [53,53]),query_start_loc_np=np.array([0,1,2]),idx_mapping_np=np.array([1,0]),idx_mapping=torch.tensor([1,0]))
 def classes(self):
  class Blocks:
   def __init__(self):self.input_block_tables=[torch.full((4,5),-7,dtype=torch.int32)];self.calls=0;self.last=None
   def gather_block_tables(self,idx,num_reqs_padded,out=None,out_ptrs=None):
    self.calls+=1;self.input_block_tables[0][:2,:2]=3;self.input_block_tables[0][2:num_reqs_padded]=0
    self.last=tuple(t[:num_reqs_padded] for t in self.input_block_tables);return self.last
  class Runner:
   pcp_manager=None
   def __init__(self):self.block_tables=Blocks();self.last=None
   def prepare_attn(self,b):self.last=(self.block_tables.gather_block_tables(b.idx_mapping,b.num_reqs_after_padding),object());return self.last
  return Runner,Blocks
 def test_exact_once_return_mutation_restore(self):
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'bank');R,B=self.classes();oldR=R.prepare_attn;oldB=B.gather_block_tables
   obs.snapshot=lambda *a:({'indices':{'storage_byte_sha256':'stable'}},{'device_view_shares_active_storage':True})
   obs.tensor=lambda *a:{}
   obs.install_classes(R,B);r=R()
   for phase in (True,False,False):
    ret=r.prepare_attn(self.batch(phase));self.assertIs(ret,r.last);self.assertIs(ret[0],r.block_tables.last)
   self.assertEqual(r.block_tables.calls,3);self.assertEqual(set(obs.samples),{'prefill','decode'});self.assertEqual(obs.seen,{'prefill':1,'decode':2})
   self.assertEqual(r.block_tables.input_block_tables[0][3].tolist(),[-7]*5)
   obs.close();self.assertIs(R.prepare_attn,oldR);self.assertIs(B.gather_block_tables,oldB)
 def test_phase_not_token_shape(self):
  b=self.batch(True);self.assertEqual(o.phase_metadata(b)['phase'],'prefill')
  b=self.batch(False);b.num_tokens=1696;self.assertEqual(o.phase_metadata(b)['phase'],'decode')
  b.is_prefilling_np=np.array([True,False]);b.has_prefill=True;self.assertEqual(o.phase_metadata(b)['phase'],'mixed')
 def test_exception_pending_cleanup(self):
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'b');R,B=self.classes();oldR=R.prepare_attn;oldB=B.gather_block_tables
   obs.snapshot=lambda *a:(_ for _ in ()).throw(ValueError('snapshot failure'))
   obs.install_classes(R,B);r=R()
   with self.assertRaises(ValueError):r.prepare_attn(self.batch(True))
   self.assertIsNone(obs.pending);self.assertEqual(r.block_tables.calls,0);obs.close();self.assertIs(R.prepare_attn,oldR);self.assertIs(B.gather_block_tables,oldB)
 def test_full_storage_offset(self):
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'b');base=torch.arange(30,dtype=torch.int32);v=base[5:15].reshape(2,5)
   row=obs.tensor(v,'test');raw=torch.load(row['storage_path'],weights_only=True)
   self.assertEqual(raw.numel(),120);self.assertEqual(row['metadata']['storage_offset'],5)
   self.assertTrue(torch.equal(raw,base.view(torch.uint8)));self.assertTrue(torch.equal(torch.load(row['path'],weights_only=True),v))
 def test_conflicting_binding_not_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'b');R,B=self.classes();obs.install_classes(R,B);foreign=lambda *a:None;R.prepare_attn=foreign
   with self.assertRaises(RuntimeError):obs.close()
   self.assertIs(R.prepare_attn,foreign)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
