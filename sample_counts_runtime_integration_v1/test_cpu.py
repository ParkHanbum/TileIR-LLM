import tempfile,unittest,types
from pathlib import Path
from unittest.mock import patch
import torch
import adapter
from tileir_vllm_sample_counts_v1.api import reference_cpu
class UvaBuffer:pass
class Tests(unittest.TestCase):
 def fixture(self):
  prefill=torch.full((32,),53,dtype=torch.int32);buf=UvaBuffer();buf.cpu=prefill;buf._uva=prefill
  owner=types.SimpleNamespace(gpu=prefill,cpu=prefill,pool=types.SimpleNamespace(_curr=0,_uva_bufs=[buf]));sampler=types.SimpleNamespace(req_states=types.SimpleNamespace(prefill_len=owner))
  args=(torch.ones(32,dtype=torch.int32),torch.full((32,),53,dtype=torch.int32),torch.arange(33,dtype=torch.int32),torch.arange(32),prefill);return sampler,args
 def test_full16_identity_mutation_counts(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');s,args=self.fixture();calls=[];last=[]
   def native():calls.append(1);ret=reference_cpu(*args);last[:]=[ret];return ret
   with patch.object(torch.Tensor,'is_pinned',return_value=True),patch('torch.cuda.synchronize'),patch('tileir_vllm_sample_counts_v1.api.get_num_sampled_and_rejected',side_effect=reference_cpu):
    a.start('shadow')
    for i in range(16):self.assertIs(a.apply(s,args,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode')),last[0])
    self.assertTrue(a.finish()['shadow_pass']);a.start('replace')
    for i in range(16):a.apply(s,args,native,dict(num_reqs=32,phase='prefill' if i==0 else 'decode'))
    self.assertEqual(a.finish()['native_gate_calls'],0);self.assertEqual(len(calls),16)
 def test_hold(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');s,args=self.fixture()
   def wrong(*args):return args[0],args[0]
   with patch.object(torch.Tensor,'is_pinned',return_value=True),patch('torch.cuda.synchronize'),patch('tileir_vllm_sample_counts_v1.api.get_num_sampled_and_rejected',side_effect=wrong):
    with self.assertRaises(AssertionError):a.apply(s,args,lambda:reference_cpu(*args),dict(num_reqs=32,phase='prefill'))
   self.assertIsNotNone(a.first_mismatch)
   with self.assertRaises(AssertionError):a.start('replace')
 def test_boundary_restore(self):
  mod=types.SimpleNamespace(get_num_sampled_and_rejected=lambda *args:object());orig=mod.get_num_sampled_and_rejected
  class S:
   def __call__(self,l,b):return mod.get_num_sampled_and_rejected(None,b.seq_lens,b.cu_num_logits,b.idx_mapping,None)
  call=S.__call__;b=types.SimpleNamespace(seq_lens=object(),cu_num_logits=object(),idx_mapping=object())
  with tempfile.TemporaryDirectory() as tmp:
   a=adapter.Adapter(Path(tmp)/'a');a.apply=lambda owner,args,native,meta:native()
   with patch.object(adapter,'phase_metadata',return_value={}):a.install_classes(S,mod);S()(None,b);self.assertIsNone(a.pending);a.close()
   self.assertIs(S.__call__,call);self.assertIs(mod.get_num_sampled_and_rejected,orig)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
