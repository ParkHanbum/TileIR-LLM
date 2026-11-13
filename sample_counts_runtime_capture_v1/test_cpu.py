import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
import observer as o
class Tests(unittest.TestCase):
 def test_bound_helper_identity_mutation_restore(self):
  module=types.SimpleNamespace();calls=[]
  def original(sampled,*a):calls.append(1);sampled.zero_();return sampled,torch.zeros_like(sampled)
  module.get_num_sampled_and_rejected=original
  class S:
   def __call__(self,logits,b):self.last=module.get_num_sampled_and_rejected(torch.ones(2,dtype=torch.int32),b.seq_lens,b.cu_num_logits,b.idx_mapping,torch.ones(2,dtype=torch.int32));return self.last
  old=S.__call__;b=types.SimpleNamespace(seq_lens=object(),cu_num_logits=object(),idx_mapping=object())
  with tempfile.TemporaryDirectory() as tmp:
   obs=o.Observer(Path(tmp)/'b');obs.snapshot=lambda *a:({'sampled':{'storage_byte_sha256':'x'},'prefill':{'storage_byte_sha256':'same'}},{})
   obs.tensor=lambda *a:{};obs.install_classes(S,module);s=S()
   for phase in ('prefill','decode','decode'):
    with patch.object(o,'phase_metadata',return_value={'phase':phase}):self.assertIs(s(None,b),s.last)
   self.assertEqual(len(calls),3);self.assertEqual(set(obs.samples),{'prefill','decode'});obs.close();self.assertIs(S.__call__,old);self.assertIs(module.get_num_sampled_and_rejected,original)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
