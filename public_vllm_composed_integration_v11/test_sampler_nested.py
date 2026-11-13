import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
from sample_counts_runtime_integration_v1 import adapter as counts
from greedy_sampler_runtime_integration_v1 import adapter as greedy
class T(unittest.TestCase):
 def exercise(self,fail=False):
  events=[];sampled=object();number=object();pair=(number,object());output=object()
  def gumbel(logits):events.append('native-greedy');return sampled
  def get_counts(*a):events.append('native-counts');return pair
  mod=types.SimpleNamespace(gumbel_sample=gumbel,get_num_sampled_and_rejected=get_counts)
  batch=types.SimpleNamespace(seq_lens=object(),cu_num_logits=object(),idx_mapping=object())
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,logits,b):
    assert mod.gumbel_sample(logits) is sampled
    if fail:raise ValueError('bodyfailure')
    assert mod.get_num_sampled_and_rejected(number,b.seq_lens,b.cu_num_logits,b.idx_mapping,object()) is pair
    return output
  original=S.__call__
  with tempfile.TemporaryDirectory() as d:
   a=counts.Adapter(Path(d)/'counts');g=greedy.Adapter(Path(d)/'greedy')
   def ca(owner,args,native,metadata):events.append('counts-wrapper');return native()
   def ga(values,native,metadata):events.append('greedy-wrapper');return native()
   a.apply=ca;g.apply=ga
   with patch.object(counts,'phase_metadata',lambda b:{}),patch.object(greedy,'phase_metadata',lambda b:{}):
    a.install_classes(S,mod);counts_wrapper=S.__call__;g.install_classes(S,mod)
    try:
     if fail:
      with self.assertRaises(ValueError):S()(object(),batch)
     else:self.assertIs(S()(object(),batch),output)
     self.assertIsNone(a.pending);self.assertIsNone(g.pending)
    finally:
     g.close();self.assertIs(S.__call__,counts_wrapper);a.close()
   self.assertIs(S.__call__,original);self.assertIs(mod.gumbel_sample,gumbel);self.assertIs(mod.get_num_sampled_and_rejected,get_counts)
   self.assertEqual(events,['greedy-wrapper','native-greedy']+([] if fail else ['counts-wrapper','native-counts']))
 def test_actual_nested_wrappers(self):self.exercise()
 def test_actual_nested_exception_undo(self):self.exercise(True)
