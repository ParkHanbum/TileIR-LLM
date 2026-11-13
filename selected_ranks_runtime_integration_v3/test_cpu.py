import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
from . import adapter as a
class Tests(unittest.TestCase):
 def test_tie_hold(self):
  x=[[1]*32 for _ in range(5)];self.assertTrue(a.unique_counts(x));x[2][7]=2;self.assertFalse(a.unique_counts(x));self.assertFalse(a.unique_counts([[1]*32]))
 def test_census(self):
  self.assertTrue(a.census([dict(phase='prefill')]+[dict(phase='decode')]*15));self.assertFalse(a.census([dict(phase='decode')]*16))
 def exercise(self,fail=False):
  ret=object();calls=[]
  def helper(logits):calls.append(logits);return ret
  module=types.SimpleNamespace(compute_topk_scores=helper,gumbel_sample=object());untouched=module.gumbel_sample
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,logits,batch):
    out=module.compute_topk_scores(logits)
    if fail:raise ValueError('native fail')
    return out
  original=S.__call__
  with tempfile.TemporaryDirectory() as d:
   ob=a.Adapter(Path(d)/'out')
   def apply(values,native,batch):
    out=native();ob.rows.append(dict(phase=batch['phase'],passed=True));ob.native_gate_calls+=1;ob.tileir_gate_calls+=1;return out
   ob.apply=apply
   with patch.object(a,'phase_metadata',lambda b:dict(phase=b,num_reqs=32)):
    ob.install_classes(S,module)
    try:
     if fail:
      with self.assertRaises(ValueError):S()(object(),'prefill')
      self.assertIsNone(ob.pending)
     else:
      for i in range(16):self.assertIs(S()(i,'prefill' if i==0 else 'decode'),ret)
      self.assertEqual(calls,list(range(16)));self.assertTrue(ob.finish()['shadow_pass']);ob.start('replace')
    finally:ob.close()
   self.assertIs(S.__call__,original);self.assertIs(module.compute_topk_scores,helper);self.assertIs(module.gumbel_sample,untouched)
 def test_native16_return_identity_restore(self):self.exercise()
 def test_exception_restore(self):self.exercise(True)
 def test_replace_before_shadow_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   ob=a.Adapter(Path(d)/'out')
   with self.assertRaises(AssertionError):ob.start('replace')
if __name__=='__main__':unittest.main()
