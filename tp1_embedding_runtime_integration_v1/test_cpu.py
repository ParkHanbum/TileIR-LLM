import unittest,tempfile
from types import SimpleNamespace
from .adapter import Adapter,census
from .result_gate import audit
from .gates import qualification
from .precision import PrecisionPolicy
class Tests(unittest.TestCase):
 def test_census(self):self.assertTrue(census([{'phase':'prefill'}]+[{'phase':'decode'}]*15));self.assertFalse(census([{'phase':'decode'}]*16))
 def test_replace_before_shadow(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(d+'/a')
   with self.assertRaises(AssertionError):a.start('replace')
 def test_close(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(d+'/a');q=SimpleNamespace();w=lambda:None;q.embedding=w;a.binding=(q,False,None,w);a.close();self.assertFalse(hasattr(q,'embedding'))
 def test_incomplete(self):self.assertFalse(audit({})['passed'])
 def test_proof(self):self.assertIn('result_sha256',qualification(None))
 def test_flags(self):
  t=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True))))
  p=PrecisionPolicy(t);p.select('native');p.restore();t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  with self.assertRaises(AssertionError):p.restore()

 def fixture(self):
  rows=[dict(phase='prefill',passed=True)]+[dict(phase='decode',passed=True)]*15
  def arm(n):return dict(selected_dispatch=dict(shadow_pass=True,supported_calls=16,tileir_gate_calls=16,native_gate_calls=n,supported_vendor_fallback_calls=0,rows=rows),cross_runtime_tokens_equal=True,output_token_ids=[[1]*16]*32,allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True)
  return dict(complete=True,engine_shutdown=True,public_logprob_gate=dict(passed=True,values=2560,exact_numeric=True,max_abs=0),adapter_restore=[dict(restored=True)],precision_policy=dict(restored=True,restored_split_k=True,policy='unchanged native'),shadow_request=arm(16),replacement_request=arm(0))
 def test_positive_gate(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_nonexact_gate(self):
  r=self.fixture();r['public_logprob_gate']['exact_numeric']=False;self.assertFalse(audit(r)['passed'])
