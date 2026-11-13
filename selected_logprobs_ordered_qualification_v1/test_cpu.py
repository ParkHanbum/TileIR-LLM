import unittest,ast
from pathlib import Path
from run import result_gate
class Gates(unittest.TestCase):
 def base(self):return dict(complete=True,records=[dict(case=c,arm=a,repeat=n,binding_pass=True,vs_native={'raw_equal':True}) for c in range(3) for a in ('native','tile') for n in (0,1)])
 def test_complete(self):self.assertTrue(result_gate(self.base())['raw_pass'])
 def test_duplicate(self):
  r=self.base();r['records'][4]=r['records'][0];self.assertFalse(result_gate(r)['complete'])
 def test_incomplete(self):
  r=self.base();r['complete']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_failure(self):
  r=self.base();r['records'][2]['binding_pass']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_api_unchanged(self):
  p=Path(__file__).resolve().parent.parent;self.assertEqual((p/'tileir_vllm_selected_logprobs_v1/api.py').read_bytes(),(p/'tileir_vllm_selected_logprobs_v2/api.py').read_bytes())
 def test_exp_log_unchanged(self):
  p=Path(__file__).resolve().parent.parent
  for name in ['ct.exp','logsum=ct.log','value=ct.sub','maximum=ct.max']:
   a=[x.strip() for x in (p/'tileir_vllm_selected_logprobs_v1/kernels.py').read_text().splitlines() if name in x];b=[x.strip() for x in (p/'tileir_vllm_selected_logprobs_v2/kernels.py').read_text().splitlines() if name in x];self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
