import os,ast,unittest,types
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import torch
from norm_runtime_qualification_v1.compare import metrics
from norm_runtime_qualification_v1.run import force_provider,result_gate
class Tests(unittest.TestCase):
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
 def test_equal(self):
  a=torch.tensor([1.,-1.,0.],dtype=torch.bfloat16);m=metrics(a,a);self.assertTrue(m['raw_equal'] and m['strict_pass']);self.assertEqual(m['max_bf16_ulp_finite'],0)
 def test_adjacent(self):
  a=torch.tensor([0x3f81],dtype=torch.int16).view(torch.bfloat16);b=torch.tensor([1.],dtype=torch.bfloat16);m=metrics(a,b);self.assertEqual(m['max_bf16_ulp_finite'],1);self.assertFalse(m['raw_equal']);self.assertFalse(m['strict_pass']);self.assertTrue(m['source_reference_pass'])
 def test_nonfinite(self):
  a=torch.tensor([float('nan'),float('inf')],dtype=torch.bfloat16);m=metrics(a,a);self.assertEqual(m['candidate_nonfinite'],2);self.assertFalse(m['strict_pass'])
 def test_dispatch_restore_success_and_failure(self):
  op=types.SimpleNamespace(dispatch=lambda: 'original');prior=op.dispatch;impl=object()
  with force_provider(op,impl) as calls:self.assertIs(op.dispatch(),impl)
  self.assertEqual(len(calls),1);self.assertIs(op.dispatch,prior)
  with self.assertRaises(RuntimeError):
   with force_provider(op,impl):raise RuntimeError('expected')
  self.assertIs(op.dispatch,prior)
 def test_result_hold(self):
  rows=[dict(kind=k,phase=p,contract_pass=True,inputs_preserved={'candidate':True,'provider':True,'native_ir':True},candidate_vs_capture=[{'raw_equal':True}],provider_vs_capture=[{'raw_equal':True}]) for k in ('no_residual','with_residual') for p in ('prefill','decode')]
  r=dict(complete=True,gpu_executed=True,records=rows);self.assertTrue(result_gate(r)['raw_pass'])
  rows[0]['candidate_vs_capture'][0]['raw_equal']=False;self.assertTrue(result_gate(r)['raw_difference_hold']);self.assertFalse(result_gate(r)['raw_pass'])
  rows[0]['contract_pass']=False;self.assertFalse(result_gate(r)['raw_difference_hold'])
if __name__=='__main__':unittest.main()
