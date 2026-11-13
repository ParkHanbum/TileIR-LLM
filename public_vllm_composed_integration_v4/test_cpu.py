import unittest,types,sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v4.precision import PrecisionPolicy
from public_vllm_composed_integration_v4 import adapter
from public_vllm_composed_integration_v3 import adapter as original
class Flags:
 @property
 def allow_bf16_reduced_precision_reduction(self):return True
 @property
 def allow_bf16_reduced_precision_reduction_split_k(self):return True
 def __setattr__(self,k,v):raise AssertionError('Backend flag assignment forbidden')
class Tests(unittest.TestCase):
 def policy(self,f):return PrecisionPolicy(types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=f))))
 def test_readonly(self):
  p=self.policy(Flags());p.select('DEFAULT');p.select('DEFAULT');p.restore();self.assertEqual(p.record['changes'],[]);self.assertTrue(p.record['restored']);self.assertEqual(len(p.record['observations']),2)
 def test_wrong_initial_blocks(self):
  f=types.SimpleNamespace(allow_bf16_reduced_precision_reduction=False,allow_bf16_reduced_precision_reduction_split_k=True)
  with self.assertRaises(AssertionError):self.policy(f)
  self.assertFalse(f.allow_bf16_reduced_precision_reduction)
 def test_no_drift_repair(self):
  f=types.SimpleNamespace(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True);p=self.policy(f);f.allow_bf16_reduced_precision_reduction=False
  with self.assertRaises(AssertionError):p.restore()
  self.assertFalse(f.allow_bf16_reduced_precision_reduction)
 def test_exact_adapter(self):
  for n in ('install','start','finish','close','PARTS'):self.assertIs(getattr(adapter,n),getattr(original,n))
 def test_ast(self):
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
