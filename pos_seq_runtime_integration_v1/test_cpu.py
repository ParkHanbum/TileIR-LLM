import ast,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pos_seq_runtime_integration_v1.adapter import Adapter
from pos_seq_runtime_integration_v1.precision import PrecisionPolicy
from types import SimpleNamespace as S
class Tests(unittest.TestCase):
 def test_count_guard(self):
  a=Adapter.__new__(Adapter);a.pending=None;a.shadow_pass=False
  with self.assertRaises(AssertionError):a.start('replace')
  a.rows=[dict(phase=p,passed=True) for p in ['prefill']+['decode']*15];a.mode='shadow';a.native_calls=a.tileir_calls=16;a.current_call=None;a.save=lambda:None
  self.assertTrue(a.finish()['shadow_pass']);a.rows.pop()
  with self.assertRaises(AssertionError):a.finish()
 def test_precision_unchanged(self):
  m=S(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=False);p=PrecisionPolicy(S(backends=S(cuda=S(matmul=m))));self.assertEqual(p.select('F'),('native_unchanged',True));p.restore();self.assertEqual(vars(m),dict(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=False))
 def test_ast(self):
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
