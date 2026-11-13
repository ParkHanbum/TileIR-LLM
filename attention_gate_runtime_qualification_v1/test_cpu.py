import ast,sys,unittest
from pathlib import Path
import run
class Tests(unittest.TestCase):
 def test_layout(self):
  self.assertEqual(run.storage_elements(dict(shape=[32,4096],stride=[12288,1],storage_offset=8192)),8192+31*12288+4096)
  with self.assertRaises(AssertionError):run.storage_elements(dict(shape=[32,4096],stride=[2048,1],storage_offset=0))
 def test_gate(self):
  c=dict(raw_equal=True,candidate_nonfinite=0,reference_nonfinite=0)
  r=dict(complete=True,gpu_executed=True,records=[dict(phase=p,contract_pass=True,inputs_preserved={'provider0':True},comparisons={'a':dict(c)}) for p in ('prefill','decode')])
  self.assertTrue(run.result_gate(r)['raw_pass']);r['records'][0]['comparisons']['a']['raw_equal']=False
  self.assertTrue(run.result_gate(r)['raw_difference_hold']);r['records'][0]['contract_pass']=False
  self.assertFalse(run.result_gate(r)['raw_difference_hold'])
 def test_cpu_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
