import unittest,json,ast
from pathlib import Path
from .run import check_manifest,result_gate
class T(unittest.TestCase):
 def test_manifest(self):
  p=Path(__file__).parent.parent/'greedy_sampler_runtime_capture_v1/greedySamplerCapture01/bank/MANIFEST.json';self.assertEqual(len(check_manifest(json.loads(p.read_text()))),2)
 def test_order(self):
  r=dict(complete=True,records=[dict(call_index=c,arm=a,repeat=i,passed=True)for c in [0,15]for a in ['native','tile']for i in [0,1]])
  self.assertTrue(result_gate(r)['raw_pass']);r['records'][0]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast(self):
  for d in ['tileir_vllm_top5_v1','top5_qualification_v1']:
   for p in (Path(__file__).parent.parent/d).glob('*.py'):ast.parse(p.read_text())
