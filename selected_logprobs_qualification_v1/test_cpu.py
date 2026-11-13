import unittest,json
from pathlib import Path
from .run import check_manifest,result_gate
class Tests(unittest.TestCase):
 def test_actual(self):
  m=json.loads((Path(__file__).parent.parent/'greedy_sampler_runtime_capture_v1/greedySamplerCapture01/bank/MANIFEST.json').read_text());self.assertEqual(len(check_manifest(m)),2)
 def test_order(self):
  rows=[dict(call_index=c,arm=a,repeat=n,passed=True) for c in (0,15) for a in ('native','tile') for n in (0,1)]
  self.assertTrue(result_gate(dict(complete=True,records=rows))['raw_pass']);rows[-1]=rows[0];self.assertFalse(result_gate(dict(complete=True,records=rows))['raw_pass'])
 def test_missing(self):self.assertFalse(result_gate({})['raw_pass'])
