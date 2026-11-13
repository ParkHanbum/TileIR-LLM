import unittest,ast,json
from pathlib import Path
from tileir_vllm_runner_constructors_v1.api import validate_num_reqs
from .run import result_gate
class Tests(unittest.TestCase):
 def test_supported(self):validate_num_reqs(32)
 def test_reject(self):
  for x in (31,33,True,32.0):
   with self.assertRaises(AssertionError):validate_num_reqs(x)
 def test_indices(self):
  self.assertEqual([i for i in range(64)if i<33],list(range(33)));self.assertEqual([0 for i in range(64)if i<32],[0]*32)
 def test_gate(self):
  r=dict(complete=True,records=[dict(case=c,arm=a,repeat=n,passed=True)for c in ('cu_num_logits','expanded_local_pos')for a in ('native','tile')for n in (0,1)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][3]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_actual_observations(self):
  r=Path(__file__).parent;rows=json.loads((r/'OBSERVED-CPU-OPERATORS.json').read_text());self.assertEqual(sum(x['name']=='aten::zeros'for x in rows),16)
  ranges=[x for x in rows if x['name']=='aten::arange'and x['args']['Concrete Inputs'][:3]==['0','33','1']];self.assertEqual(len(ranges),16)
 def test_ast(self):
  p=Path(__file__).parent
  for f in [*p.glob('*.py'),*(p.parent/'tileir_vllm_runner_constructors_v1').glob('*.py')]:ast.parse(f.read_text())
