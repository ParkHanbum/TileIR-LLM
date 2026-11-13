import unittest,ast,json
from pathlib import Path
from .run import result_gate
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_bank(self):
  m=json.loads((R/'MANIFEST.json').read_text());s,=m['samples'];self.assertEqual(len(s['pre']),9);self.assertEqual(set(s['outputs']),{'o','final_state'})
 def test_gate_fail_closed(self):
  self.assertFalse(result_gate({})['raw_pass']);self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
 def test_stage_flow(self):
  s=(R/'run.py').read_text();self.assertIn("if arm=='native' and repeat==0:",s);self.assertIn('else:out=execute(inputs,ops,observe)',s);self.assertIn('raise RawHold(label)',s);self.assertIn("compare(full[name],value.cpu(),torch)",s)
