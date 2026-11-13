import ast,unittest
from pathlib import Path
from .run import result_gate
class T(unittest.TestCase):
 def test_pending(self):self.assertFalse(result_gate({})['raw_pass'])
 def test_gate(self):
  self.assertTrue(result_gate(dict(complete=True,records=[dict(passed=True)]*8))['raw_pass'])
  self.assertFalse(result_gate(dict(complete=True,records=[dict(passed=False)]*8))['raw_pass'])
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
