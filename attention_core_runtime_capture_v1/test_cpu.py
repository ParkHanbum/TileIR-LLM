import unittest,ast
from pathlib import Path
from observer import phase_from_lists,CAP
from result_gate import audit
class Tests(unittest.TestCase):
 def test_phase(self):self.assertEqual(phase_from_lists(list(range(0,1697,53)),[53]*32),'prefill');self.assertEqual(phase_from_lists(list(range(33)),[54]*32),'decode')
 def test_bad_phase(self):
  with self.assertRaises(AssertionError):phase_from_lists(list(range(33)),[53]*32)
 def test_gate(self):self.assertFalse(audit({})['passed'])
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertEqual(CAP,256*1024*1024)
