import unittest,ast,sys
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_complete(self):self.assertTrue(result_gate({'complete':True,'records':[{'passed':True}]*8})['raw_pass'])
 def test_partial_hold(self):self.assertFalse(result_gate({'complete':True,'records':[{'passed':True}]*7})['raw_pass'])
 def test_failure(self):self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
 def test_ast_no_gpu(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
