import unittest,ast
from pathlib import Path
from .run import result_gate,span,walk_items
class Gates(unittest.TestCase):
 def test_no_partial_success(self):
  for n in (0,1,2,3):self.assertFalse(result_gate(dict(complete=True,gpu_executed=True,records=[dict(passed=True)]*n))['raw_pass'])
 def test_all_required(self):
  d=dict(complete=True,gpu_executed=True,records=[dict(passed=True)]*4);self.assertTrue(result_gate(d)['raw_pass']);d['records'][2]=dict(passed=False);self.assertFalse(result_gate(d)['raw_pass'])
 def test_hold(self):self.assertTrue(result_gate(dict(raw_difference_hold=True))['raw_difference_hold'])
 def test_minspan(self):self.assertEqual(span(dict(shape=[311,8192,3],stride=[1081344,1,8192],storage_offset=0))*2,670482432)
 def test_recursive_items(self):self.assertEqual(list(walk_items({'a':[{'path':'p','sha256':'h'}]})),[{'path':'p','sha256':'h'}])
 def test_syntax(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
