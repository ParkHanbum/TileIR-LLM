import ast,sys,unittest
from pathlib import Path
from api import reference_checked as f
class Tests(unittest.TestCase):
 def test_permutation_tail_padding(self):
  self.assertEqual(f([1,0],[1,3,4],[8,2],[99]*6,[77]*4),([99,2,3,8,99,99],[4,9,0,0]))
 def test_duplicate_zero_len(self):
  self.assertEqual(f([0,0],[0,0,2],[5],[99]*3,[77]*3),([5,6,99],[5,7,0]))
 def test_empty(self):self.assertEqual(f([], [0], [], [7], [9,9]),([7],[0,0]))
 def test_invalid(self):
  for args in [([-1],[0,1],[1],[0],[0]),([0],[1,0],[1],[0],[0]),([0],[0,2],[1],[0],[0]),([0],[0,1],[2147483647],[0],[0])]:
   with self.assertRaises(ValueError):f(*args)
 def test_cpu_ast(self):
  self.assertNotIn('torch',sys.modules)
  for x in Path(__file__).parent.glob('*.py'):ast.parse(x.read_text())
if __name__=='__main__':unittest.main()
