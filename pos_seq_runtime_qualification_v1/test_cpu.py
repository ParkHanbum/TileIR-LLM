import ast,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pos_seq_runtime_qualification_v1.run import result_gate
class Tests(unittest.TestCase):
 def test_gate(self):
  r=dict(complete=True,records=[dict(passed=True)]*8);self.assertTrue(result_gate(r)['raw_pass']);r['records'].pop();self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast(self):
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
