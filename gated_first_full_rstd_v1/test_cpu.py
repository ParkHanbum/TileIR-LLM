import ast,unittest
from pathlib import Path
class Test(unittest.TestCase):
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_binding_layout(self):
  s=(Path(__file__).parent/'run.py').read_text()
  for v in ('8192','12288','helper_vs_saved_native','clean_vs_saved_tile','instrumented_vs_clean','extra_compiledkernel_replay_launches=0'):self.assertIn(v,s)
