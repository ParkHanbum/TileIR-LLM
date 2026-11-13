import ast,unittest
from pathlib import Path
class Test(unittest.TestCase):
 def test_sources(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_bindings(self):
  s=(Path(__file__).parent/'run.py').read_text()
  for t in ['native_helper_vs_saved','instrumented_vs_original','assert mean is None','norm_before_gate=True,is_rms_norm=True']:self.assertIn(t,s)
if __name__=='__main__':unittest.main()
