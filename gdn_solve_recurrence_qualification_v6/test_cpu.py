import ast,unittest
from pathlib import Path
from .run import result_gate
P=Path(__file__).parent.parent
class Tests(unittest.TestCase):
 def test_ast(self):
  for d in ('tileir_gdn_solve_tril_v6','gdn_solve_recurrence_qualification_v6'):
   for p in (P/d).glob('*.py'):ast.parse(p.read_text())
 def test_only_recurrence_changed(self):
  a=(P/'tileir_gdn_solve_tril_v5/kernels.py').read_text();b=(P/'tileir_gdn_solve_tril_v6/kernels.py').read_text();self.assertEqual(a[a.index('    a21='):],b[b.index('    a21='):]);self.assertEqual(b.count('        total=ct.add(sum02,sum13'),4);self.assertNotIn('ct.sum(',b)
 def test_gate(self):self.assertFalse(result_gate({})['raw_pass'])
