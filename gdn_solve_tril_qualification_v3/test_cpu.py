import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_source(self):
  p=Path(__file__).parent.parent
  for d in ('tileir_gdn_solve_tril_v3','gdn_solve_tril_qualification_v3'):
   for f in (p/d).glob('*.py'):ast.parse(f.read_text())
 def test_partial(self):self.assertFalse(result_gate({'complete':False,'records':[]})['raw_pass'])
 def test_hold(self):self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
 def test_false_rows(self):
  r={'complete':True,'records':[dict(arm=a,repeat=i,passed=False) for a in ('native','tile') for i in range(2)],'initial_flags':[True,True],'final_flags':[True,True]};self.assertFalse(result_gate(r)['raw_pass'])
 def test_shape_algebra(self):
  self.assertEqual([len(range(2,min(16,53-n*16))) for n in range(4)],[14,14,14,3])
 def test_candidate_boundaries(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_solve_tril_v3/kernels.py').read_text();self.assertNotIn('tfloat32',s);self.assertEqual(s.count('astype(ct.bfloat16'),1);self.assertEqual(s.count('for row in range'),4)

 def test_cat_arity(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_solve_tril_v3/kernels.py').read_text();calls=[]
  for n in ast.walk(ast.parse(s)):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='ct' and n.func.attr=='cat':
    self.assertIsInstance(n.args[0],(ast.Tuple,ast.List));self.assertLessEqual(len(n.args[0].elts),2);calls.append(n)
  self.assertEqual(len(calls),15)
 def test_cat_layout(self):
  def cat(rows):return [sum(parts,[]) for parts in zip(*rows)]
  blocks={n:[[n]*16 for _ in range(16)] for n in ('d1','d2','d3','d4','x21','x31','x32','x41','x42','x43','zero')}
  names=[['d1','zero','zero','zero'],['x21','d2','zero','zero'],['x31','x32','d3','zero'],['x41','x42','x43','d4']]
  for ns in names:
   vals=[blocks[n] for n in ns];self.assertEqual(cat(vals),cat([cat(vals[:2]),cat(vals[2:])]))

 def test_dot_fusion_expression(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_solve_tril_v3/kernels.py').read_text();t=ast.parse(s)
  self.assertNotIn('ct.mma(',s)
  steps=[n for n in ast.walk(t) if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.Add) and isinstance(n.value.left,ast.BinOp) and isinstance(n.value.left.op,ast.Mult)]
  self.assertEqual(len(steps),256)
  for n in steps:self.assertEqual(ast.unparse(n.targets[0]),ast.unparse(n.value.right))
  init=[n for n in ast.walk(t) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Name) and n.value.id=='zero' and ast.unparse(n.targets[0]).startswith('dot_')]
  self.assertEqual(len(init),16)
