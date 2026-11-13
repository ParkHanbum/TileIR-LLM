import unittest,ast
from pathlib import Path
from .geometry import *
class Tests(unittest.TestCase):
 def test_partition_full_coverage(self):
  p=partition(256,[(8,24),(16,32),(64,96),(8,24)])
  self.assertEqual(p,[('outside',0,8),('target',8,32),('outside',32,64),('target',64,96),('outside',96,256)])
 def test_bounded_above_old_cap(self):
  n=553648128;p=partition(n+8,[(4,n+4)])
  self.assertEqual(sum(b-a for k,a,b in p if k=='target'),n)
  self.assertTrue(all(b-a<=CHUNK for k,a,b in p));self.assertEqual(p[0][1],0);self.assertEqual(p[-1][2],n+8)
 def test_invalid_bounds_alignment(self):
  for spans in [[(-4,8)],[(0,260)],[(1,4)],[(8,4)]]:
   with self.assertRaises(ContractError):partition(256,spans)
 def test_ast_only(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
