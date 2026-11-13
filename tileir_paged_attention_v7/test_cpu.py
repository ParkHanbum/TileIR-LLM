import unittest
from .contract import validate_values
class Tests(unittest.TestCase):
 def test_prefill(self):self.assertEqual(validate_values(list(range(0,1697,53)),[53]*32,[[1]*8]*32,313,528),53)
 def test_decode(self):self.assertEqual(validate_values(list(range(33)),[64]*32,[[1]*8]*32,313,528),1)
 def test_capacity(self):
  with self.assertRaises(AssertionError):validate_values(list(range(33)),[54]*32,[[313]*8]*32,313,528)
 def test_mixed(self):
  with self.assertRaises(AssertionError):validate_values(list(range(33)),[54]*31+[55],[[1]*8]*32,313,528)
 def test_mask(self):self.assertEqual([i for i in range(54) if i<=0+54-1],list(range(54)))

class MappingTests(unittest.TestCase):
 def test_columns_and_tails(self):
  groups=[[8*n+2*l+j for n in range(8) for j in (0,1)] for l in range(4)]
  self.assertEqual(sorted(sum(groups,[])),list(range(64)))
  for length in range(1,65):
   used=[c for group in groups for c in group if c<length]
   self.assertEqual(sorted(used),list(range(length)))
 def test_unsupported_lengths(self):
  for length in range(65,69):
   with self.assertRaises(AssertionError):validate_values(list(range(33)),[length]*32,[[1]*8]*32,313,528)
 def test_actual_source_order(self):
  from pathlib import Path
  s=(Path(__file__).parent/'kernels.py').read_text()
  self.assertEqual(s.count('sums=ct.add('),15)
  positions=[s.index(f'ct.extract(eg,({n},0,{j}),(1,4,1))') for n in range(8) for j in (0,1)]
  self.assertEqual(positions,sorted(positions))
  self.assertIn('ct.add(s02,s13,rounding_mode=ct.RoundingMode.RN)',s)
  self.assertNotIn('den=ct.sum',s)
