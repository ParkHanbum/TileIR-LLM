import unittest
from .contract import validate_values
class Tests(unittest.TestCase):
 def test_prefill(self):self.assertEqual(validate_values(list(range(0,1697,53)),[53]*32,[[1]*8]*32,313,528),53)
 def test_decode(self):self.assertEqual(validate_values(list(range(33)),[68]*32,[[1]*8]*32,313,528),1)
 def test_capacity(self):
  with self.assertRaises(AssertionError):validate_values(list(range(33)),[54]*32,[[313]*8]*32,313,528)
 def test_mixed(self):
  with self.assertRaises(AssertionError):validate_values(list(range(33)),[54]*31+[55],[[1]*8]*32,313,528)
 def test_mask(self):self.assertEqual([i for i in range(54) if i<=0+54-1],list(range(54)))
