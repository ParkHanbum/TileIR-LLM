import unittest,math
from .api import reference_rows,validate_descriptor
class T(unittest.TestCase):
 def test_ties(self):self.assertEqual(reference_rows([[2,2,1,3,3,0]])[0],[3,4,0,1,2])
 def test_nonfinite(self):
  for v in [float('nan'),float('inf'),float('-inf')]:
   with self.assertRaises(AssertionError):reference_rows([[1,2,3,4,v]])
 def test_zeros(self):self.assertEqual(reference_rows([[0.,-0.,0.,-0.,0.,-1.]])[0],[0,1,2,3,4])
 def test_union(self):
  import random
  r=random.Random(371)
  for n in [5,31,1025,2107]:
   x=[r.randrange(-10,11) for _ in range(n)];c=[]
   for b in range(0,n,19):c+=sorted(range(b,min(b+19,n)),key=lambda i:(-x[i],i))[:5]
   self.assertEqual(sorted(c,key=lambda i:(-x[i],i))[:5],reference_rows([x])[0])
