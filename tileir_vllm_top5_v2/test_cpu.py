import unittest,random,math
from .reference import reference,key,network
class T(unittest.TestCase):
 def test_actual_pattern(self):self.assertEqual(reference([17.875,17.875,21.,20.625,17.75]),[2,3,1,0,4])
 def test_many_ties(self):
  for row in [[3.]*13,[5,5,4,4,4,4,3,2],[9,8,7,6,6,6,6],[0.,-0.,0.,-0.,0.,-0.]]:
   ids=reference(row);self.assertEqual(len(set(ids)),5);self.assertEqual(sorted([key(row[i])for i in ids],reverse=True),sorted(map(key,row),reverse=True)[:5])
 def test_union(self):
  r=random.Random(47)
  for n in [5,15,1025,2107]:
   row=[float(r.randrange(-4,5))for _ in range(n)];c=[]
   for b in range(0,n,19):c+=sorted(range(b,min(b+19,n)),key=lambda i:(-key(row[i]),i))[:5]
   chosen=sorted(c,key=lambda i:(-key(row[i]),i))[:5];threshold=min(key(row[i])for i in chosen);order=sorted([i for i in chosen if key(row[i])>threshold])+sorted([i for i in chosen if key(row[i])==threshold])
   self.assertEqual(network([row[i]for i in order],order),reference(row))
 def test_reject(self):
  for x in [float('nan'),float('inf'),float('-inf')]:
   with self.assertRaises(AssertionError):reference([1,2,3,4,x])
 def test_signed_zero(self):self.assertGreater(key(0.),key(-0.))
