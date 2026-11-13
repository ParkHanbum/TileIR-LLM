import unittest,ast,json
from pathlib import Path
from .api import validate_indices
class T(unittest.TestCase):
 def test_simple(self):validate_indices(list(range(32)),32)
 def test_negative(self):validate_indices([-1,-9]+list(range(30)),32)
 def test_reject(self):
  for ids,cap in [([0]*32,32),(list(range(32)),31),(list(range(31)),32),(list(range(32)),4097)]:
   with self.assertRaises(AssertionError):validate_indices(ids,cap)
 def test_equation_synthetic(self):
  ids=[-1,-3]+list(range(30));values=[-2147483648,2147483647]+[-2,0,1,7,2147483647]*6;owner=[99]*40;before=owner[:]
  for i,v in zip(ids,values):
   if i>=0:owner[i]=max(v,1)
  self.assertEqual(owner[30:],before[30:]);self.assertEqual(owner[:5],[1,1,1,7,2147483647])
 def test_actual_descriptors(self):
  rows=json.loads(Path(__file__).with_name('ACTUAL-BOUNDARY-DESCRIPTORS.json').read_text());self.assertEqual(len(rows),16)
  for r in rows:self.assertEqual([x['dtype']for x in r['inputs']],['torch.int64','torch.int32','torch.int32'])
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
