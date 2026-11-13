import unittest,json,ast
from pathlib import Path
from .api import validate_contract
class T(unittest.TestCase):
 def test_observed(self):
  for n,v in ((1696,False),(32,False),(0,True)):self.assertEqual(validate_contract((n,),(1,),'torch.bool',0,2048,v),n)
 def test_offset(self):self.assertEqual(validate_contract((32,),(1,),'torch.bool',17,64,False),32)
 def test_reject(self):
  for args in [((32,),(1,),'torch.bool',0,64,True),((31,),(1,),'torch.bool',0,64,False),((32,),(2,),'torch.bool',0,64,False),((32,),(1,),'torch.int32',0,128,False),((32,),(1,),'torch.bool',40,64,False),((32,),(1,),'torch.bool',0,64,0)]:
   with self.assertRaises(AssertionError):validate_contract(*args)
 def test_sentinel_index_coverage(self):
  for n in (0,32,1696):
   owner=[True]*(n+31);before=owner[:];offset=13
   indices=[b*256+i for b in range((n+255)//256)for i in range(256)if b*256+i<n]
   for i in indices:owner[offset+i]=False
   self.assertEqual(indices,list(range(n)));self.assertEqual(owner[:offset],before[:offset]);self.assertEqual(owner[offset+n:],before[offset+n:]);self.assertEqual(owner[offset:offset+n],[False]*n)
 def test_actual_counts(self):
  rows=json.loads(Path(__file__).with_name('OBSERVED-CPU-OPERATORS.json').read_text());self.assertEqual(sum(x['args']['Input Dims'][0]==[1696] for x in rows),1);self.assertEqual(sum(x['args']['Input Dims'][0]==[32]for x in rows),15);self.assertEqual(sum(x['args']['Input Dims'][0]==[0]for x in rows),16)
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
