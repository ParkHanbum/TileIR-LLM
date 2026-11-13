import copy,unittest
from .recovery import validate_receipts
class RecoveryTests(unittest.TestCase):
 def valid(self):
  return dict(returncode=0,qualification_complete=True,result_gate=dict(passed=True),child_absent=True,
   remaining_process_group=[],lock_reacquired=True,after=dict(gpu='observed'),finished=1,result_sha256='abc',pid=123)
 def test_success(self):self.assertEqual(validate_receipts('EXIT=0\n',self.valid(),dict(complete=True),'abc'),123)
 def test_receipt_failures(self):
  for key,value in [('returncode',1),('returncode',False),('qualification_complete',False),('remaining_process_group',[123]),('child_absent',False),('lock_reacquired',False),('result_sha256','wrong'),('timeout',True),('error','failure'),('pid',0)]:
   r=self.valid();r[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(AssertionError):validate_receipts('EXIT=0\n',r,dict(complete=True),'abc')
 def test_outer_and_result_required(self):
  for outer in ['','0\n','EXIT=1\n','EXIT=0']:
   with self.subTest(outer=outer),self.assertRaises(AssertionError):validate_receipts(outer,self.valid(),dict(complete=True),'abc')
  with self.assertRaises(AssertionError):validate_receipts('EXIT=0\n',self.valid(),dict(complete=False),'abc')
if __name__=='__main__':unittest.main()
