import unittest,copy
from .modules_predicates import audit,expected_rows
class Tests(unittest.TestCase):
 def test_failure_injection(self):
  r=dict(complete=True,records=[dict(spec,passed=True,readonly=True,native=spec['expected'],candidate=spec['expected'])for spec in expected_rows()])
  self.assertTrue(audit(r))
  for field in ('candidate','readonly','passed'):
   bad=copy.deepcopy(r);bad['records'][0][field]=False;self.assertFalse(audit(bad))
  bad=copy.deepcopy(r);bad['records'].pop();self.assertFalse(audit(bad))
 def test_identity_duplicates_order_wrongcase(self):
  r=dict(complete=True,records=[dict(s,passed=True,readonly=True,native=s['expected'],candidate=s['expected'])for s in expected_rows()])
  for change in ('duplicate','reverse','wrongcase'):
   bad=copy.deepcopy(r)
   if change=='duplicate':bad['records'][1]=copy.deepcopy(bad['records'][0])
   elif change=='reverse':bad['records'].reverse()
   else:bad['records'][0]['case']='not-a-fixture'
   self.assertFalse(audit(bad))
if __name__=='__main__':unittest.main()
