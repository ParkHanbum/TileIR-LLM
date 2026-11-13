import unittest,ast
from pathlib import Path
from sampling_guard_synthetic_qualification_v1.cases import cases,expected
class Tests(unittest.TestCase):
 def test_cases(self):
  cs=cases();self.assertEqual(len(cs),118)
  for c in cs:
   self.assertIs(type(expected(c)),bool)
   self.assertTrue(all(0<=i<c['rows']*c['cols']for i,w in c['changes']))
 def test_expected(self):
  self.assertTrue(expected(dict(dtype='torch.float32',rows=2,cols=1,fill=0,changes=[[0,0x7f800001]],mode='has_nan')))
  self.assertFalse(expected(dict(dtype='torch.float32',rows=2,cols=1,fill=0,changes=[[0,0xff800000]],mode='has_posinf')))
 def test_gate(self):
  from sampling_guard_synthetic_qualification_v1.run import result_gate
  rows=[dict(kind='classification',case=i,spec=c,passed=True,native=[expected(c)]*2,candidate=[expected(c)]*2,readonly_checks=4)for i,c in enumerate(cases())]
  rows.extend(dict(kind='gather',dtype=d,native=[True]*2,candidate=[True]*2,readonly_checks=4,passed=True)for d in ('torch.float32','torch.bfloat16'))
  r=dict(complete=True,records=rows,final_synchronize=True,range_prerequisite=dict(passed=True))
  self.assertTrue(result_gate(r)['raw_pass']);r['records'][0]['readonly_checks']=3;self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
