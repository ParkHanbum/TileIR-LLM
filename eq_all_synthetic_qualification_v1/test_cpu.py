import unittest,copy,sys,ast
from pathlib import Path
from eq_all_synthetic_qualification_v1.cases import cases,expected,WORDS
from eq_all_synthetic_qualification_v1.run import result_gate
class Tests(unittest.TestCase):
 def test_edge_words(self):
  for word in (0,0x80000000,1,0x80000001,0x00800000,0x80800000,0x7fc12345,0xffc12345,0x7f800001,0xff800001,0x7f800000,0xff800000,0x3f7fffff,0x3f800000,0x3f800001):self.assertIn(word,WORDS)
 def test_contracts(self):
  rows=cases();self.assertEqual(len(rows),190)
  self.assertEqual({r['offset']for r in rows if r['kind']=='broadcast'},{0,1})
  self.assertEqual({r['index']for r in rows if r['n']==1<<24},{0,1<<23,(1<<24)-1})
  self.assertTrue(any(r['n']==1025 and r['index']==1024 for r in rows))
 def test_result_gate(self):
  rows=[dict(spec=r,passed=True,native=[expected(r)]*2,candidate=[expected(r)]*2,readonly_checks=4,stream_unchanged=True)for r in cases()]
  result=dict(complete=True,records=rows,final_synchronize=True)
  self.assertTrue(result_gate(result)['raw_pass']);bad=copy.deepcopy(result);bad['records'][0]['readonly_checks']=3;self.assertFalse(result_gate(bad)['raw_pass'])
  bad=copy.deepcopy(result);bad['records'].pop();self.assertFalse(result_gate(bad)['raw_pass'])
 def test_lifecycle_source(self):
  r=Path(__file__).parent;s=(r/'run.py').read_text();self.assertIn('signal.alarm(585)',s);self.assertIn('signal.SIGALRM,timeout',s)
  launch=(r/'launch.py').read_text();self.assertIn('LOCK_NB',launch);self.assertIn('child.wait(timeout=600)',launch);self.assertIn("rec['after']=h.admit()",launch)
 def test_parse_no_import(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
