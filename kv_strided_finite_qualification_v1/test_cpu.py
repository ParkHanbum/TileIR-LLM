import unittest,ast,sys
from pathlib import Path
from .run import result_gate
from .probe import CASES
class T(unittest.TestCase):
 def valid(self):
  rows=[]
  for p in ('prefill','decode'):
   for c in CASES:
    for a in ('native','tile'):
     for i in range(2):
      good=c not in ('first_nan','last_nan','first_posinf','last_neginf')
      rows.append(dict(phase=p,case=c,arm=a,repeat=i,value=good,expected=good,passed=True,owner_preserved=True))
  return dict(complete=True,gpu_executed=True,precision_before=[True,True],precision_after=[True,True],records=rows)
 def test_gate(self):
  r=self.valid();self.assertTrue(result_gate(r)['raw_pass']);r['records'][0]['value']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_missing_and_cleanup(self):
  r=self.valid();r['records'].pop();self.assertFalse(result_gate(r)['raw_pass'])
  r=self.valid();r['cleanup_error']='sync';self.assertFalse(result_gate(r)['raw_pass'])
 def test_source(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertFalse(any(k in sys.modules for k in ('torch','vllm','cuda.tile')))
if __name__=='__main__':unittest.main()
