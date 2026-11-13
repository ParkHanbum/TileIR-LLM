import unittest,ast,json
from pathlib import Path
from .run import result_gate,order
class T(unittest.TestCase):
 def test_order(self):self.assertEqual(len(order()),72);self.assertEqual(len(set(order())),72)
 def test_gate(self):
  rows=[dict(case=c,arm=a,repeat=n,raw_equal=True,inputs_preserved=True,layout=True,return_none=True,repeat_exact=True,passed=True)for c,a,n in order()];r=dict(complete=True,records=rows);self.assertTrue(result_gate(r)['raw_pass']);rows[0]['inputs_preserved']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_manifest(self):
  m=json.loads(Path(__file__).with_name('MANIFEST.json').read_text());self.assertEqual(sum(not c['synthetic']for c in m['cases']),16);self.assertEqual(sum(c['synthetic']for c in m['cases']),2)
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
