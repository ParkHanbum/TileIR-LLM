import unittest,ast,json
from pathlib import Path
from .run import check_manifest,result_gate
class T(unittest.TestCase):
 def test_manifest(self):self.assertEqual(len(check_manifest(json.loads((Path(__file__).parent/'MANIFEST.json').read_text()))),3)
 def test_gate(self):
  r=dict(complete=True,records=[dict(call_index=c,arm=a,repeat=i,passed=True)for c in [0,15,6]for a in ['native','tile']for i in [0,1]])
  self.assertTrue(result_gate(r)['raw_pass']);r['records'][-1]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast_api(self):
  p=Path(__file__).parent.parent
  for d in ['tileir_vllm_top5_v2','top5_qualification_v2']:
   for f in (p/d).glob('*.py'):ast.parse(f.read_text())
  s=(p/'tileir_vllm_top5_v2/kernels.py').read_text();self.assertEqual(s.count('direction='),15);self.assertEqual(s.count('ct.cat('),45);self.assertNotIn('3050',s);self.assertNotIn('854',s)
