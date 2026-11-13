import unittest,json,ast
from pathlib import Path
from .run import validate_manifest,result_gate,checked_extent,MAX_BACKING_BYTES
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def test_actual_manifest(self):
  m=json.loads((P/'attention_failure_native_capture_v3/attentionFailureNative03/bank/MANIFEST.json').read_text());s=validate_manifest(m);self.assertEqual(len(s),1);self.assertEqual(s[0]['lengths'],[53]*32)
  m['samples'][0]['prior_q_logical_match']=False
  with self.assertRaises(AssertionError):validate_manifest(m)
 def test_gate(self):
  self.assertFalse(result_gate({})['raw_pass']);r=dict(complete=True,records=[dict(passed=True) for _ in range(5)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][-1]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_arithmetic_preserved(self):
  src=(P/'tileir_paged_attention_v7/kernels.py').read_text();new=(R/'kernels.py').read_text()
  restored=new.replace('O,ST,SS,W:ct.Constant[int]','O,W:ct.Constant[int]')
  restored='\n'.join(x for x in restored.splitlines() if not x.startswith((' ct.scatter(ST,',' ct.scatter(SS,')))+'\n'
  restored=restored.replace(' numerator=ct.reshape(ct.extract(pv,(0,0),(1,256)),(256,))\n result=numerator/den',' result=ct.reshape(ct.extract(pv,(0,0),(1,256)),(256,))/den')
  self.assertEqual(restored,src)
 def test_bindings(self):
  s=(R/'run.py').read_text();self.assertIn("('native','native','clean','clean','instrumented')",s);self.assertIn("prior_tile if kind=='clean' else target",s);self.assertNotIn("kind=='clean' or",s)
 def test_telemetry(self):
  s=(R/'kernels.py').read_text();self.assertEqual(s.count('ct.scatter(ST,'),4);self.assertEqual(s.count('ct.scatter(SS,'),4);self.assertIn('ct.where(valid,scores,0.0)',s)
 def test_absolute_extent(self):
  m=json.loads((P/'attention_failure_native_capture_v3/attentionFailureNative03/bank/MANIFEST.json').read_text());s=m['samples'][0]
  self.assertEqual(checked_extent(s['tensors']['k'],s['tensors']['v'])*2,1340866560)
  self.assertEqual(MAX_BACKING_BYTES,1610612736)
 def test_bound_and_alias(self):
  k=dict(shape=[1,1,1,1],stride=[1,1,1,1],storage_offset=MAX_BACKING_BYTES//2-257);v=dict(k,storage_offset=k['storage_offset']+256)
  self.assertEqual(checked_extent(k,v)*2,MAX_BACKING_BYTES)
  k['storage_offset']+=1;v['storage_offset']+=1
  with self.assertRaises(AssertionError):checked_extent(k,v)
  k['storage_offset']=0;v['storage_offset']=255
  with self.assertRaises(AssertionError):checked_extent(k,v)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
