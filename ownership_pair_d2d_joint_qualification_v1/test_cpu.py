import unittest,json,ast,copy
from pathlib import Path
from .gates import audit
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def fixture(self):
  r=json.loads((P/'whole_lean_extensions_runtime_v5/wholeLeanExtensions05/RESULT.json').read_text());r['cache_preseed']['destination']=str(R/'cache_v1')
  r['pair_synthetic']=dict(cases=48,equal_cases=8,unequal_cases=40,all_inputs_readonly=True,passed=True)
  origin=dict(code_identity_verified=True,torch_closure_identity_verified=True);r['pair_callback_origin']=origin
  rows=[];observed=[]
  for i in range(68):
   dtype='torch.uint64'if i%2==0 else 'torch.int64';rows.append(dict(dtype=dtype,shape=[4],stride=[1],native=True,candidate=True,whole_inputs_readonly=True))
   t=dict(dtype=dtype,device='cuda:0',shape=[4],stride=[1],offset=0,storage_bytes=32,requires_grad=False,is_neg=False,is_conj=False,layout='torch.strided');observed.append(dict(a=t,b=t))
  r['pair_shadow']=dict(table_checks=34,pair_checks=68,native_diagnostic_oracles=68,pointer_values_serialized=False,rows=rows,actual_metadata=observed,actual_callback_origin=origin)
  r['copy_shadow']=dict(complete=True,calls=24,model_copies_native=True,actual_replacement=False,rows=[dict(layer=str(i),native_calls=1,candidate_calls=1,raw_exact=True,outside_preserved=True,source_readonly=True,poison_independent=True)for i in range(24)])
  r['copy_shadow_restored']=True;r['pair_shadow_restored']=dict(restored=True,original_callback_identity=True);return r
 def test_success_fixture(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_copy_poison_missing_hold(self):
  r=self.fixture();r['copy_shadow']['rows'][0]['poison_independent']=False;self.assertFalse(audit(r)['passed'])
 def test_origin_missing_hold(self):
  r=self.fixture();r['pair_shadow']['actual_callback_origin']['code_identity_verified']=False;self.assertFalse(audit(r)['passed'])
 def test_metadata_wrong_extent_hold(self):
  r=self.fixture();r['pair_shadow']['actual_metadata'][0]['a']['storage_bytes']=64;self.assertFalse(audit(r)['passed'])
 def test_synthetic_failure_hold(self):
  r=self.fixture();r['pair_synthetic']['passed']=False;self.assertFalse(audit(r)['passed'])
 def test_one_request_and_cleanup_order(self):
  s=(R/'run.py').read_text();self.assertEqual(s.count("lean=request('lean')"),1)
  self.assertLess(s.index('copy_shadow.close()'),s.index('ext.close_post()'))
  self.assertLess(s.index('pair_shadow.close()'),s.index('ext.close_post()'))
  self.assertLess(s.index("result['pair_synthetic']=pair_synthetic(torch)"),s.index('engine=LLMEngine.from_engine_args'))
  for p in R.glob('*.py'):ast.parse(p.read_text())
