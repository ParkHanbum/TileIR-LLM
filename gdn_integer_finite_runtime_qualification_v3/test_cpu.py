import unittest,json,copy,ast
from pathlib import Path
from .gates import audit
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def fixture(self):
  r=json.loads((P/'lean26_gdn_finite_runtime_v1/lean26GDNFinite01/RESULT.json').read_text());r['cache_preseed']['destination']=str(R/'cache_v1')
  r['shadow_request']=copy.deepcopy(r['lean_request']);r['shadow_request']['mode']='shadow'
  for arm,phase in [('shadow_request','shadow'),('lean_request','lean')]:
   c=r[arm]['selected_dispatch']['components']['gdn_prefill_core'];g=c['gdn_finite'];g['typed_integer_checks']=g.pop('retained_integer_checks');g['typed_integer_site_counts']=g.pop('retained_integer_site_counts');g.update(integer_gpu_kernels=0,integer_stream_synchronizations=72,integer_native_predicates=0)
   c['gdn_integer_proof']=dict(calls=72,native_diagnostic_predicates=72,candidate_gpu_kernels=0,candidate_stream_syncs=72,raw_readonly=True,phase=phase,selected_native_predicates=72 if phase=='shadow'else 0,native_oracle_predicates=72)
  return r
 def test_result_gate_fixture(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_wrong_census_rejected(self):
  r=self.fixture();r['lean_request']['selected_dispatch']['components']['gdn_prefill_core']['gdn_integer_proof']['calls']=71;self.assertFalse(audit(r)['passed'])
 def test_missing_sync_rejected(self):
  r=self.fixture();r['lean_request']['selected_dispatch']['components']['gdn_prefill_core']['gdn_finite']['integer_stream_synchronizations']=0;self.assertFalse(audit(r)['passed'])
 def test_raw_failure_rejected(self):
  r=self.fixture();r['shadow_request']['selected_dispatch']['components']['gdn_prefill_core']['gdn_integer_proof']['raw_readonly']=False;self.assertFalse(audit(r)['passed'])
 def test_shadow_before_replacement(self):
  s=(R/'run.py').read_text();self.assertLess(s.index("shadow=request('shadow')"),s.index("lean=request('lean')"))
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
