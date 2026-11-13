import unittest,json,copy,ast
from pathlib import Path
from . import gates
R=Path(__file__).parent;P=R.parent
class Composition(unittest.TestCase):
 def fixture(self):
  r=json.loads((P/'lean26_gdn_finite_runtime_v1/lean26GDNFinite01/RESULT.json').read_text())
  f=json.loads((P/'residual_four_runtime_replace_v1/residualFourReplace01/RESULT.json').read_text())
  for k in ('four_binding','four_replacement','uva_constructor_restored','uva_release'):r[k]=f[k]
  r['cache_preseed']['destination']=str(R/'cache_v1');return r
 def test_prior_actual_gates(self):self.assertTrue(gates.qualification(R)['passed'])
 def test_synthetic_composition_gate_only(self):self.assertTrue(gates.audit(self.fixture(),R)['passed'])
 def test_old_census_rejected(self):
  for k,v in [('selected_boundaries',5888),('selected_logical_operations',6056)]:
   r=self.fixture();r['lean_request']['selected_dispatch'][k]=v;self.assertFalse(gates.audit(r,R)['passed'])
 def test_guard_and_four_missing_rejected(self):
  r=self.fixture();r['lean_request']['selected_dispatch']['components']['gdn_prefill_core']['gdn_finite']['float_checks']=359;self.assertFalse(gates.audit(r,R)['passed'])
  r=self.fixture();r['four_replacement']['calls']['bias']=15;self.assertFalse(gates.audit(r,R)['passed'])
 def test_namespace_and_numeric_preservation(self):
  for name in ('ownership.py','oracle.py','cleanup.py'):
   self.assertEqual((R/name).read_bytes(),(P/'residual_four_runtime_replace_v1'/name).read_bytes())
  for name in ('run.py','adapter.py'):
   s=(R/name).read_text();self.assertIn('lean26_gdn_finite_runtime_v1',s);self.assertNotIn('from lean_public_composition_v2',s);ast.parse(s)
