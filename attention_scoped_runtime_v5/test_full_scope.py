import unittest,json,copy
from pathlib import Path
from .result_gate import audit
from .adapter import census,MODULES
P=Path(__file__).resolve().parent.parent
class FullScope(unittest.TestCase):
 def fixture(self):
  r=json.loads((P/'attention_scoped_runtime_v4/attentionScoped04/RESULT.json').read_text())
  for key,native in (('shadow_request',128),('replacement_request',0)):
   d=r[key]['selected_dispatch'];d.update(supported_calls=128,tileir_gate_calls=128,native_gate_calls=native,explicit_native_exceptions=0)
   for row in d['rows']:row.update(supported=True,tile_calls=1,native_calls=int(key=='shadow_request'))
  return r
 def test_full_fixture(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_old_partial_rejected(self):
  r=json.loads((P/'attention_scoped_runtime_v4/attentionScoped04/RESULT.json').read_text());self.assertFalse(audit(r)['passed'])
 def test_late_replacement_native_rejected(self):
  r=self.fixture();r['replacement_request']['selected_dispatch']['rows'][-1]['native_calls']=1;self.assertFalse(audit(r)['passed'])
 def test_each_late_layer_required(self):
  rows=[dict(module=m,length=L,supported=True,passed=True) for m in MODULES for L in range(53,69)]
  for m in MODULES:
   for L in (65,66,67,68):
    bad=[x for x in rows if (x['module'],x['length'])!=(m,L)]
    with self.assertRaises(AssertionError):census(bad)
 def test_numerical_gate_unchanged(self):
  r=self.fixture();r['public_logprob_gate']['max_abs']=1e-8;self.assertFalse(audit(r)['passed'])
 def test_no_replace_native_dispatch(self):
  s=Path(__file__).with_name('adapter.py').read_text();self.assertIn("if self.mode=='shadow':\n   result=orig",s);self.assertNotIn("or not ctx['supported']",s)
if __name__=='__main__':unittest.main()
