import unittest,ast
from pathlib import Path
from .cases import CASES,intervals
from zero_kv_phase2_typed_plan_v1.geometry import partition,ContractError,CHUNK
class Tests(unittest.TestCase):
 def test_cases(self):
  for c in CASES:
   chunks=partition(c['words']*4,intervals(c));self.assertEqual(sum(b-a for k,a,b in chunks),c['words']*4);self.assertTrue(all(b-a<=CHUNK for k,a,b in chunks))
  large=partition(CASES[-1]['words']*4,intervals(CASES[-1]));self.assertIn(('target',4,4+CHUNK),large)
 def test_empty(self):self.assertEqual(partition(16,[]),[('outside',0,16)])
 def test_invalid_prelaunch(self):
  for iv in [[(-4,4)],[(0,20)],[(0,1)],[(8,4)]]:
   with self.assertRaises(ContractError):partition(16,iv)
 def test_source_only(self):
  r=Path(__file__).parent
  self.assertFalse((r/'cache_v1').exists());self.assertEqual(list(r.glob('CACHE-PRESEED*')),[])
  for p in r.glob('*.py'):ast.parse(p.read_text())
class APIPrelaunchTests(unittest.TestCase):
 def test_empty_and_invalid_invoke_no_backend(self):
  from .api import zero_owned_intervals
  class A:
   size=16
   def check(self):pass
  self.assertEqual(zero_owned_intervals(A(),[],torch=None,ct=None),dict(target_bytes=0,launches=0))
  for bad in [[(-4,4)],[(0,20)],[(0,1)],[(8,4)]]:
   with self.assertRaises(ContractError):zero_owned_intervals(A(),bad,torch=None,ct=None)

class ResultGateTests(unittest.TestCase):
 def test_lifecycle_and_all_cases_required(self):
  from .gates import audit
  from .cases import CASES
  rows=[dict(case=c['name'],arm=a,expected_full_raw=True,readonly_tables_raw=True,same_current_stream=True,cross_arm_equal=True)for c in CASES for a in ('native','tile')]
  for i,c in enumerate(CASES):
   for j,arm in enumerate(('native','tile')):
    n=int(bool(c['ids']))if arm=='native'else 0;t=[1,2,0,2][i]if arm=='tile'else 0
    rows[2*i+j].update(candidate_launches=t,source_calls=dict(native_attempted=n,native_returned=n,tile=dict(attempted=t,returned=t)))
  r=dict(complete=True,synchronized=True,precision_initial=dict(bf16=True,split_k=True),precision_final=dict(bf16=True,split_k=True),records=rows)
  self.assertTrue(audit(r)['passed'])
  r['synchronized']=False;self.assertFalse(audit(r)['passed']);r['synchronized']=True
  r['records']=rows[:-1];self.assertFalse(audit(r)['passed'])
 def test_runner_cache_provenance(self):
  source=(Path(__file__).parent/'run.py').read_text()
  self.assertIn('cache=preflight(R)',source);self.assertIn('cache_preseed=cache',source)

class CallCountRegression(unittest.TestCase):
 def test_wrong_return_and_empty_launch_rejected(self):
  from .gates import call_counts_valid
  import copy
  rows=[]
  for i,c in enumerate(CASES):
   for arm in ('native','tile'):
    n=int(bool(c['ids']))if arm=='native'else 0;t=[1,2,0,2][i]if arm=='tile'else 0
    rows.append(dict(case=c['name'],arm=arm,candidate_launches=t,source_calls=dict(native_attempted=n,native_returned=n,tile=dict(attempted=t,returned=t))))
  self.assertTrue(call_counts_valid(rows))
  for idx,field,value in [(1,'returned',0),(5,'attempted',1),(7,'returned',True)]:
   bad=copy.deepcopy(rows);bad[idx]['source_calls']['tile'][field]=value;self.assertFalse(call_counts_valid(bad))
  bad=copy.deepcopy(rows);bad[4]['source_calls']['native_attempted']=1;self.assertFalse(call_counts_valid(bad))
